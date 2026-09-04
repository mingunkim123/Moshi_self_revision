from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tarfile

import pytest

from experiments.self_repair.mechanistic.core import (
    ContractError,
    MODEL_REPO,
    MODEL_REVISION,
    PatchCell,
    package_tree,
    read_json,
    read_jsonl,
    sha256_value,
    verify_archive,
    write_json,
    write_jsonl,
)
from experiments.self_repair.mechanistic.verification import (
    package_checksum_manifest,
    verify_artifact_manifest,
    verify_or_create_artifact_manifest,
    verify_package_checksums,
    verify_patch_artifacts,
)
from experiments.self_repair.mechanistic.readiness import (
    AUTHORIZATION_STAGES,
    build_authorization_artifact,
    target_binding_sha256,
)


SHA = "a" * 64
COMMIT = "b" * 40


def _provenance(run_hash: str, readout_hash: str, *, synthetic: bool) -> dict[str, object]:
    return {
        "code_commit": COMMIT,
        "harness_version": "test",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "config_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "encoded_manifest_sha256": None if synthetic else "3" * 64,
        "anchor_map_sha256": "4" * 64,
        "readout_sha256": readout_hash,
        "scan_spec_sha256": None if synthetic else "5" * 64,
        "selection_file_sha256": None,
        "data_sha256": None if synthetic else "6" * 64,
        "run_identity_sha256": run_hash,
    }


def _assignment(donor: str, recipient: str) -> dict[str, object]:
    return {
        "requested_arm": "clean_current",
        "relation": "clean_current",
        "donor_trial_id": donor,
        "recipient_trial_id": recipient,
        "selection_tier": "exact_same_scenario_direction_speaker_values",
        "scenario_matched": True,
        "direction_matched": True,
        "speaker_matched": True,
        "current_value_matched": True,
        "stale_value_matched": True,
    }


def _scan_fixture(root: Path, *, synthetic: bool = True) -> tuple[Path, dict[str, object]]:
    directory = root / "discovery" / "residual"
    directory.mkdir(parents=True)
    readout_hash = "7" * 64
    run_hash = "8" * 64
    cell = PatchCell(
        run_hash,
        "clean-1",
        "repair-1",
        "resid_post",
        3,
        None,
        (4,),
        (4,),
        readout_hash,
    )
    provenance = _provenance(run_hash, readout_hash, synthetic=synthetic)
    assignment = _assignment(cell.donor_trial_id, cell.recipient_trial_id)
    plan_row = {
        "schema_version": "1.0.0",
        "cell_id": cell.cell_id,
        **asdict(cell),
        "donor_arm": "clean_current",
        "relation": "clean_current",
        "anchor": "new_end",
        "query_end_frame_exclusive": 12,
        "donor_assignment": assignment,
        "path": None,
    }
    result_row = {
        **plan_row,
        "status": "completed",
        "role": "discovery",
        "scenario_id": "scenario-1",
        "direction_id": "boston-to-seattle",
        "speaker_id": "speaker-1",
        "old_value": "Boston",
        "new_value": "Seattle",
        "source_frame": 4,
        "target_frame": 4,
        "attempt_index": 1,
        "readout_id": "query_value",
        "provenance": provenance,
        "synthetic": synthetic,
        "baseline_M": 0.25,
        "patched_M": 0.75,
        "delta_M": 0.5,
        "feedback_sha256": "9" * 64,
        "path_evidence": None,
    }
    write_jsonl(directory / "planned_cells.jsonl", [plan_row])
    write_json(directory / "scan_plan.json", {
        "schema_version": "1.0.0",
        "kind": "residual",
        "role": "discovery",
        "planned_cell_count": 1,
        "planned_cells_sha256": sha256_value([plan_row]),
        "result_uri": "residual_patch_results.jsonl",
        "provenance": provenance,
    })
    write_jsonl(directory / "residual_patch_results.jsonl", [result_row])
    write_json(directory / "cells" / f"{cell.cell_id}.json", result_row)
    (directory / "failures").mkdir()
    write_jsonl(directory / "failures.jsonl", [])
    write_json(directory / "resume_summary.json", {
        "planned_cells": 1,
        "completed_cells": 1,
        "unresolved_failed_cells": 0,
        "failure_attempts": 0,
        "duplicate_cells": 0,
        "skipped_existing_cells": 0,
        "run_identity_sha256": run_hash,
        "planned_cells_sha256": sha256_value([plan_row]),
    })
    return directory, result_row


def test_existing_artifact_manifest_is_verified_not_regenerated(tmp_path: Path) -> None:
    write_json(tmp_path / "reports" / "summary.json", {"status": "synthetic"})
    count, created = verify_or_create_artifact_manifest(tmp_path)
    assert (count, created) == (1, True)
    manifest_path = tmp_path / "artifact_sha256.json"
    frozen = manifest_path.read_bytes()

    write_json(tmp_path / "reports" / "summary.json", {"status": "changed"})
    with pytest.raises(ContractError, match="byte count mismatch|SHA-256 mismatch"):
        verify_or_create_artifact_manifest(tmp_path)
    assert manifest_path.read_bytes() == frozen


def test_artifact_manifest_excludes_and_rejects_self_hash(tmp_path: Path) -> None:
    write_json(tmp_path / "result.json", {"ok": True})
    verify_or_create_artifact_manifest(tmp_path)
    value = read_json(tmp_path / "artifact_sha256.json")
    value["artifacts"].append({
        "uri": "artifact_sha256.json",
        "sha256": SHA,
        "bytes": 1,
    })
    with pytest.raises(ContractError, match="must not hash itself"):
        verify_artifact_manifest(tmp_path, value)


def test_patch_verifier_checks_exact_plan_atomic_rows_and_provenance(tmp_path: Path) -> None:
    _scan_fixture(tmp_path)
    assert verify_patch_artifacts(tmp_path) == (1, True)

    result_path = tmp_path / "discovery/residual/residual_patch_results.jsonl"
    row = read_json(result_path)
    # read_json on JSONL with one row is still a valid object; changing the
    # merge without changing the atom must be rejected.
    row["delta_M"] = 0.4
    write_jsonl(result_path, [row])
    with pytest.raises(ContractError, match="delta_M disagrees"):
        verify_patch_artifacts(tmp_path)


def test_patch_verifier_requires_matching_empirical_go_authorization(tmp_path: Path) -> None:
    _scan_fixture(tmp_path, synthetic=False)
    with pytest.raises(ContractError, match="no matching GO authorization"):
        verify_patch_artifacts(tmp_path)


def test_patch_verifier_accepts_exact_hash_bound_empirical_authorization(tmp_path: Path) -> None:
    _, row = _scan_fixture(tmp_path, synthetic=False)
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    binding = {
        "code_commit": provenance["code_commit"],
        "code_sha256": sha256_value({"git_commit": provenance["code_commit"]}),
        "model_repo": provenance["model_repo"],
        "model_revision": provenance["model_revision"],
        "model_sha256": sha256_value({
            "repo": provenance["model_repo"],
            "revision": provenance["model_revision"],
        }),
        "manifest_sha256": provenance["manifest_sha256"],
        "data_sha256": provenance["data_sha256"],
        "encoded_manifest_sha256": provenance["encoded_manifest_sha256"],
        "config_sha256": provenance["config_sha256"],
        "scan_spec_sha256": provenance["scan_spec_sha256"],
    }
    # The PatchCell run identity is the target-binding hash for every paid scan.
    binding_hash = target_binding_sha256(binding)
    old_hash = str(row["run_identity_sha256"])
    assert old_hash != binding_hash

    # Rewrite the fixture with the correct content-addressed cell identity.
    directory = tmp_path / "discovery/residual"
    old_plan = read_jsonl(directory / "planned_cells.jsonl")[0]
    cell = PatchCell(
        binding_hash,
        str(row["donor_trial_id"]),
        str(row["recipient_trial_id"]),
        str(row["component"]),
        int(row["layer"]),
        row["head"],
        tuple(row["source_frames"]),
        tuple(row["target_frames"]),
        str(row["readout_sha256"]),
    )
    for value in (row, old_plan):
        value["run_identity_sha256"] = binding_hash
        value["cell_id"] = cell.cell_id
    provenance["run_identity_sha256"] = binding_hash
    row["provenance"] = provenance
    write_jsonl(directory / "planned_cells.jsonl", [old_plan])
    plan = read_json(directory / "scan_plan.json")
    plan["planned_cells_sha256"] = sha256_value([old_plan])
    plan["provenance"] = provenance
    write_json(directory / "scan_plan.json", plan)
    write_jsonl(directory / "residual_patch_results.jsonl", [row])
    # The original atomic filename is the old cell_id, not the old run hash.
    for path in (directory / "cells").glob("*.json"):
        path.unlink()
    write_json(directory / "cells" / f"{cell.cell_id}.json", row)
    resume = read_json(directory / "resume_summary.json")
    resume["run_identity_sha256"] = binding_hash
    resume["planned_cells_sha256"] = plan["planned_cells_sha256"]
    write_json(directory / "resume_summary.json", resume)

    evidence = {"target_binding_sha256": binding_hash}
    assessment = {
        "decision": "GO",
        "blockers": [],
        "stages": [{"stage": stage, "decision": "GO"} for stage in AUTHORIZATION_STAGES],
    }
    authorization = build_authorization_artifact(binding, evidence, assessment)
    write_json(tmp_path / "preflight/paid_scan_authorization.json", authorization)
    assert verify_patch_artifacts(tmp_path) == (1, False)


def test_patch_verifier_fails_closed_on_missing_cell(tmp_path: Path) -> None:
    directory, row = _scan_fixture(tmp_path)
    second_cell = PatchCell(
        "8" * 64, "clean-2", "repair-2", "resid_post", 3, None,
        (4,), (4,), "7" * 64,
    )
    second = {
        key: value for key, value in row.items()
        if key not in {
            "status", "role", "scenario_id", "direction_id", "speaker_id", "old_value",
            "new_value", "source_frame", "target_frame", "readout_id", "provenance",
            "synthetic", "baseline_M", "patched_M", "delta_M", "feedback_sha256",
            "path_evidence",
        }
    }
    second.update({"cell_id": second_cell.cell_id, **asdict(second_cell)})
    second["donor_assignment"] = _assignment("clean-2", "repair-2")
    plans = [
        {key: value for key, value in row.items() if key in second},
        second,
    ]
    # Reuse the exact original plan rather than deriving it from the richer
    # result row.
    plans[0] = read_jsonl(directory / "planned_cells.jsonl")[0]
    write_jsonl(directory / "planned_cells.jsonl", plans)
    plan = read_json(directory / "scan_plan.json")
    plan["planned_cell_count"] = 2
    plan["planned_cells_sha256"] = sha256_value(plans)
    write_json(directory / "scan_plan.json", plan)
    with pytest.raises(ContractError, match="coverage is incomplete"):
        verify_patch_artifacts(tmp_path)


def test_public_package_excludes_all_private_classes_and_absolute_paths(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_json(run / "reports" / "public.json", {"status": "synthetic"})
    write_json(run / "reports" / "host-path.json", {"source": "/mnt/run/input.json"})
    write_json(run / "blind_map.json", {"A": "patched"})
    write_json(run / "activations" / "capture.dat", {"tensor": [1, 2, 3]})
    write_json(run / "model_cache" / "metadata.json", {"repo": MODEL_REPO})
    (run / "audio").mkdir(parents=True)
    (run / "audio" / "response.ogg").write_bytes(b"OggS")
    write_json(run / "artifact_sha256.json", {"schema_version": "1.0.0", "artifacts": []})
    public = tmp_path / "public.tar.gz"
    private = tmp_path / "private.tar.gz"
    package_tree(run, public, private)
    verify_archive(public, public=True)
    verify_archive(private, public=False)
    with tarfile.open(public, "r:gz") as archive:
        assert archive.getnames() == ["reports/public.json"]
    with tarfile.open(private, "r:gz") as archive:
        assert {
            "artifact_sha256.json",
            "reports/host-path.json",
            "blind_map.json",
            "activations/capture.dat",
            "model_cache/metadata.json",
            "audio/response.ogg",
        } <= set(archive.getnames())


def test_package_checksums_are_reverified_after_write(tmp_path: Path) -> None:
    public = tmp_path / "public.tar.gz"
    private = tmp_path / "private.tar.gz"
    public.write_bytes(b"public")
    private.write_bytes(b"private")
    manifest = package_checksum_manifest(public, private)
    verify_package_checksums(manifest, public_path=public, private_path=private)
    public.write_bytes(b"tampered")
    with pytest.raises(ContractError, match="byte count mismatch|SHA-256 mismatch"):
        verify_package_checksums(manifest, public_path=public, private_path=private)


def test_package_cli_requires_prior_run_verification(tmp_path: Path) -> None:
    from experiments.self_repair.mechanistic.scripts._cli import package_results

    run = tmp_path / "run"
    write_json(run / "reports/summary.json", {"status": "synthetic"})
    with pytest.raises(ContractError, match="verify_mechanistic_run.py"):
        package_results([
            "--run-root", str(run),
            "--public-output", str(tmp_path / "public.tar.gz"),
            "--private-output", str(tmp_path / "private.tar.gz"),
            "--synthetic",
        ])


def test_public_archive_verifier_rejects_forbidden_path_even_if_handcrafted(tmp_path: Path) -> None:
    source = tmp_path / "weights.bin"
    source.write_bytes(b"checkpoint")
    archive_path = tmp_path / "public.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname="model_cache/weights.bin")
    with pytest.raises(ContractError, match="private artifact leaked"):
        verify_archive(archive_path, public=True)


def test_archive_verifier_fails_cleanly_on_corrupt_input(tmp_path: Path) -> None:
    archive_path = tmp_path / "broken.tar.gz"
    archive_path.write_bytes(b"not a gzip tar archive")
    with pytest.raises(ContractError, match="cannot verify result archive"):
        verify_archive(archive_path, public=True)
