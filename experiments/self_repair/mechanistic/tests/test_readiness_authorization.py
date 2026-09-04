from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.self_repair.mechanistic.core import (
    ContractError,
    MODEL_REPO,
    MODEL_REVISION,
    sha256_file,
    sha256_value,
)
from experiments.self_repair.mechanistic.readiness import (
    ReadinessError,
    build_authorization_artifact,
    target_binding_sha256,
    verify_authorization_artifact,
)
from experiments.self_repair.mechanistic.scripts.readiness_cli import (
    assemble_evidence,
    build_target_binding_from_files,
    run_gpu_canary,
    select_canary_manifest,
    validate_scan_execution,
)


COMMIT = "1" * 40


def _binding() -> dict[str, str]:
    return {
        "code_commit": COMMIT,
        "code_sha256": sha256_value({"git_commit": COMMIT}),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_sha256": sha256_value({"repo": MODEL_REPO, "revision": MODEL_REVISION}),
        "manifest_sha256": "2" * 64,
        "data_sha256": "3" * 64,
        "encoded_manifest_sha256": "4" * 64,
        "config_sha256": "5" * 64,
        "scan_spec_sha256": "6" * 64,
    }


def _assessment() -> dict:
    return {
        "schema_version": "1.0.0",
        "decision": "GO",
        "stages": [{"stage": name, "decision": "GO", "blockers": []}
                   for name in (
                       "static_plan", "budget", "evidence_binding", "model_contract", "open_loop",
                       "conversation_canary", "gpu_canary", "paid_scan",
                   )],
        "blockers": [],
    }


def test_go_authorization_is_content_addressed_and_tamper_evident() -> None:
    binding = _binding()
    evidence = {"target_binding_sha256": target_binding_sha256(binding), "marker": "measured"}
    artifact = build_authorization_artifact(binding, evidence, _assessment())
    assert verify_authorization_artifact(artifact, binding) == binding

    tampered = deepcopy(artifact)
    tampered["evidence"]["marker"] = "invented"
    with pytest.raises(ReadinessError, match="authorization SHA-256"):
        verify_authorization_artifact(tampered, binding)

    wrong_target = deepcopy(binding)
    wrong_target["manifest_sha256"] = "7" * 64
    with pytest.raises(ReadinessError, match="different model/code/data/config/scan"):
        verify_authorization_artifact(artifact, wrong_target)


def test_no_go_assessment_cannot_be_used_as_authorization() -> None:
    binding = _binding()
    evidence = {"target_binding_sha256": target_binding_sha256(binding)}
    assessment = _assessment()
    assessment["decision"] = "NO_GO"
    assessment["blockers"] = [{"code": "missing_canary"}]
    artifact = build_authorization_artifact(binding, evidence, assessment)
    assert artifact["decision"] == "NO_GO"
    with pytest.raises(ReadinessError, match="decision is not GO"):
        verify_authorization_artifact(artifact, binding)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_file_binding_covers_source_and_encoded_data(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    manifest = tmp_path / "manifest.jsonl"
    encoded = tmp_path / "encoded.jsonl"
    scan_spec = tmp_path / "scan.json"
    _write_json(config, {
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "audio": {"sample_rate": 24_000, "mimi_frame_samples": 1_920},
    })
    manifest.write_text(json.dumps({
        "trial_id": "trial-1", "audio_sha256": "8" * 64,
        "sample_count": 1_920, "frame_count": 1,
    }) + "\n", encoding="utf-8")
    encoded.write_text(json.dumps({
        "trial_id": "trial-1", "source_audio_sha256": "8" * 64,
        "codes_sha256": "9" * 64, "shape": [1, 8, 1],
        "model_revision": MODEL_REVISION,
    }) + "\n", encoding="utf-8")
    _write_json(scan_spec, {"execution": {}})
    first = build_target_binding_from_files(
        config_path=config, manifest_path=manifest, encoded_manifest_path=encoded,
        scan_spec_path=scan_spec, code_commit=COMMIT, require_clean=False,
    )
    row = json.loads(encoded.read_text(encoding="utf-8"))
    row["codes_sha256"] = "a" * 64
    encoded.write_text(json.dumps(row) + "\n", encoding="utf-8")
    second = build_target_binding_from_files(
        config_path=config, manifest_path=manifest, encoded_manifest_path=encoded,
        scan_spec_path=scan_spec, code_commit=COMMIT, require_clean=False,
    )
    assert first["data_sha256"] != second["data_sha256"]
    assert first["encoded_manifest_sha256"] != second["encoded_manifest_sha256"]


def test_file_binding_rejects_prepared_only_cache_for_conversation(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    manifest = tmp_path / "manifest.jsonl"
    encoded = tmp_path / "encoded.jsonl"
    scan_spec = tmp_path / "scan.json"
    _write_json(config, {
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "audio": {"sample_rate": 24_000, "mimi_frame_samples": 1_920},
    })
    manifest.write_text(json.dumps({
        "trial_id": "trial-1", "audio_sha256": "8" * 64,
        "sample_count": 1_920, "frame_count": 1,
        "conversation_contract": {"user_frame_count": 1, "target_end_frame_count": 3},
    }) + "\n", encoding="utf-8")
    encoded.write_text(json.dumps({
        "trial_id": "trial-1", "source_audio_sha256": "8" * 64,
        "codes_sha256": "9" * 64, "shape": [1, 8, 1],
        "model_revision": MODEL_REVISION,
    }) + "\n", encoding="utf-8")
    _write_json(scan_spec, {"execution": {}})
    with pytest.raises(Exception, match="conversation cache lacks"):
        build_target_binding_from_files(
            config_path=config, manifest_path=manifest, encoded_manifest_path=encoded,
            scan_spec_path=scan_spec, code_commit=COMMIT, require_clean=False,
        )


def test_file_binding_accepts_only_canonical_conversation_shape_fields(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    manifest = tmp_path / "manifest.jsonl"
    encoded = tmp_path / "encoded.jsonl"
    scan_spec = tmp_path / "scan.json"
    _write_json(config, {
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "audio": {"sample_rate": 24_000, "mimi_frame_samples": 1_920},
    })
    source = {
        "trial_id": "trial-1", "audio_sha256": "8" * 64,
        "sample_count": 1_920, "frame_count": 1,
        "conversation_contract": {"user_frame_count": 1, "target_end_frame_count": 3},
    }
    manifest.write_text(json.dumps(source) + "\n", encoding="utf-8")
    encoded_row = {
        "trial_id": "trial-1", "source_audio_sha256": "8" * 64,
        "codes_sha256": "9" * 64, "shape": [1, 8, 1],
        "conversation_codes_sha256": "a" * 64,
        "assistant_silence_codes_sha256": "b" * 64,
        "conversation_codes_shape": [1, 8, 3],
        "assistant_silence_codes_shape": [1, 8, 3],
        "model_revision": MODEL_REVISION,
    }
    encoded.write_text(json.dumps(encoded_row) + "\n", encoding="utf-8")
    _write_json(scan_spec, {"execution": {}})
    binding = build_target_binding_from_files(
        config_path=config, manifest_path=manifest, encoded_manifest_path=encoded,
        scan_spec_path=scan_spec, code_commit=COMMIT, require_clean=False,
    )
    assert len(binding["data_sha256"]) == 64

    encoded_row["conversation_codes_shape"] = [1, 8, 2]
    encoded.write_text(json.dumps(encoded_row) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="does not cover 3 frozen frames"):
        build_target_binding_from_files(
            config_path=config, manifest_path=manifest, encoded_manifest_path=encoded,
            scan_spec_path=scan_spec, code_commit=COMMIT, require_clean=False,
        )


def test_scan_execution_requires_exact_lists_and_selection_hash() -> None:
    actual = {
        "kind": "residual", "role": "discovery", "layers": [0, 15, 31],
        "anchors": ["new_end", "query_end"], "donors": ["clean_current"],
        "controls": ["self"], "components": ["resid_post"],
        "limit_scenarios": None, "selection_sha256": None,
    }
    validate_scan_execution({"execution": deepcopy(actual)}, actual)
    changed = deepcopy(actual)
    changed["layers"] = [0, 31]
    with pytest.raises(Exception, match="differs from authorized execution"):
        validate_scan_execution({"execution": actual}, changed)


@pytest.mark.parametrize(
    "stale_target",
    [
        "gpu_source_mismatch",
        "model_code_missing",
        "model_config_mismatch",
        "model_manifest_mismatch",
        "model_run_identity_mismatch",
    ],
)
def test_evidence_assembler_rejects_stale_canary_provenance(
    tmp_path: Path, stale_target: str,
) -> None:
    config = tmp_path / "config.json"
    manifest = tmp_path / "manifest.jsonl"
    encoded = tmp_path / "encoded.jsonl"
    scan_spec = tmp_path / "scan.json"
    canary = tmp_path / "canary.jsonl"
    canary_encoded = tmp_path / "canary-encoded.jsonl"
    model_contract = tmp_path / "model.json"
    model_identity = tmp_path / "identity.json"
    open_loop = tmp_path / "open.json"
    conversation = tmp_path / "conversation.json"
    gpu = tmp_path / "gpu.json"
    output = tmp_path / "evidence.json"
    config_value = {
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "audio": {"sample_rate": 24_000, "mimi_frame_samples": 1_920},
    }
    source_rows = [
        {"trial_id": "trial-clean", "audio_sha256": "8" * 64,
         "sample_count": 1_920, "frame_count": 1, "scenario_id": "s1",
         "direction_id": "a_to_b", "speaker_id": "spk1", "new_value": "Seattle",
         "condition": "clean_final"},
        {"trial_id": "trial-repair", "audio_sha256": "a" * 64,
         "sample_count": 1_920, "frame_count": 1, "scenario_id": "s1",
         "direction_id": "a_to_b", "speaker_id": "spk1", "new_value": "Seattle",
         "condition": "delayed_three_dependencies"},
    ]
    encoded_rows = [
        {"trial_id": "trial-clean", "source_audio_sha256": "8" * 64,
         "codes_sha256": "9" * 64, "shape": [1, 8, 1], "model_revision": MODEL_REVISION},
        {"trial_id": "trial-repair", "source_audio_sha256": "a" * 64,
         "codes_sha256": "b" * 64, "shape": [1, 8, 1], "model_revision": MODEL_REVISION},
    ]
    _write_json(config, config_value)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")
    encoded.write_text("".join(json.dumps(row) + "\n" for row in encoded_rows), encoding="utf-8")
    canary.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")
    canary_encoded.write_text(
        "".join(json.dumps(row) + "\n" for row in encoded_rows), encoding="utf-8")
    _write_json(scan_spec, {"execution": {}})
    _write_json(canary.with_suffix(canary.suffix + ".selection.json"), {
        "source_manifest_sha256": sha256_file(manifest),
        "canary_manifest_sha256": sha256_file(canary),
        "trial_ids": ["trial-clean", "trial-repair"],
        "trial_count": 2,
        "bounded_max_trials": 4,
        "matched_group": {"scenario_id": "s1", "direction_id": "a_to_b",
                          "speaker_id": "spk1", "current_value": "Seattle"},
        "clean_trial_id": "trial-clean", "repair_trial_id": "trial-repair",
        "clean_condition": "clean_final", "repair_condition": "delayed_three_dependencies",
    })
    identity_body = {
        "schema_version": "1.0.0", "harness_version": "test",
        "code_commit": COMMIT, "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "config_sha256": sha256_value(config_value), "manifest_sha256": sha256_file(manifest),
        "open_loop_policy_sha256": "7" * 64, "data_status": "synthetic_fixture",
    }
    identity_sha = sha256_value(identity_body)
    model_contract_value = {
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "run_identity_sha256": identity_sha, "code_commit": COMMIT,
        "config_sha256": sha256_file(config), "manifest_sha256": sha256_file(manifest),
        "checks": {},
    }
    if stale_target == "model_code_missing":
        del model_contract_value["code_commit"]
    elif stale_target == "model_config_mismatch":
        model_contract_value["config_sha256"] = "c" * 64
    elif stale_target == "model_manifest_mismatch":
        model_contract_value["manifest_sha256"] = "d" * 64
    elif stale_target == "model_run_identity_mismatch":
        model_contract_value["run_identity_sha256"] = "e" * 64
    _write_json(model_contract, model_contract_value)
    _write_json(model_identity, {**identity_body, "run_identity_sha256": identity_sha})
    common = {
        "code_commit": COMMIT, "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "config_sha256": sha256_file(config), "checks": {},
    }
    _write_json(open_loop, {**common, "encoded_manifest_sha256": sha256_file(canary_encoded)})
    _write_json(conversation, {
        **common, "source_manifest_sha256": sha256_file(manifest),
        "canary_manifest_sha256": sha256_file(canary),
    })
    gpu_source_hash = "f" * 64 if stale_target == "gpu_source_mismatch" else sha256_file(manifest)
    _write_json(gpu, {
        **common, "canary_manifest_sha256": sha256_file(canary),
        "source_manifest_sha256": gpu_source_hash,
    })
    with pytest.raises(ContractError, match="stale readiness evidence"):
        assemble_evidence([
            "--config", str(config), "--manifest", str(manifest),
            "--encoded-manifest", str(encoded), "--scan-spec", str(scan_spec),
            "--model-contract", str(model_contract), "--model-run-identity", str(model_identity),
            "--open-loop", str(open_loop), "--conversation-canary", str(conversation),
            "--gpu-canary", str(gpu), "--canary-manifest", str(canary),
            "--canary-encoded-manifest", str(canary_encoded), "--output", str(output),
            "--code-commit", COMMIT, "--allow-dirty-for-tests",
        ])


def test_canary_selection_prefers_strict_matched_delayed_three_pair(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "canary.jsonl"

    def row(trial: str, scenario: str, direction: str, speaker: str, value: str, condition: str) -> dict:
        return {
            "trial_id": trial,
            "scenario_id": scenario,
            "direction_id": direction,
            "speaker_id": speaker,
            "new_value": value,
            "condition": condition,
        }

    rows = [
        row("a-clean", "s0", "a_to_b", "spk0", "Seattle", "clean_final"),
        row("a-repair", "s0", "a_to_b", "spk0", "Seattle", "immediate_repair"),
        row("z-extra", "s1", "b_to_a", "spk1", "Boston", "delayed_neutral"),
        row("z-clean", "s1", "b_to_a", "spk1", "Boston", "clean_final"),
        row("z-repair", "s1", "b_to_a", "spk1", "Boston", "delayed_three_dependencies"),
    ]
    manifest.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    assert select_canary_manifest([
        "--manifest", str(manifest), "--output", str(output), "--max-trials", "4",
    ]) == 0
    selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [item["trial_id"] for item in selected[:2]] == ["z-clean", "z-repair"]
    assert {item["scenario_id"] for item in selected} == {"s1"}
    assert {item["direction_id"] for item in selected} == {"b_to_a"}
    assert {item["speaker_id"] for item in selected} == {"spk1"}
    assert {item["new_value"] for item in selected} == {"Boston"}
    sidecar = json.loads(
        output.with_suffix(output.suffix + ".selection.json").read_text(encoding="utf-8"))
    assert sidecar["matched_group"] == {
        "scenario_id": "s1", "direction_id": "b_to_a",
        "speaker_id": "spk1", "current_value": "Boston",
    }
    assert sidecar["source_manifest_sha256"] == sha256_file(manifest)


def test_canary_selection_rejects_near_but_unmatched_pair(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "canary.jsonl"
    rows = [
        {"trial_id": "clean", "scenario_id": "s1", "direction_id": "a_to_b",
         "speaker_id": "spk1", "new_value": "Seattle", "condition": "clean_final"},
        {"trial_id": "repair-direction", "scenario_id": "s1", "direction_id": "b_to_a",
         "speaker_id": "spk1", "new_value": "Seattle", "condition": "delayed_three_dependencies"},
        {"trial_id": "repair-speaker", "scenario_id": "s1", "direction_id": "a_to_b",
         "speaker_id": "spk2", "new_value": "Seattle", "condition": "delayed_three_dependencies"},
        {"trial_id": "repair-value", "scenario_id": "s1", "direction_id": "a_to_b",
         "speaker_id": "spk1", "new_value": "Boston", "condition": "delayed_three_dependencies"},
    ]
    manifest.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    with pytest.raises(ContractError, match="no scenario/direction/speaker/current-value group"):
        select_canary_manifest(["--manifest", str(manifest), "--output", str(output)])


def test_canary_selection_filters_immutable_role_before_grouping(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "formal-canary.jsonl"
    rows = [
        {"trial_id": "a-clean", "scenario_id": "discovery", "direction_id": "a_to_b",
         "speaker_id": "spk", "new_value": "Seattle", "condition": "clean_final",
         "role": "discovery"},
        {"trial_id": "a-repair", "scenario_id": "discovery", "direction_id": "a_to_b",
         "speaker_id": "spk", "new_value": "Seattle", "condition": "repair_immediate",
         "role": "discovery"},
        {"trial_id": "z-clean", "scenario_id": "formal", "direction_id": "b_to_a",
         "speaker_id": "spk", "new_value": "Boston", "condition": "clean_current",
         "role": "formal_confirmation"},
        {"trial_id": "z-repair", "scenario_id": "formal", "direction_id": "b_to_a",
         "speaker_id": "spk", "new_value": "Boston", "condition": "repair_delayed_640",
         "role": "formal_confirmation"},
    ]
    manifest.write_text(
        "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
    )
    assert select_canary_manifest([
        "--manifest", str(manifest), "--output", str(output),
        "--role", "formal_confirmation",
    ]) == 0
    selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {row["role"] for row in selected} == {"formal_confirmation"}
    assert {row["scenario_id"] for row in selected} == {"formal"}
    sidecar = json.loads(
        output.with_suffix(output.suffix + ".selection.json").read_text(encoding="utf-8")
    )
    assert sidecar["role_filter"] == "formal_confirmation"
    assert sidecar["source_manifest_sha256"] == sha256_file(manifest)

    with pytest.raises(ContractError, match="role filter selects no trials"):
        select_canary_manifest([
            "--manifest", str(manifest), "--output", str(tmp_path / "missing.jsonl"),
            "--role", "internal_validation",
        ])


def test_gpu_canary_rejects_missing_source_manifest_hash_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    manifest = tmp_path / "canary.jsonl"
    output = tmp_path / "gpu.json"
    _write_json(config, {
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION, "layers": 32},
    })
    rows = [
        {"trial_id": "clean", "scenario_id": "s1", "direction_id": "a_to_b",
         "speaker_id": "spk1", "new_value": "Seattle", "condition": "clean_final"},
        {"trial_id": "repair", "scenario_id": "s1", "direction_id": "a_to_b",
         "speaker_id": "spk1", "new_value": "Seattle", "condition": "delayed_three_dependencies",
         "frame_count": 2, "conversation_contract": {"query_end_frame": 2}},
    ]
    manifest.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    _write_json(manifest.with_suffix(manifest.suffix + ".selection.json"), {
        "canary_manifest_sha256": sha256_file(manifest),
        "matched_group": {
            "scenario_id": "s1", "direction_id": "a_to_b",
            "speaker_id": "spk1", "current_value": "Seattle",
        },
        "clean_trial_id": "clean",
        "repair_trial_id": "repair",
    })
    monkeypatch.setenv("NO_TORCH_COMPILE", "1")
    monkeypatch.setenv("NO_CUDA_GRAPH", "1")
    with pytest.raises(ContractError, match="no valid source manifest hash"):
        run_gpu_canary([
            "--config", str(config), "--manifest", str(manifest),
            "--input-artifact-root", str(tmp_path), "--output", str(output),
        ])
