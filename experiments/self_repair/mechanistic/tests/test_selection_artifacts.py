from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.self_repair.mechanistic.causal_scan import parse_path_specification
from experiments.self_repair.mechanistic.core import (
    ContractError,
    MODEL_REPO,
    MODEL_REVISION,
    freeze_selection,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
    write_jsonl,
)
from experiments.self_repair.mechanistic.scripts import _cli
from experiments.self_repair.mechanistic.selection_artifacts import (
    build_path_selection,
    freeze_path_selection_main,
    rebind_mechanistic_selection,
    rebind_mechanistic_selection_main,
)


def _config(root: Path) -> Path:
    path = root / "config.json"
    write_json(path, {
        "schema_version": "1.0.0",
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "layers": 4,
            "heads": 4,
        },
        "anchors": {"primary": ["old_end", "new_end", "query_end"]},
        "gates": {"self_patch_abs_delta_max": 1e-5},
    })
    return path


def _writer_selection(root: Path, config: Path) -> Path:
    path = root / "writer-selection.json"
    body = {
        "schema_version": "1.0.0",
        "status": "frozen_discovery_selection",
        "config_sha256": sha256_file(config),
        "component": "k_only",
        "layer": 1,
        "head": 1,
        "anchor": "new_end",
        "direction": "target_minus_stale",
        "donor_arm": "clean_current",
        "relation": "clean_current",
        "readout_sha256": "1" * 64,
        "selection_source_cell_id": "2" * 64,
    }
    body["selection_sha256"] = sha256_value(body)
    write_json(path, body)
    return path


def _bound_readouts(root: Path, config: Path) -> Path:
    path = root / "formal-readouts.bound.json"
    body = {
        "schema_version": "1.0.0",
        "candidate_scoring": "mean_log_probability_per_token",
        "candidate_branching": "restore_identical_query_snapshot_before_each_candidate",
        "schedule_aggregation": "logmeanexp_over_all_preregistered_schedules",
        "readouts": [{
            "id": "root",
            "prefix": "The current destination is",
            "anchor": "query_end",
            "prefix_token_ids": [10, 11],
        }],
        "emission_schedules": [{
            "id": "immediate",
            "prefix_start_offset_frames": 0,
            "pad_frames_between_tokens": 0,
        }],
        "candidate_token_ids": {
            "Boston": [20],
            "Seattle": [21],
            "Chicago": [22],
            "Denver": [23],
        },
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "config_sha256": sha256_file(config),
        "manifest_sha256": "3" * 64,
        "run_identity_sha256": "4" * 64,
    }
    body["bound_readout_sha256"] = sha256_value(body)
    write_json(path, body)
    return path


def test_path_selection_maps_single_kv_writer_and_freezes_explicit_mediator(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    writer = _writer_selection(tmp_path, config)
    source_bytes = writer.read_bytes()

    selection = build_path_selection(
        config_path=config,
        writer_selection_path=writer,
        mediator_site="head_z",
        mediator_layer=3,
        mediator_anchor="query_end",
        mediator_head=2,
    )

    assert selection["status"] == "frozen_discovery_selection"
    assert selection["component"] == "path"
    assert selection["layer"] == 1
    assert selection["head"] == 1
    assert selection["anchor"] == "new_end"
    assert selection["donor_arm"] == "clean_current"
    assert selection["readout_sha256"] == "1" * 64
    assert selection["config_sha256"] == sha256_file(config)
    assert selection["writer_selection_file_sha256"] == sha256_file(writer)
    assert selection["writer_selection_identity_sha256"] == read_json(writer)["selection_sha256"]
    assert parse_path_specification(selection).identity == {
        "writer": {
            "site": "k_pre_rope", "layer": 1, "anchor": "new_end", "head": 1,
        },
        "mediator": {
            "site": "head_z", "layer": 3, "anchor": "query_end", "head": 2,
        },
    }
    assert selection["selection_sha256"] == sha256_value({
        key: value for key, value in selection.items() if key != "selection_sha256"
    })

    output = tmp_path / "path-selection.json"
    assert freeze_path_selection_main([
        "--config", str(config),
        "--writer-selection", str(writer),
        "--mediator-site", "head_z",
        "--mediator-layer", "3",
        "--mediator-anchor", "query_end",
        "--mediator-head", "2",
        "--output", str(output),
    ]) == 0
    assert read_json(output) == selection
    assert writer.read_bytes() == source_bytes

    joint = read_json(writer)
    joint["component"] = "kv"
    joint["selection_sha256"] = sha256_value({
        key: value for key, value in joint.items() if key != "selection_sha256"
    })
    joint_path = tmp_path / "joint-writer.json"
    write_json(joint_path, joint)
    with pytest.raises(ContractError, match="exactly one tensor site"):
        build_path_selection(
            config_path=config, writer_selection_path=joint_path,
            mediator_site="head_z", mediator_layer=3,
            mediator_anchor="query_end", mediator_head=2,
        )


def test_rebind_preserves_intervention_and_formal_confirmation_uses_new_readout(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    writer = _writer_selection(tmp_path, config)
    path_source = tmp_path / "path-source.json"
    write_json(path_source, build_path_selection(
        config_path=config, writer_selection_path=writer,
        mediator_site="head_z", mediator_layer=3,
        mediator_anchor="query_end", mediator_head=2,
    ))
    source = read_json(path_source)
    source_bytes = path_source.read_bytes()
    bound = _bound_readouts(tmp_path, config)

    transported = rebind_mechanistic_selection(
        config_path=config,
        source_selection_path=path_source,
        bound_readout_path=bound,
    )
    for field in ("component", "layer", "head", "anchor", "donor_arm", "relation", "path"):
        assert transported[field] == source[field]
    assert transported["config_sha256"] == source["config_sha256"]
    assert transported["readout_sha256"] == sha256_file(bound)
    assert transported["source_readout_sha256"] == source["readout_sha256"]
    assert transported["source_selection_file_sha256"] == sha256_file(path_source)
    assert transported["source_selection_identity_sha256"] == source["selection_sha256"]
    assert transported["bound_readout_identity_sha256"] == read_json(bound)[
        "bound_readout_sha256"]
    assert transported["selection_sha256"] == sha256_value({
        key: value for key, value in transported.items() if key != "selection_sha256"
    })

    output = tmp_path / "formal-selection.json"
    assert rebind_mechanistic_selection_main([
        "--config", str(config), "--selection", str(path_source),
        "--readouts", str(bound), "--output", str(output),
    ]) == 0
    assert read_json(output) == transported
    assert path_source.read_bytes() == source_bytes

    role_rows = [
        {"trial_id": trial_id, "role": "formal_confirmation"}
        for trial_id in ("clean", "repair")
    ]
    role_manifest = tmp_path / "formal-role.jsonl"
    write_jsonl(role_manifest, role_rows)
    role_hash = sha256_file(role_manifest)
    manifest = tmp_path / "formal-manifest.jsonl"
    trial_rows = []
    for role_row, condition in zip(role_rows, ("clean_current", "repair"), strict=True):
        trial_rows.append({
            "trial_id": role_row["trial_id"],
            "scenario_id": "formal-scenario",
            "direction_id": "boston_to_seattle",
            "speaker_id": "speaker-1",
            "condition": condition,
            "old_value": "Boston",
            "new_value": "Seattle",
            "role": "formal_confirmation",
            "analysis_fold": 1,
            "frame_count": 4,
            "role_manifest_sha256": role_hash,
            "role_binding_sha256": sha256_value(role_row),
        })
    write_jsonl(manifest, trial_rows)
    anchors = tmp_path / "formal-anchors.jsonl"
    write_jsonl(anchors, [
        {"trial_id": trial_id, "anchor": anchor, "frame": frame}
        for trial_id in ("clean", "repair")
        for anchor, frame in (("new_end", 1), ("query_end", 3))
    ])
    confirmation = tmp_path / "formal-confirmation"
    assert _cli.run_confirmatory([
        "--synthetic",
        "--config", str(config),
        "--selection", str(output),
        "--manifest", str(manifest),
        "--role-manifest", str(role_manifest),
        "--anchors", str(anchors),
        "--readouts", str(bound),
        "--role", "formal_confirmation",
        "--output-root", str(confirmation),
    ]) == 0
    assert read_json(confirmation / "scan_plan.json")["selection_sha256"] == transported[
        "selection_sha256"]

    tampered = deepcopy(read_json(bound))
    tampered["candidate_token_ids"]["Seattle"] = [999]
    write_json(bound, tampered)
    with pytest.raises(ContractError, match="self-hash mismatch"):
        rebind_mechanistic_selection(
            config_path=config, source_selection_path=path_source,
            bound_readout_path=bound,
        )


def test_freeze_selection_uses_canonical_scenario_balanced_primary() -> None:
    config_sha = "a" * 64
    readout_sha = "b" * 64
    provenance = {"config_sha256": config_sha, "code_commit": "c" * 40}

    def row(
        cell_id: str, *, layer: int, scenario: str, effect: float,
        donor_arm: str = "clean_current", relation: str = "clean_current",
        component: str = "resid_post",
    ) -> dict[str, object]:
        return {
            "status": "completed", "cell_id": cell_id,
            "component": component, "layer": layer, "head": None,
            "anchor": "query_end", "donor_arm": donor_arm, "relation": relation,
            "readout_sha256": readout_sha, "scenario_id": scenario,
            "delta_M": effect, "provenance": provenance,
        }

    rows = [
        row("a1", layer=0, scenario="s1", effect=20.0),
        row("a2", layer=0, scenario="s1", effect=20.0),
        row("a3", layer=0, scenario="s1", effect=20.0),
        row("a4", layer=0, scenario="s2", effect=-20.0),
        row("b1", layer=1, scenario="s1", effect=5.0, donor_arm="current"),
        row("b2", layer=1, scenario="s2", effect=5.0, donor_arm="current"),
        row("decoy", layer=2, scenario="s1", effect=999.0,
            donor_arm="self", relation="self"),
        row("filtered", layer=3, scenario="s1", effect=1000.0, component="kv"),
    ]
    selection = freeze_selection(rows, config_sha, components=["resid_post"])

    assert selection["layer"] == 1
    assert selection["donor_arm"] == "clean_current"
    assert selection["relation"] == "clean_current"
    assert selection["selection_source_cell_ids"] == ["b1", "b2"]
    assert selection["selection_source_cell_count"] == 2
    assert selection["selection_source_scenario_count"] == 2
    assert selection["selection_aggregate_delta_M"] == pytest.approx(5.0)
    assert selection["selection_eligibility_policy"] == {
        "version": "1.0.0",
        "required_relation": "clean_current",
        "eligible_components": ["resid_post"],
        "aggregation": "mean_within_scenario_then_mean_across_scenarios",
        "ranking": "absolute_aggregate_delta_M",
        "tie_break": "identity_sha256",
    }
    assert selection["selection_scenario_means"] == [
        {"scenario_id": "s1", "cell_count": 1, "mean_delta_M": 5.0},
        {"scenario_id": "s2", "cell_count": 1, "mean_delta_M": 5.0},
    ]
    assert selection["selection_source_provenance"] == [provenance]
    assert selection["selection_sha256"] == sha256_value({
        key: value for key, value in selection.items() if key != "selection_sha256"
    })
    assert freeze_selection(
        list(reversed(rows)), config_sha, components=["resid_post"]
    ) == selection


def test_selection_cli_help_exposes_staged_and_transport_arguments() -> None:
    root = Path(__file__).parents[4]
    scripts = Path(__file__).parents[1] / "scripts"
    expected = {
        "freeze_paid_scan_spec.py": ("--selection", "--upstream-selection"),
        "freeze_path_selection.py": ("--writer-selection", "--mediator-head"),
        "rebind_mechanistic_selection.py": ("--source-selection", "--bound-readouts"),
        "freeze_mechanistic_selection.py": ("--components",),
    }
    for name, flags in expected.items():
        result = subprocess.run(
            [sys.executable, str(scripts / name), "--help"],
            cwd=root, text=True, capture_output=True, check=True,
        )
        for flag in flags:
            assert flag in result.stdout
