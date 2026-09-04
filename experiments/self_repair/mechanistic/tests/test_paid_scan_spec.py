from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import jsonschema
import pytest

from experiments.self_repair.mechanistic.core import (
    ContractError,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
    write_jsonl,
)
from experiments.self_repair.mechanistic.paid_scan_spec import (
    build_paid_scan_spec,
    main,
    verify_paid_scan_spec,
)
from experiments.self_repair.mechanistic.readiness import ReadinessError, estimate_workload


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "config.json"
    write_json(config, {
        "schema_version": "1.0.0",
        "model": {"layers": 4, "heads": 4, "hidden_size": 16},
        "audio": {"sample_rate": 24_000, "mimi_frame_samples": 1_920, "frame_ms": 80},
        "anchors": {"primary": ["old_end", "new_end", "query_end"]},
        "manifest": {"discovery_generation_seeds": [17, 29]},
        "conversation": {
            "required_modes": ["common_handshake_then_request", "greeting_suppressed"],
            "startup": {"natural_max_ms": 160},
            "response": {"post_user_max_ms": 160},
        },
    })
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for scenario in ("scenario-a", "scenario-b"):
        for suffix, condition, frames in (
            ("clean", "clean_current", 10),
            ("repair", "repair_immediate", 12),
        ):
            target = frames + 2
            rows.append({
                "trial_id": f"{scenario}-{suffix}",
                "scenario_id": scenario,
                "direction_id": "boston_to_seattle",
                "speaker_id": "speaker-1",
                "old_value": "Boston",
                "new_value": "Seattle",
                "condition": condition,
                "role": "discovery",
                "analysis_fold": 1,
                "sample_rate": 24_000,
                "frame_count": frames,
                "sample_count": frames * 1_920,
                "conversation_contract": {
                    "user_frame_count": frames,
                    "user_end_frame": frames,
                    "response_capture_frames": 2,
                    "target_end_frame_count": target,
                    "target_end_sample_count": target * 1_920,
                    "appended_zero_frame_count": 2,
                },
            })
    write_jsonl(manifest, rows)
    return config, manifest


def _build(config: Path, manifest: Path, **overrides: object):
    kwargs = {
        "config_path": config,
        "manifest_path": manifest,
        "role": "discovery",
        "kind": "residual",
        "layers": [0, 3],
        "anchors": ["new_end", "query_end"],
        "donors": ["clean_current", "self"],
        "controls": ["self"],
        "components": ["resid_post"],
        "full_replays_per_cell": 3,
        "readout_steps_per_cell": 4,
        "limit_scenarios": 1,
        "include_generation": True,
        "generation_seeds": None,
        "generation_branches": ["baseline", "patched"],
    }
    kwargs.update(overrides)
    return build_paid_scan_spec(**kwargs)


def test_bounded_discovery_spec_freezes_exact_arithmetic_and_bytes(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)
    spec, estimate = _build(config, manifest)

    assert spec["execution"] == {
        "kind": "residual",
        "role": "discovery",
        "layers": [0, 3],
        "anchors": ["new_end", "query_end"],
        "donors": ["clean_current", "self"],
        "controls": ["self"],
        "components": ["resid_post"],
        "limit_scenarios": 1,
        "selection_sha256": None,
    }
    assert spec["scans"][0]["expected_cell_count"] == 8
    assert estimate.cell_count == 8
    assert estimate.recipient_trial_count == 1
    assert estimate.replay_pass_count == 24
    assert estimate.replay_frame_count == 336
    assert estimate.readout_frame_count == 32
    assert estimate.generation_count == 8
    assert estimate.generation_frame_count == 120
    assert estimate.total_model_frames == 488
    assert estimate.activation_tensor_bytes == 512
    assert spec["declared_workload"] == estimate.to_dict()
    assert spec["declared_workload_sha256"] == sha256_value(estimate.to_dict())
    assert spec["scan_spec_identity_sha256"] == sha256_value({
        key: value for key, value in spec.items() if key != "scan_spec_identity_sha256"
    })

    schema = read_json(Path(__file__).parents[1] / "schemas/paid-scan-spec.schema.json")
    jsonschema.Draft202012Validator(schema).validate(spec)

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert main([
        "--config", str(config), "--manifest", str(manifest),
        "--kind", "residual", "--role", "discovery",
        "--layers", "0,3", "--anchors", "new_end,query_end",
        "--donors", "clean_current,self", "--controls", "self",
        "--components", "resid_post", "--limit-scenarios", "1",
        "--full-replays-per-cell", "3", "--readout-steps-per-cell", "4",
        "--include-generation", "--generation-branches", "baseline,patched",
        "--output", str(first),
    ]) == 0
    assert main([
        "--config", str(config), "--manifest", str(manifest),
        "--kind", "residual", "--role", "discovery",
        "--layers", "0,3", "--anchors", "new_end,query_end",
        "--donors", "clean_current,self", "--controls", "self",
        "--components", "resid_post", "--limit-scenarios", "1",
        "--full-replays-per-cell", "3", "--readout-steps-per-cell", "4",
        "--include-generation", "--generation-branches", "baseline,patched",
        "--output", str(second),
    ]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert read_json(first) == spec


def test_generation_is_opt_in_and_uses_frozen_config_dimensions(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)
    spec, estimate = _build(config, manifest, include_generation=False)
    assert "generation" not in spec
    assert estimate.generation_count == 0
    assert spec["frozen_dimensions"] == {
        "model_layers": 4,
        "model_heads": 4,
        "hidden_size": 16,
        "generation_included": False,
    }


def test_cli_accepts_same_exclusive_layer_range_syntax_as_scan_runner(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)
    output = tmp_path / "range.json"
    assert main([
        "--config", str(config), "--manifest", str(manifest),
        "--kind", "residual", "--role", "discovery",
        "--layers", "0:4", "--anchors", "query_end",
        "--donors", "clean_current", "--controls", "self",
        "--components", "resid_post", "--limit-scenarios", "1",
        "--full-replays-per-cell", "3", "--readout-steps-per-cell", "8",
        "--output", str(output),
    ]) == 0
    spec = read_json(output)
    assert spec["execution"]["layers"] == [0, 1, 2, 3]
    assert spec["scans"][0]["expected_cell_count"] == 4


def test_selection_derives_single_confirmatory_head_and_file_binding(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)
    selection = tmp_path / "selection.json"
    body = {
        "schema_version": "1.0.0",
        "status": "frozen_discovery_selection",
        "config_sha256": sha256_file(config),
        "component": "head_z",
        "layer": 3,
        "head": 2,
        "anchor": "new_end",
        "direction": "target_minus_stale",
        "donor_arm": "self",
        "relation": "self",
        "readout_sha256": "a" * 64,
        "selection_source_cell_id": "b" * 64,
    }
    body["selection_sha256"] = sha256_value(body)
    write_json(selection, body)

    spec, estimate = build_paid_scan_spec(
        config_path=config,
        manifest_path=manifest,
        role="discovery",
        kind=None,
        layers=None,
        anchors=None,
        donors=None,
        controls=None,
        components=None,
        full_replays_per_cell=3,
        readout_steps_per_cell=4,
        selection_path=selection,
    )
    assert spec["execution"]["kind"] == "component"
    assert spec["execution"]["layers"] == [3]
    assert spec["execution"]["anchors"] == ["new_end"]
    assert spec["execution"]["controls"] == ["self"]
    assert spec["execution"]["selection_sha256"] == sha256_file(selection)
    assert spec["scans"][0]["components"] == [{"name": "head_z", "heads": [2]}]
    assert estimate.cell_count == 2
    assert verify_paid_scan_spec(
        spec, config_path=config, manifest_path=manifest, selection_path=selection,
    ).cell_count == 2

    with pytest.raises(ContractError, match="explicit layers differs"):
        build_paid_scan_spec(
            config_path=config, manifest_path=manifest, role="discovery", kind=None,
            layers=[2], anchors=None, donors=None, controls=None, components=None,
            full_replays_per_cell=3, readout_steps_per_cell=4,
            selection_path=selection,
        )


def test_selection_appends_primary_then_confirmation_controls(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)
    selection = tmp_path / "selection.json"
    body = {
        "schema_version": "1.0.0",
        "status": "frozen_discovery_selection",
        "config_sha256": sha256_file(config),
        "component": "resid_post",
        "layer": 3,
        "head": None,
        "anchor": "new_end",
        "direction": "target_minus_stale",
        "donor_arm": "clean_current",
        "relation": "clean_current",
        "readout_sha256": "a" * 64,
        "selection_source_cell_id": "b" * 64,
    }
    body["selection_sha256"] = sha256_value(body)
    write_json(selection, body)

    spec, estimate = build_paid_scan_spec(
        config_path=config,
        manifest_path=manifest,
        role="discovery",
        kind=None,
        layers=None,
        anchors=None,
        donors=None,
        controls=None,
        components=None,
        full_replays_per_cell=3,
        readout_steps_per_cell=4,
        selection_path=selection,
        confirmation_control_arms=["self"],
    )
    assert spec["execution"]["donors"] == ["clean_current", "self"]
    assert spec["scans"][0]["donor_arms"] == ["clean_current", "self"]
    assert estimate.recipient_trial_count == 2
    assert estimate.cell_count == 4

    component_selection = tmp_path / "component-selection.json"
    component_body = {
        **{key: value for key, value in body.items() if key != "selection_sha256"},
        "component": "attn_out",
    }
    component_body["selection_sha256"] = sha256_value(component_body)
    write_json(component_selection, component_body)
    component_spec, component_estimate = build_paid_scan_spec(
        config_path=config,
        manifest_path=manifest,
        role="discovery",
        kind=None,
        layers=None,
        anchors=None,
        donors=None,
        controls=None,
        components=None,
        full_replays_per_cell=3,
        readout_steps_per_cell=4,
        selection_path=component_selection,
        confirmation_control_arms=["self"],
    )
    assert component_spec["execution"]["kind"] == "component"
    assert component_spec["execution"]["controls"] == ["clean_current", "self"]
    assert component_spec["scans"][0]["donor_arms"] == ["clean_current", "self"]
    assert component_estimate.cell_count == 4

    with pytest.raises(ContractError, match="exclude the frozen primary"):
        build_paid_scan_spec(
            config_path=config, manifest_path=manifest, role="discovery", kind=None,
            layers=None, anchors=None, donors=None, controls=None, components=None,
            full_replays_per_cell=3, readout_steps_per_cell=4,
            selection_path=selection,
            confirmation_control_arms=["clean_current"],
        )


def test_upstream_selection_binds_site_location_but_allows_new_kv_stage(
    tmp_path: Path,
) -> None:
    config, manifest = _fixture(tmp_path)
    upstream = tmp_path / "component-selection.json"
    body = {
        "schema_version": "1.0.0",
        "status": "frozen_discovery_selection",
        "config_sha256": sha256_file(config),
        "component": "head_z",
        "layer": 2,
        "head": 1,
        "anchor": "new_end",
        "direction": "target_minus_stale",
        "donor_arm": "self",
        "relation": "self",
        "readout_sha256": "a" * 64,
        "selection_source_cell_id": "b" * 64,
    }
    body["selection_sha256"] = sha256_value(body)
    write_json(upstream, body)

    spec, estimate = build_paid_scan_spec(
        config_path=config,
        manifest_path=manifest,
        role="discovery",
        kind="kv",
        layers=None,
        anchors=None,
        donors=["clean_current"],
        controls=["self"],
        components=["k_only", "v_only", "kv"],
        full_replays_per_cell=3,
        readout_steps_per_cell=4,
        upstream_selection_path=upstream,
    )

    assert spec["execution"] == {
        "kind": "kv",
        "role": "discovery",
        "layers": [2],
        "anchors": ["new_end"],
        "donors": ["clean_current"],
        "controls": ["self"],
        "components": ["k_only", "v_only", "kv"],
        "limit_scenarios": None,
        "selection_sha256": sha256_file(upstream),
    }
    assert spec["scans"][0]["components"] == [
        {"name": "k_only", "heads": [1]},
        {"name": "v_only", "heads": [1]},
        {"name": "kv", "heads": [1]},
    ]
    assert estimate.scan_breakdown[0]["component_instances"] == 3
    assert estimate.cell_count == 6
    assert verify_paid_scan_spec(
        spec, config_path=config, manifest_path=manifest, selection_path=upstream,
    ).cell_count == 6

    schema = read_json(Path(__file__).parents[1] / "schemas/paid-scan-spec.schema.json")
    jsonschema.Draft202012Validator(schema).validate(spec)

    with pytest.raises(ContractError, match="explicit layers differs"):
        build_paid_scan_spec(
            config_path=config, manifest_path=manifest, role="discovery", kind="kv",
            layers=[1], anchors=None, donors=["clean_current"], controls=["self"],
            components=["kv"], full_replays_per_cell=3, readout_steps_per_cell=4,
            upstream_selection_path=upstream,
        )
    with pytest.raises(ContractError, match="mutually exclusive"):
        build_paid_scan_spec(
            config_path=config, manifest_path=manifest, role="discovery", kind="kv",
            layers=None, anchors=None, donors=["clean_current"], controls=["self"],
            components=["kv"], full_replays_per_cell=3, readout_steps_per_cell=4,
            selection_path=upstream, upstream_selection_path=upstream,
        )


def test_tamper_and_grid_mismatches_fail_closed(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)
    spec, _ = _build(config, manifest)
    tampered = deepcopy(spec)
    tampered["scans"][0]["expected_cell_count"] = 7
    with pytest.raises(ReadinessError, match="expected_cell_count"):
        estimate_workload(read_jsonl(manifest), read_json(config), tampered)

    mismatched = deepcopy(spec)
    mismatched["execution"]["donors"] = ["clean_current"]
    with pytest.raises(ReadinessError, match="kind-active execution.donors"):
        estimate_workload(read_jsonl(manifest), read_json(config), mismatched)

    identity_tamper = deepcopy(spec)
    identity_tamper["storage"]["fixed_reserved_bytes"] += 1
    with pytest.raises(ContractError, match="content identity mismatch"):
        verify_paid_scan_spec(
            identity_tamper, config_path=config, manifest_path=manifest)

    with pytest.raises(ContractError, match="donors must be non-empty"):
        _build(config, manifest, donors=[])
    with pytest.raises(ContractError, match="full_replays_per_cell must be positive"):
        _build(config, manifest, full_replays_per_cell=0)


def test_import_path_never_loads_model_or_gpu_backend() -> None:
    code = """
import json, sys
import experiments.self_repair.mechanistic.paid_scan_spec
forbidden = [
    name for name in sys.modules
    if name == 'torch'
    or name.startswith('transformers')
    or name.startswith('moshi.moshi.models')
    or name.endswith('.scripts._cli')
]
print(json.dumps(forbidden))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, text=True,
        capture_output=True, cwd=Path(__file__).parents[4],
    )
    assert json.loads(result.stdout) == []
