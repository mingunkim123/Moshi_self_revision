from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest

from experiments.self_repair.mechanistic.causal_scan import (
    CausalCellPlan,
    parse_path_specification,
)
from experiments.self_repair.mechanistic.core import ContractError, read_json, read_jsonl, sha256_file, sha256_value, write_json, write_jsonl
from experiments.self_repair.mechanistic.scripts import _cli


class _ArrayTorch:
    @staticmethod
    def as_tensor(value: object, *, device: str | None = None) -> np.ndarray:
        del device
        return np.asarray(value)


class _FakeLMGen:
    def __init__(self, backend: "_PatchBackend") -> None:
        self.backend = backend

    def snapshot_streaming_state(self) -> float:
        return self.backend.current_margin


class _PatchBackend:
    def __init__(self) -> None:
        self.torch = _ArrayTorch()
        self.device = "cpu"
        self.lm_gen = _FakeLMGen(self)
        self.current_margin = 0.0
        self.calls: list[dict[str, object]] = []

    def replay_codes(self, codes: object, **kwargs: object) -> SimpleNamespace:
        array = np.asarray(codes)
        sites = tuple(kwargs.get("sites", ()))
        layers = tuple(kwargs.get("capture_layers") or ())
        frames = tuple(kwargs.get("capture_frames") or ())
        replacement = dict(kwargs.get("replacement") or {})
        self.calls.append({"sites": sites, "layers": layers, "frames": frames,
                           "replacement": replacement, **kwargs})
        events: dict[tuple[str, int, int], np.ndarray] = {}
        for site in sites:
            for layer in layers:
                for frame in frames:
                    value = 10.0 if array.reshape(-1)[0] == 9 else 2.0
                    if replacement and site == "head_z":
                        value = 7.0
                    events[(str(site), int(layer), int(frame))] = np.full(
                        (1, 2, 1, 3), value, dtype=np.float32)
        if kwargs.get("hook_enabled") is False:
            self.current_margin = 1.0
        elif replacement and not sites:
            self.current_margin = 3.0 if any(key[0] == "head_z" for key in replacement) else 2.0
        end = int(kwargs["end_frame_exclusive"])
        return SimpleNamespace(
            event_tensors=events, feedback_sha256="f" * 64, frame_count=end,
        )

    @staticmethod
    def score_candidates(snapshot: float, candidates: dict[str, str], **_: object) -> dict[str, float]:
        return {name: float(snapshot) if name == "target" else 0.0 for name in candidates}


def _plan(component: str) -> CausalCellPlan:
    return CausalCellPlan(
        recipient_trial_id="repair", donor_trial_id="clean", requested_arm="clean_current",
        relation="clean_current", component=component, layer=1, anchor="new_end",
        source_frame=2, target_frame=2, query_end_frame_exclusive=6, head=None,
    )


def _readout() -> tuple[dict[str, object], list[dict[str, object]]]:
    return (
        {"id": "root", "prefix": "Current", "anchor": "query_end"},
        [{"id": "immediate", "prefix_start_offset_frames": 0,
          "pad_frames_between_tokens": 0}],
    )


def test_real_kv_patch_captures_and_replaces_k_and_v_jointly() -> None:
    backend = _PatchBackend()
    readout, schedules = _readout()
    metric = _cli._real_patch_metric(
        backend,
        donor_codes=np.full((1, 8, 6), 9),
        recipient_codes=np.full((1, 8, 6), 1),
        plan=_plan("kv"), readout=readout, schedules=schedules,
        target="Seattle", stale="Boston", path=None, anchor_rows_by_key={},
    )
    donor_call = backend.calls[0]
    assert donor_call["sites"] == ("k_pre_rope", "v_pre_rope")
    assert donor_call["layers"] == (1,)
    assert donor_call["frames"] == (2,)
    assert donor_call["end_frame_exclusive"] == 3
    patched_call = backend.calls[-1]
    assert set(patched_call["replacement"]) == {
        ("k_pre_rope", 1, 2), ("v_pre_rope", 1, 2),
    }
    assert patched_call["end_frame_exclusive"] == 6
    assert metric["delta_M"] == pytest.approx(1.0)


def test_path_patch_is_two_stage_writer_then_mediator() -> None:
    backend = _PatchBackend()
    readout, schedules = _readout()
    path = parse_path_specification({
        "path": {
            "writer": {"site": "resid_post", "layer": 1, "anchor": "new_end"},
            "mediator": {"site": "head_z", "layer": 2, "anchor": "query_end", "head": 1},
        }
    })
    anchors = {
        ("clean", "new_end"): {"frame": 2},
        ("repair", "new_end"): {"frame": 2},
        ("repair", "query_end"): {"frame": 5},
    }
    metric = _cli._real_patch_metric(
        backend,
        donor_codes=np.full((1, 8, 6), 9),
        recipient_codes=np.full((1, 8, 6), 1),
        plan=_plan("path"), readout=readout, schedules=schedules,
        target="Seattle", stale="Boston", path=path, anchor_rows_by_key=anchors,
    )
    assert tuple(backend.calls[0]["sites"]) == ("resid_post",)
    stage_one = backend.calls[1]
    assert tuple(stage_one["sites"]) == ("head_z",)
    assert set(stage_one["replacement"]) == {("resid_post", 1, 2)}
    stage_two = backend.calls[-1]
    assert tuple(stage_two["sites"]) == ()
    assert set(stage_two["replacement"]) == {("head_z", 2, 5)}
    assert metric["path_evidence"]["algorithm"] == "two_stage_writer_to_mediator_path_patch_v1"
    assert metric["delta_M"] == pytest.approx(2.0)


def _minimal_scan_files(root: Path) -> tuple[Path, Path, Path]:
    config = root / "config.json"
    manifest = root / "manifest.jsonl"
    anchors = root / "anchors.jsonl"
    write_json(config, {"model": {"heads": 1}, "gates": {"self_patch_abs_delta_max": 1e-5}})
    write_json(root / "readouts.json", {
        "schema_version": "1.0.0",
        "candidate_scoring": "mean_log_probability_per_token",
        "candidate_branching": "restore_identical_query_snapshot_before_each_candidate",
        "schedule_aggregation": "logmeanexp_over_all_preregistered_schedules",
        "readouts": [{"id": "root", "prefix": "Current", "anchor": "query_end"}],
        "emission_schedules": [{"id": "immediate", "prefix_start_offset_frames": 0,
                                "pad_frames_between_tokens": 0}],
    })
    rows = [
        {"trial_id": "clean", "scenario_id": "s1", "direction_id": "a_to_b",
         "speaker_id": "spk", "condition": "clean_current", "old_value": "Boston",
         "new_value": "Seattle", "role": "local_validation", "frame_count": 4},
        {"trial_id": "repair", "scenario_id": "s1", "direction_id": "a_to_b",
         "speaker_id": "spk", "condition": "repair", "old_value": "Boston",
         "new_value": "Seattle", "role": "local_validation", "frame_count": 4},
    ]
    write_jsonl(manifest, rows)
    write_jsonl(anchors, [
        {"trial_id": "clean", "anchor": "new_end", "frame": 1},
        {"trial_id": "clean", "anchor": "query_end", "frame": 3},
        {"trial_id": "repair", "anchor": "new_end", "frame": 1},
        {"trial_id": "repair", "anchor": "query_end", "frame": 3},
    ])
    return config, manifest, anchors


def test_failed_cell_is_preserved_and_resume_retries_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest, anchors = _minimal_scan_files(tmp_path)
    output = tmp_path / "out"

    class Failing:
        def patch(self, *_: object, **__: object) -> dict[str, float]:
            raise RuntimeError("transient fixture error")

    monkeypatch.setattr(_cli, "SyntheticBackend", Failing)
    assert _cli._scan([
        "--synthetic", "--config", str(config), "--manifest", str(manifest),
        "--anchor-map", str(anchors), "--role", "local_validation", "--layers", "0",
        "--anchors", "query_end", "--donors", "clean_current",
        "--output-root", str(output),
    ], "residual") == 0
    assert read_jsonl(output / "residual_patch_results.jsonl")[0]["status"] == "failed"
    assert len(read_jsonl(output / "failures.jsonl")) == 1

    class Passing:
        def patch(self, *_: object, **__: object) -> dict[str, object]:
            return {"baseline_M": 0.0, "patched_M": 1.0, "delta_M": 1.0,
                    "feedback_sha256": "a" * 64}

    monkeypatch.setattr(_cli, "SyntheticBackend", Passing)
    assert _cli._scan([
        "--synthetic", "--config", str(config), "--manifest", str(manifest),
        "--anchor-map", str(anchors), "--role", "local_validation", "--layers", "0",
        "--anchors", "query_end", "--donors", "clean_current",
        "--output-root", str(output), "--resume",
    ], "residual") == 0
    assert read_jsonl(output / "residual_patch_results.jsonl")[0]["status"] == "completed"
    assert len(read_jsonl(output / "failures.jsonl")) == 1
    assert read_json(output / "resume_summary.json")["failure_attempts"] == 1


def test_missing_anchor_stops_before_backend_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest, anchors = _minimal_scan_files(tmp_path)
    rows = [row for row in read_jsonl(anchors) if row["trial_id"] == "clean"]
    write_jsonl(anchors, rows)
    constructed = False

    def forbidden() -> None:
        nonlocal constructed
        constructed = True

    monkeypatch.setattr(_cli, "SyntheticBackend", forbidden)
    with pytest.raises(ContractError, match="required semantic anchor"):
        _cli._scan([
            "--synthetic", "--config", str(config), "--manifest", str(manifest),
            "--anchor-map", str(anchors), "--role", "local_validation", "--layers", "0",
            "--anchors", "query_end", "--donors", "clean_current",
            "--output-root", str(tmp_path / "out"),
        ], "residual")
    assert constructed is False


def test_patch_result_schema_requires_status_specific_evidence(tmp_path: Path) -> None:
    schema = json.loads(Path(
        "experiments/self_repair/mechanistic/schemas/patch-result.schema.json"
    ).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    config, manifest, anchors = _minimal_scan_files(tmp_path)
    output = tmp_path / "schema-out"
    assert _cli._scan([
        "--synthetic", "--config", str(config), "--manifest", str(manifest),
        "--anchor-map", str(anchors), "--role", "local_validation", "--layers", "0",
        "--anchors", "query_end", "--donors", "clean_current",
        "--output-root", str(output),
    ], "residual") == 0
    row = read_jsonl(output / "residual_patch_results.jsonl")[0]
    validator.validate(row)
    invalid = dict(row)
    del invalid["delta_M"]
    assert list(validator.iter_errors(invalid))


def test_path_scan_and_freeze_preserve_explicit_two_stage_specification(tmp_path: Path) -> None:
    config, manifest, anchors = _minimal_scan_files(tmp_path)
    readouts = tmp_path / "readouts.json"
    selection = {
        "schema_version": "1.0.0", "status": "frozen_path_candidate",
        "config_sha256": sha256_file(config), "component": "path", "layer": 1,
        "head": None, "anchor": "new_end", "direction": "target_minus_stale",
        "donor_arm": "clean_current", "relation": "clean_current",
        "readout_sha256": sha256_file(readouts), "selection_source_cell_id": "a" * 64,
        "path": {
            "writer": {"site": "resid_post", "layer": 1, "anchor": "new_end"},
            "mediator": {"site": "head_z", "layer": 2, "anchor": "query_end", "head": 0},
        },
    }
    selection["selection_sha256"] = sha256_value(selection)
    selection_path = tmp_path / "path-selection.json"
    write_json(selection_path, selection)
    output = tmp_path / "path-output"
    assert _cli._scan([
        "--synthetic", "--config", str(config), "--manifest", str(manifest),
        "--anchor-map", str(anchors), "--readouts", str(readouts),
        "--selection", str(selection_path), "--role", "local_validation",
        "--donors", "clean_current", "--output-root", str(output),
    ], "path") == 0
    row = read_jsonl(output / "path_patch_results.jsonl")[0]
    normalized_path = parse_path_specification(selection).identity
    assert row["path"] == normalized_path
    assert row["path_evidence"]["algorithm"] == "two_stage_writer_to_mediator_path_patch_v1"
    assert row["source_frames"] == [1]
    assert row["target_frames"] == [1, 3]
    frozen = tmp_path / "frozen.json"
    assert _cli.freeze_mechanistic_selection([
        "--synthetic", "--config", str(config), "--discovery-root", str(output),
        "--output", str(frozen),
    ]) == 0
    assert read_json(frozen)["path"] == normalized_path


def test_real_confirmatory_requires_hash_bound_authorization_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest, anchors = _minimal_scan_files(tmp_path)
    rows = read_jsonl(manifest)
    for row in rows:
        row["role"] = "confirmation_test"
    write_jsonl(manifest, rows)
    encoded = tmp_path / "encoded.jsonl"
    write_jsonl(encoded, [{"trial_id": row["trial_id"]} for row in rows])
    readouts = Path("experiments/self_repair/mechanistic/config/readouts.json")
    selection = {
        "schema_version": "1.0.0", "status": "frozen_discovery_selection",
        "config_sha256": sha256_file(config), "component": "resid_post", "layer": 0,
        "head": None, "anchor": "query_end", "direction": "target_minus_stale",
        "donor_arm": "clean_current", "relation": "clean_current",
        "readout_sha256": sha256_file(readouts), "selection_source_cell_id": "a" * 64,
    }
    selection["selection_sha256"] = sha256_value(selection)
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)
    constructed = False

    def forbidden() -> None:
        nonlocal constructed
        constructed = True

    monkeypatch.setattr(_cli, "MoshiBackend", forbidden)
    with pytest.raises(ContractError, match="hash-bound"):
        _cli.run_confirmatory([
            "--config", str(config), "--selection", str(selection_path),
            "--manifest", str(manifest), "--encoded-manifest", str(encoded),
            "--anchors", str(anchors), "--readouts", str(readouts),
            "--role", "confirmation_test", "--output-root", str(tmp_path / "confirm"),
        ])
    assert constructed is False


def test_confirmatory_materializes_frozen_primary_and_preregistered_self_control(
    tmp_path: Path,
) -> None:
    config, manifest, anchors = _minimal_scan_files(tmp_path)
    discovery = tmp_path / "discovery"
    assert _cli._scan([
        "--synthetic", "--config", str(config), "--manifest", str(manifest),
        "--anchor-map", str(anchors), "--role", "local_validation",
        "--layers", "0", "--anchors", "query_end",
        "--donors", "clean_current,self", "--output-root", str(discovery),
    ], "residual") == 0
    selection = tmp_path / "selection.json"
    assert _cli.freeze_mechanistic_selection([
        "--synthetic", "--config", str(config), "--discovery-root", str(discovery),
        "--output", str(selection),
    ]) == 0
    output = tmp_path / "confirmation"
    assert _cli.run_confirmatory([
        "--synthetic", "--config", str(config), "--selection", str(selection),
        "--manifest", str(manifest), "--anchors", str(anchors),
        "--readouts", str(tmp_path / "readouts.json"),
        "--role", "local_validation", "--control-arms", "self",
        "--output-root", str(output),
    ]) == 0
    rows = read_jsonl(output / "patch_results.jsonl")
    assert {row["donor_arm"] for row in rows} == {"clean_current", "self"}
    assert len(rows) == 2
    self_row = next(row for row in rows if row["donor_arm"] == "self")
    assert self_row["relation"] == "self"
    assert self_row["delta_M"] == pytest.approx(0.0)
    assert read_json(output / "scan_plan.json")["planned_cell_count"] == 2
