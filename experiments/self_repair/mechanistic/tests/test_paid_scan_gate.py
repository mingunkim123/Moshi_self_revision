from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.self_repair.mechanistic.core import (
    AtomicCellStore,
    ContractError,
    PatchCell,
    sha256_file,
    sha256_value,
)
from experiments.self_repair.mechanistic.readiness import target_binding_sha256
from experiments.self_repair.mechanistic.scripts import _cli


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_real_scan_refuses_missing_go_before_backend_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    manifest = tmp_path / "manifest.jsonl"
    encoded = tmp_path / "encoded.jsonl"
    _json(config, {"model": {"heads": 1}, "readouts": {}})
    manifest.touch()
    encoded.touch()
    constructed = False

    def forbidden_backend() -> None:
        nonlocal constructed
        constructed = True
        raise AssertionError("backend must not be constructed")

    monkeypatch.setattr(_cli, "MoshiBackend", forbidden_backend)
    with pytest.raises(ContractError, match="readiness-go"):
        _cli._scan([
            "--config", str(config), "--manifest", str(manifest),
            "--encoded-manifest", str(encoded), "--role", "discovery",
            "--output-root", str(tmp_path / "out"),
        ], "residual")
    assert constructed is False


def test_resume_skips_verified_cell_before_any_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.jsonl"
    encoded_path = tmp_path / "encoded.jsonl"
    anchors_path = tmp_path / "anchors.jsonl"
    readouts_path = tmp_path / "readouts.bound.json"
    spec_path = tmp_path / "scan.json"
    go_path = tmp_path / "go.json"
    output_root = tmp_path / "out"
    config = {"model": {"heads": 1}, "readouts": {}}
    _json(config_path, config)
    rows = [
        {"trial_id": "clean", "scenario_id": "s1", "condition": "clean_current",
         "direction_id": "boston_to_seattle", "speaker_id": "speaker-1",
         "new_value": "Seattle", "old_value": "Boston", "role": "discovery",
         "analysis_fold": 1, "frame_count": 1},
        {"trial_id": "repair", "scenario_id": "s1", "condition": "repair",
         "direction_id": "boston_to_seattle", "speaker_id": "speaker-1",
         "new_value": "Seattle", "old_value": "Boston", "role": "discovery",
         "analysis_fold": 1, "frame_count": 1},
    ]
    _jsonl(manifest_path, rows)
    _jsonl(encoded_path, [{"trial_id": row["trial_id"], "synthetic": False,
                           "conversation_frame_end_exclusive": 1} for row in rows])
    _jsonl(anchors_path, [
        {"trial_id": "clean", "anchor": "query_end", "frame": 0},
        {"trial_id": "repair", "anchor": "query_end", "frame": 0},
    ])
    execution = {
        "kind": "residual", "role": "discovery", "layers": [0],
        "anchors": ["query_end"], "donors": ["clean_current"],
        "controls": ["self"], "components": ["resid_post"],
        "limit_scenarios": None, "selection_sha256": None,
    }
    _json(spec_path, {"execution": execution})
    _json(go_path, {"placeholder": True})
    readouts = {
        "schema_version": "1.0.0",
        "candidate_scoring": "mean_log_probability_per_token",
        "candidate_branching": "restore_identical_query_snapshot_before_each_candidate",
        "schedule_aggregation": "logmeanexp_over_all_preregistered_schedules",
        "readouts": [{"id": "root", "prefix": "Current", "anchor": "query_end",
                      "prefix_token_ids": [1]}],
        "emission_schedules": [{"id": "immediate", "prefix_start_offset_frames": 0,
                                "pad_frames_between_tokens": 0}],
        "candidate_token_ids": {"Boston": [2], "Seattle": [3]},
        "model_revision": _cli.MODEL_REVISION,
    }
    readouts["bound_readout_sha256"] = sha256_value(readouts)
    _json(readouts_path, readouts)
    binding = {
        "code_commit": "1" * 40, "code_sha256": sha256_value({"git_commit": "1" * 40}),
        "model_repo": "repo", "model_revision": "2" * 40,
        "model_sha256": sha256_value({"repo": "repo", "revision": "2" * 40}),
        "manifest_sha256": "3" * 64, "data_sha256": "4" * 64,
        "encoded_manifest_sha256": "5" * 64, "config_sha256": "6" * 64,
        "scan_spec_sha256": "7" * 64,
    }
    monkeypatch.setattr(_cli, "build_target_binding_from_files", lambda **_: binding)
    monkeypatch.setattr(_cli, "verify_authorization_artifact", lambda *_: binding)
    monkeypatch.setattr(_cli, "estimate_workload", lambda *_: SimpleNamespace(cell_count=1))

    class Backend:
        metadata = {"heads": 1}

        def replay_codes(self, *_: object, **__: object) -> None:
            raise AssertionError("resume must skip before replay")

    monkeypatch.setattr(_cli, "MoshiBackend", Backend)
    run_hash = target_binding_sha256(binding)
    cell = PatchCell(
        run_hash, "clean", "repair", "resid_post", 0, None, (0,), (0,),
        sha256_file(readouts_path),
    )
    AtomicCellStore(output_root).record(cell, {
        "status": "completed", "anchor": "query_end", "role": "discovery",
        "scenario_id": "s1", "synthetic": False, "delta_M": 0.0,
    })
    assert _cli._scan([
        "--config", str(config_path), "--manifest", str(manifest_path),
        "--encoded-manifest", str(encoded_path), "--anchor-map", str(anchors_path),
        "--role", "discovery", "--layers", "0", "--anchors", "query_end",
        "--donors", "clean_current", "--controls", "self",
        "--readouts", str(readouts_path),
        "--scan-spec", str(spec_path), "--readiness-go", str(go_path),
        "--output-root", str(output_root), "--resume",
    ], "residual") == 0
    assert len(AtomicCellStore(output_root).rows()) == 1
