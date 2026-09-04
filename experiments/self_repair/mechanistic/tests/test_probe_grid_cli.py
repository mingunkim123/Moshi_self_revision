from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.self_repair.mechanistic.core import (
    ContractError,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
    write_jsonl,
)
from experiments.self_repair.mechanistic.scripts import _cli


CONFIG = Path(__file__).resolve().parents[1] / "config/mechanistic.json"


def _manifest(root: Path, *, role: str, prefix: str) -> tuple[Path, Path | None, list[dict]]:
    rows: list[dict] = []
    for scenario in range(4):
        for city in ("Boston", "Seattle"):
            row = {
                "trial_id": f"{prefix}-{scenario}-{city.lower()}",
                "scenario_id": f"{prefix}-scenario-{scenario}",
                "condition": "clean_current",
                "old_value": "Denver",
                "new_value": city,
                "frame_count": 4,
                "role": role,
            }
            if role == "discovery":
                row["analysis_fold"] = 1
            rows.append(row)
    role_manifest = None
    if role == "formal_confirmation":
        role_manifest = root / f"{prefix}-roles.jsonl"
        role_rows = [{"trial_id": row["trial_id"], "role": role} for row in rows]
        write_jsonl(role_manifest, role_rows)
        role_sha = sha256_file(role_manifest)
        for row, role_row in zip(rows, role_rows, strict=True):
            row["role_manifest_sha256"] = role_sha
            row["role_binding_sha256"] = sha256_value(role_row)
    path = root / f"{prefix}-manifest.jsonl"
    write_jsonl(path, rows)
    return path, role_manifest, rows


def _capture_grid(
    root: Path, manifest: Path, rows: list[dict], *, role: str,
) -> Path:
    anchors = root / f"{role}-anchors.jsonl"
    write_jsonl(anchors, [
        {
            "trial_id": row["trial_id"],
            "anchor": anchor,
            "frame": frame,
            "time_ms": frame * 80.0,
            "timebase": "prepared_stream_relative",
        }
        for row in rows
        for anchor, frame in (("new_end", 1), ("query_end", 2))
    ])
    capture_root = root / f"{role}-captures"
    assert _cli.capture_activations([
        "--config", str(CONFIG),
        "--manifest", str(manifest),
        "--anchor-map", str(anchors),
        "--role", role,
        "--sites", "resid_post",
        "--layers", "1,2",
        "--anchors", "new_end,query_end",
        "--output-root", str(capture_root),
        "--synthetic",
    ]) == 0
    _rewrite_grid_vectors(capture_root)
    return capture_root


def _rewrite_grid_vectors(capture_root: Path) -> None:
    """Make two grid cells diagnostic and two cells intentionally uninformative."""

    rows = read_jsonl(capture_root / "capture_manifest.jsonl")
    for row in rows:
        path = capture_root / str(row["feature_uri"])
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        metadata = json.loads(np.asarray(arrays["capture_metadata_json"]).item())
        positive = 6.0 if row["label"] == "Boston" else -6.0
        for descriptor in metadata["tensors"]:
            if descriptor["site"] != "resid_post":
                continue
            coordinate = (descriptor["layer"], descriptor["anchors"][0])
            value = np.zeros(descriptor["shape"], dtype=np.dtype(descriptor["dtype"]))
            if coordinate in {(1, "new_end"), (2, "query_end")}:
                value.reshape(-1)[0] = positive
            arrays[descriptor["key"]] = value
            descriptor["sha256"] = _cli._array_sha256(value)
        arrays["capture_metadata_json"] = np.asarray(canonical_json(metadata))
        _cli._atomic_savez(path, **arrays)
        row["feature_sha256"] = sha256_file(path)
    write_jsonl(capture_root / "capture_manifest.jsonl", rows)


def _selection(path: Path) -> Path:
    body = {
        "schema_version": "1.0.0",
        "status": "frozen_discovery_selection",
        "config_sha256": sha256_file(CONFIG),
        "component": "resid_post",
        "layer": 1,
        "head": None,
        "anchor": "new_end",
    }
    write_json(path, {**body, "selection_sha256": sha256_value(body)})
    return path


def _fit_grid(
    *, manifest: Path, capture_root: Path, selection: Path, output_root: Path,
    freeze_output: Path | None = None,
) -> None:
    args = [
        "--config", str(CONFIG),
        "--manifest", str(manifest),
        "--capture-root", str(capture_root),
        "--role", "discovery",
        "--probe-grid",
        "--sites", "resid_post",
        "--layers", "2,1",
        "--anchors", "query_end,new_end",
        "--site-selection", str(selection),
        "--folds", "4",
        "--output-root", str(output_root),
        "--synthetic",
    ]
    if freeze_output is not None:
        args.extend(["--freeze-output", str(freeze_output)])
    assert _cli.fit_probes(args) == 0


def test_probe_grid_localizes_two_layers_by_two_anchors_and_freezes_exact_cell(
    tmp_path: Path,
) -> None:
    manifest, _, rows = _manifest(tmp_path, role="discovery", prefix="discovery")
    capture_root = _capture_grid(tmp_path, manifest, rows, role="discovery")
    selection = _selection(tmp_path / "selection.json")
    frozen_path = tmp_path / "frozen-probe.json"
    output_root = tmp_path / "grid"
    _fit_grid(
        manifest=manifest,
        capture_root=capture_root,
        selection=selection,
        output_root=output_root,
        freeze_output=frozen_path,
    )

    reports = read_jsonl(output_root / "probe_grid_metrics.jsonl")
    grid_manifest = read_json(output_root / "probe_grid_manifest.json")
    assert len(reports) == grid_manifest["expected_cell_count"] == 4
    by_coordinate = {
        (row["probe_coordinate"]["layer"], row["probe_coordinate"]["anchor"]): row
        for row in reports
    }
    assert by_coordinate[(1, "new_end")]["cv_accuracy"] == 1.0
    assert by_coordinate[(2, "query_end")]["cv_accuracy"] == 1.0
    assert by_coordinate[(1, "query_end")]["cv_accuracy"] == 0.5
    assert by_coordinate[(2, "new_end")]["cv_accuracy"] == 0.5
    assert all(
        row["capture_contract"]["feature_policy"]
        == "flatten_one_exact_captured_tensor"
        for row in reports
    )
    frozen = read_json(frozen_path)
    assert frozen["probe_coordinate"] == {
        "site": "resid_post", "layer": 1, "anchor": "new_end"
    }
    assert frozen["capture_contract"]["probe_coordinate"] == frozen["probe_coordinate"]


def test_probe_grid_refuses_to_choose_a_probe_when_freezing(tmp_path: Path) -> None:
    manifest, _, rows = _manifest(tmp_path, role="discovery", prefix="discovery")
    capture_root = _capture_grid(tmp_path, manifest, rows, role="discovery")
    with pytest.raises(ContractError, match="requires a frozen --selection"):
        _cli.fit_probes([
            "--config", str(CONFIG),
            "--manifest", str(manifest),
            "--capture-root", str(capture_root),
            "--role", "discovery",
            "--probe-grid",
            "--sites", "resid_post",
            "--layers", "1,2",
            "--anchors", "new_end,query_end",
            "--freeze-output", str(tmp_path / "must-not-exist.json"),
            "--output-root", str(tmp_path / "grid"),
            "--synthetic",
        ])


def test_probe_grid_fit_is_deterministic_under_capture_row_reordering(
    tmp_path: Path,
) -> None:
    manifest, _, rows = _manifest(tmp_path, role="discovery", prefix="discovery")
    capture_root = _capture_grid(tmp_path, manifest, rows, role="discovery")
    selection = _selection(tmp_path / "selection.json")
    first = tmp_path / "first"
    _fit_grid(
        manifest=manifest, capture_root=capture_root, selection=selection, output_root=first
    )
    capture_manifest = capture_root / "capture_manifest.jsonl"
    write_jsonl(capture_manifest, list(reversed(read_jsonl(capture_manifest))))
    second = tmp_path / "second"
    _fit_grid(
        manifest=manifest, capture_root=capture_root, selection=selection, output_root=second
    )
    first_rows = read_jsonl(first / "probe_grid_metrics.jsonl")
    second_rows = read_jsonl(second / "probe_grid_metrics.jsonl")
    for left, right in zip(first_rows, second_rows, strict=True):
        assert left["probe_coordinate"] == right["probe_coordinate"]
        assert left["training_rows"] == right["training_rows"]
        assert left["cv_predictions"] == right["cv_predictions"]
        assert left["probe"] == right["probe"]
        assert left["probe_feature_dataset_sha256"] == right["probe_feature_dataset_sha256"]
        assert left["probe_grid_cell_sha256"] == right["probe_grid_cell_sha256"]
    assert (
        read_json(first / "probe_grid_manifest.json")["probe_grid_semantic_identity_sha256"]
        == read_json(second / "probe_grid_manifest.json")["probe_grid_semantic_identity_sha256"]
    )


def test_frozen_exact_probe_application_loads_selected_tensor_without_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery_manifest, _, discovery_rows = _manifest(
        tmp_path, role="discovery", prefix="discovery"
    )
    discovery_capture = _capture_grid(
        tmp_path, discovery_manifest, discovery_rows, role="discovery"
    )
    selection = _selection(tmp_path / "selection.json")
    frozen_path = tmp_path / "frozen-probe.json"
    _fit_grid(
        manifest=discovery_manifest,
        capture_root=discovery_capture,
        selection=selection,
        output_root=tmp_path / "fit",
        freeze_output=frozen_path,
    )

    formal_manifest, role_manifest, formal_rows = _manifest(
        tmp_path, role="formal_confirmation", prefix="formal"
    )
    assert role_manifest is not None
    formal_capture = _capture_grid(
        tmp_path, formal_manifest, formal_rows, role="formal_confirmation"
    )

    def forbidden_refit(*args, **kwargs):
        del args, kwargs
        raise AssertionError("frozen application attempted to refit")

    monkeypatch.setattr(_cli, "fit_grouped_ridge_probe", forbidden_refit)
    output_root = tmp_path / "apply"
    assert _cli.fit_probes([
        "--config", str(CONFIG),
        "--manifest", str(formal_manifest),
        "--role-manifest", str(role_manifest),
        "--capture-root", str(formal_capture),
        "--role", "formal_confirmation",
        "--site-selection", str(selection),
        "--frozen-probe", str(frozen_path),
        "--output-root", str(output_root),
        "--synthetic",
    ]) == 0
    predictions = read_jsonl(output_root / "probe_predictions.jsonl")
    metrics = read_json(output_root / "probe_metrics.json")
    assert metrics["mode"] == "apply_without_refit"
    assert metrics["accuracy"] == 1.0
    assert metrics["probe_coordinate"] == {
        "site": "resid_post", "layer": 1, "anchor": "new_end"
    }
    assert all(row["probe_tensor_sha256"] for row in predictions)
