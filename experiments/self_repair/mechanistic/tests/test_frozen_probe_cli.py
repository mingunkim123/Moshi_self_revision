from __future__ import annotations

from pathlib import Path

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
from experiments.self_repair.mechanistic.scripts import _cli


CONFIG = Path(__file__).resolve().parents[1] / "config/mechanistic.json"


def _rows(role: str, prefix: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_index in range(4):
        for city in ("Boston", "Seattle"):
            row: dict[str, object] = {
                "trial_id": f"{prefix}-{scenario_index}-{city.lower()}",
                "scenario_id": f"{prefix}-scenario-{scenario_index}",
                "condition": "clean_current",
                "old_value": "Denver",
                "new_value": city,
                "frame_count": 4,
                "role": role,
            }
            if role == "discovery":
                row["analysis_fold"] = 1
            rows.append(row)
    return rows


def _manifest_and_roles(
    root: Path, role: str, prefix: str,
) -> tuple[Path, Path | None, list[dict[str, object]]]:
    rows = _rows(role, prefix)
    role_manifest: Path | None = None
    if role == "formal_confirmation":
        role_manifest = root / f"{prefix}-roles.jsonl"
        role_rows = [
            {"trial_id": row["trial_id"], "role": role} for row in rows
        ]
        write_jsonl(role_manifest, role_rows)
        role_sha = sha256_file(role_manifest)
        for row, role_row in zip(rows, role_rows, strict=True):
            row["role_manifest_sha256"] = role_sha
            row["role_binding_sha256"] = sha256_value(role_row)
    manifest = root / f"{prefix}-manifest.jsonl"
    write_jsonl(manifest, rows)
    return manifest, role_manifest, rows


def _capture(root: Path, manifest: Path, rows: list[dict[str, object]], role: str) -> Path:
    anchors = root / f"{role}-anchors.jsonl"
    write_jsonl(anchors, [
        {
            "trial_id": row["trial_id"],
            "anchor": "query_end",
            "frame": 2,
            "time_ms": 240.0,
            "timebase": "prepared_stream_relative",
        }
        for row in rows
    ])
    capture_root = root / f"{role}-captures"
    assert _cli.capture_activations([
        "--config", str(CONFIG),
        "--manifest", str(manifest),
        "--anchor-map", str(anchors),
        "--role", role,
        "--sites", "resid_post",
        "--layers", "1",
        "--anchors", "query_end",
        "--output-root", str(capture_root),
        "--synthetic",
    ]) == 0
    return capture_root


def test_formal_probe_application_never_refits_and_records_diagnostic_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery_manifest, _, discovery_rows = _manifest_and_roles(
        tmp_path, "discovery", "discovery"
    )
    discovery_capture = _capture(
        tmp_path, discovery_manifest, discovery_rows, "discovery"
    )
    frozen_path = tmp_path / "probe-frozen.json"
    assert _cli.fit_probes([
        "--config", str(CONFIG),
        "--manifest", str(discovery_manifest),
        "--capture-root", str(discovery_capture),
        "--role", "discovery",
        "--freeze-output", str(frozen_path),
        "--output-root", str(tmp_path / "probe-fit"),
        "--synthetic",
    ]) == 0
    frozen = read_json(frozen_path)
    assert frozen["training_role"] == "discovery"
    assert {row["row_id"] for row in frozen["training_rows"]} == {
        row["trial_id"] for row in discovery_rows
    }

    formal_manifest, role_manifest, formal_rows = _manifest_and_roles(
        tmp_path, "formal_confirmation", "formal"
    )
    assert role_manifest is not None
    formal_capture = _capture(
        tmp_path, formal_manifest, formal_rows, "formal_confirmation"
    )

    def forbidden_refit(*args, **kwargs):
        del args, kwargs
        raise AssertionError("formal rows reached the probe fitting routine")

    monkeypatch.setattr(_cli, "fit_grouped_ridge_probe", forbidden_refit)
    output_root = tmp_path / "formal-probe"
    assert _cli.fit_probes([
        "--config", str(CONFIG),
        "--manifest", str(formal_manifest),
        "--role-manifest", str(role_manifest),
        "--capture-root", str(formal_capture),
        "--role", "formal_confirmation",
        "--frozen-probe", str(frozen_path),
        "--output-root", str(output_root),
        "--synthetic",
    ]) == 0
    predictions = read_jsonl(output_root / "probe_predictions.jsonl")
    metrics = read_json(output_root / "probe_metrics.json")
    assert len(predictions) == len(formal_rows)
    assert {row["trial_id"] for row in predictions} == {
        row["trial_id"] for row in formal_rows
    }
    assert all(row["diagnostic_only"] is True for row in predictions)
    assert all(row["causal_use_prohibited"] is True for row in predictions)
    assert metrics["mode"] == "apply_without_refit"
    assert metrics["role"] == "formal_confirmation"
    assert metrics["diagnostic_only"] is True
    assert metrics["causal_use_prohibited"] is True


def test_formal_probe_training_and_tampered_frozen_artifact_fail_closed(
    tmp_path: Path,
) -> None:
    formal_manifest, role_manifest, formal_rows = _manifest_and_roles(
        tmp_path, "formal_confirmation", "formal"
    )
    assert role_manifest is not None
    formal_capture = _capture(
        tmp_path, formal_manifest, formal_rows, "formal_confirmation"
    )
    with pytest.raises(ContractError, match="may not train"):
        _cli.fit_probes([
            "--config", str(CONFIG),
            "--manifest", str(formal_manifest),
            "--role-manifest", str(role_manifest),
            "--capture-root", str(formal_capture),
            "--role", "formal_confirmation",
            "--output-root", str(tmp_path / "forbidden-fit"),
            "--synthetic",
        ])

    discovery_manifest, _, discovery_rows = _manifest_and_roles(
        tmp_path, "discovery", "discovery"
    )
    discovery_capture = _capture(
        tmp_path, discovery_manifest, discovery_rows, "discovery"
    )
    frozen_path = tmp_path / "probe-frozen.json"
    assert _cli.fit_probes([
        "--config", str(CONFIG),
        "--manifest", str(discovery_manifest),
        "--capture-root", str(discovery_capture),
        "--role", "discovery",
        "--freeze-output", str(frozen_path),
        "--output-root", str(tmp_path / "probe-fit"),
        "--synthetic",
    ]) == 0
    tampered = read_json(frozen_path)
    tampered["probe"]["weights"][0][0] += 1.0
    tampered_path = tmp_path / "probe-tampered.json"
    write_json(tampered_path, tampered)
    with pytest.raises(ContractError, match="self-hash mismatch"):
        _cli.fit_probes([
            "--config", str(CONFIG),
            "--manifest", str(formal_manifest),
            "--role-manifest", str(role_manifest),
            "--capture-root", str(formal_capture),
            "--role", "formal_confirmation",
            "--frozen-probe", str(tampered_path),
            "--output-root", str(tmp_path / "tampered-apply"),
            "--synthetic",
        ])


def test_fit_and_apply_modes_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _cli.fit_probes([
            "--config", str(CONFIG),
            "--role", "discovery",
            "--freeze-output", str(tmp_path / "fit.json"),
            "--frozen-probe", str(tmp_path / "apply.json"),
            "--output-root", str(tmp_path / "output"),
            "--synthetic",
        ])
