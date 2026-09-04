from __future__ import annotations

from pathlib import Path

import jsonschema
import pytest

from experiments.self_repair.mechanistic.analysis_protocol import (
    analyze_frozen_contrasts,
    freeze_analysis_artifacts,
    load_frozen_analysis_inputs,
)
from experiments.self_repair.mechanistic.core import (
    ContractError,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
)
from experiments.self_repair.mechanistic.scripts import _cli
from experiments.self_repair.mechanistic.core import write_jsonl
from experiments.self_repair.mechanistic.verification import verify_analysis_provenance


def _rows(*, missing: bool = False, failed: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in range(8):
        for intervention, arm, delta in (
            ("patched", "relevant", 1.4),
            ("baseline", "relevant", 0.1),
            ("patched", "control", 0.2),
            ("baseline", "control", 0.1),
        ):
            cell_id = f"s{scenario}:{intervention}:{arm}"
            if missing and cell_id == "s7:baseline:control":
                continue
            rows.append({
                "cell_id": cell_id,
                "scenario_id": f"s{scenario}",
                "status": "failed" if failed and cell_id == "s7:baseline:control" else "completed",
                "intervention": intervention,
                "donor_arm": arm,
                "delta_M": delta + scenario * 0.01,
                "synthetic": True,
            })
    return rows


def _hypothesis() -> list[dict[str, object]]:
    return [{
        "id": "primary_did",
        "family": "primary",
        "direction": "positive",
        "sesoi": 0.1,
        "terms": [
            {"selector": {"intervention": "patched", "donor_arm": "relevant"}, "weight": 1},
            {"selector": {"intervention": "baseline", "donor_arm": "relevant"}, "weight": -1},
            {"selector": {"intervention": "patched", "donor_arm": "control"}, "weight": -1},
            {"selector": {"intervention": "baseline", "donor_arm": "control"}, "weight": 1},
        ],
    }]


def test_did_bootstrap_holm_and_sesoi_gate() -> None:
    rows = _rows()
    result = analyze_frozen_contrasts(
        rows, _hypothesis(), expected_cell_ids=[str(row["cell_id"]) for row in rows],
        bootstrap_replicates=500, seed=19,
    )
    assert result["analysis_status"] == "synthetic_local_validation"
    assert result["all_cells_complete"] is True
    assert result["registry"][0]["estimate"] == pytest.approx(1.2)
    assert result["registry"][0]["holm_p"] <= 0.05
    assert result["registry"][0]["passes_sesoi_ci"] is True
    assert result["passed"] is True


@pytest.mark.parametrize("mode", ["missing", "failed"])
def test_analysis_fails_closed_for_incomplete_cells(mode: str) -> None:
    complete = _rows()
    rows = _rows(missing=mode == "missing", failed=mode == "failed")
    with pytest.raises(ContractError):
        analyze_frozen_contrasts(
            rows,
            _hypothesis(),
            expected_cell_ids=[str(row["cell_id"]) for row in complete],
            bootstrap_replicates=100,
        )


def test_analysis_never_labels_real_rows_synthetic() -> None:
    rows = _rows()
    for row in rows:
        row["synthetic"] = False
    result = analyze_frozen_contrasts(rows, _hypothesis(), bootstrap_replicates=100)
    assert result["analysis_status"] == "empirical_requires_gate_review"


@pytest.mark.parametrize(
    ("filename", "role", "prefix"),
    [
        ("analysis_plan.template.json", "formal_confirmation", "formal_"),
        ("internal_analysis_plan.template.json", "internal_validation", "internal_"),
    ],
)
def test_tracked_analysis_templates_bind_one_role(
    filename: str, role: str, prefix: str,
) -> None:
    template = read_json(Path(__file__).resolve().parents[1] / "config" / filename)
    assert template["analysis_kind"] == "empirical_preregistered"
    assert template["hypotheses"]
    assert all(item["id"].startswith(prefix) for item in template["hypotheses"])
    selectors = [
        term["selector"]
        for hypothesis in template["hypotheses"]
        for term in hypothesis["terms"]
    ]
    assert selectors
    assert {selector["role"] for selector in selectors} == {role}


def test_analysis_rejects_mixed_synthetic_provenance() -> None:
    rows = _rows()
    rows[0]["synthetic"] = False
    with pytest.raises(ContractError, match="cannot mix synthetic and empirical"):
        analyze_frozen_contrasts(rows, _hypothesis(), bootstrap_replicates=100)


def test_empirical_cli_requires_frozen_spec_and_expected_grid(tmp_path) -> None:
    result_root = tmp_path / "run" / "formal"
    result_root.mkdir(parents=True)
    rows = _rows()
    for row in rows:
        row["synthetic"] = False
    write_jsonl(result_root / "residual_patch_results.jsonl", rows)
    with pytest.raises(ContractError, match="--analysis-spec"):
        _cli.analyze([
            "--run-root", str(tmp_path / "run"),
            "--bootstrap-replicates", "100",
        ])


def _frozen_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, list[Path]]:
    root = tmp_path / "run"
    config = root / "config.json"
    template = root / "analysis.template.json"
    selection = root / "mechanistic_frozen_selection.json"
    write_json(config, {"experiment_id": "fixture"})
    selection_body = {
        "schema_version": "1.0.0",
        "status": "frozen_discovery_selection",
        "config_sha256": sha256_file(config),
        "component": "resid_post",
        "layer": 17,
        "head": None,
        "anchor": "new_end",
        "direction": "target_minus_stale",
        "donor_arm": "clean_current",
        "relation": "clean_current",
        "readout_sha256": "1" * 64,
        "selection_source_cell_id": "2" * 64,
    }
    selection_body["selection_sha256"] = sha256_value(selection_body)
    write_json(selection, selection_body)
    write_json(template, {
        "schema_version": "1.0.0",
        "analysis_kind": "empirical_preregistered",
        "alpha": 0.05,
        "bootstrap_replicates": 500,
        "bootstrap_seed": 19,
        "cluster_key": "scenario_id",
        "hypotheses": [{
            "id": "formal_did",
            "family": "primary",
            "direction": "positive",
            "sesoi": 0.1,
            "terms": [
                {"selector": {"role": "formal_confirmation", "donor_arm": "clean_current"},
                 "metric": "patched_M", "weight": 1},
                {"selector": {"role": "formal_confirmation", "donor_arm": "clean_current"},
                 "metric": "baseline_M", "weight": -1},
                {"selector": {"role": "formal_confirmation", "donor_arm": "self"},
                 "metric": "patched_M", "weight": -1},
                {"selector": {"role": "formal_confirmation", "donor_arm": "self"},
                 "metric": "baseline_M", "weight": 1},
            ],
        }],
    })
    planned_paths: list[Path] = []
    for arm in ("clean_current", "self"):
        directory = root / "formal" / arm
        rows = []
        for scenario in range(8):
            identity = {
                "run_identity_sha256": "3" * 64,
                "donor_trial_id": f"donor-{arm}-{scenario}",
                "recipient_trial_id": f"repair-{scenario}",
                "component": "resid_post",
                "layer": 17,
                "head": None,
                "source_frames": [10],
                "target_frames": [10],
                "readout_sha256": "1" * 64,
                "donor_arm": arm,
                "relation": arm,
                "anchor": "new_end",
            }
            rows.append({"schema_version": "1.0.0", "cell_id": sha256_value(identity), **identity})
        planned = directory / "planned_cells.jsonl"
        write_jsonl(planned, rows)
        write_json(directory / "scan_plan.json", {
            "schema_version": "1.0.0",
            "kind": "residual",
            "role": "formal_confirmation",
            "planned_cell_count": len(rows),
            "planned_cells_sha256": sha256_value(rows),
            "result_uri": "residual_patch_results.jsonl",
            "provenance": {"selection_file_sha256": sha256_file(selection)},
        })
        planned_paths.append(planned)
    return root, config, template, selection, planned_paths


def _materialize_results(root: Path, planned_paths: list[Path]) -> None:
    for planned in planned_paths:
        rows = []
        for index, plan in enumerate(read_jsonl(planned)):
            arm = str(plan["donor_arm"])
            baseline = 0.2 + index * 0.01
            patched = baseline + (1.5 if arm == "clean_current" else 0.1)
            rows.append({
                **plan,
                "status": "completed",
                "role": "formal_confirmation",
                "scenario_id": f"s{index}",
                "baseline_M": baseline,
                "patched_M": patched,
                "delta_M": patched - baseline,
                "synthetic": False,
            })
        write_jsonl(planned.parent / "residual_patch_results.jsonl", rows)


def test_freeze_multi_plan_and_run_exact_empirical_did(tmp_path: Path) -> None:
    root, config, template, selection, planned = _frozen_fixture(tmp_path)
    spec_path = root / "analysis" / "analysis_spec.json"
    expected_path = root / "analysis" / "expected_cells.json"
    assert _cli.freeze_analysis_plan([
        "--run-root", str(root),
        "--config", str(config),
        "--template", str(template),
        "--selection", str(selection),
        "--planned-cells", str(planned[1]),
        "--planned-cells", str(planned[0]),
        "--output-spec", str(spec_path),
        "--output-expected-cells", str(expected_path),
    ]) == 0
    spec = read_json(spec_path)
    expected = read_json(expected_path)
    assert expected["cell_count"] == 16
    assert len(expected["source_plans"]) == 2
    assert spec["analysis_spec_sha256"] == sha256_value({
        key: value for key, value in spec.items() if key != "analysis_spec_sha256"
    })

    schemas = Path(__file__).resolve().parents[1] / "schemas"
    jsonschema.validate(spec, read_json(schemas / "analysis-spec.schema.json"))
    jsonschema.validate(expected, read_json(schemas / "expected-cells.schema.json"))

    _materialize_results(root, planned)
    assert _cli.analyze([
        "--run-root", str(root),
        "--config", str(config),
        "--analysis-spec", str(spec_path),
    ]) == 0
    summary = read_json(root / "reports/mechanistic_discovery_summary.json")
    assert summary["analysis_status"] == "empirical_requires_gate_review"
    assert summary["n_cells"] == 16
    assert summary["registry"][0]["estimate"] == pytest.approx(1.4)
    assert summary["registry"][0]["passed"] is True
    assert summary["expected_cells_value_sha256"] == expected["expected_cells_sha256"]
    verify_analysis_provenance(root, summary)


def test_freeze_requires_pristine_plans_and_is_immutable(tmp_path: Path) -> None:
    root, config, template, selection, planned = _frozen_fixture(tmp_path)
    write_jsonl(planned[0].parent / "residual_patch_results.jsonl", [{"cell_id": "late"}])
    with pytest.raises(ContractError, match="before results exist"):
        freeze_analysis_artifacts(
            run_root=root,
            config_path=config,
            template_path=template,
            selection_paths=[selection],
            planned_cell_paths=planned,
            output_spec=root / "analysis/spec.json",
            output_expected_cells=root / "analysis/expected.json",
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "role"])
def test_empirical_analysis_rejects_missing_extra_or_relabeled_cells(
    tmp_path: Path, mutation: str,
) -> None:
    root, config, template, selection, planned = _frozen_fixture(tmp_path)
    spec_path = root / "analysis/spec.json"
    expected_path = root / "analysis/expected.json"
    freeze_analysis_artifacts(
        run_root=root, config_path=config, template_path=template,
        selection_paths=[selection], planned_cell_paths=planned,
        output_spec=spec_path, output_expected_cells=expected_path,
    )
    _materialize_results(root, planned)
    target = planned[0].parent / "residual_patch_results.jsonl"
    rows = read_jsonl(target)
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append({**rows[0], "cell_id": "f" * 64})
    else:
        rows[0]["role"] = "discovery"
    write_jsonl(target, rows)
    with pytest.raises(ContractError, match=(
        "coverage mismatch" if mutation != "role" else "changed frozen identity field role"
    )):
        _cli.analyze([
            "--run-root", str(root), "--config", str(config),
            "--analysis-spec", str(spec_path),
        ])


def test_empirical_analysis_rejects_posthoc_statistics_override(tmp_path: Path) -> None:
    root, config, template, selection, planned = _frozen_fixture(tmp_path)
    spec_path = root / "analysis/spec.json"
    expected_path = root / "analysis/expected.json"
    freeze_analysis_artifacts(
        run_root=root, config_path=config, template_path=template,
        selection_paths=[selection], planned_cell_paths=planned,
        output_spec=spec_path, output_expected_cells=expected_path,
    )
    _materialize_results(root, planned)
    with pytest.raises(ContractError, match="differs from the frozen"):
        _cli.analyze([
            "--run-root", str(root), "--config", str(config),
            "--analysis-spec", str(spec_path), "--bootstrap-seed", "20",
        ])


def test_frozen_inputs_reauthenticate_source_plans(tmp_path: Path) -> None:
    root, config, template, selection, planned = _frozen_fixture(tmp_path)
    spec_path = root / "analysis/spec.json"
    expected_path = root / "analysis/expected.json"
    freeze_analysis_artifacts(
        run_root=root, config_path=config, template_path=template,
        selection_paths=[selection], planned_cell_paths=planned,
        output_spec=spec_path, output_expected_cells=expected_path,
    )
    _materialize_results(root, planned)
    plan_rows = read_jsonl(planned[0])
    plan_rows[0]["anchor"] = "query_end"
    write_jsonl(planned[0], plan_rows)
    with pytest.raises(ContractError, match="planned-cells file is missing or changed"):
        load_frozen_analysis_inputs(
            run_root=root, analysis_spec_path=spec_path,
        )


def test_scan_plan_only_never_constructs_backend(tmp_path: Path, monkeypatch) -> None:
    config = Path(__file__).resolve().parents[1] / "config/mechanistic.json"

    class ForbiddenBackend:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("plan-only constructed a backend")

    monkeypatch.setattr(_cli, "SyntheticBackend", ForbiddenBackend)
    output = tmp_path / "planned"
    assert _cli._scan([
        "--config", str(config),
        "--role", "formal_confirmation",
        "--output-root", str(output),
        "--layers", "17",
        "--anchors", "new_end",
        "--donors", "clean_current,self",
        "--synthetic",
        "--plan-only",
    ], "residual") == 0
    assert (output / "planned_cells.jsonl").is_file()
    assert (output / "scan_plan.json").is_file()
    assert not (output / "residual_patch_results.jsonl").exists()
    assert not list((output / "cells").glob("*.json"))
