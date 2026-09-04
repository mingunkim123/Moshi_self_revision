from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.self_repair.mechanistic.core import ContractError, read_jsonl, write_json
from experiments.self_repair.mechanistic.scripts._cli import build_multivalue_controls, validate_multivalue_controls


def test_multivalue_builder_is_role_isolated_and_review_fail_closed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    source = json.loads((root / "experiments/self_repair/mechanistic/config/multivalue_cities.json").read_text())
    for city in source["cities"]:
        city["eligible"] = True
        city["screen_status"] = "passed_on_excluded_calibration_set"
    config = tmp_path / "cities.json"
    write_json(config, source)
    output = tmp_path / "controls"
    assert build_multivalue_controls([
        "--city-config", str(config),
        "--scenario-blueprints", str(root / "experiments/self_repair/dataset_v2/blueprints/scenarios.jsonl"),
        "--output-root", str(output),
        "--synthetic",
    ]) == 0
    roles = read_jsonl(output / "role_manifest.jsonl")
    pairs = {}
    scenarios = {}
    for row in roles:
        pairs.setdefault(row["ordered_pair"], set()).add(row["role"])
        scenarios.setdefault(row["scenario_id"], set()).add(row["role"])
    assert all(len(value) == 1 for value in pairs.values())
    assert all(len(value) == 1 for value in scenarios.values())
    with pytest.raises(ContractError, match="reviews.jsonl is missing"):
        validate_multivalue_controls(["--input-root", str(output), "--require-double-listen-review"])
