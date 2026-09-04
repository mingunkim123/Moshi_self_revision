from __future__ import annotations

import json
from pathlib import Path
import wave

import pytest

from experiments.self_repair.mechanistic.core import (
    ContractError,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
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


def test_reviewed_multivalue_materialization_freezes_full_conversation_contract(
    tmp_path: Path,
) -> None:
    city_config = {
        "split_seed": 17,
        "cities": [
            {
                "value": city,
                "eligible": True,
                "screen_status": "passed_on_excluded_calibration_set",
            }
            for city in ("Boston", "Seattle", "Chicago", "Denver")
        ],
        "design": {
            "formal_scenario_clusters": 1,
            "calibration_scenario_clusters": 1,
            "minimum_speakers": 2,
            "speaker_ids": ["speaker-1", "speaker-2"],
            "conditions": ["clean_current", "repair_immediate", "repair_delayed_640"],
        },
    }
    scenarios = [
        {
            "scenario_id": f"scenario-{index}",
            "root_template": "I want to visit {value}",
            "repair_template": "no, {new}, not {old}",
            "dependent_units": [],
            "closing_prompt": "Can you help me plan it",
        }
        for index in range(2)
    ]
    config_path = tmp_path / "cities.json"
    scenario_path = tmp_path / "scenarios.jsonl"
    output = tmp_path / "controls"
    write_json(config_path, city_config)
    write_jsonl(scenario_path, scenarios)
    assert build_multivalue_controls([
        "--city-config", str(config_path),
        "--scenario-blueprints", str(scenario_path),
        "--output-root", str(output),
    ]) == 0

    reviews = []
    timings = []
    for script in read_jsonl(output / "source_scripts.jsonl"):
        trial_id = script["trial_id"]
        wav = output / "audio" / f"{trial_id}.wav"
        with wave.open(str(wav), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24_000)
            handle.writeframes(b"\0\0" * (12 * 1_920))
        reviews.append({
            "trial_id": trial_id,
            "wav_sha256": sha256_file(wav),
            "alignment_reviewer": "aligner-1",
            "listener_1": "listener-1",
            "listener_1_decision": "passed",
            "listener_2": "listener-2",
            "listener_2_decision": "passed",
            "adjudicator": None,
            "adjudication_decision": None,
            "reviewed_at": "2026-09-04T12:00:00+09:00",
            "status": "passed",
        })
        timings.append({
            "trial_id": trial_id,
            "timebase": "prepared_stream_relative",
            "utterance_end_ms": 640,
            "unit_spans": [],
        })
    write_jsonl(output / "reviews.jsonl", reviews)
    write_jsonl(output / "timing.jsonl", timings)

    assert build_multivalue_controls([
        "--city-config", str(config_path),
        "--scenario-blueprints", str(scenario_path),
        "--output-root", str(output),
    ]) == 0
    prepared = read_jsonl(output / "prepared_stimuli.jsonl")
    assert len(prepared) == len(reviews)
    assert all(
        row["conversation_contract_source"]
        == "reviewed_multivalue_frozen_capture_contract"
        for row in prepared
    )
    assert all(row["capture_contract"]["response_capture_ms"] == 40_000 for row in prepared)
    assert all(
        row["capture_contract"]["target_end_frame_count"] == 508
        for row in prepared
    )
    assert validate_multivalue_controls([
        "--input-root", str(output),
        "--require-independent-alignment",
        "--require-double-listen-review",
    ]) == 0

    reviews[0]["listener_2_decision"] = "failed"
    write_jsonl(output / "reviews.jsonl", reviews)
    with pytest.raises(ContractError, match="disagreement lacks adjudication"):
        validate_multivalue_controls([
            "--input-root", str(output),
            "--require-independent-alignment",
            "--require-double-listen-review",
        ])


def test_multivalue_builder_rejects_unscreened_eligible_city(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    source = json.loads(
        (root / "experiments/self_repair/mechanistic/config/multivalue_cities.json").read_text()
    )
    for city in source["cities"]:
        city["eligible"] = True
        city["screen_status"] = "passed_on_excluded_calibration_set"
    source["cities"][0]["screen_status"] = "pending"
    config = tmp_path / "cities.json"
    write_json(config, source)
    with pytest.raises(ContractError, match="excluded clean-recognition calibration screen"):
        build_multivalue_controls([
            "--city-config", str(config),
            "--scenario-blueprints",
            str(root / "experiments/self_repair/dataset_v2/blueprints/scenarios.jsonl"),
            "--output-root", str(tmp_path / "controls"),
            "--synthetic",
        ])
