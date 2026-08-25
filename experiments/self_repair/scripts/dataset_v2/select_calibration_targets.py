#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
from typing import Any

from common import CONDITIONS, DATASET_ROOT, DEFAULT_SCRIPTS, read_config, read_jsonl, word_count, write_json, write_jsonl
from ids import matched_audio_bundle_id, rendition_target_id


DEFAULT_OUTPUT = DATASET_ROOT / "calibration/edge_private_targets.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select shortest/median/longest private calibration targets.")
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def select_targets(scripts: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for script in scripts:
        by_scenario[str(script["scenario_id"])].append(script)
    if len(by_scenario) != 30 or any(len(rows) != 10 for rows in by_scenario.values()):
        raise ValueError("calibration selection requires 30 scenarios with 10 scripts each")
    lengths = sorted(
        (statistics.mean(word_count(row["transcript"]) for row in rows), scenario_id)
        for scenario_id, rows in by_scenario.items()
    )
    selected = [lengths[0], lengths[(len(lengths) - 1) // 2], lengths[-1]]
    calibration = config["engineering_calibration"]
    source_track_id = str(calibration["source_track_id"])
    targets: list[dict[str, Any]] = []
    for rank, (mean_words, scenario_id) in zip(("shortest", "median", "longest"), selected):
        for script in sorted(by_scenario[scenario_id], key=lambda row: row["script_id"]):
            for speaker in calibration["speakers"]:
                speaker_id = str(speaker["speaker_id"])
                target_id = rendition_target_id(str(script["script_id"]), source_track_id, speaker_id)
                targets.append(
                    {
                        "schema_version": "2.0.0",
                        "rendition_target_id": target_id,
                        "script_id": script["script_id"],
                        "text_bundle_id": script["text_bundle_id"],
                        "matched_audio_bundle_id": matched_audio_bundle_id(
                            str(script["text_bundle_id"]), source_track_id, speaker_id
                        ),
                        "scenario_id": scenario_id,
                        "direction_id": script["direction_id"],
                        "condition": script["condition"],
                        "source_track_id": source_track_id,
                        "speaker_id": speaker_id,
                        "voice": speaker["voice"],
                        "inferential_role": "engineering_calibration_only",
                        "length_rank": rank,
                        "scenario_mean_words": mean_words,
                        "generation_seed": config["generation_seed"],
                    }
                )
    if len(targets) != 60 or {row["condition"] for row in targets} != set(CONDITIONS):
        raise AssertionError("expected 60 calibration targets covering all conditions")
    report = {
        "schema_version": "2.0.0",
        "source_track_id": source_track_id,
        "release_eligible": False,
        "selection": [
            {"length_rank": rank, "scenario_id": scenario_id, "mean_words": mean_words}
            for rank, (mean_words, scenario_id) in zip(("shortest", "median", "longest"), selected)
        ],
        "speaker_ids": [row["speaker_id"] for row in calibration["speakers"]],
        "target_count": len(targets),
        "candidate_count_if_fully_run": len(targets) * int(calibration["candidates_per_target"]),
    }
    return sorted(targets, key=lambda row: row["rendition_target_id"]), report


def main() -> None:
    args = parse_args()
    targets, report = select_targets(read_jsonl(args.scripts), read_config())
    write_jsonl(args.output, targets)
    write_json(args.report or args.output.with_suffix(".report.json"), report)
    print(f"Selected {len(targets)} private calibration targets -> {args.output}")


if __name__ == "__main__":
    main()
