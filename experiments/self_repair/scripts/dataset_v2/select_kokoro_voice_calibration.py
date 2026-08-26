#!/usr/bin/env python3
"""Create one matched audition target per frozen Kokoro voice.

This is deliberately smaller than the production matrix.  It compares every
candidate voice on the same median-length, dependency-heavy script before any
voice is admitted to a release-eligible source track.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
from typing import Any

from common import DATASET_ROOT, DEFAULT_SCRIPTS, read_config, read_jsonl, word_count, write_json, write_jsonl
from ids import matched_audio_bundle_id, rendition_target_id


DEFAULT_OUTPUT = DATASET_ROOT / "calibration/kokoro_voice_targets.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the frozen common-script Kokoro voice calibration matrix."
    )
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def select_targets(
    scripts: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration = config.get("open_source_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("dataset config is missing open_source_calibration")
    speakers = calibration.get("speakers")
    if not isinstance(speakers, list) or len(speakers) != 10:
        raise ValueError("Kokoro calibration requires exactly 10 frozen speakers")
    speaker_ids = [str(row.get("speaker_id")) for row in speakers]
    voices = [str(row.get("voice")) for row in speakers]
    if len(set(speaker_ids)) != 10 or len(set(voices)) != 10:
        raise ValueError("Kokoro speaker IDs and voices must be unique")

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for script in scripts:
        by_scenario[str(script["scenario_id"])].append(script)
    if len(by_scenario) != 30 or any(len(rows) != 10 for rows in by_scenario.values()):
        raise ValueError("voice calibration requires the frozen 30x10 script matrix")
    lengths = sorted(
        (
            statistics.mean(word_count(row["transcript"]) for row in rows),
            scenario_id,
        )
        for scenario_id, rows in by_scenario.items()
    )
    scenario_rank = str(calibration["voice_calibration_script"]["scenario_rank"])
    if scenario_rank != "median":
        raise ValueError("only the preregistered median scenario rank is supported")
    mean_words, scenario_id = lengths[(len(lengths) - 1) // 2]
    direction_id = str(calibration["voice_calibration_script"]["direction_id"])
    condition = str(calibration["voice_calibration_script"]["condition"])
    matching = [
        row
        for row in by_scenario[scenario_id]
        if row["direction_id"] == direction_id and row["condition"] == condition
    ]
    if len(matching) != 1:
        raise ValueError("frozen Kokoro calibration script did not resolve uniquely")
    script = matching[0]
    source_track_id = str(calibration["source_track_id"])
    targets: list[dict[str, Any]] = []
    for speaker in speakers:
        speaker_id = str(speaker["speaker_id"])
        targets.append(
            {
                "schema_version": "2.0.0",
                "rendition_target_id": rendition_target_id(
                    str(script["script_id"]), source_track_id, speaker_id
                ),
                "script_id": script["script_id"],
                "text_bundle_id": script["text_bundle_id"],
                "matched_audio_bundle_id": matched_audio_bundle_id(
                    str(script["text_bundle_id"]), source_track_id, speaker_id
                ),
                "scenario_id": script["scenario_id"],
                "direction_id": direction_id,
                "condition": condition,
                "source_track_id": source_track_id,
                "speaker_id": speaker_id,
                "voice": speaker["voice"],
                "voice_sha256": speaker["voice_sha256"],
                "published_grade": speaker["published_grade"],
                "inferential_role": "open_source_voice_calibration_only",
                "scenario_rank": scenario_rank,
                "scenario_mean_words": mean_words,
                "generation_seed": config["generation_seed"],
            }
        )
    report = {
        "schema_version": "2.0.0",
        "source_track_id": source_track_id,
        "provider": calibration["provider"],
        "release_eligible": False,
        "model_repo": calibration["model_repo"],
        "model_revision": calibration["model_revision"],
        "model_sha256": calibration["model_sha256"],
        "selected_script_id": script["script_id"],
        "selected_script_word_count": word_count(script["transcript"]),
        "scenario_mean_words": mean_words,
        "speaker_ids": speaker_ids,
        "voices": voices,
        "target_count": len(targets),
        "candidate_count": len(targets),
        "candidate_policy": calibration["candidate_policy"],
    }
    return sorted(targets, key=lambda row: row["rendition_target_id"]), report


def main() -> None:
    args = parse_args()
    targets, report = select_targets(read_jsonl(args.scripts), read_config())
    write_jsonl(args.output, targets)
    write_json(args.report or args.output.with_suffix(".report.json"), report)
    print(f"Selected {len(targets)} Kokoro voice targets -> {args.output}")


if __name__ == "__main__":
    main()
