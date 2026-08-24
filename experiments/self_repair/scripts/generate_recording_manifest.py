#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    EXPERIMENT_ROOT,
    parse_bool,
    read_csv,
    resolve_experiment_path,
    validate_id,
    write_csv,
)


FIELDS = [
    "trial_id",
    "speaker_id",
    "condition_id",
    "raw_audio_path",
    "repair_marker_onset_ms",
    "repair_onset_ms",
    "repair_end_ms",
    "user_end_ms",
    "insert_silence_at_ms",
    "insert_silence_ms",
    "accepted",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand experiment conditions into a recording manifest."
    )
    parser.add_argument(
        "--conditions",
        type=Path,
        default=EXPERIMENT_ROOT / "data/conditions.csv",
    )
    speakers = parser.add_mutually_exclusive_group(required=True)
    speakers.add_argument("--speakers", help="Comma-separated speaker IDs, e.g. S01,S02")
    speakers.add_argument("--speakers-file", type=Path, help="One speaker ID per line")
    parser.add_argument(
        "--condition-set",
        choices=("smoke", "pilot", "all"),
        default="pilot",
    )
    parser.add_argument(
        "--languages",
        default="ko",
        help="Comma-separated language codes, or 'all'",
    )
    parser.add_argument(
        "--tracks",
        default="natural",
        help="Comma-separated tracks, or 'all'",
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Raw audio root, relative to the experiment directory unless absolute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "data/recordings.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_speakers(args: argparse.Namespace) -> list[str]:
    if args.speakers:
        values = [item.strip() for item in args.speakers.split(",") if item.strip()]
    else:
        values = [
            line.strip()
            for line in args.speakers_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not values:
        raise ValueError("At least one speaker ID is required")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate speaker IDs are not allowed")
    return [validate_id(value, "speaker_id") for value in values]


def selected(value: str, requested: set[str] | None) -> bool:
    return requested is None or value in requested


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")

    conditions = read_csv(args.conditions)
    speakers = load_speakers(args)
    languages = None if args.languages == "all" else set(args.languages.split(","))
    tracks = None if args.tracks == "all" else set(args.tracks.split(","))
    raw_root = resolve_experiment_path(args.raw_root)

    chosen: list[dict[str, str]] = []
    seen_conditions: set[str] = set()
    for condition in conditions:
        condition_id = validate_id(condition["condition_id"], "condition_id")
        if condition_id in seen_conditions:
            raise ValueError(f"Duplicate condition_id: {condition_id}")
        seen_conditions.add(condition_id)
        if args.condition_set != "all" and not parse_bool(
            condition.get(args.condition_set), default=False
        ):
            continue
        if not selected(condition["language"], languages):
            continue
        if not selected(condition["track"], tracks):
            continue
        chosen.append(condition)

    if not chosen:
        raise ValueError("No conditions matched the requested filters")

    rows = []
    for speaker_id in speakers:
        for condition in chosen:
            condition_id = condition["condition_id"]
            relative_audio = raw_root / speaker_id / f"{condition_id}.wav"
            try:
                relative_audio = relative_audio.relative_to(EXPERIMENT_ROOT)
            except ValueError:
                pass
            rows.append(
                {
                    "trial_id": f"{speaker_id}__{condition_id}",
                    "speaker_id": speaker_id,
                    "condition_id": condition_id,
                    "raw_audio_path": str(relative_audio),
                    "repair_marker_onset_ms": "",
                    "repair_onset_ms": "",
                    "repair_end_ms": "",
                    "user_end_ms": "",
                    "insert_silence_at_ms": "",
                    "insert_silence_ms": "",
                    "accepted": "1",
                    "notes": "",
                }
            )

    write_csv(args.output, rows, FIELDS)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Place recordings under {raw_root}/<speaker_id>/<condition_id>.wav")


if __name__ == "__main__":
    main()
