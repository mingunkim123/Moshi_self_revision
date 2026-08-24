#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil

from common import EXPERIMENT_ROOT, write_csv


FIELDS = [
    "blind_id",
    "annotator_id",
    "adjudicator",
    "response_audio_path",
    "inner_text",
    "label",
    "final_target_correct",
    "early_stale_before_repair",
    "stale_after_repair",
    "recovered",
    "response_onset_ms",
    "first_target_ms",
    "first_stale_ms",
    "output_language",
    "intelligible",
    "notes",
]

KEY_FIELDS = ["blind_id", "trial_id", "seed", "speaker_id", "condition_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a condition-blinded annotation CSV from predictions."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=EXPERIMENT_ROOT / "results/predictions.jsonl",
    )
    parser.add_argument(
        "--annotators",
        default="A1,A2",
        help="Comma-separated annotator IDs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "annotations/annotations.csv",
    )
    parser.add_argument(
        "--key-output",
        type=Path,
        default=EXPERIMENT_ROOT / "annotations/annotation_key.csv",
        help="Private unblinding key; do not give this file to annotators.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=EXPERIMENT_ROOT / "annotations/audio",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260824)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output_path in (args.output, args.key_output):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_path}; pass --overwrite"
            )
    annotators = [value.strip() for value in args.annotators.split(",") if value.strip()]
    if not annotators or len(annotators) != len(set(annotators)):
        raise ValueError("Annotator IDs must be a non-empty unique list")

    predictions = []
    with args.predictions.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                predictions.append(json.loads(line))
    if not predictions:
        raise ValueError(f"No predictions found in {args.predictions}")

    random.Random(args.shuffle_seed).shuffle(predictions)
    rows = []
    key_rows = []
    args.audio_root.mkdir(parents=True, exist_ok=True)
    for ordinal, prediction in enumerate(predictions, start=1):
        blind_id = f"B{ordinal:05d}"
        source_audio = Path(prediction["response_audio_path"])
        blind_audio = args.audio_root / f"{blind_id}.wav"
        if blind_audio.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite {blind_audio}; pass --overwrite"
                )
            blind_audio.unlink()
        try:
            blind_audio.hardlink_to(source_audio)
        except OSError:
            shutil.copy2(source_audio, blind_audio)
        key_rows.append(
            {
                "blind_id": blind_id,
                "trial_id": prediction["trial_id"],
                "seed": prediction["seed"],
                "speaker_id": prediction["speaker_id"],
                "condition_id": prediction["condition_id"],
            }
        )
        for annotator in annotators:
            rows.append(
                {
                    "blind_id": blind_id,
                    "annotator_id": annotator,
                    "adjudicator": "0",
                    "response_audio_path": str(blind_audio),
                    "inner_text": prediction.get("inner_text", ""),
                    "label": "",
                    "final_target_correct": "",
                    "early_stale_before_repair": "",
                    "stale_after_repair": "",
                    "recovered": "",
                    "response_onset_ms": "",
                    "first_target_ms": "",
                    "first_stale_ms": "",
                    "output_language": "",
                    "intelligible": "",
                    "notes": "",
                }
            )
    write_csv(args.output, rows, FIELDS)
    write_csv(args.key_output, key_rows, KEY_FIELDS)
    print(f"Wrote {len(rows)} annotation rows to {args.output}")
    print(f"Private unblinding key: {args.key_output}")
    print(f"Blinded audio: {args.audio_root}")
    print("Allowed labels are documented in experiments/self_repair/README.md")


if __name__ == "__main__":
    main()
