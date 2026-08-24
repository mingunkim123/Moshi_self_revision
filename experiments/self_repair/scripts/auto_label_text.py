#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from common import EXPERIMENT_ROOT, write_csv
from make_annotation_sheet import FIELDS, KEY_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create conservative preliminary labels from Moshi inner-text."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=EXPERIMENT_ROOT / "results_en/predictions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "annotations/annotations.en.auto.csv",
    )
    parser.add_argument(
        "--key-output",
        type=Path,
        default=EXPERIMENT_ROOT / "annotations/annotation_key.en.auto.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def occurrences(text: str, term: str) -> list[int]:
    return [
        match.start()
        for match in re.finditer(rf"\b{re.escape(term.lower())}\b", text.lower())
    ]


def classify(record: dict[str, object]) -> dict[str, object]:
    text = str(record.get("inner_text") or "")
    targets = [value.strip().lower() for value in str(record["target"]).split("|")]
    stale = str(record.get("stale") or "").strip().lower()
    target_positions = {
        target: occurrences(text, target) for target in targets
    }
    stale_positions = occurrences(text, stale) if stale else []
    all_targets_present = all(target_positions[target] for target in targets)
    any_target_present = any(target_positions[target] for target in targets)
    review = False

    if len(targets) > 1:
        if all_targets_present:
            label = "both"
            correct = True
        elif any_target_present:
            label = "target_only"
            correct = False
            review = True
        else:
            label = "irrelevant"
            correct = False
            review = True
        recovered = False
    else:
        target_hits = target_positions[targets[0]]
        if target_hits and not stale_positions:
            label = "target_only"
            correct = True
            recovered = False
        elif stale_positions and not target_hits:
            label = "stale_only"
            correct = False
            recovered = False
        elif target_hits and stale_positions:
            review = True
            if max(target_hits) > max(stale_positions):
                label = "recovered"
                correct = True
                recovered = True
            else:
                label = "both"
                correct = False
                recovered = False
        else:
            label = "irrelevant"
            correct = False
            recovered = False
            review = True

    return {
        "label": label,
        "final_target_correct": int(correct),
        "recovered": int(recovered),
        "intelligible": int(bool(text.strip())),
        "notes": "AUTO_HEURISTIC_REVIEW" if review else "AUTO_HIGH_CONFIDENCE",
    }


def main() -> None:
    args = parse_args()
    for path in (args.output, args.key_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    predictions = []
    with args.predictions.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                predictions.append(json.loads(line))
    if not predictions:
        raise ValueError(f"No predictions found in {args.predictions}")

    annotations = []
    keys = []
    review_count = 0
    for ordinal, prediction in enumerate(predictions, start=1):
        blind_id = f"AUTO{ordinal:05d}"
        automatic = classify(prediction)
        review_count += int(automatic["notes"] == "AUTO_HEURISTIC_REVIEW")
        keys.append(
            {
                "blind_id": blind_id,
                "trial_id": prediction["trial_id"],
                "seed": prediction["seed"],
                "speaker_id": prediction["speaker_id"],
                "condition_id": prediction["condition_id"],
            }
        )
        annotations.append(
            {
                "blind_id": blind_id,
                "annotator_id": "AUTO_TEXT",
                "adjudicator": "1",
                "response_audio_path": prediction["response_audio_path"],
                "inner_text": prediction.get("inner_text", ""),
                "label": automatic["label"],
                "final_target_correct": automatic["final_target_correct"],
                "early_stale_before_repair": "",
                "stale_after_repair": "",
                "recovered": automatic["recovered"],
                "response_onset_ms": "",
                "first_target_ms": "",
                "first_stale_ms": "",
                "output_language": "en",
                "intelligible": automatic["intelligible"],
                "notes": automatic["notes"],
            }
        )
    write_csv(args.output, annotations, FIELDS)
    write_csv(args.key_output, keys, KEY_FIELDS)
    print(f"Auto-labeled {len(annotations)} outputs")
    print(f"Needs text review: {review_count}")
    print(f"Annotations: {args.output}")


if __name__ == "__main__":
    main()
