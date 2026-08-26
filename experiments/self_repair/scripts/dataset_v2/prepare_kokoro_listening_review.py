#!/usr/bin/env python3
"""Prepare two identity-blind local listening sheets for Kokoro voice admission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from common import DATASET_ROOT, read_config, read_jsonl, sha256_file, write_json


DEFAULT_INPUT = DATASET_ROOT / "release_evidence/kokoro_voice_qc_candidates.jsonl"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "release_evidence/kokoro_voice_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def prepare(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    if len(rows) != 10:
        raise ValueError("listening review requires exactly 10 Kokoro candidates")
    config = read_config()
    seed = int(config["generation_seed"])
    mapping: list[dict[str, Any]] = []
    for row in rows:
        if (row.get("qc") or {}).get("automatic_status") != "passed":
            raise ValueError("all listening candidates must pass automatic QC")
        raw = row.get("raw_candidate")
        if not isinstance(raw, dict):
            raise ValueError("candidate is missing raw audio")
        path = Path(str(raw.get("uri", "")))
        if not path.is_file() or raw.get("sha256") != sha256_file(path):
            raise ValueError("candidate raw audio is missing or has a hash mismatch")
        try:
            relative = path.resolve().relative_to(DATASET_ROOT.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("listening audio must live inside dataset_v2") from error
        digest = hashlib.sha256(
            f"{seed}|{row['candidate_id']}".encode("utf-8")
        ).hexdigest()
        mapping.append(
            {
                "blind_item_id": f"KV-{digest[:8].upper()}",
                "candidate_id": row["candidate_id"],
                "speaker_id": row["speaker_id"],
                "voice": row["voice"],
                "audio_uri": relative,
                "audio_sha256": raw["sha256"],
            }
        )
    if len({row["blind_item_id"] for row in mapping}) != 10:
        raise ValueError("blind listening IDs collided")
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "blind_item_id",
        "audio_uri",
        "pronunciation_ok",
        "complete_no_truncation",
        "artifact_free",
        "naturalness_1_to_5",
        "pace_1_to_5",
        "admit_voice",
        "reviewer_id",
        "notes",
    ]
    sheets: list[dict[str, str]] = []
    for reviewer_index, reviewer_label in enumerate(("reviewer_a", "reviewer_b")):
        ordered = list(mapping)
        random.Random(seed + reviewer_index).shuffle(ordered)
        path = output_dir / f"{reviewer_label}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in ordered:
                writer.writerow(
                    {
                        "blind_item_id": row["blind_item_id"],
                        "audio_uri": row["audio_uri"],
                        "pronunciation_ok": "",
                        "complete_no_truncation": "",
                        "artifact_free": "",
                        "naturalness_1_to_5": "",
                        "pace_1_to_5": "",
                        "admit_voice": "",
                        "reviewer_id": "",
                        "notes": "",
                    }
                )
        sheets.append({"reviewer": reviewer_label, "path": path.as_posix(), "sha256": sha256_file(path)})
    map_path = output_dir / "private_voice_map.json"
    write_json(
        map_path,
        {
            "schema_version": "2.0.0",
            "condition_blind": True,
            "voice_identity_blind": True,
            "mapping": sorted(mapping, key=lambda row: row["blind_item_id"]),
        },
    )
    return {
        "schema_version": "2.0.0",
        "status": "awaiting_two_independent_reviews",
        "item_count": 10,
        "required_reviewers": 2,
        "sheets": sheets,
        "private_map": {"path": map_path.as_posix(), "sha256": sha256_file(map_path)},
    }


def main() -> None:
    args = parse_args()
    package = prepare(read_jsonl(args.input), args.output_dir)
    manifest = args.output_dir / "review_package.json"
    write_json(manifest, package)
    print(f"Prepared blind Kokoro review package -> {manifest}")


if __name__ == "__main__":
    main()
