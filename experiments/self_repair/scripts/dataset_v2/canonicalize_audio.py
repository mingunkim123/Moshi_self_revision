#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from audio_utils import duration_ms, read_pcm16_mono, resample_linear, write_pcm16_mono
from common import DATASET_ROOT, portable_path, read_config, read_jsonl, sha256_file, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw PCM16 mono WAV candidates to canonical WAV.")
    parser.add_argument("--input", type=Path, default=DATASET_ROOT / "manifests/raw_candidates.jsonl")
    parser.add_argument("--output", type=Path, default=DATASET_ROOT / "manifests/canonical_candidates.jsonl")
    parser.add_argument("--audio-root", type=Path, default=DATASET_ROOT / "artifacts/canonical_candidates")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def canonicalize_rows(
    rows: list[dict[str, Any]], config: dict[str, Any], audio_root: Path
) -> list[dict[str, Any]]:
    target_rate = int(config["audio"]["canonical_sample_rate"])
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        candidate = str(row["candidate_id"])
        if candidate in seen:
            raise ValueError(f"duplicate candidate_id: {candidate}")
        seen.add(candidate)
        raw = row.get("raw_candidate") or {}
        raw_path = Path(str(raw.get("uri", "")))
        if not raw_path.is_file():
            raise FileNotFoundError(f"{candidate}: missing raw WAV {raw_path}")
        if raw.get("sha256") != sha256_file(raw_path):
            raise ValueError(f"{candidate}: raw candidate hash mismatch")
        audio, source_rate = read_pcm16_mono(raw_path)
        canonical = resample_linear(audio, source_rate, target_rate)
        output_path = audio_root / f"{candidate}.wav"
        if output_path.exists():
            raise FileExistsError(f"immutable canonical candidate already exists: {output_path}")
        write_pcm16_mono(output_path, canonical, target_rate)
        item = dict(row)
        item["lifecycle_status"] = "canonical_candidate"
        item["canonical_candidate"] = {
            "uri": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "duration_ms": duration_ms(canonical, target_rate),
            "sample_rate": target_rate,
            "channels": 1,
            "sample_width_bytes": 2,
            "timeline": "content_relative",
        }
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    config = read_config()
    output = canonicalize_rows(read_jsonl(args.input), config, args.audio_root)
    write_jsonl(args.output, output)
    report = {"schema_version": "2.0.0", "canonical_candidate_count": len(output), "output": portable_path(args.output)}
    if args.report:
        write_json(args.report, report)
    print(f"Canonicalized {len(output)} candidates -> {args.output}")


if __name__ == "__main__":
    main()
