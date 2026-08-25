#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from audio_utils import append_silence_and_frame_pad, duration_ms, read_pcm16_mono, write_pcm16_mono
from common import DATASET_ROOT, portable_path, read_config, read_jsonl, sha256_file, sha256_value, write_json, write_jsonl
from ids import prepared_stimulus_id
from timing import shift_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create frame-aligned Moshi stimuli from accepted canonical utterances.")
    parser.add_argument("--input", type=Path, default=DATASET_ROOT / "manifests/accepted_audio.jsonl")
    parser.add_argument("--output", type=Path, default=DATASET_ROOT / "manifests/prepared_stimuli.jsonl")
    parser.add_argument("--audio-root", type=Path, default=DATASET_ROOT / "artifacts/prepared")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def prepare_rows(
    rows: list[dict[str, Any]], config: dict[str, Any], audio_root: Path
) -> list[dict[str, Any]]:
    audio_config = config["audio"]
    rate = int(audio_config["canonical_sample_rate"])
    prefix_ms = float(audio_config["prefix_silence_ms"])
    frame_samples = int(audio_config["mimi_frame_samples"])
    preparation = {
        "sample_rate": rate,
        "prefix_silence_ms": prefix_ms,
        "mimi_frame_samples": frame_samples,
        "normalization_stage": "accepted_canonical",
    }
    preparation_hash = sha256_value(preparation)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        accepted_id = str(row["accepted_audio_id"])
        if accepted_id in seen:
            raise ValueError(f"duplicate accepted_audio_id: {accepted_id}")
        seen.add(accepted_id)
        artifact = row.get("accepted_utterance") or {}
        source_path = Path(str(artifact.get("uri", "")))
        if not source_path.is_file() or artifact.get("sha256") != sha256_file(source_path):
            raise ValueError(f"{accepted_id}: missing or hash-mismatched accepted utterance")
        audio, source_rate = read_pcm16_mono(source_path)
        if source_rate != rate:
            raise ValueError(f"{accepted_id}: accepted sample rate must be {rate}")
        prepared, padding = append_silence_and_frame_pad(audio, rate, prefix_ms, frame_samples)
        stimulus_id = prepared_stimulus_id(accepted_id, preparation_hash)
        target_path = audio_root / f"{stimulus_id}.wav"
        if target_path.exists():
            raise FileExistsError(f"immutable prepared stimulus already exists: {target_path}")
        write_pcm16_mono(target_path, prepared, rate)
        item = dict(row)
        item["lifecycle_status"] = "prepared"
        item["prepared_stimulus_id"] = stimulus_id
        item["preparation_hash"] = preparation_hash
        item["prepared_stimulus"] = {
            "uri": str(target_path.resolve()),
            "sha256": sha256_file(target_path),
            "duration_ms": duration_ms(prepared, rate),
            "sample_rate": rate,
            "channels": 1,
            "sample_width_bytes": 2,
            "timeline": "prepared_stream_relative",
        }
        item["preparation"] = {**preparation, **padding}
        item["prepared_timing"] = shift_events(row["timing"], float(padding["prefix_ms_actual"]))
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    config = read_config()
    output = prepare_rows(read_jsonl(args.input), config, args.audio_root)
    write_jsonl(args.output, output)
    report = {"schema_version": "2.0.0", "prepared_count": len(output), "output": portable_path(args.output)}
    if args.report:
        write_json(args.report, report)
    print(f"Prepared {len(output)} stimuli -> {args.output}")


if __name__ == "__main__":
    main()
