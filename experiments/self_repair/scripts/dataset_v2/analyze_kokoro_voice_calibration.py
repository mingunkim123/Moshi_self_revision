#!/usr/bin/env python3
"""Summarize the private 10-voice Kokoro technical calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
from typing import Any

from common import DATASET_ROOT, DEFAULT_SCRIPTS, read_config, read_jsonl, sha256_file, word_count, write_json


DEFAULT_INPUT = DATASET_ROOT / "release_evidence/kokoro_voice_qc_candidates.jsonl"
DEFAULT_TARGETS = DATASET_ROOT / "calibration/kokoro_voice_targets.jsonl"
DEFAULT_OUTPUT = DATASET_ROOT / "reports/kokoro_voice_calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def analyze(
    rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    scripts: list[dict[str, Any]],
    config: dict[str, Any],
    input_path: Path,
) -> dict[str, Any]:
    calibration = config.get("open_source_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("dataset config is missing open_source_calibration")
    if len(rows) != len(targets) or len(rows) != 10:
        raise ValueError("Kokoro voice calibration requires exactly 10 rows")
    target_ids = {str(row["rendition_target_id"]) for row in targets}
    if {str(row.get("rendition_target_id")) for row in rows} != target_ids:
        raise ValueError("QC rows do not exactly cover the frozen calibration targets")
    script_ids = {str(row["script_id"]) for row in targets}
    if len(script_ids) != 1:
        raise ValueError("all voices must use the same calibration script")
    script_id = next(iter(script_ids))
    script_map = {str(row["script_id"]): row for row in scripts}
    script = script_map.get(script_id)
    if script is None:
        raise ValueError("calibration script is absent from the frozen script manifest")
    expected_voices = {
        str(row["speaker_id"]): row for row in calibration["speakers"]
    }
    voice_rows: list[dict[str, Any]] = []
    technical_errors: list[str] = []
    for row in sorted(rows, key=lambda item: str(item["speaker_id"])):
        speaker_id = str(row["speaker_id"])
        expected = expected_voices.get(speaker_id)
        if expected is None or row.get("voice") != expected.get("voice"):
            technical_errors.append(f"{speaker_id}: frozen voice mismatch")
            continue
        synthesis = row.get("synthesis") if isinstance(row.get("synthesis"), dict) else {}
        provider_artifact = (
            synthesis.get("provider_artifact")
            if isinstance(synthesis.get("provider_artifact"), dict)
            else {}
        )
        if synthesis.get("provider") != "kokoro_local_v1_0":
            technical_errors.append(f"{speaker_id}: provider mismatch")
        if provider_artifact.get("model_sha256") != calibration["model_sha256"]:
            technical_errors.append(f"{speaker_id}: model hash mismatch")
        if provider_artifact.get("voice_sha256") != expected["voice_sha256"]:
            technical_errors.append(f"{speaker_id}: voice hash mismatch")
        qc = row.get("qc") if isinstance(row.get("qc"), dict) else {}
        if qc.get("automatic_status") != "passed":
            technical_errors.append(f"{speaker_id}: automatic QC failed")
        metrics = qc.get("metrics") if isinstance(qc.get("metrics"), dict) else {}
        duration_ms = float(metrics.get("duration_ms", 0.0))
        if duration_ms <= 0:
            technical_errors.append(f"{speaker_id}: nonpositive duration")
            continue
        alignment = row.get("alignment") if isinstance(row.get("alignment"), dict) else {}
        mapping = alignment.get("transcript_mapping")
        if not isinstance(mapping, list) or not mapping:
            technical_errors.append(f"{speaker_id}: provider timing seed missing")
        voice_rows.append(
            {
                "speaker_id": speaker_id,
                "voice": row["voice"],
                "published_grade": expected["published_grade"],
                "voice_sha256": expected["voice_sha256"],
                "duration_ms": duration_ms,
                "words_per_minute": word_count(script["transcript"])
                / duration_ms
                * 60000.0,
                "active_rms_dbfs": metrics.get("active_rms_dbfs"),
                "peak_dbfs": metrics.get("peak_dbfs"),
                "clipping_fraction": metrics.get("clipping_fraction"),
                "leading_silence_proxy_ms": metrics.get("leading_silence_proxy_ms"),
                "trailing_silence_proxy_ms": metrics.get("trailing_silence_proxy_ms"),
                "provider_seed_token_count": len(mapping) if isinstance(mapping, list) else 0,
                "automatic_qc": qc.get("automatic_status"),
            }
        )
    if len(voice_rows) != 10:
        technical_errors.append("not all 10 voices produced analyzable metrics")
    durations = [float(row["duration_ms"]) for row in voice_rows]
    ordered = sorted(voice_rows, key=lambda row: float(row["duration_ms"]))
    return {
        "schema_version": "2.0.0",
        "report_kind": "kokoro_open_source_voice_calibration",
        "release_eligible": False,
        "technical_gate_passed": not technical_errors,
        "technical_errors": technical_errors,
        "production_voice_selection_frozen": False,
        "human_double_listen_status": "pending",
        "required_next_gate": (
            "Two outcome-blind reviewers must listen for pronunciation, truncation, "
            "artifacts, and voice-quality imbalance before production assignment."
        ),
        "source": {
            "provider": calibration["provider"],
            "source_track_id": calibration["source_track_id"],
            "model_repo": calibration["model_repo"],
            "model_revision": calibration["model_revision"],
            "model_sha256": calibration["model_sha256"],
            "model_license": calibration["model_license"],
            "candidate_policy": calibration["candidate_policy"],
            "sample_rate": calibration["sample_rate"],
            "speed": calibration["speed"],
        },
        "input": {
            "private_qc_manifest_sha256": sha256_file(input_path),
            "script_id": script_id,
            "script_word_count": word_count(script["transcript"]),
        },
        "summary": {
            "voice_count": len(voice_rows),
            "automatic_qc_passed": sum(
                row["automatic_qc"] == "passed" for row in voice_rows
            ),
            "duration_ms_min": min(durations) if durations else None,
            "duration_ms_median": statistics.median(durations) if durations else None,
            "duration_ms_max": max(durations) if durations else None,
            "duration_ratio_max_over_min": max(durations) / min(durations)
            if durations
            else None,
            "fastest_speaker_id": ordered[0]["speaker_id"] if ordered else None,
            "slowest_speaker_id": ordered[-1]["speaker_id"] if ordered else None,
        },
        "voices": voice_rows,
    }


def main() -> None:
    args = parse_args()
    report = analyze(
        read_jsonl(args.input),
        read_jsonl(args.targets),
        read_jsonl(args.scripts),
        read_config(),
        args.input,
    )
    write_json(args.output, report)
    print(
        f"Analyzed {report['summary']['voice_count']} Kokoro voices; "
        f"technical_gate_passed={report['technical_gate_passed']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
