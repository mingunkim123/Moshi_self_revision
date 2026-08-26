#!/usr/bin/env python3
"""Materialize engineering-only stimuli when human review was not recorded.

This escape hatch deliberately does not modify or satisfy the release alignment
gate.  It exists only to let the model runner and its artifact plumbing be
tested while the structured human-review record remains missing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from audio_utils import duration_ms, normalize_audio, read_pcm16_mono, write_pcm16_mono
from common import (
    DATASET_ROOT,
    portable_path,
    read_config,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
    write_jsonl,
)
from ids import accepted_audio_id
from prepare_v2_stimuli import prepare_rows
from timing import validate_timing


PROVISIONAL_STATUS = "user_directed_continue_without_structured_review_record"
PROVISIONAL_PURPOSE = "provisional_engineering_smoke_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create engineering-only accepted/prepared stimuli without claiming "
            "that unrecorded human review passed the release gate."
        )
    )
    parser.add_argument("--input", type=Path, default=DATASET_ROOT / "manifests/qc_candidates.jsonl")
    parser.add_argument(
        "--accepted-output",
        type=Path,
        default=DATASET_ROOT / "manifests/provisional_accepted_audio.jsonl",
    )
    parser.add_argument(
        "--prepared-output",
        type=Path,
        default=DATASET_ROOT / "manifests/provisional_prepared_stimuli.jsonl",
    )
    parser.add_argument(
        "--accepted-root", type=Path, default=DATASET_ROOT / "artifacts/provisional_accepted"
    )
    parser.add_argument(
        "--prepared-root", type=Path, default=DATASET_ROOT / "artifacts/provisional_prepared"
    )
    parser.add_argument(
        "--waiver",
        type=Path,
        default=DATASET_ROOT / "release_evidence/provisional_review_waiver.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DATASET_ROOT / "release_evidence/provisional_materialization_report.json",
    )
    parser.add_argument("--expected-count", type=int, default=600)
    parser.add_argument(
        "--acknowledge-missing-review-record",
        action="store_true",
        help="Required acknowledgement; never implies that any candidate passed human review.",
    )
    return parser.parse_args()


def _artifact(row: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    artifact = row.get("canonical_candidate")
    if not isinstance(artifact, dict):
        raise ValueError(f"{row.get('candidate_id')}: missing canonical_candidate")
    source_path = Path(str(artifact.get("uri", "")))
    if not source_path.is_file():
        raise ValueError(f"{row.get('candidate_id')}: canonical WAV is missing: {source_path}")
    observed_hash = sha256_file(source_path)
    if artifact.get("sha256") != observed_hash:
        raise ValueError(f"{row.get('candidate_id')}: canonical WAV hash mismatch")
    return artifact, source_path


def validate_provisional_source_rows(
    rows: list[dict[str, Any]], config: dict[str, Any], expected_count: int
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} QC candidates, found {len(rows)}")
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    rate = int(config["audio"]["canonical_sample_rate"])
    target_ids: set[str] = set()
    candidate_ids: set[str] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        target_id = str(row.get("rendition_target_id", ""))
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError(f"duplicate or missing candidate_id: {candidate_id!r}")
        if not target_id or target_id in target_ids:
            raise ValueError(f"duplicate or missing rendition_target_id: {target_id!r}")
        candidate_ids.add(candidate_id)
        target_ids.add(target_id)
        if row.get("lifecycle_status") != "canonical_candidate":
            raise ValueError(f"{candidate_id}: source lifecycle must be canonical_candidate")
        qc = row.get("qc")
        if not isinstance(qc, dict) or qc.get("automatic_status") != "passed":
            raise ValueError(f"{candidate_id}: automatic QC has not passed")
        alignment = row.get("alignment")
        if not isinstance(alignment, dict) or alignment.get("independent_forced_alignment") is not True:
            raise ValueError(f"{candidate_id}: independent forced alignment is missing")
        review = alignment.get("manual_review")
        if not isinstance(review, dict) or review.get("status") != "pending":
            raise ValueError(
                f"{candidate_id}: provisional path only accepts an explicitly pending review record"
            )
        artifact, source_path = _artifact(row)
        audio, observed_rate = read_pcm16_mono(source_path)
        if observed_rate != rate:
            raise ValueError(f"{candidate_id}: expected {rate} Hz, found {observed_rate} Hz")
        observed_duration = duration_ms(audio, observed_rate)
        if abs(float(artifact.get("duration_ms", -1)) - observed_duration) > 1:
            raise ValueError(f"{candidate_id}: canonical duration metadata mismatch")
        timing_errors = validate_timing(str(row.get("condition")), row.get("timing") or {}, observed_duration)
        if timing_errors:
            raise ValueError(f"{candidate_id}: invalid timing: {'; '.join(timing_errors)}")


def materialize_provisional_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    accepted_root: Path,
    waiver_hash: str,
) -> list[dict[str, Any]]:
    audio_config = config["audio"]
    rate = int(audio_config["canonical_sample_rate"])
    target_rms = float(audio_config["target_active_rms_dbfs"])
    peak_limit = float(audio_config["peak_limit_dbfs"])
    tail_ms = 200.0
    output: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["rendition_target_id"])):
        candidate_id = str(row["candidate_id"])
        target_id = str(row["rendition_target_id"])
        artifact, source_path = _artifact(row)
        audio, observed_rate = read_pcm16_mono(source_path)
        if observed_rate != rate:
            raise ValueError(f"{candidate_id}: unexpected sample rate")
        utterance_end_ms = float(row["timing"]["utterance_end_ms"])
        target_samples = round((utterance_end_ms + tail_ms) * rate / 1000.0)
        if target_samples < 1:
            raise ValueError(f"{candidate_id}: invalid target sample count")
        if audio.size < target_samples:
            fixed_tail = np.pad(audio, (0, target_samples - audio.size))
            tail_action = "zero_extended_to_fixed_tail"
        else:
            fixed_tail = audio[:target_samples].copy()
            tail_action = "trimmed_to_fixed_tail" if audio.size > target_samples else "already_fixed_tail"
        normalized, normalization = normalize_audio(fixed_tail, target_rms, peak_limit)
        accepted_id = accepted_audio_id(target_id)
        target_path = accepted_root / f"{accepted_id}.wav"
        if target_path.exists():
            raise FileExistsError(f"immutable provisional accepted WAV already exists: {target_path}")
        write_pcm16_mono(target_path, normalized, rate)
        accepted_duration = duration_ms(normalized, rate)
        item = dict(row)
        item["candidate_id"] = None
        item["selected_candidate_id"] = candidate_id
        item["accepted_audio_id"] = accepted_id
        item["lifecycle_status"] = "accepted"
        item["release_eligible"] = False
        item["inferential_role"] = PROVISIONAL_PURPOSE
        item["provisional_engineering"] = {
            "status": PROVISIONAL_STATUS,
            "purpose": PROVISIONAL_PURPOSE,
            "waiver_sha256": waiver_hash,
            "structured_review_record_present": False,
            "all_items_passed_claimed": False,
            "release_eligible": False,
        }
        item["accepted_utterance"] = {
            "uri": str(target_path.resolve()),
            "sha256": sha256_file(target_path),
            "duration_ms": accepted_duration,
            "sample_rate": rate,
            "channels": 1,
            "sample_width_bytes": 2,
            "timeline": "content_relative",
            "source_canonical_sha256": artifact["sha256"],
        }
        item["selection"] = {
            "status": "provisional_single_candidate_user_directed_waiver",
            "selected_candidate_id": candidate_id,
            "selected_canonical_uri": str(source_path.resolve()),
            "selected_canonical_sha256": artifact["sha256"],
            "candidate_pool_size": 1,
            "outcome_blind": True,
            "release_eligible": False,
            "materialization_mode": "gain_normalized_transformed_copy",
            "normalization": {
                **normalization,
                "target_active_rms_dbfs": target_rms,
                "peak_limit_dbfs": peak_limit,
            },
            "tail_policy": {
                "utterance_end_ms": utterance_end_ms,
                "fixed_tail_ms": tail_ms,
                "tail_after_utterance_ms_actual": accepted_duration - utterance_end_ms,
                "target_samples": target_samples,
                "action": tail_action,
                "leading_coordinate_shift_samples": 0,
                "frame_padding_applied": False,
                "prefix_silence_applied": False,
            },
        }
        output.append(item)
    return output


def run_materialization(
    *,
    input_path: Path,
    accepted_output: Path,
    prepared_output: Path,
    accepted_root: Path,
    prepared_root: Path,
    waiver_path: Path,
    report_path: Path,
    expected_count: int,
    acknowledge_missing_review_record: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not acknowledge_missing_review_record:
        raise ValueError("--acknowledge-missing-review-record is required")
    for path in (accepted_output, prepared_output, waiver_path, report_path):
        if path.exists():
            raise FileExistsError(f"immutable provisional output already exists: {path}")
    config = read_config()
    rows = read_jsonl(input_path)
    validate_provisional_source_rows(rows, config, expected_count)
    qc_manifest_hash = sha256_file(input_path)
    waiver = {
        "schema_version": "2.0.0",
        "status": PROVISIONAL_STATUS,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "user_direction": "Review was completed but results were not recorded; continue to the next step.",
        "structured_review_record_present": False,
        "all_items_passed_claimed": False,
        "failed_item_ids_recorded": False,
        "release_eligible": False,
        "purpose": PROVISIONAL_PURPOSE,
        "qc_manifest_sha256": qc_manifest_hash,
    }
    waiver_hash = sha256_value(waiver)
    waiver["waiver_sha256"] = waiver_hash
    accepted = materialize_provisional_rows(rows, config, accepted_root, waiver_hash)
    prepared = prepare_rows(accepted, config, prepared_root)
    for item in prepared:
        item["release_eligible"] = False
        item["inferential_role"] = PROVISIONAL_PURPOSE
        item["provisional_engineering"]["release_eligible"] = False
    write_json(waiver_path, waiver)
    write_jsonl(accepted_output, accepted)
    write_jsonl(prepared_output, prepared)
    report = {
        "schema_version": "2.0.0",
        "status": PROVISIONAL_STATUS,
        "purpose": PROVISIONAL_PURPOSE,
        "release_eligible": False,
        "human_review_gate_satisfied": False,
        "all_items_passed_claimed": False,
        "source_qc_count": len(rows),
        "provisional_accepted_count": len(accepted),
        "provisional_prepared_count": len(prepared),
        "source_qc_manifest": portable_path(input_path),
        "source_qc_manifest_sha256": qc_manifest_hash,
        "waiver": portable_path(waiver_path),
        "waiver_sha256": waiver_hash,
        "accepted_manifest": portable_path(accepted_output),
        "accepted_manifest_sha256": sha256_file(accepted_output),
        "prepared_manifest": portable_path(prepared_output),
        "prepared_manifest_sha256": sha256_file(prepared_output),
        "next_allowed_use": "Moshi runner engineering validation only",
        "blocked_use": "release, publication, or confirmatory inference",
    }
    write_json(report_path, report)
    return accepted, prepared


def main() -> None:
    args = parse_args()
    accepted, prepared = run_materialization(
        input_path=args.input,
        accepted_output=args.accepted_output,
        prepared_output=args.prepared_output,
        accepted_root=args.accepted_root,
        prepared_root=args.prepared_root,
        waiver_path=args.waiver,
        report_path=args.report,
        expected_count=args.expected_count,
        acknowledge_missing_review_record=args.acknowledge_missing_review_record,
    )
    print(
        f"Created {len(accepted)} provisional accepted and {len(prepared)} prepared stimuli; "
        "release eligibility remains false."
    )


if __name__ == "__main__":
    main()
