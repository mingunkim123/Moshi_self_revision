#!/usr/bin/env python3
"""Validate raw -> canonical -> accepted -> prepared audio lineage."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from alignment_evidence import (
    validate_alignment_gate_contract,
    validate_downstream_alignment_evidence,
)
from audio_utils import (
    append_silence_and_frame_pad,
    duration_ms,
    normalize_audio,
    read_pcm16_mono,
)
from common import (
    DATASET_ROOT,
    DEFAULT_SCRIPTS,
    read_config,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
)
from ids import accepted_audio_id, prepared_stimulus_id
from select_candidates import validate_selection_policy
from timing import shift_events, validate_timing


VALIDATION_VERSION = "2.2.0"
PCM_TOLERANCE = 1.5 / 32768.0
DEFAULT_QC_INPUT = DATASET_ROOT / "manifests/qc_candidates.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate complete v2 audio lifecycle lineage.")
    parser.add_argument(
        "--raw", type=Path, default=DATASET_ROOT / "manifests/raw_candidates.jsonl"
    )
    parser.add_argument(
        "--qc",
        "--canonical",
        dest="canonical",
        type=Path,
        default=DEFAULT_QC_INPUT,
        help="Post-QC aligned candidate manifest (--canonical is a compatibility alias).",
    )
    parser.add_argument(
        "--accepted", type=Path, default=DATASET_ROOT / "manifests/accepted_audio.jsonl"
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=DATASET_ROOT / "manifests/prepared_stimuli.jsonl",
    )
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--no-enforce-config-counts",
        action="store_true",
        help="Useful only for smoke fixtures; production validation enforces all configured counts.",
    )
    return parser.parse_args()


def _index(
    rows: Iterable[dict[str, Any]], field: str, label: str, errors: list[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"{label} row {position}: missing {field}")
            continue
        if value in result:
            errors.append(f"{label}: duplicate {field} {value}")
            continue
        result[value] = row
    return result


def _check_artifact(
    row_id: str,
    artifact: Any,
    field: str,
    errors: list[str],
) -> tuple[np.ndarray, int] | None:
    if not isinstance(artifact, dict):
        errors.append(f"{row_id}: missing {field}")
        return None
    path = Path(str(artifact.get("uri", "")))
    if not path.is_file():
        errors.append(f"{row_id}: {field} file does not exist: {path}")
        return None
    actual_hash = sha256_file(path)
    if artifact.get("sha256") != actual_hash:
        errors.append(f"{row_id}: {field} hash mismatch")
    try:
        audio, sample_rate = read_pcm16_mono(path)
    except (OSError, ValueError) as error:
        errors.append(f"{row_id}: invalid {field} PCM WAV: {error}")
        return None
    expected_rate = artifact.get("sample_rate")
    if expected_rate != sample_rate:
        errors.append(f"{row_id}: {field} manifest/sample-rate mismatch")
    if artifact.get("channels") != 1:
        errors.append(f"{row_id}: {field} channels must be 1")
    if artifact.get("sample_width_bytes", 2) != 2:
        errors.append(f"{row_id}: {field} sample width must be two bytes")
    observed_duration = duration_ms(audio, sample_rate)
    declared_duration = artifact.get("duration_ms")
    if not isinstance(declared_duration, (int, float)) or abs(
        float(declared_duration) - observed_duration
    ) > 1000.0 / sample_rate:
        errors.append(f"{row_id}: {field} duration mismatch")
    return audio, sample_rate


def _same_artifact(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return left.get("uri") == right.get("uri") and left.get("sha256") == right.get("sha256")


def _timing_equal(left: Any, right: Any, tolerance_ms: float = 0.05) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if a is None or b is None:
            if a is not None or b is not None:
                return False
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) > tolerance_ms:
                return False
        elif a != b:
            return False
    return True


def _validate_alignment(row_id: str, row: dict[str, Any], errors: list[str]) -> None:
    alignment = row.get("alignment")
    timing = row.get("timing")
    if not isinstance(alignment, dict):
        errors.append(f"{row_id}: missing alignment")
        return
    if not isinstance(timing, dict):
        errors.append(f"{row_id}: missing timing")
        return
    if alignment.get("method") == "provider_word_boundaries_seed":
        if alignment.get("independent_forced_alignment") is not False:
            errors.append(f"{row_id}: provider-boundary seed is mislabeled as independent alignment")
        review = alignment.get("manual_review")
        if not isinstance(review, dict) or review.get("required") is not True:
            errors.append(f"{row_id}: provider-boundary seed must require manual review")
    mapping = alignment.get("transcript_mapping")
    if not isinstance(mapping, list) or not mapping:
        errors.append(f"{row_id}: empty transcript alignment mapping")
    else:
        indices = [item.get("token_index") for item in mapping if isinstance(item, dict)]
        if indices != list(range(len(mapping))):
            errors.append(f"{row_id}: transcript token indices are not contiguous")
        previous = -1.0
        for item in mapping:
            if not isinstance(item, dict):
                errors.append(f"{row_id}: non-object transcript mapping row")
                continue
            onset, offset = item.get("onset_ms"), item.get("offset_ms")
            if not isinstance(onset, (int, float)) or not isinstance(offset, (int, float)):
                errors.append(f"{row_id}: transcript mapping interval is not numeric")
            elif float(onset) < previous or float(offset) <= float(onset):
                errors.append(f"{row_id}: transcript mapping is not monotonic")
            else:
                previous = float(onset)
    event_spans = alignment.get("event_spans")
    if not isinstance(event_spans, dict):
        errors.append(f"{row_id}: missing event spans")
    unit_spans = alignment.get("unit_spans")
    if not isinstance(unit_spans, list):
        errors.append(f"{row_id}: missing unit spans")
        return
    cue_onset = timing.get("repair_cue_onset_ms")
    for span in unit_spans:
        if not isinstance(span, dict):
            errors.append(f"{row_id}: non-object unit span")
            continue
        onset, offset = span.get("onset_ms"), span.get("offset_ms")
        if (
            not isinstance(onset, (int, float))
            or not isinstance(offset, (int, float))
            or float(offset) <= float(onset)
        ):
            errors.append(f"{row_id}: invalid unit interval for {span.get('unit_id')}")
        expected_age = None
        if (
            isinstance(cue_onset, (int, float))
            and span.get("repair_position") == "pre"
            and str(span.get("unit_id", "")).startswith("D")
            and isinstance(offset, (int, float))
        ):
            expected_age = float(cue_onset) - float(offset)
        observed_age = span.get("stale_dependency_age_ms")
        if expected_age is None:
            if observed_age is not None:
                errors.append(f"{row_id}: unexpected stale dependency age for {span.get('unit_id')}")
        elif not isinstance(observed_age, (int, float)) or abs(float(observed_age) - expected_age) > 0.05:
            errors.append(f"{row_id}: stale dependency age mismatch for {span.get('unit_id')}")


def validate_audio_lifecycle(
    raw_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    prepared_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    selection_policy: dict[str, Any] | None = None,
    scripts: list[dict[str, Any]] | None = None,
    enforce_config_counts: bool = False,
) -> list[str]:
    errors: list[str] = []
    selection_policy_hash: str | None = None
    if isinstance(selection_policy, dict):
        try:
            selection_policy_hash = validate_selection_policy(selection_policy)
        except ValueError as error:
            errors.append(f"selection policy: {error}")
    else:
        errors.append("selection policy: frozen policy object is required")
    alignment_gate = (
        selection_policy.get("alignment_gate")
        if isinstance(selection_policy, dict)
        else None
    )
    gate_errors = validate_alignment_gate_contract(alignment_gate)
    errors.extend(f"selection policy: {error}" for error in gate_errors)
    scripts_by_id: dict[str, dict[str, Any]] = {}
    for script in scripts or []:
        script_id = str(script.get("script_id", ""))
        if not script_id or script_id in scripts_by_id:
            errors.append(f"scripts: missing or duplicate script_id {script_id!r}")
            continue
        scripts_by_id[script_id] = script
    raw = _index(raw_rows, "candidate_id", "raw", errors)
    canonical = _index(canonical_rows, "candidate_id", "canonical", errors)
    accepted = _index(accepted_rows, "accepted_audio_id", "accepted", errors)
    prepared = _index(prepared_rows, "prepared_stimulus_id", "prepared", errors)
    rate = int(config["audio"]["canonical_sample_rate"])
    frame_samples = int(config["audio"]["mimi_frame_samples"])
    prefix_ms = float(config["audio"]["prefix_silence_ms"])
    prefix_samples = round(prefix_ms * rate / 1000.0)
    target_rms = float(config["audio"]["target_active_rms_dbfs"])
    peak_limit = float(config["audio"]["peak_limit_dbfs"])

    if set(raw) != set(canonical):
        errors.append(
            "raw/canonical candidate ID sets differ: "
            f"missing_canonical={sorted(set(raw) - set(canonical))[:5]}, "
            f"missing_raw={sorted(set(canonical) - set(raw))[:5]}"
        )

    canonical_hash_target: dict[str, str] = {}
    for candidate_id, row in canonical.items():
        if row.get("lifecycle_status") != "canonical_candidate":
            errors.append(f"{candidate_id}: canonical lifecycle status is wrong")
        raw_row = raw.get(candidate_id)
        if raw_row is not None:
            if not _same_artifact(row.get("raw_candidate"), raw_row.get("raw_candidate")):
                errors.append(f"{candidate_id}: raw -> canonical artifact lineage mismatch")
            _check_artifact(candidate_id, raw_row.get("raw_candidate"), "raw_candidate", errors)
        checked = _check_artifact(candidate_id, row.get("canonical_candidate"), "canonical_candidate", errors)
        artifact = row.get("canonical_candidate")
        if checked is not None:
            _, observed_rate = checked
            if observed_rate != rate:
                errors.append(f"{candidate_id}: canonical sample rate is not {rate}")
            if isinstance(artifact, dict) and artifact.get("timeline") != "content_relative":
                errors.append(f"{candidate_id}: canonical timeline must be content_relative")
        _validate_alignment(candidate_id, row, errors)
        if isinstance(artifact, dict) and isinstance(artifact.get("sha256"), str):
            target_id = str(row.get("rendition_target_id", ""))
            prior = canonical_hash_target.get(artifact["sha256"])
            if prior is not None and prior != target_id:
                errors.append(f"{candidate_id}: canonical hash reused across rendition targets")
            canonical_hash_target[artifact["sha256"]] = target_id

    accepted_by_target: dict[str, dict[str, Any]] = {}
    selected_ids: set[str] = set()
    for accepted_id, row in accepted.items():
        target_id = str(row.get("rendition_target_id", ""))
        expected_accepted_id = accepted_audio_id(target_id) if target_id else ""
        if accepted_id != expected_accepted_id:
            errors.append(f"{accepted_id}: non-canonical accepted ID")
        if target_id in accepted_by_target:
            errors.append(f"{accepted_id}: multiple accepted rows for rendition target {target_id}")
        accepted_by_target[target_id] = row
        selected_id = row.get("selected_candidate_id")
        if not isinstance(selected_id, str) or selected_id not in canonical:
            errors.append(f"{accepted_id}: selected_candidate_id does not reference a canonical row")
            continue
        if selected_id in selected_ids:
            errors.append(f"{accepted_id}: selected candidate is reused")
        selected_ids.add(selected_id)
        source = canonical[selected_id]
        if source.get("rendition_target_id") != target_id:
            errors.append(f"{accepted_id}: selected candidate belongs to another target")
        source_alignment = source.get("alignment")
        if not isinstance(source_alignment, dict):
            errors.append(f"{accepted_id}: selected candidate has no alignment provenance")
        elif not gate_errors:
            script = scripts_by_id.get(str(source.get("script_id")))
            evidence_errors = validate_downstream_alignment_evidence(
                source,
                script,
                alignment_gate,
            )
            errors.extend(
                f"{accepted_id}: selected alignment evidence: {error}"
                for error in evidence_errors
            )
        source_qc = source.get("qc")
        if not isinstance(source_qc, dict) or source_qc.get("automatic_status") != "passed":
            errors.append(f"{accepted_id}: selected candidate automatic QC is not passed")
        if sha256_value(row.get("timing")) != sha256_value(source.get("timing")):
            errors.append(f"{accepted_id}: accepted timing does not exactly match selected QC row")
        if sha256_value(row.get("alignment")) != sha256_value(source.get("alignment")):
            errors.append(f"{accepted_id}: accepted alignment does not exactly match selected QC row")
        if not _same_artifact(row.get("canonical_candidate"), source.get("canonical_candidate")):
            errors.append(f"{accepted_id}: canonical -> accepted artifact lineage mismatch")
        if row.get("lifecycle_status") != "accepted":
            errors.append(f"{accepted_id}: accepted lifecycle status is wrong")
        if "preparation" in row or row.get("prepared_stimulus") is not None:
            errors.append(f"{accepted_id}: accepted row must not contain preparation/frame padding")
        checked = _check_artifact(accepted_id, row.get("accepted_utterance"), "accepted_utterance", errors)
        if checked is None:
            continue
        accepted_audio, accepted_rate = checked
        artifact = row.get("accepted_utterance")
        if accepted_rate != rate:
            errors.append(f"{accepted_id}: accepted sample rate is not {rate}")
        if isinstance(artifact, dict) and artifact.get("timeline") != "content_relative":
            errors.append(f"{accepted_id}: accepted timeline must be content_relative")
        timing = row.get("timing")
        if not isinstance(timing, dict):
            errors.append(f"{accepted_id}: accepted timing is missing")
            continue
        for timing_error in validate_timing(
            str(row.get("condition")), timing, duration_ms(accepted_audio, accepted_rate)
        ):
            errors.append(f"{accepted_id}: {timing_error}")
        selection = row.get("selection")
        if not isinstance(selection, dict):
            errors.append(f"{accepted_id}: selection metadata is missing")
            continue
        if selection.get("alignment_gate_hash") != sha256_value(alignment_gate):
            errors.append(f"{accepted_id}: selection alignment gate hash mismatch")
        if selection.get("policy_hash") != selection_policy_hash:
            errors.append(f"{accepted_id}: selection policy hash mismatch")
        canonical_artifact = source.get("canonical_candidate")
        if not isinstance(canonical_artifact, dict) or selection.get(
            "selected_canonical_sha256"
        ) != canonical_artifact.get("sha256"):
            errors.append(f"{accepted_id}: selection canonical hash lineage mismatch")
        tail_policy = selection.get("tail_policy")
        if not isinstance(tail_policy, dict) or float(tail_policy.get("fixed_tail_ms", -1)) != 200.0:
            errors.append(f"{accepted_id}: accepted tail policy is not fixed at 200 ms")
            continue
        if tail_policy.get("leading_coordinate_shift_samples") != 0:
            errors.append(f"{accepted_id}: accepted audio has a declared leading shift")
        utterance_end = timing.get("utterance_end_ms")
        if not isinstance(utterance_end, (int, float)):
            errors.append(f"{accepted_id}: utterance_end_ms is not numeric")
            continue
        expected_samples = round((float(utterance_end) + 200.0) * rate / 1000.0)
        if accepted_audio.size != expected_samples:
            errors.append(f"{accepted_id}: accepted tail is not exactly 200 ms at sample precision")
        source_checked = _check_artifact(
            selected_id, source.get("canonical_candidate"), "canonical_candidate", errors
        )
        if source_checked is None:
            continue
        source_audio, source_rate = source_checked
        if source_audio.size < expected_samples:
            expected_audio = np.pad(source_audio, (0, expected_samples - source_audio.size))
        else:
            expected_audio = source_audio[:expected_samples].copy()
        try:
            expected_audio, _ = normalize_audio(expected_audio, target_rms, peak_limit)
        except ValueError as error:
            errors.append(f"{accepted_id}: cannot reproduce normalization: {error}")
            continue
        if expected_audio.shape != accepted_audio.shape or not np.allclose(
            expected_audio, accepted_audio, atol=PCM_TOLERANCE, rtol=0
        ):
            errors.append(
                f"{accepted_id}: accepted samples are not a gain-only, zero-leading-shift "
                "transformation of the selected canonical candidate"
            )

    prepared_by_accepted: Counter[str] = Counter()
    expected_preparation = {
        "sample_rate": rate,
        "prefix_silence_ms": prefix_ms,
        "mimi_frame_samples": frame_samples,
        "normalization_stage": "accepted_canonical",
    }
    expected_preparation_hash = sha256_value(expected_preparation)
    for prepared_id, row in prepared.items():
        accepted_id = str(row.get("accepted_audio_id", ""))
        prepared_by_accepted[accepted_id] += 1
        source = accepted.get(accepted_id)
        if source is None:
            errors.append(f"{prepared_id}: prepared row references unknown accepted audio")
            continue
        expected_id = prepared_stimulus_id(accepted_id, expected_preparation_hash)
        if prepared_id != expected_id or row.get("preparation_hash") != expected_preparation_hash:
            errors.append(f"{prepared_id}: preparation ID/hash is not canonical")
        if row.get("lifecycle_status") != "prepared":
            errors.append(f"{prepared_id}: prepared lifecycle status is wrong")
        if not _same_artifact(row.get("accepted_utterance"), source.get("accepted_utterance")):
            errors.append(f"{prepared_id}: accepted -> prepared artifact lineage mismatch")
        accepted_checked = _check_artifact(
            accepted_id, source.get("accepted_utterance"), "accepted_utterance", errors
        )
        prepared_checked = _check_artifact(
            prepared_id, row.get("prepared_stimulus"), "prepared_stimulus", errors
        )
        if accepted_checked is None or prepared_checked is None:
            continue
        accepted_audio, accepted_rate = accepted_checked
        prepared_audio, prepared_rate = prepared_checked
        if accepted_rate != rate or prepared_rate != rate:
            errors.append(f"{prepared_id}: prepared/accepted sample-rate mismatch")
        expected_audio, padding = append_silence_and_frame_pad(
            accepted_audio, rate, prefix_ms, frame_samples
        )
        if prepared_audio.shape != expected_audio.shape or not np.array_equal(
            prepared_audio, expected_audio
        ):
            errors.append(f"{prepared_id}: prepared samples do not equal exact prefix + accepted + frame pad")
        if prepared_audio.size % frame_samples:
            errors.append(f"{prepared_id}: prepared audio is not a Mimi frame multiple")
        if not np.all(prepared_audio[:prefix_samples] == 0):
            errors.append(f"{prepared_id}: prepared prefix is not exact digital silence")
        preparation = row.get("preparation")
        if not isinstance(preparation, dict):
            errors.append(f"{prepared_id}: preparation metadata is missing")
        else:
            for key, value in {**expected_preparation, **padding}.items():
                if preparation.get(key) != value:
                    errors.append(f"{prepared_id}: preparation.{key} mismatch")
        expected_timing = shift_events(source["timing"], float(padding["prefix_ms_actual"]))
        if not _timing_equal(row.get("prepared_timing"), expected_timing):
            errors.append(f"{prepared_id}: prepared timestamps are not the exact prefix shift")

    if set(prepared_by_accepted) != set(accepted):
        errors.append(
            "accepted/prepared ID sets differ: "
            f"missing_prepared={sorted(set(accepted) - set(prepared_by_accepted))[:5]}, "
            f"unknown_accepted={sorted(set(prepared_by_accepted) - set(accepted))[:5]}"
        )
    for accepted_id, count in prepared_by_accepted.items():
        if count != 1:
            errors.append(f"{accepted_id}: expected one prepared stimulus, found {count}")

    if enforce_config_counts:
        counts = config["counts"]
        expected_candidates_min = int(counts["rendition_targets_per_track"])
        expected_accepted = int(counts["rendition_targets_per_track"])
        if len(canonical) < expected_candidates_min:
            errors.append(
                f"canonical candidates: expected at least {expected_candidates_min}, found {len(canonical)}"
            )
        if len(accepted) != expected_accepted:
            errors.append(f"accepted audio: expected {expected_accepted}, found {len(accepted)}")
        if len(prepared) != expected_accepted:
            errors.append(f"prepared stimuli: expected {expected_accepted}, found {len(prepared)}")
        expected_bundles = int(counts["matched_audio_bundles_per_track"])
        observed_bundles = {str(row.get("matched_audio_bundle_id", "")) for row in accepted.values()}
        if len(observed_bundles) != expected_bundles:
            errors.append(
                f"matched audio bundles: expected {expected_bundles}, found {len(observed_bundles)}"
            )
    return errors


def main() -> None:
    args = parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise SystemExit("selection policy must be a JSON object")
    errors = validate_audio_lifecycle(
        read_jsonl(args.raw),
        read_jsonl(args.canonical),
        read_jsonl(args.accepted),
        read_jsonl(args.prepared),
        read_config(),
        selection_policy=policy,
        scripts=read_jsonl(args.scripts),
        enforce_config_counts=not args.no_enforce_config_counts,
    )
    report = {
        "schema_version": "2.0.0",
        "validation_version": VALIDATION_VERSION,
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
    }
    if args.report:
        write_json(args.report, report)
    if errors:
        raise SystemExit("Audio lifecycle validation failed:\n" + "\n".join(errors))
    print("Audio lifecycle validation passed")


if __name__ == "__main__":
    main()
