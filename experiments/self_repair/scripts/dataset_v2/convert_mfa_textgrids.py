#!/usr/bin/env python3
"""Convert hash-bound MFA TextGrids into the v2 independent-alignment contract."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Any

from align_from_boundaries import boundary_tokens, transcript_tokens
from alignment_evidence import alignment_input_binding_sha256, frozen_transcript_sha256
from common import DATASET_ROOT, read_jsonl, sha256_file, sha256_value, write_json, write_jsonl


DEFAULT_TEXTGRIDS = DATASET_ROOT / "artifacts/mfa_output/aligned"
DEFAULT_INPUT_MANIFEST = DATASET_ROOT / "artifacts/mfa_input/mfa_input_manifest.jsonl"
DEFAULT_SCRIPTS = DATASET_ROOT / "generated/scripts.jsonl"
DEFAULT_OUTPUT = DATASET_ROOT / "artifacts/mfa_output/external_alignments.jsonl"
DEFAULT_REPORT = DATASET_ROOT / "reports/mfa_local_alignment.json"
TIER_RE = re.compile(
    r'item \[\d+\]:\s+class = "IntervalTier"\s+name = "words"(?P<body>.*?)(?=\n\s*item \[\d+\]:|\Z)',
    re.DOTALL,
)
INTERVAL_RE = re.compile(
    r'intervals \[\d+\]:\s+xmin = (?P<xmin>[-+0-9.eE]+)\s+'
    r'xmax = (?P<xmax>[-+0-9.eE]+)\s+text = "(?P<text>(?:[^"]|"")*)"',
    re.DOTALL,
)
DIAGNOSTIC_FIELDS = (
    "overall_log_likelihood",
    "speech_log_likelihood",
    "phone_duration_deviation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrids", type=Path, default=DEFAULT_TEXTGRIDS)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--tool-version", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--alignment-run-id", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def parse_word_tier(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    tier = TIER_RE.search(text)
    if tier is None:
        raise ValueError(f"{path}: words IntervalTier is missing")
    words: list[dict[str, Any]] = []
    previous_offset = 0.0
    for match in INTERVAL_RE.finditer(tier.group("body")):
        label = match.group("text").replace('""', '"').strip()
        if not label:
            continue
        onset_ms = float(match.group("xmin")) * 1000.0
        offset_ms = float(match.group("xmax")) * 1000.0
        if not math.isfinite(onset_ms + offset_ms) or onset_ms < previous_offset - 1e-6:
            raise ValueError(f"{path}: word intervals are not finite and monotonic")
        if offset_ms <= onset_ms:
            raise ValueError(f"{path}: word interval has non-positive duration")
        words.append(
            {
                "type": "word",
                "text": label,
                "offset_ms": onset_ms,
                "duration_ms": offset_ms - onset_ms,
            }
        )
        previous_offset = offset_ms
    if not words:
        raise ValueError(f"{path}: words tier contains no lexical intervals")
    return words


def _lexical_signature(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]", value.casefold()))


def resegment_words_to_transcript(
    words: list[dict[str, Any]], script: dict[str, Any]
) -> tuple[list[dict[str, Any]], bool]:
    """Map MFA's punctuation-dependent tokens back to the frozen tokenization.

    MFA commonly emits ``first`` + ``time`` for the frozen token ``first-time``.
    The lexical character stream is required to be identical; only boundaries
    inside that stream may be resegmented.  Times within one MFA word are
    interpolated by character position when the inverse split is needed.
    """
    expected = transcript_tokens(script)
    observed = boundary_tokens(words)
    expected_signature = "".join(_lexical_signature(token.normalized) for token in expected)
    observed_signature = "".join(_lexical_signature(token.normalized) for token in observed)
    if not expected_signature or observed_signature != expected_signature:
        raise ValueError(
            f"{script.get('script_id')}: MFA lexical stream does not equal frozen transcript"
        )
    observed_spans: list[tuple[int, int, float, float]] = []
    cursor = 0
    for token in observed:
        length = len(_lexical_signature(token.normalized))
        if length <= 0:
            raise ValueError("MFA lexical token has an empty comparison signature")
        observed_spans.append((cursor, cursor + length, token.onset_ms, token.offset_ms))
        cursor += length

    def start_time(position: int) -> float:
        for start, end, onset, offset in observed_spans:
            if start <= position < end:
                return onset + (offset - onset) * (position - start) / (end - start)
        if position == cursor:
            return observed_spans[-1][3]
        raise ValueError(f"cannot map transcript start character {position}")

    def end_time(position: int) -> float:
        for start, end, onset, offset in observed_spans:
            if start < position <= end:
                return onset + (offset - onset) * (position - start) / (end - start)
        if position == 0:
            return observed_spans[0][2]
        raise ValueError(f"cannot map transcript end character {position}")

    result: list[dict[str, Any]] = []
    cursor = 0
    for token in expected:
        length = len(_lexical_signature(token.normalized))
        onset = start_time(cursor)
        offset = end_time(cursor + length)
        if offset <= onset:
            raise ValueError(f"{script.get('script_id')}: resegmented word has invalid duration")
        result.append(
            {
                "type": "word",
                "text": token.text,
                "offset_ms": onset,
                "duration_ms": offset - onset,
                "timing_source": "mfa_word_tier_resegmented_to_frozen_token",
            }
        )
        cursor += length
    changed = [token.normalized for token in observed] != [token.normalized for token in expected]
    return result, changed


def _analysis_rows(path: Path, expected_ids: set[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            candidate_id = str(row.get("file", ""))
            if not candidate_id or candidate_id in result:
                raise ValueError(f"{path}: missing or duplicate file identity {candidate_id!r}")
            metrics: dict[str, float] = {}
            for field in DIAGNOSTIC_FIELDS:
                value = float(str(row.get(field, "")))
                if not math.isfinite(value):
                    raise ValueError(f"{path}: {candidate_id} has non-finite {field}")
                metrics[field] = value
            result[candidate_id] = metrics
    if set(result) != expected_ids:
        raise ValueError("alignment analysis rows do not exactly cover MFA input candidates")
    return result


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "mean": fmean(ordered),
        "p90": percentile(0.90),
        "max": ordered[-1],
    }


def convert(
    textgrid_root: Path,
    input_rows: list[dict[str, Any]],
    scripts: list[dict[str, Any]],
    analysis_path: Path,
    *,
    tool_version: str,
    model_id: str,
    alignment_run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(input_rows) != 600:
        raise ValueError(f"production conversion requires exactly 600 MFA inputs, got {len(input_rows)}")
    for label, value in (
        ("tool_version", tool_version),
        ("model_id", model_id),
        ("alignment_run_id", alignment_run_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
    input_map = {str(row.get("candidate_id", "")): row for row in input_rows}
    if "" in input_map or len(input_map) != len(input_rows):
        raise ValueError("MFA input manifest contains missing or duplicate candidate IDs")
    script_map = {str(row.get("script_id", "")): row for row in scripts}
    if "" in script_map or len(script_map) != len(scripts):
        raise ValueError("frozen scripts contain missing or duplicate script IDs")
    expected_ids = set(input_map)
    all_textgrids = list(textgrid_root.rglob("*.TextGrid"))
    textgrid_paths = {path.stem: path for path in all_textgrids}
    if len(textgrid_paths) != len(all_textgrids):
        raise ValueError("TextGrid output contains duplicate candidate basenames")
    if set(textgrid_paths) != expected_ids:
        raise ValueError("TextGrid files do not exactly cover MFA input candidates")
    diagnostics = _analysis_rows(analysis_path, expected_ids)
    speaker_counts: dict[str, int] = {}
    for row in input_rows:
        speaker_id = str(row.get("speaker_id", ""))
        if not speaker_id:
            raise ValueError(f"{row.get('candidate_id')}: MFA input speaker_id is missing")
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
    if len(speaker_counts) != 10 or set(speaker_counts.values()) != {60}:
        raise ValueError(f"MFA conversion requires 10 voices x 60 files, got {speaker_counts}")
    output: list[dict[str, Any]] = []
    textgrid_hashes: list[dict[str, str]] = []
    resegmented_candidate_count = 0
    for candidate_id in sorted(expected_ids):
        source = input_map[candidate_id]
        script_id = str(source.get("script_id", ""))
        script = script_map.get(script_id)
        if script is None:
            raise ValueError(f"{candidate_id}: frozen script is missing")
        transcript_hash = frozen_transcript_sha256(script)
        if source.get("transcript_sha256") != transcript_hash:
            raise ValueError(f"{candidate_id}: MFA input transcript hash mismatch")
        audio_hash = str(source.get("canonical_audio_sha256", ""))
        path = textgrid_paths[candidate_id]
        mfa_words = parse_word_tier(path)
        words, resegmented = resegment_words_to_transcript(mfa_words, script)
        resegmented_candidate_count += int(resegmented)
        expected_tokens = [token.normalized for token in transcript_tokens(script)]
        observed_tokens = [token.normalized for token in boundary_tokens(words)]
        if observed_tokens != expected_tokens:
            raise ValueError(
                f"{candidate_id}: MFA/transcript token mismatch; "
                f"expected={expected_tokens[:8]}, observed={observed_tokens[:8]}"
            )
        input_binding = alignment_input_binding_sha256(
            candidate_id=candidate_id,
            script_id=script_id,
            canonical_audio_sha256=audio_hash,
            transcript_sha256=transcript_hash,
            alignment_run_id=alignment_run_id,
            tool="montreal_forced_aligner",
            tool_version=tool_version,
            model_id=model_id,
        )
        textgrid_hash = sha256_file(path)
        textgrid_hashes.append({"candidate_id": candidate_id, "sha256": textgrid_hash})
        output.append(
            {
                "schema_version": "2.0.0",
                "candidate_id": candidate_id,
                "script_id": script_id,
                "audio_sha256": audio_hash,
                "transcript_sha256": transcript_hash,
                "alignment_run_id": alignment_run_id,
                "tool": "montreal_forced_aligner",
                "tool_version": tool_version,
                "model_id": model_id,
                "input_binding_sha256": input_binding,
                "words": words,
                "aggregate_confidence": 0.0,
                "minimum_word_confidence": None,
                "confidence_calibrated": False,
                "confidence_kind": "mfa_alignment_analysis_diagnostics_not_probability",
                "textgrid_sha256": textgrid_hash,
                "mfa_diagnostics": diagnostics[candidate_id],
            }
        )
    report = {
        "schema_version": "2.0.0",
        "status": "independent_alignment_completed_manual_review_required",
        "release_eligible": False,
        "candidate_count": len(output),
        "speaker_count": len(speaker_counts),
        "candidates_by_speaker": dict(sorted(speaker_counts.items())),
        "textgrid_count": len(textgrid_hashes),
        "resegmented_candidate_count": resegmented_candidate_count,
        "oov_word_type_count": 0,
        "tool": "montreal_forced_aligner",
        "tool_version": tool_version,
        "model_id": model_id,
        "alignment_run_id": alignment_run_id,
        "textgrid_set_sha256": sha256_value(textgrid_hashes),
        "analysis_csv_sha256": sha256_file(analysis_path),
        "external_alignment_set_sha256": sha256_value(output),
        "confidence_calibrated": False,
        "automatic_confidence_pass_count": 0,
        "manual_review_required_count": len(output),
        "diagnostics_are_not_probability_confidence": True,
        "diagnostic_summary": {
            field: _summary([row["mfa_diagnostics"][field] for row in output])
            for field in DIAGNOSTIC_FIELDS
        },
        "next_gate": (
            "Import with a positive frozen confidence threshold, rerun automatic QC, "
            "and complete hash-bound human alignment/audio review before selection."
        ),
    }
    return output, report


def main() -> None:
    args = parse_args()
    analysis = args.analysis or args.textgrids / "alignment_analysis.csv"
    output, report = convert(
        args.textgrids,
        read_jsonl(args.input_manifest),
        read_jsonl(args.scripts),
        analysis,
        tool_version=args.tool_version,
        model_id=args.model_id,
        alignment_run_id=args.alignment_run_id,
    )
    write_jsonl(args.output, output)
    write_json(args.report, report)
    print(f"Converted {len(output)} MFA TextGrids -> {args.output}")


if __name__ == "__main__":
    main()
