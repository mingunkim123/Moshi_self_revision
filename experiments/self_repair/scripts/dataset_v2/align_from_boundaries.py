#!/usr/bin/env python3
"""Extract deterministic alignment seeds from provider word-boundary events.

Provider events are useful for locating transcript tokens, but they are produced by
the same system that synthesized the audio.  Consequently this module deliberately
does *not* describe its output as independent forced alignment.  A seed must either
be replaced by an independent aligner or receive an audited manual review before it
can be selected for an accepted rendition.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Iterable

from alignment_evidence import set_manual_review_evidence_binding
from common import DATASET_ROOT, normalized_text, portable_path, read_jsonl, sha256_file, write_json, write_jsonl
from timing import derived_timing


ALIGNMENT_VERSION = "2.1.0"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’‐‑-][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class TranscriptToken:
    index: int
    segment_index: int
    segment_role: str
    unit_id: str | None
    segment_token_index: int
    text: str
    normalized: str


@dataclass(frozen=True)
class BoundaryToken:
    boundary_index: int
    boundary_token_index: int
    text: str
    normalized: str
    onset_ms: float
    offset_ms: float
    confidence: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map provider word boundaries to v2 script events (seed only)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATASET_ROOT / "manifests/canonical_candidates.jsonl",
    )
    parser.add_argument(
        "--scripts", type=Path, default=DATASET_ROOT / "generated/scripts.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_ROOT / "manifests/aligned_candidates.jsonl",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _token_parts(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    for match in TOKEN_RE.finditer(text):
        surface = match.group(0)
        normalized = normalized_text(surface).replace("’", "'").replace("‐", "-").replace("‑", "-")
        if normalized:
            parts.append((surface, normalized))
    return parts


def transcript_tokens(script: dict[str, Any]) -> list[TranscriptToken]:
    segments = script.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{script.get('script_id')}: segments must be a non-empty list")
    result: list[TranscriptToken] = []
    for list_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"{script.get('script_id')}: segment {list_index} is not an object")
        segment_index = segment.get("segment_index")
        if segment_index != list_index:
            raise ValueError(
                f"{script.get('script_id')}: segment indices must be contiguous from zero"
            )
        role = str(segment.get("role", ""))
        unit_id = segment.get("unit_id")
        for segment_token_index, (surface, token) in enumerate(
            _token_parts(str(segment.get("text", "")))
        ):
            result.append(
                TranscriptToken(
                    index=len(result),
                    segment_index=list_index,
                    segment_role=role,
                    unit_id=str(unit_id) if unit_id is not None else None,
                    segment_token_index=segment_token_index,
                    text=surface,
                    normalized=token,
                )
            )
    rendered = " ".join(token.normalized for token in result)
    declared = (
        normalized_text(str(script.get("transcript", "")))
        .replace("’", "'")
        .replace("‐", "-")
        .replace("‑", "-")
    )
    if rendered != declared:
        raise ValueError(
            f"{script.get('script_id')}: transcript tokens do not equal the segment rendering"
        )
    return result


def _boundary_source(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    synthesis = row.get("synthesis")
    candidates: tuple[tuple[Any, str], ...] = (
        (
            synthesis.get("provider_word_boundaries") if isinstance(synthesis, dict) else None,
            "synthesis.provider_word_boundaries",
        ),
        (
            synthesis.get("word_boundaries") if isinstance(synthesis, dict) else None,
            "synthesis.word_boundaries",
        ),
        (row.get("provider_word_boundaries"), "provider_word_boundaries"),
        (row.get("word_boundaries"), "word_boundaries"),
    )
    for value, path in candidates:
        if isinstance(value, list) and value:
            if not all(isinstance(item, dict) for item in value):
                raise ValueError(f"{row.get('candidate_id')}: {path} must contain objects")
            return value, path
    alignment = row.get("alignment")
    if isinstance(alignment, dict) and alignment.get("provider_event_uri"):
        event_path = Path(str(alignment["provider_event_uri"]))
        if not event_path.is_file():
            raise ValueError(f"{row.get('candidate_id')}: missing provider event sidecar {event_path}")
        declared_hash = alignment.get("provider_event_sha256")
        if declared_hash != sha256_file(event_path):
            raise ValueError(f"{row.get('candidate_id')}: provider event sidecar hash mismatch")
        import json

        payload = json.loads(event_path.read_text(encoding="utf-8"))
        events = payload.get("events") if isinstance(payload, dict) else None
        if not isinstance(events, list) or not events:
            raise ValueError(f"{row.get('candidate_id')}: provider event sidecar has no events")
        words = [event for event in events if isinstance(event, dict) and event.get("type") == "word"]
        if not words:
            raise ValueError(f"{row.get('candidate_id')}: provider event sidecar has no word events")
        return words, "alignment.provider_event_uri"
    raise ValueError(f"{row.get('candidate_id')}: no provider word boundaries")


def boundary_tokens(boundaries: list[dict[str, Any]]) -> list[BoundaryToken]:
    result: list[BoundaryToken] = []
    previous_onset = -1.0
    for boundary_index, boundary in enumerate(boundaries):
        text = str(boundary.get("text", boundary.get("word", "")))
        parts = _token_parts(text)
        if not parts:
            # Providers sometimes emit punctuation-only boundary records.  They do
            # not take part in lexical alignment and are retained in provenance.
            continue
        onset_value = boundary.get("offset_ms", boundary.get("start_ms"))
        if not isinstance(onset_value, (int, float)):
            raise ValueError(f"boundary {boundary_index}: missing numeric offset_ms/start_ms")
        onset_ms = float(onset_value)
        if "duration_ms" in boundary and isinstance(boundary["duration_ms"], (int, float)):
            offset_ms = onset_ms + float(boundary["duration_ms"])
        elif isinstance(boundary.get("end_ms"), (int, float)):
            offset_ms = float(boundary["end_ms"])
        else:
            raise ValueError(f"boundary {boundary_index}: missing duration_ms/end_ms")
        if onset_ms < 0 or offset_ms <= onset_ms:
            raise ValueError(f"boundary {boundary_index}: invalid interval {onset_ms}..{offset_ms}")
        if onset_ms < previous_onset:
            raise ValueError("provider word boundaries are not monotonic")
        previous_onset = onset_ms
        confidence_value = boundary.get("confidence")
        confidence = (
            float(confidence_value)
            if isinstance(confidence_value, (int, float))
            else None
        )
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(f"boundary {boundary_index}: confidence must be in [0, 1]")
        part_duration = (offset_ms - onset_ms) / len(parts)
        for part_index, (surface, token) in enumerate(parts):
            part_onset = onset_ms + part_index * part_duration
            part_offset = onset_ms + (part_index + 1) * part_duration
            result.append(
                BoundaryToken(
                    boundary_index=boundary_index,
                    boundary_token_index=part_index,
                    text=surface,
                    normalized=token,
                    onset_ms=part_onset,
                    offset_ms=part_offset,
                    confidence=confidence,
                )
            )
    if not result:
        raise ValueError("provider word boundaries contain no lexical tokens")
    return result


def _map_tokens(
    expected: list[TranscriptToken], observed: list[BoundaryToken]
) -> dict[int, BoundaryToken]:
    expected_words = [token.normalized for token in expected]
    observed_words = [token.normalized for token in observed]
    matcher = SequenceMatcher(None, expected_words, observed_words, autojunk=False)
    mapping: dict[int, BoundaryToken] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = observed[block.b + offset]
    missing = [expected[index].text for index in range(len(expected)) if index not in mapping]
    matched_observed = {
        (value.boundary_index, value.boundary_token_index) for value in mapping.values()
    }
    extra = [
        token.text
        for token in observed
        if (token.boundary_index, token.boundary_token_index) not in matched_observed
    ]
    if missing or extra:
        raise ValueError(
            "provider/transcript token mismatch; "
            f"missing transcript tokens={missing[:8]}, extra provider tokens={extra[:8]}"
        )
    return mapping


def _segment_token_indices(
    tokens: list[TranscriptToken], role: str
) -> list[int]:
    matches = [token.index for token in tokens if token.segment_role == role]
    if not matches:
        raise ValueError(f"script has no {role!r} segment")
    segment_ids = {tokens[index].segment_index for index in matches}
    if len(segment_ids) != 1:
        raise ValueError(f"script has multiple {role!r} segments")
    return matches


def _phrase_indices(
    tokens: list[TranscriptToken], segment_indices: list[int], phrase: str
) -> list[int]:
    needle = [normalized for _, normalized in _token_parts(phrase)]
    if not needle:
        raise ValueError(f"empty phrase cannot be aligned: {phrase!r}")
    haystack = [tokens[index].normalized for index in segment_indices]
    starts = [
        offset
        for offset in range(len(haystack) - len(needle) + 1)
        if haystack[offset : offset + len(needle)] == needle
    ]
    if len(starts) != 1:
        raise ValueError(
            f"phrase {phrase!r} occurs {len(starts)} times in segment "
            f"{tokens[segment_indices[0]].segment_index}"
        )
    start = starts[0]
    return segment_indices[start : start + len(needle)]


def _span(
    label: str,
    indices: list[int],
    tokens: list[TranscriptToken],
    mapping: dict[int, BoundaryToken],
) -> dict[str, Any]:
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"{label}: token span is empty or non-contiguous")
    first = mapping[indices[0]]
    last = mapping[indices[-1]]
    return {
        "label": label,
        "segment_index": tokens[indices[0]].segment_index,
        "token_start_index": indices[0],
        "token_end_index_exclusive": indices[-1] + 1,
        "text": " ".join(tokens[index].text for index in indices),
        "onset_ms": first.onset_ms,
        "offset_ms": last.offset_ms,
    }


def _provider_identity(row: dict[str, Any]) -> dict[str, Any]:
    synthesis = row.get("synthesis") if isinstance(row.get("synthesis"), dict) else {}
    return {
        key: synthesis.get(key)
        for key in (
            "provider",
            "provider_version",
            "model",
            "voice",
            "request_id",
        )
        if synthesis.get(key) is not None
    }


def align_candidate_row(row: dict[str, Any], script: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id", ""))
    if row.get("lifecycle_status") != "canonical_candidate":
        raise ValueError(f"{candidate_id}: expected lifecycle_status=canonical_candidate")
    if row.get("script_id") != script.get("script_id"):
        raise ValueError(f"{candidate_id}: candidate/script ID mismatch")
    condition = str(script.get("condition", ""))
    tokens = transcript_tokens(script)
    raw_boundaries, source_path = _boundary_source(row)
    observed = boundary_tokens(raw_boundaries)
    mapping = _map_tokens(tokens, observed)

    if condition == "clean_final":
        root_indices = _segment_token_indices(tokens, "clean_root")
        new_indices = _phrase_indices(tokens, root_indices, str(script["new_value"]))
        event_spans: dict[str, dict[str, Any] | None] = {
            "old_value": None,
            "repair_cue": None,
            "new_value": _span("new_value", new_indices, tokens, mapping),
            "repeated_old": None,
        }
    else:
        root_indices = _segment_token_indices(tokens, "initial_old_root")
        cue_indices = _segment_token_indices(tokens, "repair_cue")
        old_indices = _phrase_indices(tokens, root_indices, str(script["old_value"]))
        new_indices = _phrase_indices(tokens, cue_indices, str(script["new_value"]))
        repeated_indices = _phrase_indices(tokens, cue_indices, str(script["old_value"]))
        event_spans = {
            "old_value": _span("old_value", old_indices, tokens, mapping),
            "repair_cue": _span("repair_cue", cue_indices, tokens, mapping),
            "new_value": _span("new_value", new_indices, tokens, mapping),
            "repeated_old": _span("repeated_old", repeated_indices, tokens, mapping),
        }

    unit_spans: list[dict[str, Any]] = []
    pre_units = {str(value) for value in script.get("pre_repair_units", [])}
    for segment in script["segments"]:
        unit_id = segment.get("unit_id")
        if unit_id is None:
            continue
        segment_index = int(segment["segment_index"])
        indices = [token.index for token in tokens if token.segment_index == segment_index]
        span = _span(str(unit_id), indices, tokens, mapping)
        span.update(
            {
                "unit_id": str(unit_id),
                "relation": segment.get("relation"),
                "binding": segment.get("binding"),
                "repair_position": "pre" if str(unit_id) in pre_units else "post",
                "stale_dependency_age_ms": None,
            }
        )
        unit_spans.append(span)

    closing_indices = _segment_token_indices(tokens, "closing_prompt")
    closing_span = _span("closing_prompt", closing_indices, tokens, mapping)
    event_spans["closing_prompt"] = closing_span
    last_boundary = mapping[len(tokens) - 1]
    new_span = event_spans["new_value"]
    assert new_span is not None
    if condition == "clean_final":
        base_timing: dict[str, float | None] = {
            "old_value_onset_ms": None,
            "old_value_offset_ms": None,
            "repair_cue_onset_ms": None,
            "new_value_onset_ms": float(new_span["onset_ms"]),
            "new_value_offset_ms": float(new_span["offset_ms"]),
            "repeated_old_onset_ms": None,
            "repeated_old_offset_ms": None,
            "repair_cue_offset_ms": None,
            "closing_prompt_onset_ms": float(closing_span["onset_ms"]),
            "closing_prompt_offset_ms": float(closing_span["offset_ms"]),
            "utterance_end_ms": last_boundary.offset_ms,
        }
    else:
        old_span = event_spans["old_value"]
        cue_span = event_spans["repair_cue"]
        repeated_span = event_spans["repeated_old"]
        assert old_span is not None and cue_span is not None and repeated_span is not None
        base_timing = {
            "old_value_onset_ms": float(old_span["onset_ms"]),
            "old_value_offset_ms": float(old_span["offset_ms"]),
            "repair_cue_onset_ms": float(cue_span["onset_ms"]),
            "new_value_onset_ms": float(new_span["onset_ms"]),
            "new_value_offset_ms": float(new_span["offset_ms"]),
            "repeated_old_onset_ms": float(repeated_span["onset_ms"]),
            "repeated_old_offset_ms": float(repeated_span["offset_ms"]),
            "repair_cue_offset_ms": float(cue_span["offset_ms"]),
            "closing_prompt_onset_ms": float(closing_span["onset_ms"]),
            "closing_prompt_offset_ms": float(closing_span["offset_ms"]),
            "utterance_end_ms": last_boundary.offset_ms,
        }
    timing = derived_timing(condition, base_timing)
    cue_onset = timing.get("repair_cue_onset_ms")
    if isinstance(cue_onset, (int, float)):
        for span in unit_spans:
            if (
                span["repair_position"] == "pre"
                and str(span["unit_id"]).startswith("D")
            ):
                age = float(cue_onset) - float(span["offset_ms"])
                if age < 0:
                    raise ValueError(f"{candidate_id}: pre-repair unit ends after the cue")
                span["stale_dependency_age_ms"] = age

    reported_confidences = [
        value.confidence for value in mapping.values() if value.confidence is not None
    ]
    provider_confidence = fmean(reported_confidences) if reported_confidences else None
    token_mapping = [
        {
            "token_index": token.index,
            "segment_index": token.segment_index,
            "segment_role": token.segment_role,
            "unit_id": token.unit_id,
            "text": token.text,
            "normalized": token.normalized,
            "provider_boundary_index": mapping[token.index].boundary_index,
            "provider_boundary_token_index": mapping[token.index].boundary_token_index,
            "onset_ms": mapping[token.index].onset_ms,
            "offset_ms": mapping[token.index].offset_ms,
            "provider_confidence": mapping[token.index].confidence,
        }
        for token in tokens
    ]

    item = dict(row)
    item["timing"] = timing
    item["alignment"] = {
        "alignment_version": ALIGNMENT_VERSION,
        "method": "provider_word_boundaries_seed",
        "source_kind": "provider_generated_word_boundaries",
        "source_path": source_path,
        "independent_forced_alignment": False,
        "suitable_for_final_acceptance_without_review": False,
        "provider_identity": _provider_identity(row),
        "confidence": {
            # Exact token coverage is a mapping diagnostic, not acoustic alignment
            # confidence.  Leave aggregate unset when the provider supplies no
            # confidence so selection cannot silently treat coverage as certainty.
            "aggregate": provider_confidence,
            "minimum": min(reported_confidences) if reported_confidences else None,
            "source": "provider_reported" if reported_confidences else "exact_token_coverage_only",
            "calibrated": False,
            "provider_reported_token_count": len(reported_confidences),
            "matched_transcript_token_count": len(tokens),
            "transcript_token_coverage": 1.0,
        },
        "manual_review": {
            "required": True,
            "reason": "provider boundaries are an extraction seed, not independent forced alignment",
            "status": "pending",
            "reviewer_id": None,
            "reviewed_at": None,
            "audit_log": [],
        },
        "raw_provider_boundaries": raw_boundaries,
        "transcript_mapping": token_mapping,
        "event_spans": event_spans,
        "unit_spans": unit_spans,
    }
    set_manual_review_evidence_binding(item, script)
    return item


def align_rows(
    candidates: Iterable[dict[str, Any]], scripts: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    script_index: dict[str, dict[str, Any]] = {}
    for script in scripts:
        script_id = str(script.get("script_id", ""))
        if not script_id or script_id in script_index:
            raise ValueError(f"missing or duplicate script_id: {script_id!r}")
        script_index[script_id] = script
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            raise ValueError(f"missing or duplicate candidate_id: {candidate_id!r}")
        seen.add(candidate_id)
        script_id = str(row.get("script_id", ""))
        if script_id not in script_index:
            raise ValueError(f"{candidate_id}: unknown script_id {script_id!r}")
        output.append(align_candidate_row(row, script_index[script_id]))
    return sorted(output, key=lambda value: str(value["candidate_id"]))


def main() -> None:
    args = parse_args()
    output = align_rows(read_jsonl(args.input), read_jsonl(args.scripts))
    write_jsonl(args.output, output)
    report = {
        "schema_version": "2.0.0",
        "alignment_version": ALIGNMENT_VERSION,
        "candidate_count": len(output),
        "method": "provider_word_boundaries_seed",
        "independent_forced_alignment": False,
        "manual_review_required_count": sum(
            row["alignment"]["manual_review"]["required"] for row in output
        ),
        "output": portable_path(args.output),
    }
    if args.report:
        write_json(args.report, report)
    print(f"Mapped {len(output)} provider-boundary alignment seeds -> {args.output}")


if __name__ == "__main__":
    main()
