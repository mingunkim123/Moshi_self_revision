"""Shared fail-closed contracts for alignment and manual-review evidence."""

from __future__ import annotations

from datetime import datetime
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from common import sha256_file, sha256_value


ALIGNMENT_INPUT_BINDING_VERSION = "2.0.0"
MANUAL_REVIEW_EVIDENCE_BINDING_VERSION = "2.1.0"
TRANSCRIPT_HASH_ENCODING = "exact_utf8_sha256"
ALIGNMENT_GATE_FIELDS = {
    "minimum_aggregate_confidence",
    "require_calibrated_confidence",
    "allow_audited_manual_review",
}
EXTERNAL_PROVENANCE_FIELDS = {
    "candidate_id",
    "script_id",
    "canonical_audio_sha256",
    "transcript_sha256",
    "alignment_run_id",
    "tool",
    "tool_version",
    "model_id",
    "input_binding_sha256",
    "transcript_hash_encoding",
    "binding_version",
    "verified_against_local_inputs",
    "external_row_content_sha256",
    "alignment_payload_sha256",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _finite_probability(value: Any, *, strictly_positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    lower_ok = number > 0 if strictly_positive else number >= 0
    return math.isfinite(number) and lower_ok and number <= 1


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _candidate_id(row: dict[str, Any]) -> str:
    candidate_id = row.get("candidate_id")
    if not _nonempty_string(candidate_id):
        candidate_id = row.get("selected_candidate_id")
    if not _nonempty_string(candidate_id):
        raise ValueError("alignment evidence requires candidate_id or selected_candidate_id")
    return str(candidate_id)


def validate_alignment_gate_contract(gate: Any) -> list[str]:
    if not isinstance(gate, dict):
        return ["alignment_gate must be an object"]
    if set(gate) != ALIGNMENT_GATE_FIELDS:
        return [f"alignment_gate fields must be exactly {sorted(ALIGNMENT_GATE_FIELDS)}"]
    errors: list[str] = []
    minimum = gate.get("minimum_aggregate_confidence")
    if not _finite_probability(minimum, strictly_positive=True):
        errors.append("alignment_gate.minimum_aggregate_confidence must be finite and in (0,1]")
    if not isinstance(gate.get("require_calibrated_confidence"), bool):
        errors.append("alignment_gate.require_calibrated_confidence must be boolean")
    if gate.get("allow_audited_manual_review") is not True:
        errors.append("alignment_gate.allow_audited_manual_review must be true")
    return errors


def frozen_transcript_sha256(script: dict[str, Any]) -> str:
    """Hash the exact frozen UTF-8 transcript after checking segment rendering."""
    transcript = script.get("transcript")
    segments = script.get("segments")
    if not _nonempty_string(transcript):
        raise ValueError(f"{script.get('script_id')}: frozen transcript is missing")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{script.get('script_id')}: frozen transcript segments are missing")
    rendered_parts: list[str] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or segment.get("segment_index") != index:
            raise ValueError(f"{script.get('script_id')}: frozen segments are not contiguous")
        text = segment.get("text")
        if not _nonempty_string(text):
            raise ValueError(f"{script.get('script_id')}: segment {index} text is missing")
        rendered_parts.append(str(text).strip())
    if "; ".join(rendered_parts) != transcript:
        raise ValueError(f"{script.get('script_id')}: frozen transcript/segment mismatch")
    return hashlib.sha256(str(transcript).encode("utf-8")).hexdigest()


def alignment_input_binding_sha256(
    *,
    candidate_id: str,
    script_id: str,
    canonical_audio_sha256: str,
    transcript_sha256: str,
    alignment_run_id: str,
    tool: str,
    tool_version: str,
    model_id: str,
) -> str:
    """Bind one independent alignment to exact audio, transcript and run inputs."""
    return sha256_value(
        {
            "binding_version": ALIGNMENT_INPUT_BINDING_VERSION,
            "candidate_id": candidate_id,
            "script_id": script_id,
            "canonical_audio_sha256": canonical_audio_sha256,
            "transcript_sha256": transcript_sha256,
            "transcript_hash_encoding": TRANSCRIPT_HASH_ENCODING,
            "alignment_run_id": alignment_run_id,
            "tool": tool,
            "tool_version": tool_version,
            "model_id": model_id,
        }
    )


def alignment_without_review_sha256(alignment: dict[str, Any]) -> str:
    if not isinstance(alignment, dict):
        raise ValueError("alignment must be an object")
    return sha256_value(
        {key: value for key, value in alignment.items() if key != "manual_review"}
    )


def independent_alignment_payload_sha256(alignment: dict[str, Any]) -> str:
    """Hash imported alignment output without its self-referential provenance/review."""
    if not isinstance(alignment, dict):
        raise ValueError("alignment must be an object")
    return sha256_value(
        {
            key: value
            for key, value in alignment.items()
            if key not in {"manual_review", "external_provenance"}
        }
    )


def _verified_canonical_audio_sha256(
    row: dict[str, Any], actual_canonical_audio_sha256: str | None
) -> str:
    artifact = row.get("canonical_candidate")
    if not isinstance(artifact, dict):
        raise ValueError("canonical_candidate artifact is missing")
    declared = artifact.get("sha256")
    if not isinstance(declared, str) or not SHA256_RE.fullmatch(declared):
        raise ValueError("canonical_candidate SHA-256 is invalid")
    if actual_canonical_audio_sha256 is None:
        path = Path(str(artifact.get("uri", "")))
        if not path.is_file():
            raise ValueError(f"canonical WAV does not exist: {path}")
        actual_canonical_audio_sha256 = sha256_file(path)
    if not SHA256_RE.fullmatch(str(actual_canonical_audio_sha256)):
        raise ValueError("actual canonical WAV SHA-256 is invalid")
    if declared != actual_canonical_audio_sha256:
        raise ValueError("canonical WAV manifest hash does not match actual input")
    return declared


def manual_review_evidence_binding(
    row: dict[str, Any],
    script: dict[str, Any],
    *,
    actual_canonical_audio_sha256: str | None = None,
) -> dict[str, str]:
    alignment = row.get("alignment")
    timing = row.get("timing")
    if not isinstance(alignment, dict):
        raise ValueError("manual review evidence requires alignment")
    if not isinstance(timing, dict):
        raise ValueError("manual review evidence requires timing")
    script_id = row.get("script_id")
    if not _nonempty_string(script_id):
        raise ValueError("manual review evidence requires script_id")
    if not isinstance(script, dict) or script.get("script_id") != script_id:
        raise ValueError("manual review evidence frozen script identity mismatch")
    transcript_hash = frozen_transcript_sha256(script)
    candidate_id = _candidate_id(row)
    audio_hash = _verified_canonical_audio_sha256(row, actual_canonical_audio_sha256)
    alignment_hash = alignment_without_review_sha256(alignment)
    timing_hash = sha256_value(timing)
    binding_sha256 = sha256_value(
        {
            "binding_version": MANUAL_REVIEW_EVIDENCE_BINDING_VERSION,
            "candidate_id": candidate_id,
            "script_id": script_id,
            "transcript_sha256": transcript_hash,
            "canonical_audio_sha256": audio_hash,
            "alignment_without_review_sha256": alignment_hash,
            "timing_sha256": timing_hash,
        }
    )
    return {
        "binding_version": MANUAL_REVIEW_EVIDENCE_BINDING_VERSION,
        "candidate_id": candidate_id,
        "script_id": str(script_id),
        "transcript_sha256": transcript_hash,
        "canonical_audio_sha256": audio_hash,
        "alignment_without_review_sha256": alignment_hash,
        "timing_sha256": timing_hash,
        "evidence_binding_sha256": binding_sha256,
    }


def set_manual_review_evidence_binding(
    row: dict[str, Any],
    script: dict[str, Any],
    *,
    actual_canonical_audio_sha256: str | None = None,
) -> dict[str, str]:
    alignment = row.get("alignment")
    if not isinstance(alignment, dict) or not isinstance(alignment.get("manual_review"), dict):
        raise ValueError("alignment.manual_review is missing")
    binding = manual_review_evidence_binding(
        row,
        script,
        actual_canonical_audio_sha256=actual_canonical_audio_sha256,
    )
    review = alignment["manual_review"]
    review["evidence_binding_version"] = binding["binding_version"]
    review["evidence_binding_sha256"] = binding["evidence_binding_sha256"]
    return binding


def _manual_review_errors(
    row: dict[str, Any],
    script: dict[str, Any],
    *,
    actual_canonical_audio_sha256: str | None,
) -> list[str]:
    alignment = row.get("alignment")
    review = alignment.get("manual_review") if isinstance(alignment, dict) else None
    if not isinstance(review, dict):
        return ["audited manual review is missing"]
    errors: list[str] = []
    reviewer_id = review.get("reviewer_id")
    reviewed_at = review.get("reviewed_at")
    try:
        parsed = datetime.fromisoformat(str(reviewed_at).replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if review.get("required") is not True:
        errors.append("manual_review.required must be true")
    if review.get("status") != "passed":
        errors.append("manual_review.status must be passed")
    if not _nonempty_string(reviewer_id):
        errors.append("manual_review.reviewer_id is missing")
    if parsed is None or parsed.tzinfo is None:
        errors.append("manual_review.reviewed_at must be timezone-aware ISO timestamp")
    if not _nonempty_string(review.get("reason")):
        errors.append("manual_review.reason is missing")
    try:
        binding = manual_review_evidence_binding(
            row,
            script,
            actual_canonical_audio_sha256=actual_canonical_audio_sha256,
        )
    except ValueError as error:
        errors.append(str(error))
        binding = None
    if binding is not None:
        if review.get("evidence_binding_version") != binding["binding_version"]:
            errors.append("manual_review evidence binding version mismatch")
        if review.get("evidence_binding_sha256") != binding["evidence_binding_sha256"]:
            errors.append("manual_review evidence binding is stale or mismatched")
    audit_log = review.get("audit_log")
    if not isinstance(audit_log, list) or not audit_log:
        errors.append("manual_review.audit_log must be non-empty")
    elif binding is not None:
        for index, entry in enumerate(audit_log):
            if not isinstance(entry, dict):
                errors.append(f"manual_review.audit_log[{index}] must be an object")
                continue
            if not _nonempty_string(entry.get("action")):
                errors.append(f"manual_review.audit_log[{index}].action is missing")
            if entry.get("reviewer_id") != reviewer_id:
                errors.append(f"manual_review.audit_log[{index}] reviewer mismatch")
            if entry.get("evidence_binding_sha256") != binding["evidence_binding_sha256"]:
                errors.append(f"manual_review.audit_log[{index}] evidence binding mismatch")
    return errors


def _independent_provenance_errors(
    row: dict[str, Any],
    script: dict[str, Any] | None,
    *,
    actual_canonical_audio_sha256: str | None,
) -> list[str]:
    alignment = row.get("alignment")
    if not isinstance(alignment, dict):
        return ["alignment is missing"]
    provenance = alignment.get("external_provenance")
    if not isinstance(provenance, dict):
        return ["independent alignment external_provenance is missing"]
    errors: list[str] = []
    if set(provenance) != EXTERNAL_PROVENANCE_FIELDS:
        errors.append(
            "independent alignment external_provenance fields must be exactly "
            f"{sorted(EXTERNAL_PROVENANCE_FIELDS)}"
        )
    if provenance.get("verified_against_local_inputs") is not True:
        errors.append("independent alignment provenance verified flag is not true")
    try:
        candidate_id = _candidate_id(row)
        audio_hash = _verified_canonical_audio_sha256(
            row, actual_canonical_audio_sha256
        )
    except ValueError as error:
        errors.append(str(error))
        candidate_id = ""
        audio_hash = ""
    script_id = row.get("script_id")
    if not _nonempty_string(script_id):
        errors.append("independent alignment row script_id is missing")
        script_id = ""
    if script is None:
        errors.append("frozen script row is required for independent alignment")
        transcript_hash = ""
    else:
        if script.get("script_id") != script_id:
            errors.append("independent alignment script identity mismatch")
        try:
            transcript_hash = frozen_transcript_sha256(script)
        except ValueError as error:
            errors.append(str(error))
            transcript_hash = ""
    expected_values = {
        "candidate_id": candidate_id,
        "script_id": script_id,
        "canonical_audio_sha256": audio_hash,
        "transcript_sha256": transcript_hash,
        "transcript_hash_encoding": TRANSCRIPT_HASH_ENCODING,
        "binding_version": ALIGNMENT_INPUT_BINDING_VERSION,
    }
    for field, expected in expected_values.items():
        if provenance.get(field) != expected:
            errors.append(f"independent alignment provenance {field} mismatch")
    for field in ("alignment_run_id", "tool", "tool_version", "model_id"):
        if not _nonempty_string(provenance.get(field)):
            errors.append(f"independent alignment provenance {field} is missing")
    if provenance.get("tool") != alignment.get("method"):
        errors.append("independent alignment method/tool mismatch")
    if provenance.get("tool_version") != alignment.get("tool_version"):
        errors.append("independent alignment tool_version mismatch")
    if provenance.get("model_id") != alignment.get("model_id"):
        errors.append("independent alignment model_id mismatch")
    if not isinstance(provenance.get("external_row_content_sha256"), str) or not SHA256_RE.fullmatch(
        str(provenance.get("external_row_content_sha256"))
    ):
        errors.append("independent alignment external row content hash is invalid")
    expected_payload_hash = independent_alignment_payload_sha256(alignment)
    if provenance.get("alignment_payload_sha256") != expected_payload_hash:
        errors.append("independent alignment payload hash mismatch")
    if all(_nonempty_string(provenance.get(field)) for field in ("alignment_run_id", "tool", "tool_version", "model_id")):
        expected_binding = alignment_input_binding_sha256(
            candidate_id=candidate_id,
            script_id=str(script_id),
            canonical_audio_sha256=audio_hash,
            transcript_sha256=transcript_hash,
            alignment_run_id=str(provenance["alignment_run_id"]),
            tool=str(provenance["tool"]),
            tool_version=str(provenance["tool_version"]),
            model_id=str(provenance["model_id"]),
        )
        if provenance.get("input_binding_sha256") != expected_binding:
            errors.append("independent alignment input binding mismatch")
    return errors


def validate_downstream_alignment_evidence(
    row: dict[str, Any],
    script: dict[str, Any],
    alignment_gate: dict[str, Any],
    *,
    actual_canonical_audio_sha256: str | None = None,
) -> list[str]:
    """Validate the automatic-independent or bound-manual acceptance path.

    ``actual_canonical_audio_sha256`` is an escape hatch for callers that have
    already hashed a relocated artifact.  Omitting it makes this helper hash the
    URI in ``row.canonical_candidate`` itself.
    """
    gate_errors = validate_alignment_gate_contract(alignment_gate)
    if gate_errors:
        return gate_errors
    alignment = row.get("alignment")
    if not isinstance(alignment, dict):
        return ["alignment is missing"]
    independent = alignment.get("independent_forced_alignment") is True
    if independent:
        provenance_errors = _independent_provenance_errors(
            row,
            script,
            actual_canonical_audio_sha256=actual_canonical_audio_sha256,
        )
        if provenance_errors:
            return provenance_errors
        confidence = alignment.get("confidence")
        automatic_errors: list[str] = []
        if not isinstance(confidence, dict):
            automatic_errors.append("independent alignment confidence is missing")
        else:
            aggregate = confidence.get("aggregate")
            threshold = confidence.get("threshold")
            minimum = float(alignment_gate["minimum_aggregate_confidence"])
            if confidence.get("threshold_passed") is not True:
                automatic_errors.append("independent alignment threshold_passed is not true")
            if not _finite_probability(aggregate):
                automatic_errors.append("independent alignment aggregate confidence is invalid")
            elif float(aggregate) < minimum:
                automatic_errors.append("independent alignment aggregate confidence is below policy minimum")
            if not _finite_probability(threshold) or float(threshold) != minimum:
                automatic_errors.append("independent importer threshold does not equal policy minimum")
            if not isinstance(confidence.get("calibrated"), bool):
                automatic_errors.append("independent alignment calibrated flag must be boolean")
            elif (
                alignment_gate["require_calibrated_confidence"] is True
                and confidence.get("calibrated") is not True
            ):
                automatic_errors.append("independent alignment confidence is not calibrated")
        if not automatic_errors:
            return []
        manual_errors = _manual_review_errors(
            row,
            script,
            actual_canonical_audio_sha256=actual_canonical_audio_sha256,
        )
        if not manual_errors:
            return []
        return [*automatic_errors, *manual_errors]
    return _manual_review_errors(
        row,
        script,
        actual_canonical_audio_sha256=actual_canonical_audio_sha256,
    )
