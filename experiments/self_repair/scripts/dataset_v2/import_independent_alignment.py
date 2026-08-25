#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from align_from_boundaries import align_candidate_row
from alignment_evidence import (
    ALIGNMENT_INPUT_BINDING_VERSION,
    TRANSCRIPT_HASH_ENCODING,
    alignment_input_binding_sha256,
    frozen_transcript_sha256,
    independent_alignment_payload_sha256,
    set_manual_review_evidence_binding,
)
from common import (
    DATASET_ROOT,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
    write_jsonl,
)

SUPPORTED_TOOLS = ("montreal_forced_aligner", "whisperx_known_transcript")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import independently generated known-transcript word alignments.")
    parser.add_argument("--input", type=Path, default=DATASET_ROOT / "manifests/canonical_candidates.jsonl")
    parser.add_argument("--scripts", type=Path, default=DATASET_ROOT / "generated/scripts.jsonl")
    parser.add_argument("--external", type=Path, required=True, help="JSONL from MFA/WhisperX conversion")
    parser.add_argument("--tool", required=True, choices=SUPPORTED_TOOLS)
    parser.add_argument("--tool-version", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--alignment-run-id",
        required=True,
        help="Run ID independently supplied to the importer and matched to every external row.",
    )
    parser.add_argument("--minimum-confidence", type=float, required=True)
    parser.add_argument("--output", type=Path, default=DATASET_ROOT / "manifests/aligned_candidates.jsonl")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _required_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _actual_local_input_binding(
    candidate: dict[str, Any],
    script: dict[str, Any],
    *,
    alignment_run_id: str,
    tool: str,
    tool_version: str,
    model_id: str,
) -> dict[str, str]:
    candidate_id = str(candidate.get("candidate_id", ""))
    if candidate.get("lifecycle_status") != "canonical_candidate":
        raise ValueError(f"{candidate_id}: expected lifecycle_status=canonical_candidate")
    script_id = str(candidate.get("script_id", ""))
    if script.get("script_id") != script_id:
        raise ValueError(f"{candidate_id}: candidate/script identity mismatch")
    artifact = candidate.get("canonical_candidate")
    if not isinstance(artifact, dict):
        raise ValueError(f"{candidate_id}: canonical_candidate is missing")
    path = Path(str(artifact.get("uri", "")))
    if not path.is_file():
        raise ValueError(f"{candidate_id}: canonical WAV does not exist: {path}")
    actual_audio_sha256 = sha256_file(path)
    if artifact.get("sha256") != actual_audio_sha256:
        raise ValueError(f"{candidate_id}: canonical WAV manifest hash mismatch")
    transcript_sha256 = frozen_transcript_sha256(script)
    binding_sha256 = alignment_input_binding_sha256(
        candidate_id=candidate_id,
        script_id=script_id,
        canonical_audio_sha256=actual_audio_sha256,
        transcript_sha256=transcript_sha256,
        alignment_run_id=alignment_run_id,
        tool=tool,
        tool_version=tool_version,
        model_id=model_id,
    )
    return {
        "candidate_id": candidate_id,
        "script_id": script_id,
        "canonical_audio_sha256": actual_audio_sha256,
        "transcript_sha256": transcript_sha256,
        "alignment_run_id": alignment_run_id,
        "tool": tool,
        "tool_version": tool_version,
        "model_id": model_id,
        "input_binding_sha256": binding_sha256,
    }


def _verify_external_provenance(
    external: dict[str, Any], actual: dict[str, str]
) -> None:
    candidate_id = actual["candidate_id"]
    comparisons = {
        "script_id": actual["script_id"],
        "audio_sha256": actual["canonical_audio_sha256"],
        "transcript_sha256": actual["transcript_sha256"],
        "alignment_run_id": actual["alignment_run_id"],
        "tool": actual["tool"],
        "tool_version": actual["tool_version"],
        "model_id": actual["model_id"],
        "input_binding_sha256": actual["input_binding_sha256"],
    }
    for field, expected in comparisons.items():
        observed = external.get(field)
        if observed != expected:
            raise ValueError(
                f"{candidate_id}: external {field} does not match verified import input"
            )


def verified_run_input_sha256(rows: list[dict[str, Any]]) -> str:
    bindings = []
    for row in sorted(rows, key=lambda item: str(item.get("candidate_id", ""))):
        alignment = row.get("alignment")
        provenance = alignment.get("external_provenance") if isinstance(alignment, dict) else None
        if not isinstance(provenance, dict) or provenance.get("verified_against_local_inputs") is not True:
            raise ValueError(f"{row.get('candidate_id')}: verified external provenance is missing")
        bindings.append(
            {
                key: provenance[key]
                for key in (
                    "candidate_id",
                    "script_id",
                    "canonical_audio_sha256",
                    "transcript_sha256",
                    "alignment_run_id",
                    "tool",
                    "tool_version",
                    "model_id",
                    "input_binding_sha256",
                )
            }
        )
    return sha256_value(bindings)


def import_alignments(
    candidates: list[dict[str, Any]],
    scripts: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    *,
    tool: str,
    tool_version: str,
    model_id: str,
    alignment_run_id: str,
    minimum_confidence: float,
) -> list[dict[str, Any]]:
    if (
        isinstance(minimum_confidence, bool)
        or not isinstance(minimum_confidence, (int, float))
        or not math.isfinite(float(minimum_confidence))
        or not 0 <= float(minimum_confidence) <= 1
    ):
        raise ValueError("minimum_confidence must be in [0,1]")
    tool = _required_identity(tool, "tool")
    if tool not in SUPPORTED_TOOLS:
        raise ValueError(f"unsupported independent alignment tool: {tool!r}")
    tool_version = _required_identity(tool_version, "tool_version")
    model_id = _required_identity(model_id, "model_id")
    alignment_run_id = _required_identity(alignment_run_id, "alignment_run_id")
    script_map: dict[str, dict[str, Any]] = {}
    for script in scripts:
        script_id = str(script.get("script_id", ""))
        if not script_id or script_id in script_map:
            raise ValueError(f"missing or duplicate script_id: {script_id!r}")
        script_map[script_id] = script
    external_map: dict[str, dict[str, Any]] = {}
    for row in external_rows:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in external_map:
            raise ValueError(f"missing or duplicate external candidate_id: {candidate_id!r}")
        external_map[candidate_id] = row
    candidate_ids = [str(row.get("candidate_id", "")) for row in candidates]
    if not candidate_ids:
        raise ValueError("canonical candidate input must not be empty")
    if any(not value for value in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("canonical candidates contain missing or duplicate candidate_id")
    extras = sorted(set(external_map) - set(candidate_ids))
    missing = sorted(set(candidate_ids) - set(external_map))
    if extras or missing:
        raise ValueError(
            "external/canonical candidate sets differ: "
            f"missing_external={missing[:5]}, unknown_external={extras[:5]}"
        )
    output: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: str(row["candidate_id"])):
        candidate_id = str(candidate.get("candidate_id", ""))
        external = external_map[candidate_id]
        words = external.get("words")
        if not isinstance(words, list) or not words:
            raise ValueError(f"{candidate_id}: external words must be a non-empty array")
        confidence = external.get("aggregate_confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            raise ValueError(f"{candidate_id}: aggregate_confidence must be in [0,1]")
        script = script_map.get(str(candidate.get("script_id")))
        if script is None:
            raise ValueError(f"{candidate_id}: unknown script")
        actual = _actual_local_input_binding(
            candidate,
            script,
            alignment_run_id=alignment_run_id,
            tool=tool,
            tool_version=tool_version,
            model_id=model_id,
        )
        _verify_external_provenance(external, actual)
        minimum_word_confidence = external.get("minimum_word_confidence")
        if minimum_word_confidence is not None and (
            isinstance(minimum_word_confidence, bool)
            or not isinstance(minimum_word_confidence, (int, float))
            or not 0 <= float(minimum_word_confidence) <= 1
        ):
            raise ValueError(f"{candidate_id}: minimum_word_confidence must be in [0,1]")
        confidence_calibrated = external.get("confidence_calibrated", False)
        if not isinstance(confidence_calibrated, bool):
            raise ValueError(f"{candidate_id}: confidence_calibrated must be boolean")
        seeded = dict(candidate)
        # Canonical candidates can retain provider boundaries from synthesis.
        # Remove every earlier boundary source so align_candidate_row cannot
        # accidentally prefer the provider seed over the independently supplied
        # words being imported here.
        synthesis = dict(seeded.get("synthesis") or {})
        synthesis.pop("provider_word_boundaries", None)
        synthesis.pop("word_boundaries", None)
        seeded["synthesis"] = synthesis
        seeded.pop("word_boundaries", None)
        seeded["provider_word_boundaries"] = words
        aligned = align_candidate_row(seeded, script)
        aligned.pop("provider_word_boundaries", None)
        prior = aligned["alignment"]
        confidence_passed = float(confidence) >= minimum_confidence
        prior.update(
            {
                "method": tool,
                "source_kind": "independent_known_transcript_forced_alignment",
                "source_path": "external_alignment_manifest",
                "independent_forced_alignment": True,
                "suitable_for_final_acceptance_without_review": confidence_passed,
                "tool_version": tool_version,
                "model_id": model_id,
                "confidence": {
                    "aggregate": float(confidence),
                    "minimum": minimum_word_confidence,
                    "source": str(external.get("confidence_kind", "tool_diagnostic_normalized")),
                    "calibrated": confidence_calibrated,
                    "threshold": minimum_confidence,
                    "threshold_passed": confidence_passed,
                    "transcript_token_coverage": 1.0,
                },
                "manual_review": {
                    "required": not confidence_passed,
                    "reason": None if confidence_passed else "independent alignment confidence below frozen threshold",
                    "status": "not_required" if confidence_passed else "pending",
                    "reviewer_id": None,
                    "reviewed_at": None,
                    "audit_log": [],
                },
                "external_provenance": {
                    **actual,
                    "transcript_hash_encoding": TRANSCRIPT_HASH_ENCODING,
                    "binding_version": ALIGNMENT_INPUT_BINDING_VERSION,
                    "verified_against_local_inputs": True,
                    "external_row_content_sha256": sha256_value(external),
                },
            }
        )
        prior["external_provenance"]["alignment_payload_sha256"] = (
            independent_alignment_payload_sha256(prior)
        )
        set_manual_review_evidence_binding(
            aligned,
            script,
            actual_canonical_audio_sha256=actual["canonical_audio_sha256"],
        )
        output.append(aligned)
    return output


def main() -> None:
    args = parse_args()
    output = import_alignments(
        read_jsonl(args.input),
        read_jsonl(args.scripts),
        read_jsonl(args.external),
        tool=args.tool,
        tool_version=args.tool_version,
        model_id=args.model_id,
        alignment_run_id=args.alignment_run_id,
        minimum_confidence=args.minimum_confidence,
    )
    write_jsonl(args.output, output)
    report = {
        "schema_version": "2.0.0",
        "candidate_count": len(output),
        "independent_forced_alignment": True,
        "tool": args.tool,
        "tool_version": args.tool_version,
        "model_id": args.model_id,
        "alignment_run_id": args.alignment_run_id,
        "verified_run_input_sha256": verified_run_input_sha256(output),
        "all_external_provenance_verified_against_local_inputs": all(
            row["alignment"]["external_provenance"]["verified_against_local_inputs"]
            is True
            for row in output
        ),
        "minimum_confidence": args.minimum_confidence,
        "manual_review_required_count": sum(row["alignment"]["manual_review"]["required"] for row in output),
    }
    if args.report:
        write_json(args.report, report)
    print(f"Imported {len(output)} independent alignments -> {args.output}")


if __name__ == "__main__":
    main()
