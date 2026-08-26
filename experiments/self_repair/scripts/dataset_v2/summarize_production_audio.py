#!/usr/bin/env python3
"""Verify and summarize the private 600-candidate Kokoro production run."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import statistics
from typing import Any

from common import DATASET_ROOT, read_config, read_jsonl, sha256_file, write_json


DEFAULT_RAW = DATASET_ROOT / "manifests/raw_candidates.jsonl"
DEFAULT_QC = DATASET_ROOT / "manifests/qc_candidates.jsonl"
DEFAULT_TARGETS = DATASET_ROOT / "assignments/rendition_targets.jsonl"
DEFAULT_OUTPUT = DATASET_ROOT / "reports/kokoro_production_audio.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--qc", type=Path, default=DEFAULT_QC)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def summarize(
    raw_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    config: dict[str, Any],
    raw_manifest: Path,
    qc_manifest: Path,
    targets_manifest: Path,
) -> dict[str, Any]:
    expected = int(config["counts"]["rendition_targets_per_track"])
    if not (len(raw_rows) == len(qc_rows) == len(targets) == expected == 600):
        raise ValueError(
            f"expected exact 600-row lifecycle, got raw={len(raw_rows)}, "
            f"qc={len(qc_rows)}, targets={len(targets)}"
        )
    target_map = {str(row["rendition_target_id"]): row for row in targets}
    raw_map = {str(row["candidate_id"]): row for row in raw_rows}
    qc_map = {str(row["candidate_id"]): row for row in qc_rows}
    if len(target_map) != 600 or len(raw_map) != 600 or set(raw_map) != set(qc_map):
        raise ValueError("target/candidate IDs are duplicate or raw/QC coverage differs")

    source_tracks = {str(row["source_track_id"]) for row in targets}
    if len(source_tracks) != 1:
        raise ValueError("production summary requires exactly one source track")
    source_track_id = next(iter(source_tracks))
    source_track = config.get("source_tracks", {}).get(source_track_id)
    if not isinstance(source_track, dict) or source_track.get("provider") != "kokoro_local_v1_0":
        raise ValueError("frozen target source track is not the configured Kokoro track")

    raw_bytes = canonical_bytes = provider_event_bytes = 0
    durations: list[float] = []
    conditions: Counter[str] = Counter()
    speakers: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    provider_seed_only = 0
    independent_alignment = 0
    independent_alignment_review_pending = 0
    for candidate, raw in raw_map.items():
        qc = qc_map[candidate]
        target_id = str(raw["rendition_target_id"])
        target = target_map.get(target_id)
        if target is None:
            raise ValueError(f"{candidate}: unknown rendition target")
        for field in (
            "script_id",
            "source_track_id",
            "speaker_id",
            "voice",
            "condition",
            "direction_id",
        ):
            if raw.get(field) != target.get(field) or qc.get(field) != target.get(field):
                raise ValueError(f"{candidate}: lifecycle/target {field} mismatch")
        raw_artifact = raw.get("raw_candidate")
        canonical = qc.get("canonical_candidate")
        if not isinstance(raw_artifact, dict) or not isinstance(canonical, dict):
            raise ValueError(f"{candidate}: raw/canonical artifact metadata missing")
        raw_path = Path(str(raw_artifact.get("uri", "")))
        canonical_path = Path(str(canonical.get("uri", "")))
        for label, artifact, path in (
            ("raw", raw_artifact, raw_path),
            ("canonical", canonical, canonical_path),
        ):
            if not path.is_file() or artifact.get("sha256") != sha256_file(path):
                raise ValueError(f"{candidate}: {label} artifact missing or hash mismatch")
            if artifact.get("sample_rate") != 24000 or artifact.get("channels") != 1:
                raise ValueError(f"{candidate}: {label} audio contract mismatch")
        raw_alignment = raw.get("alignment")
        if not isinstance(raw_alignment, dict):
            raise ValueError(f"{candidate}: provider event evidence missing")
        provider_event_path = Path(str(raw_alignment.get("provider_event_uri", "")))
        if (
            not provider_event_path.is_file()
            or raw_alignment.get("provider_event_sha256")
            != sha256_file(provider_event_path)
        ):
            raise ValueError(f"{candidate}: provider event sidecar hash mismatch")
        if (qc.get("qc") or {}).get("automatic_status") != "passed":
            raise ValueError(f"{candidate}: automatic QC did not pass")
        synthesis = qc.get("synthesis") or {}
        provider_artifact = synthesis.get("provider_artifact") or {}
        if synthesis.get("provider") != "kokoro_local_v1_0":
            raise ValueError(f"{candidate}: provider mismatch")
        if provider_artifact.get("model_sha256") != source_track["model_sha256"]:
            raise ValueError(f"{candidate}: model hash mismatch")
        alignment = qc.get("alignment") or {}
        if alignment.get("independent_forced_alignment") is True:
            independent_alignment += 1
            if (alignment.get("manual_review") or {}).get("status") == "pending":
                independent_alignment_review_pending += 1
        elif (
            alignment.get("independent_forced_alignment") is False
            and alignment.get("method") == "provider_word_boundaries_seed"
            and (alignment.get("manual_review") or {}).get("status") == "pending"
        ):
            provider_seed_only += 1
        else:
            raise ValueError(f"{candidate}: unexpected alignment lifecycle status")
        raw_bytes += raw_path.stat().st_size
        canonical_bytes += canonical_path.stat().st_size
        provider_event_bytes += provider_event_path.stat().st_size
        durations.append(float(canonical["duration_ms"]))
        conditions[str(target["condition"])] += 1
        speakers[str(target["speaker_id"])] += 1
        directions[str(target["direction_id"])] += 1

    if set(conditions.values()) != {120} or set(speakers.values()) != {60}:
        raise ValueError("condition or speaker counts are not balanced")
    if set(directions.values()) != {300}:
        raise ValueError("direction counts are not balanced")
    duration_total_ms = sum(durations)
    independent_complete = independent_alignment == expected and provider_seed_only == 0
    return {
        "schema_version": "2.0.0",
        "report_kind": "kokoro_private_production_audio",
        "status": (
            "raw_600_independent_alignment_and_automatic_qc_passed_human_review_pending"
            if independent_complete
            else "raw_600_automatic_qc_passed_independent_alignment_pending"
        ),
        "release_eligible": False,
        "accepted_audio_selected": False,
        "prepared_stimuli_created": False,
        "required_next_gate": (
            "Complete hash-bound human alignment review and double-listen before selection."
            if independent_complete
            else "Run independent MFA alignment and complete human double-listen before selection."
        ),
        "source": {
            "source_track_id": source_track_id,
            "provider": source_track["provider"],
            "model_repo": source_track["model_repo"],
            "model_revision": source_track["model_revision"],
            "model_sha256": source_track["model_sha256"],
            "candidate_policy": "one_deterministic_candidate_per_rendition_target",
        },
        "manifests": {
            "targets_sha256": sha256_file(targets_manifest),
            "raw_candidates_sha256": sha256_file(raw_manifest),
            "qc_candidates_sha256": sha256_file(qc_manifest),
        },
        "counts": {
            "rendition_targets": len(targets),
            "raw_candidates": len(raw_rows),
            "canonical_candidates": len(qc_rows),
            "automatic_qc_passed": len(qc_rows),
            "provider_seed_only": provider_seed_only,
            "independent_alignment_complete": independent_alignment,
            "independent_alignment_review_pending": independent_alignment_review_pending,
            "by_condition": dict(sorted(conditions.items())),
            "by_speaker": dict(sorted(speakers.items())),
            "by_direction": dict(sorted(directions.items())),
        },
        "audio": {
            "sample_rate": 24000,
            "channels": 1,
            "sample_width_bytes": 2,
            "canonical_duration_ms": _distribution(durations),
            "canonical_duration_total_ms": duration_total_ms,
            "canonical_duration_total_hours": duration_total_ms / 3_600_000.0,
            "raw_wav_bytes": raw_bytes,
            "canonical_wav_bytes": canonical_bytes,
            "provider_event_sidecar_bytes": provider_event_bytes,
            "raw_plus_canonical_gib": (raw_bytes + canonical_bytes) / 1024**3,
        },
    }


def main() -> None:
    args = parse_args()
    report = summarize(
        read_jsonl(args.raw),
        read_jsonl(args.qc),
        read_jsonl(args.targets),
        read_config(),
        args.raw,
        args.qc,
        args.targets,
    )
    write_json(args.output, report)
    print(
        f"Verified {report['counts']['raw_candidates']} Kokoro candidates; "
        f"automatic QC passed={report['counts']['automatic_qc_passed']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
