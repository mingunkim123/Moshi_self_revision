#!/usr/bin/env python3
"""Prepare a hash-bound 600-item MFA corpus using local hard links."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from typing import Any

from alignment_evidence import frozen_transcript_sha256
from common import DATASET_ROOT, read_jsonl, sha256_file, sha256_value, write_json, write_jsonl


DEFAULT_CANDIDATES = DATASET_ROOT / "manifests/canonical_candidates.jsonl"
DEFAULT_SCRIPTS = DATASET_ROOT / "generated/scripts.jsonl"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "artifacts/mfa_input"
DEFAULT_REPORT = DATASET_ROOT / "release_evidence/mfa_input_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def prepare(
    candidates: list[dict[str, Any]],
    scripts: list[dict[str, Any]],
    output_root: Path,
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(candidates) != 600:
        raise ValueError(f"production MFA corpus requires exactly 600 candidates, got {len(candidates)}")
    script_map = {str(row["script_id"]): row for row in scripts}
    if len(script_map) != len(scripts):
        raise ValueError("frozen scripts contain duplicate IDs")
    corpus = output_root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    speaker_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: str(row["candidate_id"])):
        candidate_id = str(candidate["candidate_id"])
        script_id = str(candidate["script_id"])
        speaker_id = str(candidate.get("speaker_id", ""))
        if not speaker_id or "/" in speaker_id or speaker_id in {".", ".."}:
            raise ValueError(f"{candidate_id}: invalid speaker_id for MFA corpus")
        speaker_counts[speaker_id] += 1
        script = script_map.get(script_id)
        if script is None:
            raise ValueError(f"{candidate_id}: frozen script missing")
        artifact = candidate.get("canonical_candidate")
        if not isinstance(artifact, dict):
            raise ValueError(f"{candidate_id}: canonical artifact missing")
        source = Path(str(artifact.get("uri", "")))
        if not source.is_file() or artifact.get("sha256") != sha256_file(source):
            raise ValueError(f"{candidate_id}: canonical WAV missing or hash mismatch")
        speaker_root = corpus / speaker_id
        speaker_root.mkdir(parents=True, exist_ok=True)
        wav = speaker_root / f"{candidate_id}.wav"
        lab = speaker_root / f"{candidate_id}.lab"
        transcript = str(script["transcript"]).strip()
        if not transcript:
            raise ValueError(f"{candidate_id}: transcript is empty")
        if wav.exists():
            if not resume or sha256_file(wav) != artifact["sha256"]:
                raise FileExistsError(f"{candidate_id}: existing MFA WAV is not a verified resume artifact")
        else:
            os.link(source, wav)
        if lab.exists():
            if not resume or lab.read_text(encoding="utf-8") != transcript + "\n":
                raise FileExistsError(f"{candidate_id}: existing MFA transcript is not a verified resume artifact")
        else:
            lab.write_text(transcript + "\n", encoding="utf-8")
        rows.append(
            {
                "schema_version": "2.0.0",
                "candidate_id": candidate_id,
                "script_id": script_id,
                "speaker_id": speaker_id,
                "corpus_wav": f"corpus/{speaker_id}/{candidate_id}.wav",
                "corpus_lab": f"corpus/{speaker_id}/{candidate_id}.lab",
                "canonical_audio_sha256": artifact["sha256"],
                "transcript_sha256": frozen_transcript_sha256(script),
                "sample_rate": artifact["sample_rate"],
                "channels": artifact["channels"],
            }
        )
    if len(speaker_counts) != 10 or set(speaker_counts.values()) != {60}:
        raise ValueError(f"MFA speaker layout must be 10 voices x 60 files, got {dict(speaker_counts)}")
    manifest = output_root / "mfa_input_manifest.jsonl"
    write_jsonl(manifest, rows)
    report = {
        "schema_version": "2.0.0",
        "status": "ready_for_independent_mfa_alignment",
        "release_eligible": False,
        "candidate_count": len(rows),
        "speaker_count": len(speaker_counts),
        "candidates_by_speaker": dict(sorted(speaker_counts.items())),
        "manifest": "mfa_input_manifest.jsonl",
        "manifest_sha256": sha256_file(manifest),
        "input_set_sha256": sha256_value(rows),
        "audio_storage": "speaker_subdirectory_hard_links_to_canonical_wavs_on_local_host",
        "transcript_format": "UTF-8 lab, one exact frozen transcript per WAV",
        "recommended_linux_commands": [
            "mfa model download acoustic english_us_arpa",
            "mfa model download dictionary english_us_arpa",
            "mfa validate corpus english_us_arpa english_us_arpa --clean",
            "mfa align corpus english_us_arpa english_us_arpa aligned --clean",
        ],
        "next_gate": (
            "Convert MFA TextGrids to the external alignment contract, bind the exact "
            "tool/model/run IDs, then import_independent_alignment.py."
        ),
    }
    return rows, report


def main() -> None:
    args = parse_args()
    rows, report = prepare(
        read_jsonl(args.candidates),
        read_jsonl(args.scripts),
        args.output_root,
        resume=args.resume,
    )
    write_json(args.report, report)
    print(f"Prepared {len(rows)} hash-bound MFA corpus items -> {args.output_root}")


if __name__ == "__main__":
    main()
