#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from audio_utils import active_rms, dbfs, duration_ms, read_pcm16_mono
from common import DATASET_ROOT, read_config, read_jsonl, sha256_file, write_json, write_jsonl
from timing import validate_timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run outcome-blind automatic QC on aligned canonical candidates.")
    parser.add_argument("--input", type=Path, default=DATASET_ROOT / "manifests/aligned_candidates.jsonl")
    parser.add_argument("--output", type=Path, default=DATASET_ROOT / "manifests/qc_candidates.jsonl")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def qc_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id", ""))
    artifact = row.get("canonical_candidate")
    if not isinstance(artifact, dict):
        raise ValueError(f"{candidate_id}: missing canonical candidate")
    path = Path(str(artifact.get("uri", "")))
    if not path.is_file() or artifact.get("sha256") != sha256_file(path):
        raise ValueError(f"{candidate_id}: canonical artifact missing or hash mismatch")
    audio, rate = read_pcm16_mono(path)
    expected_rate = int(config["audio"]["canonical_sample_rate"])
    if rate != expected_rate:
        raise ValueError(f"{candidate_id}: sample rate {rate}, expected {expected_rate}")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = active_rms(audio, -50.0) if audio.size else 0.0
    clipping_fraction = float(np.mean(np.abs(audio) >= 32767.0 / 32768.0)) if audio.size else 0.0
    timing = row.get("timing")
    timing_errors = (
        validate_timing(str(row.get("condition")), timing, duration_ms(audio, rate))
        if isinstance(timing, dict)
        else ["timing missing"]
    )
    mapping = (row.get("alignment") or {}).get("transcript_mapping")
    boundary_errors: list[str] = []
    leading_ms = trailing_ms = None
    if isinstance(mapping, list) and mapping:
        leading_ms = float(mapping[0]["onset_ms"])
        trailing_ms = duration_ms(audio, rate) - float(mapping[-1]["offset_ms"])
        if leading_ms < 5.0:
            boundary_errors.append("first word begins within 5 ms of file start")
        if trailing_ms < 50.0:
            boundary_errors.append("last word ends within 50 ms of file end")
    else:
        boundary_errors.append("alignment transcript mapping missing")
    errors = [*timing_errors, *boundary_errors]
    if audio.size == 0 or rms == 0:
        errors.append("empty or inactive audio")
    if clipping_fraction > 0:
        errors.append("digital clipping detected")
    peak_limit = float(config["audio"]["peak_limit_dbfs"])
    peak_dbfs = dbfs(peak)
    # Raw/canonical candidates are not normalized yet, so crossing the accepted
    # peak target is diagnostic but is not itself a failure unless clipped.
    item = dict(row)
    item["qc"] = {
        "automatic_status": "passed" if not errors else "failed",
        "errors": errors,
        "metrics": {
            "duration_ms": duration_ms(audio, rate),
            "active_rms_dbfs": dbfs(rms),
            "peak_dbfs": peak_dbfs,
            "accepted_peak_limit_dbfs": peak_limit,
            "clipping_fraction": clipping_fraction,
            "leading_silence_proxy_ms": leading_ms,
            "trailing_silence_proxy_ms": trailing_ms,
        },
        "clipping": clipping_fraction > 0,
        "noise_penalty": 0.0 if str(row.get("source_track_id", "")).startswith(("tts", "edge", "azure")) else None,
        "outcome_blind": True,
    }
    return item


def main() -> None:
    args = parse_args()
    config = read_config()
    rows = [qc_row(row, config) for row in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    failed = [row["candidate_id"] for row in rows if row["qc"]["automatic_status"] != "passed"]
    report = {
        "schema_version": "2.0.0",
        "candidate_count": len(rows),
        "passed_count": len(rows) - len(failed),
        "failed_count": len(failed),
        "failed_candidate_ids": failed,
    }
    if args.report:
        write_json(args.report, report)
    if failed:
        raise SystemExit(f"Automatic QC failed for {len(failed)} candidates; manifest was written for review")
    print(f"Automatic QC passed {len(rows)} candidates -> {args.output}")


if __name__ == "__main__":
    main()
