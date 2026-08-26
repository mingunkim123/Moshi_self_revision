#!/usr/bin/env python3
"""Merge disjoint raw-candidate shards and verify exact frozen-target coverage."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import DATASET_ROOT, read_jsonl, sha256_file, write_jsonl
from ids import candidate_id


DEFAULT_TARGETS = DATASET_ROOT / "assignments/rendition_targets.jsonl"
DEFAULT_OUTPUT = DATASET_ROOT / "manifests/raw_candidates.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def merge_rows(
    shards: list[list[dict[str, Any]]], targets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    target_map = {str(row["rendition_target_id"]): row for row in targets}
    if len(target_map) != len(targets):
        raise ValueError("frozen target manifest contains duplicate rendition_target_id")
    rows = [row for shard in shards for row in shard]
    ids = [str(row.get("candidate_id")) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate shards overlap or contain duplicate candidate_id")
    seen_targets: set[str] = set()
    for row in rows:
        target_id = str(row.get("rendition_target_id"))
        target = target_map.get(target_id)
        if target is None:
            raise ValueError(f"candidate references an unknown target: {target_id}")
        if row.get("candidate_id") != candidate_id(target_id, 1):
            raise ValueError(f"{target_id}: expected deterministic candidate index 1")
        for field in ("script_id", "source_track_id", "speaker_id", "voice"):
            if row.get(field) != target.get(field):
                raise ValueError(f"{target_id}: candidate/target {field} mismatch")
        raw = row.get("raw_candidate")
        if not isinstance(raw, dict):
            raise ValueError(f"{target_id}: raw_candidate is missing")
        path = Path(str(raw.get("uri", "")))
        if not path.is_file() or raw.get("sha256") != sha256_file(path):
            raise ValueError(f"{target_id}: raw audio is missing or has a hash mismatch")
        alignment = row.get("alignment")
        if not isinstance(alignment, dict):
            raise ValueError(f"{target_id}: provider event evidence is missing")
        boundary_path = Path(str(alignment.get("provider_event_uri", "")))
        if (
            not boundary_path.is_file()
            or alignment.get("provider_event_sha256") != sha256_file(boundary_path)
        ):
            raise ValueError(f"{target_id}: provider event sidecar is missing or mismatched")
        synthesis = row.get("synthesis")
        if not isinstance(synthesis, dict) or synthesis.get("provider") != "kokoro_local_v1_0":
            raise ValueError(f"{target_id}: candidate is not from the frozen Kokoro provider")
        seen_targets.add(target_id)
    missing = sorted(set(target_map) - seen_targets)
    if missing or len(rows) != len(targets):
        raise ValueError(
            f"candidate shards do not exactly cover frozen targets: rows={len(rows)}, "
            f"targets={len(targets)}, missing={missing[:3]}"
        )
    return sorted(rows, key=lambda row: str(row["candidate_id"]))


def main() -> None:
    args = parse_args()
    rows = merge_rows([read_jsonl(path) for path in args.inputs], read_jsonl(args.targets))
    write_jsonl(args.output, rows)
    print(f"Merged and verified {len(rows)} raw candidates -> {args.output}")


if __name__ == "__main__":
    main()
