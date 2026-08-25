#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from common import DATASET_ROOT, dotted_state, read_jsonl, write_jsonl


DEFAULT_OUTPUT = DATASET_ROOT / "blueprints/scenarios.jsonl"
DEFAULT_PATCHES = DATASET_ROOT / "blueprints/independent_review_patches.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize independently reviewed blueprint drafts.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Reviewed authoring draft; the canonical finalized blueprints are versioned in Git.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    parser.add_argument("--reviewed-at", required=True, help="ISO-8601 review timestamp")
    return parser.parse_args()


def apply_review_patches(
    rows: list[dict[str, Any]], patches: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    revised = copy.deepcopy(rows)
    scenarios = {str(row["scenario_id"]): row for row in revised}
    if len(scenarios) != len(revised):
        raise ValueError("duplicate scenario_id in draft")
    seen: set[tuple[str, str]] = set()
    for patch in patches:
        scenario_id = str(patch["scenario_id"])
        unit_id = str(patch["unit_id"])
        key = (scenario_id, unit_id)
        if key in seen:
            raise ValueError(f"duplicate review patch target: {scenario_id}/{unit_id}")
        seen.add(key)
        scenario = scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"review patch has unknown scenario: {scenario_id}")
        matches = [
            unit
            for unit in [*scenario["dependent_units"], *scenario["neutral_units"]]
            if unit.get("unit_id") == unit_id
        ]
        if len(matches) != 1:
            raise ValueError(f"review patch target is not unique: {scenario_id}/{unit_id}")
        unit = matches[0]
        if unit.get("text") != patch.get("expected_text"):
            raise ValueError(f"stale review patch text: {scenario_id}/{unit_id}")
        unit["text"] = patch["text"]
        state_patch = unit.get("state_patch")
        if not isinstance(state_patch, dict):
            raise ValueError(f"missing state patch: {scenario_id}/{unit_id}")
        for state_key in patch.get("state_patch_deletes", []):
            if state_key not in state_patch:
                raise ValueError(
                    f"review patch cannot delete missing key {state_key}: {scenario_id}/{unit_id}"
                )
            del state_patch[state_key]
        state_patch.update(patch.get("state_patch_updates", {}))
    return revised


def finalize(
    rows: list[dict[str, Any]], reviewed_at: str, patches: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if patches is not None:
        rows = apply_review_patches(rows, patches)
    output: list[dict[str, Any]] = []
    for index, draft in enumerate(rows, 1):
        item = {
            key: draft[key]
            for key in (
                "schema_version",
                "scenario_id",
                "context_label",
                "language",
                "domain",
                "root_slot",
                "value_a",
                "value_b",
                "root_template",
                "repair_template",
                "closing_prompt",
                "one_dependency_unit",
                "one_dependency_pre_position",
            )
        }
        item["dependent_units"] = [_normalize_unit(unit) for unit in draft["dependent_units"]]
        item["neutral_units"] = [_normalize_unit(unit) for unit in draft["neutral_units"]]
        item["gold_state_template"] = {str(item["root_slot"]): "{root_value}"}
        item["gold_state_template"].update(
            dotted_state(
                unit["state_patch"]
                for unit in [*item["dependent_units"], *item["neutral_units"]]
            )
        )
        item["rotation_id"] = f"R{((index - 1) % 3) + 1}"
        item["reviews"] = [
            {
                "reviewer_id": "codex_agent_root_semantic_review",
                "decision": "approved",
                "reviewed_at": reviewed_at,
                "notes": "Primary structural, counterbalance, state-patch, safety, and wording review; automated agent review, not human-subject review.",
            },
            {
                "reviewer_id": "codex_agent_independent_quality",
                "decision": "approved",
                "reviewed_at": reviewed_at,
                "notes": "Independent semantic, pragmatic speech-act, and bidirectional naturalness review; automated agent review, not human-subject review.",
            },
        ]
        item["review_status"] = "approved"
        item["source"] = {
            "authoring_method": "original_structured_draft_with_two_automated_agent_reviews",
            "license": "repository_license_pending_dataset_release_confirmation",
            "notes": "No external sentences copied; independent-review patches are versioned; human review remains a release gate.",
        }
        output.append(item)
    return output


def _normalize_unit(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: unit[key]
        for key in (
            "unit_id",
            "text",
            "relation",
            "binding",
            "state_patch",
            "balance_pair_id",
            "speech_act",
            "boundary_type",
            "planning_frame_id",
            "pragmatic_function",
        )
    }


def main() -> None:
    args = parse_args()
    patch_payload = json.loads(args.patches.read_text(encoding="utf-8"))
    patches = patch_payload.get("patches") if isinstance(patch_payload, dict) else None
    if not isinstance(patches, list):
        raise ValueError("review patch file must contain a patches array")
    rows = finalize(read_jsonl(args.input), args.reviewed_at, patches)
    write_jsonl(args.output, rows)
    print(f"Finalized {len(rows)} blueprints -> {args.output}")


if __name__ == "__main__":
    main()
