#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import DATASET_ROOT, DEFAULT_BLUEPRINTS, portable_path, read_jsonl, substitute, write_json, write_jsonl


DEFAULT_EVIDENCE = DATASET_ROOT / "config/value_evidence.json"
DEFAULT_OUTPUT = DATASET_ROOT / "answer_keys/answer_keys.jsonl"


SAFETY_NOTES = {
    "travel_003": "Do not treat allergy safety as guaranteed; credit a plan only if it preserves the stated restrictions and recommends direct venue verification.",
    "travel_006": "Accessibility claims require the stated measurable features; generic 'accessible' wording alone is unresolved.",
    "travel_019": "Recommendations should acknowledge pollen exposure and medication planning without offering medical guarantees.",
    "travel_027": "Never credit an unqualified claim that a venue is safe for severe allergy; require policy/advance-contact evidence and direct verification advice.",
    "travel_029": "Do not treat allergy safety as guaranteed; the group dietary constraints must remain explicit.",
    "travel_030": "Quiet/low-stimulation claims require concrete quiet-hour, entry, noise, or access evidence.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate condition-blind scenario/direction answer keys.")
    parser.add_argument("--blueprints", type=Path, default=DEFAULT_BLUEPRINTS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def build_answer_keys(
    blueprints: list[dict[str, Any]], evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = evidence["values"]
    for blueprint in sorted(blueprints, key=lambda item: item["scenario_id"]):
        for direction_id, old_value, new_value in (
            ("a_to_b", blueprint["value_a"], blueprint["value_b"]),
            ("b_to_a", blueprint["value_b"], blueprint["value_a"]),
        ):
            relations = []
            for unit in blueprint["dependent_units"]:
                relations.append(
                    {
                        "unit_id": unit["unit_id"],
                        "relation": unit["relation"],
                        "planning_constraint": unit["text"],
                        "new_bound_state": substitute(unit["state_patch"], {"{root_value}": new_value}),
                        "old_bound_state": substitute(unit["state_patch"], {"{root_value}": old_value}),
                    }
                )
            rows.append(
                {
                    "schema_version": "2.0.0",
                    "answer_key_id": f"{blueprint['scenario_id']}__{direction_id}",
                    "scenario_id": blueprint["scenario_id"],
                    "context_label": blueprint["context_label"],
                    "direction_id": direction_id,
                    "target_value": new_value,
                    "stale_value": old_value,
                    "target_evidence": values[new_value],
                    "stale_evidence": values[old_value],
                    "dependent_relations": relations,
                    "root_invariant_constraints": [
                        {"unit_id": unit["unit_id"], "relation": unit["relation"], "state": unit["state_patch"]}
                        for unit in blueprint["neutral_units"]
                    ],
                    "final_window_rule": "Use only evidence after closing_prompt_offset_ms for primary final_target_correct.",
                    "partial_response_rule": "Label every unaddressed D relation not_addressed; never infer it from another relation.",
                    "generic_response_rule": evidence["annotation_rule"],
                    "safety_note": SAFETY_NOTES.get(blueprint["scenario_id"]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    rows = build_answer_keys(read_jsonl(args.blueprints), evidence)
    if len(rows) != 60:
        raise ValueError(f"expected 60 answer keys, found {len(rows)}")
    write_jsonl(args.output, rows)
    if args.report:
        write_json(args.report, {"schema_version": "2.0.0", "answer_key_count": len(rows), "output": portable_path(args.output)})
    print(f"Generated {len(rows)} answer keys -> {args.output}")


if __name__ == "__main__":
    main()
