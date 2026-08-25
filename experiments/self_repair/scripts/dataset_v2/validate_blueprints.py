#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

from common import (
    DEFAULT_BLUEPRINTS,
    DEFAULT_CONFIG,
    DEPENDENT_IDS,
    NEUTRAL_IDS,
    contains_term,
    dotted_state,
    iter_duplicates,
    read_config,
    read_jsonl,
    portable_path,
    sha256_file,
    substitute,
    units_by_id,
    word_count,
    write_json,
)


REQUIRED_UNIT_FIELDS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset v2 semantic blueprints.")
    parser.add_argument("--input", type=Path, default=DEFAULT_BLUEPRINTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def parse_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return True


def patch_contains_placeholder(patch: dict[str, Any]) -> bool:
    return "{root_value}" in str(patch)


def validate_blueprints(
    blueprints: list[dict[str, Any]], config: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    expected_scenarios = int(config["counts"]["scenarios"])
    required_reviews = int(config["text_qc"]["required_reviews"])
    max_word_delta = int(config["text_qc"]["maximum_balance_pair_word_delta"])
    expected_ids = [f"travel_{index:03d}" for index in range(1, expected_scenarios + 1)]

    if len(blueprints) != expected_scenarios:
        errors.append(f"expected {expected_scenarios} blueprints, found {len(blueprints)}")

    scenario_ids = [str(item.get("scenario_id", "")) for item in blueprints]
    duplicates = list(iter_duplicates(scenario_ids))
    if duplicates:
        errors.append(f"duplicate scenario IDs: {duplicates}")
    if sorted(scenario_ids) != expected_ids:
        missing = sorted(set(expected_ids) - set(scenario_ids))
        extra = sorted(set(scenario_ids) - set(expected_ids))
        errors.append(f"scenario ID sequence mismatch; missing={missing}, extra={extra}")

    identity_counts: Counter[str] = Counter()
    position_counts: Counter[int] = Counter()
    cross_counts: Counter[tuple[str, int]] = Counter()
    rotation_counts: Counter[str] = Counter()
    frames_by_binding: Counter[tuple[str, str]] = Counter()
    frames_by_identity: Counter[tuple[str, str]] = Counter()
    semantic_fingerprints: list[str] = []

    for blueprint in blueprints:
        scenario_id = str(blueprint.get("scenario_id", "<missing>"))
        prefix = f"{scenario_id}: "
        for key, expected in (
            ("schema_version", "2.0.0"),
            ("language", config["language"]),
            ("domain", config["domain"]),
            ("root_slot", config["root_slot"]),
            ("value_a", config["value_a"]),
            ("value_b", config["value_b"]),
            ("repair_template", config["repair_template"]),
            ("closing_prompt", config["closing_prompt"]),
            ("review_status", "approved"),
        ):
            if blueprint.get(key) != expected:
                errors.append(f"{prefix}{key} must equal {expected!r}")

        root_template = str(blueprint.get("root_template", ""))
        if root_template.count("{value}") != 1:
            errors.append(f"{prefix}root_template must contain {{value}} exactly once")
        if root_template.rstrip().endswith((".", "?", "!", ";")):
            errors.append(f"{prefix}root_template must be a nonterminal clause")

        repair_template = str(blueprint.get("repair_template", ""))
        if repair_template.count("{new}") != 1 or repair_template.count("{old}") != 1:
            errors.append(f"{prefix}repair_template needs one {{new}} and one {{old}}")
        if repair_template.rstrip().endswith((".", "?", "!", ";")):
            errors.append(f"{prefix}repair_template must be a nonterminal clause")

        closing = str(blueprint.get("closing_prompt", ""))
        if "{" in closing or any(
            contains_term(closing, value) for value in (config["value_a"], config["value_b"])
        ):
            errors.append(f"{prefix}closing_prompt must be root invariant")
        if not closing.rstrip().endswith("?"):
            errors.append(f"{prefix}closing_prompt must be a terminal question")
        context_label = str(blueprint.get("context_label", ""))
        if not context_label or not context_label.replace("_", "").isalnum() or not context_label.isascii():
            errors.append(f"{prefix}context_label must be a non-empty ASCII slug")

        dependent = blueprint.get("dependent_units")
        neutral = blueprint.get("neutral_units")
        if not isinstance(dependent, list) or not isinstance(neutral, list):
            errors.append(f"{prefix}dependent_units and neutral_units must be arrays")
            continue
        if len(dependent) != 3 or len(neutral) != 3:
            errors.append(f"{prefix}requires exactly three dependent and three neutral units")
            continue

        unit_map = units_by_id(blueprint)
        if set(unit_map) != set((*DEPENDENT_IDS, *NEUTRAL_IDS)):
            errors.append(f"{prefix}unit IDs must be D1-D3 and N1-N3 exactly once")
            continue

        pair_to_ids: dict[str, list[str]] = {"P1": [], "P2": [], "P3": []}
        for unit_id, unit in unit_map.items():
            missing_fields = REQUIRED_UNIT_FIELDS - set(unit)
            if missing_fields:
                errors.append(f"{prefix}{unit_id} missing fields {sorted(missing_fields)}")
                continue
            text = str(unit["text"])
            if text.rstrip().endswith((".", "?", "!", ";")):
                errors.append(f"{prefix}{unit_id} must be a nonterminal clause")
            if unit.get("speech_act") != "statement" or unit.get("boundary_type") != "nonterminal":
                errors.append(f"{prefix}{unit_id} must be a nonterminal statement")
            if unit.get("pragmatic_function") != "planning_constraint":
                errors.append(f"{prefix}{unit_id} must have planning_constraint pragmatic function")
            expected_binding = "root_dependent" if unit_id.startswith("D") else "root_invariant"
            if unit.get("binding") != expected_binding:
                errors.append(f"{prefix}{unit_id} binding must be {expected_binding}")
            frame_id = str(unit.get("planning_frame_id", ""))
            if frame_id not in config["text_qc"]["planning_frames"]:
                errors.append(f"{prefix}{unit_id} invalid planning_frame_id {frame_id!r}")
            else:
                frames_by_binding[(expected_binding, frame_id)] += 1
                frames_by_identity[(unit_id, frame_id)] += 1
            patch = unit.get("state_patch")
            if not isinstance(patch, dict) or not patch:
                errors.append(f"{prefix}{unit_id} state_patch must be a non-empty object")
            elif unit_id.startswith("D") and not patch_contains_placeholder(patch):
                errors.append(f"{prefix}{unit_id} state_patch must bind {{root_value}}")
            elif unit_id.startswith("N") and patch_contains_placeholder(patch):
                errors.append(f"{prefix}{unit_id} state_patch must be root invariant")
            pair_id = str(unit.get("balance_pair_id", ""))
            if pair_id not in pair_to_ids:
                errors.append(f"{prefix}{unit_id} has invalid balance_pair_id {pair_id!r}")
            else:
                pair_to_ids[pair_id].append(unit_id)

        for pair_id, ids in pair_to_ids.items():
            expected_pair = {f"D{pair_id[-1]}", f"N{pair_id[-1]}"}
            if set(ids) != expected_pair:
                errors.append(f"{prefix}{pair_id} must pair {sorted(expected_pair)}, found {ids}")
                continue
            if len({unit_map[unit_id].get("planning_frame_id") for unit_id in ids}) != 1:
                errors.append(f"{prefix}{pair_id} D/N units must use the same planning frame")
            delta = abs(word_count(unit_map[next(i for i in ids if i.startswith("D"))]["text"]) -
                        word_count(unit_map[next(i for i in ids if i.startswith("N"))]["text"]))
            if delta > max_word_delta:
                errors.append(f"{prefix}{pair_id} word-count delta {delta} exceeds {max_word_delta}")

        forbidden = [
            *config["text_qc"]["neutral_forbidden_terms"],
            config["value_a"],
            config["value_b"],
        ]
        for unit in neutral:
            hits = sorted({term for term in forbidden if contains_term(str(unit["text"]), str(term))})
            if hits:
                errors.append(f"{prefix}{unit['unit_id']} forbidden neutral terms: {hits}")

        try:
            expected_state = {str(blueprint["root_slot"]): "{root_value}"}
            expected_state.update(dotted_state(unit["state_patch"] for unit in unit_map.values()))
            if blueprint.get("gold_state_template") != expected_state:
                errors.append(f"{prefix}gold_state_template does not equal merged state patches")
            for root_value in (config["value_a"], config["value_b"]):
                resolved = substitute(blueprint.get("gold_state_template"), {"{root_value}": root_value})
                if resolved.get(config["root_slot"]) != root_value:
                    errors.append(f"{prefix}gold_state_template fails root substitution for {root_value}")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{prefix}cannot construct gold_state_template: {error}")

        selected = str(blueprint.get("one_dependency_unit", ""))
        position = blueprint.get("one_dependency_pre_position")
        rotation = str(blueprint.get("rotation_id", ""))
        if selected not in DEPENDENT_IDS:
            errors.append(f"{prefix}invalid one_dependency_unit {selected!r}")
        if position not in (1, 2, 3):
            errors.append(f"{prefix}one_dependency_pre_position must be 1, 2, or 3")
        if rotation not in ("R1", "R2", "R3"):
            errors.append(f"{prefix}rotation_id must be R1, R2, or R3")
        if selected in DEPENDENT_IDS and position in (1, 2, 3):
            identity_counts[selected] += 1
            position_counts[int(position)] += 1
            cross_counts[(selected, int(position))] += 1
        rotation_counts[rotation] += 1

        reviews = blueprint.get("reviews")
        if not isinstance(reviews, list) or len(reviews) != required_reviews:
            errors.append(f"{prefix}requires exactly {required_reviews} reviews")
        else:
            reviewer_ids = [str(review.get("reviewer_id", "")) for review in reviews]
            if len(set(reviewer_ids)) != required_reviews or any(not item for item in reviewer_ids):
                errors.append(f"{prefix}reviews require distinct non-empty reviewer IDs")
            for review in reviews:
                if review.get("decision") != "approved":
                    errors.append(f"{prefix}all reviews must be approved")
                if not parse_datetime(str(review.get("reviewed_at", ""))):
                    errors.append(f"{prefix}reviewed_at must be an ISO date-time")

        source = blueprint.get("source")
        if not isinstance(source, dict) or not source.get("authoring_method") or not source.get("license"):
            errors.append(f"{prefix}source requires authoring_method and license")

        semantic_fingerprints.append(
            "|".join(str(unit_map[unit_id]["text"]).casefold() for unit_id in (*DEPENDENT_IDS, *NEUTRAL_IDS))
        )

    duplicate_semantics = list(iter_duplicates(semantic_fingerprints))
    if duplicate_semantics:
        errors.append(f"duplicate six-unit semantic designs: {len(duplicate_semantics)}")

    target_identity = int(config["counterbalance"]["target_per_identity"])
    target_position = int(config["counterbalance"]["target_per_position"])
    for identity in DEPENDENT_IDS:
        if identity_counts[identity] != target_identity:
            errors.append(f"{identity} counterbalance={identity_counts[identity]}, expected {target_identity}")
    for position in (1, 2, 3):
        if position_counts[position] != target_position:
            errors.append(f"pre-position {position} count={position_counts[position]}, expected {target_position}")
    cell_min = int(config["counterbalance"]["cross_cell_min"])
    cell_max = int(config["counterbalance"]["cross_cell_max"])
    for identity in DEPENDENT_IDS:
        for position in (1, 2, 3):
            count = cross_counts[(identity, position)]
            if count < cell_min or count > cell_max:
                errors.append(f"cross cell {identity}×{position}={count}, expected {cell_min}-{cell_max}")
    for rotation in ("R1", "R2", "R3"):
        if rotation_counts[rotation] != 10:
            errors.append(f"{rotation} count={rotation_counts[rotation]}, expected 10")
    for frame_id in config["text_qc"]["planning_frames"]:
        for binding in ("root_dependent", "root_invariant"):
            expected = int(config["text_qc"]["target_frame_count_per_binding"])
            count = frames_by_binding[(binding, frame_id)]
            if count != expected:
                errors.append(f"planning frame {frame_id}/{binding}={count}, expected {expected}")
        for unit_id in (*DEPENDENT_IDS, *NEUTRAL_IDS):
            expected = int(config["text_qc"]["target_frame_count_per_unit_identity"])
            count = frames_by_identity[(unit_id, frame_id)]
            if count != expected:
                errors.append(f"planning frame {frame_id}/{unit_id}={count}, expected {expected}")

    return errors


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    blueprints = read_jsonl(args.input)
    errors = validate_blueprints(blueprints, config)
    report = {
        "schema_version": "2.0.0",
        "input": portable_path(args.input),
        "input_sha256": sha256_file(args.input),
        "blueprint_count": len(blueprints),
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
    }
    if args.report:
        write_json(args.report, report)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated {len(blueprints)} blueprints ({report['input_sha256']})")


if __name__ == "__main__":
    main()
