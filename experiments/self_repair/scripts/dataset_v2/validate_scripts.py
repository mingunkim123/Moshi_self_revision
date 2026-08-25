#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

from common import (
    CONDITIONS,
    DEFAULT_BLUEPRINTS,
    DEFAULT_CONFIG,
    DEFAULT_SCRIPTS,
    DEPENDENT_IDS,
    NEUTRAL_IDS,
    normalized_text,
    portable_path,
    read_config,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated dataset v2 scripts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--blueprints", type=Path, default=DEFAULT_BLUEPRINTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def exact_term_count(text: str, term: str) -> int:
    words = normalized_text(text).split()
    needle = normalized_text(term).split()
    if not needle:
        return 0
    return sum(words[index : index + len(needle)] == needle for index in range(len(words) - len(needle) + 1))


def validate_scripts(
    scripts: list[dict[str, Any]],
    blueprints: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_scripts = int(config["counts"]["scripts"])
    expected_bundles = int(config["counts"]["text_bundles"])
    if len(scripts) != expected_scripts:
        errors.append(f"expected {expected_scripts} scripts, found {len(scripts)}")

    ids = [str(item.get("script_id", "")) for item in scripts]
    if len(set(ids)) != len(ids):
        errors.append("script IDs are not unique")

    blueprint_map = {item["scenario_id"]: item for item in blueprints}
    by_bundle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    condition_counts: Counter[str] = Counter()
    scenario_directions: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for script in scripts:
        script_id = str(script.get("script_id", "<missing>"))
        prefix = f"{script_id}: "
        by_bundle[str(script.get("text_bundle_id", ""))].append(script)
        condition = str(script.get("condition", ""))
        condition_counts[condition] += 1
        blueprint = blueprint_map.get(script.get("scenario_id"))
        if blueprint is None:
            errors.append(f"{prefix}unknown scenario_id")
            continue
        if script.get("blueprint_hash") != sha256_value(blueprint):
            errors.append(f"{prefix}blueprint_hash mismatch")
        if script.get("config_hash") != sha256_value(config):
            errors.append(f"{prefix}config_hash mismatch")

        expected_id = f"{script['scenario_id']}__{script['direction_id']}__{condition}"
        if script_id != expected_id:
            errors.append(f"{prefix}script_id does not match fields")
        if script.get("text_bundle_id") != f"{script['scenario_id']}__{script['direction_id']}":
            errors.append(f"{prefix}text_bundle_id does not match fields")
        scenario_directions[str(script["scenario_id"])].add(
            (str(script["old_value"]), str(script["new_value"]))
        )

        segments = script.get("segments")
        if not isinstance(segments, list) or not segments:
            errors.append(f"{prefix}segments must be a non-empty list")
            continue
        if [item.get("segment_index") for item in segments] != list(range(len(segments))):
            errors.append(f"{prefix}segment indexes are not contiguous")
        if segments[-1].get("role") != "closing_prompt" or segments[-1].get("text") != script.get("closing_prompt"):
            errors.append(f"{prefix}closing prompt must be the final segment")
        if sum(item.get("role") == "closing_prompt" for item in segments) != 1:
            errors.append(f"{prefix}requires exactly one closing prompt")
        rendered = "; ".join(str(item.get("text", "")).strip() for item in segments)
        if rendered != script.get("transcript"):
            errors.append(f"{prefix}transcript is not the canonical segment join")
        if normalized_text(rendered) != script.get("normalized_transcript"):
            errors.append(f"{prefix}normalized_transcript mismatch")

        semantic_ids = [item.get("unit_id") for item in segments if item.get("role") == "semantic_unit"]
        expected_unit_set = set((*DEPENDENT_IDS, *NEUTRAL_IDS))
        if len(semantic_ids) != 6 or set(semantic_ids) != expected_unit_set:
            errors.append(f"{prefix}must contain D1-D3 and N1-N3 exactly once")

        pre = list(script.get("pre_repair_units", []))
        post = list(script.get("post_repair_units", []))
        if set(pre + post) != expected_unit_set or len(pre + post) != 6:
            errors.append(f"{prefix}pre/post unit arrays do not partition the six units")
        if condition.startswith("delayed_") and (len(pre), len(post)) != (3, 3):
            errors.append(f"{prefix}delayed conditions require 3 pre and 3 post units")
        if condition in ("clean_final", "immediate_repair") and (len(pre), len(post)) != (0, 6):
            errors.append(f"{prefix}clean/immediate require 0 pre and 6 post units")

        expected_dependency = {
            "clean_final": 0,
            "immediate_repair": 0,
            "delayed_neutral": 0,
            "delayed_one_dependency": 1,
            "delayed_three_dependencies": 3,
        }.get(condition)
        if script.get("dependency_count") != expected_dependency:
            errors.append(f"{prefix}dependency_count must be {expected_dependency}")
        rebindings = script.get("repair_rebindings")
        if not isinstance(rebindings, list) or len(rebindings) != expected_dependency:
            errors.append(f"{prefix}repair_rebindings must match dependency_count")
        else:
            for rebinding in rebindings:
                if rebinding.get("from") != script["old_value"] or rebinding.get("to") != script["new_value"]:
                    errors.append(f"{prefix}invalid rebinding values")

        repair_segments = [item for item in segments if item.get("role") == "repair_cue"]
        old_value, new_value = str(script["old_value"]), str(script["new_value"])
        if condition == "clean_final":
            if repair_segments or script.get("repair_cue") is not None:
                errors.append(f"{prefix}clean condition must not contain repair")
            if exact_term_count(rendered, old_value) != 0 or exact_term_count(rendered, new_value) != 1:
                errors.append(f"{prefix}clean value occurrence mismatch")
            if segments[0].get("role") != "clean_root":
                errors.append(f"{prefix}clean root role mismatch")
        else:
            if len(repair_segments) != 1 or repair_segments[0].get("text") != script.get("repair_cue"):
                errors.append(f"{prefix}requires exactly one canonical repair cue")
            if exact_term_count(rendered, old_value) != 2 or exact_term_count(rendered, new_value) != 1:
                errors.append(f"{prefix}repair value occurrence mismatch")
            if segments[0].get("role") != "initial_old_root":
                errors.append(f"{prefix}initial old root role mismatch")

        if script.get("gold_state", {}).get(script["root_slot"]) != new_value:
            errors.append(f"{prefix}gold root must equal new_value")

    if len(by_bundle) != expected_bundles:
        errors.append(f"expected {expected_bundles} text bundles, found {len(by_bundle)}")
    for bundle_id, items in by_bundle.items():
        conditions = {item["condition"] for item in items}
        if len(items) != 5 or conditions != set(CONDITIONS):
            errors.append(f"{bundle_id}: must contain all five conditions exactly once")
        gold_hashes = {sha256_value(item["gold_state"]) for item in items}
        if len(gold_hashes) != 1:
            errors.append(f"{bundle_id}: gold_state differs across conditions")
        repair_cues = {item["repair_cue"] for item in items if item["condition"] != "clean_final"}
        if len(repair_cues) != 1:
            errors.append(f"{bundle_id}: repair cue differs across repair conditions")

    for condition in CONDITIONS:
        if condition_counts[condition] != 60:
            errors.append(f"{condition} count={condition_counts[condition]}, expected 60")
    expected_pairs = {
        (str(config["value_a"]), str(config["value_b"])),
        (str(config["value_b"]), str(config["value_a"])),
    }
    for scenario_id, pairs in scenario_directions.items():
        if pairs != expected_pairs:
            errors.append(f"{scenario_id}: direction value pairs are not exact reversals")
    return errors


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    scripts = read_jsonl(args.input)
    blueprints = read_jsonl(args.blueprints)
    errors = validate_scripts(scripts, blueprints, config)
    report = {
        "schema_version": "2.0.0",
        "input": portable_path(args.input),
        "input_sha256": sha256_file(args.input),
        "script_count": len(scripts),
        "text_bundle_count": len({item.get("text_bundle_id") for item in scripts}),
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
    print(f"Validated {len(scripts)} scripts ({report['input_sha256']})")


if __name__ == "__main__":
    main()
