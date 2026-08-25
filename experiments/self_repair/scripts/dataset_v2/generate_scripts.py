#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from common import (
    CONDITIONS,
    DEFAULT_BLUEPRINTS,
    DEFAULT_CONFIG,
    DEFAULT_SCRIPTS,
    DEPENDENT_IDS,
    NEUTRAL_IDS,
    dotted_state,
    normalized_text,
    portable_path,
    read_config,
    read_jsonl,
    sha256_value,
    substitute,
    units_by_id,
    write_json,
    write_jsonl,
)
from validate_blueprints import validate_blueprints


GENERATOR_VERSION = "2.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 300 matched self-repair scripts.")
    parser.add_argument("--blueprints", type=Path, default=DEFAULT_BLUEPRINTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def rotate(values: list[str], rotation_id: str) -> list[str]:
    offset = {"R1": 0, "R2": 1, "R3": 2}[rotation_id]
    return values[offset:] + values[:offset]


def delayed_one_orders(blueprint: dict[str, Any]) -> tuple[list[str], list[str]]:
    selected = str(blueprint["one_dependency_unit"])
    index = int(selected[-1])
    paired_neutral = f"N{index}"
    pre = [unit_id for unit_id in NEUTRAL_IDS if unit_id != paired_neutral]
    pre.insert(int(blueprint["one_dependency_pre_position"]) - 1, selected)
    post = [unit_id for unit_id in DEPENDENT_IDS if unit_id != selected] + [paired_neutral]
    return pre, rotate(post, str(blueprint["rotation_id"]))


def condition_orders(blueprint: dict[str, Any], condition: str) -> tuple[list[str], list[str]]:
    if condition in ("clean_final", "immediate_repair"):
        return [], [*NEUTRAL_IDS, *DEPENDENT_IDS]
    if condition == "delayed_neutral":
        return list(NEUTRAL_IDS), list(DEPENDENT_IDS)
    if condition == "delayed_one_dependency":
        return delayed_one_orders(blueprint)
    if condition == "delayed_three_dependencies":
        return list(DEPENDENT_IDS), list(NEUTRAL_IDS)
    raise ValueError(f"Unknown condition: {condition}")


def unit_patch(unit: dict[str, Any], root_value: str) -> dict[str, Any]:
    return substitute(unit["state_patch"], {"{root_value}": root_value})


def semantic_state(
    blueprint: dict[str, Any], unit_ids: Iterable[str], root_value: str
) -> dict[str, Any]:
    unit_map = units_by_id(blueprint)
    patches = [unit_patch(unit_map[unit_id], root_value) for unit_id in unit_ids]
    return dotted_state(patches)


def segment(
    index: int,
    role: str,
    text: str,
    *,
    unit: dict[str, Any] | None = None,
    terminal: bool = False,
) -> dict[str, Any]:
    return {
        "segment_index": index,
        "role": role,
        "unit_id": unit["unit_id"] if unit else None,
        "text": text,
        "binding": unit["binding"] if unit else None,
        "relation": unit["relation"] if unit else None,
        "boundary_after": "terminal" if terminal else "nonterminal_semicolon",
    }


def render_transcript(segments: list[dict[str, Any]]) -> str:
    return "; ".join(str(item["text"]).strip() for item in segments)


def build_script(
    blueprint: dict[str, Any],
    config: dict[str, Any],
    direction_id: str,
    condition: str,
) -> dict[str, Any]:
    if direction_id == "a_to_b":
        old_value, new_value = blueprint["value_a"], blueprint["value_b"]
    elif direction_id == "b_to_a":
        old_value, new_value = blueprint["value_b"], blueprint["value_a"]
    else:
        raise ValueError(f"Unknown direction: {direction_id}")

    text_bundle_id = f"{blueprint['scenario_id']}__{direction_id}"
    script_id = f"{text_bundle_id}__{condition}"
    pre_ids, post_ids = condition_orders(blueprint, condition)
    unit_map = units_by_id(blueprint)
    repair_cue = None if condition == "clean_final" else blueprint["repair_template"].format(
        new=new_value, old=old_value
    )
    root_value = new_value if condition == "clean_final" else old_value
    root_text = blueprint["root_template"].format(value=root_value)
    root_role = "clean_root" if condition == "clean_final" else "initial_old_root"

    raw_segments: list[tuple[str, str, dict[str, Any] | None]] = [(root_role, root_text, None)]
    if condition == "immediate_repair":
        raw_segments.append(("repair_cue", str(repair_cue), None))
    for unit_id in pre_ids:
        raw_segments.append(("semantic_unit", str(unit_map[unit_id]["text"]), unit_map[unit_id]))
    if condition not in ("clean_final", "immediate_repair"):
        raw_segments.append(("repair_cue", str(repair_cue), None))
    for unit_id in post_ids:
        raw_segments.append(("semantic_unit", str(unit_map[unit_id]["text"]), unit_map[unit_id]))
    raw_segments.append(("closing_prompt", str(blueprint["closing_prompt"]), None))

    segments = [
        segment(index, role, text, unit=unit, terminal=role == "closing_prompt")
        for index, (role, text, unit) in enumerate(raw_segments)
    ]
    transcript = render_transcript(segments)

    pre_state_root = new_value if condition == "clean_final" else old_value
    pre_state = {blueprint["root_slot"]: pre_state_root}
    pre_state.update(semantic_state(blueprint, pre_ids, pre_state_root))

    pre_dependent = [unit_id for unit_id in pre_ids if unit_id.startswith("D")]
    rebindings = [
        {
            "unit_id": unit_id,
            "relation": unit_map[unit_id]["relation"],
            "from": old_value,
            "to": new_value,
        }
        for unit_id in pre_dependent
    ]

    gold_state = {blueprint["root_slot"]: new_value}
    gold_state.update(
        semantic_state(blueprint, [*DEPENDENT_IDS, *NEUTRAL_IDS], new_value)
    )

    return {
        "schema_version": "2.0.0",
        "text_bundle_id": text_bundle_id,
        "script_id": script_id,
        "scenario_id": blueprint["scenario_id"],
        "direction_id": direction_id,
        "condition": condition,
        "root_slot": blueprint["root_slot"],
        "old_value": old_value,
        "new_value": new_value,
        "segments": segments,
        "pre_repair_units": pre_ids,
        "post_repair_units": post_ids,
        "dependency_count": len(pre_dependent),
        "repair_cue": repair_cue,
        "closing_prompt": blueprint["closing_prompt"],
        "transcript": transcript,
        "normalized_transcript": normalized_text(transcript),
        "pre_repair_state": pre_state,
        "repair_rebindings": rebindings,
        "gold_state": gold_state,
        "one_dependency_unit": (
            blueprint["one_dependency_unit"] if condition == "delayed_one_dependency" else None
        ),
        "one_dependency_pre_position": (
            blueprint["one_dependency_pre_position"]
            if condition == "delayed_one_dependency"
            else None
        ),
        "rotation_id": (
            blueprint["rotation_id"] if condition == "delayed_one_dependency" else None
        ),
        "blueprint_hash": sha256_value(blueprint),
        "generator_version": GENERATOR_VERSION,
        "config_hash": sha256_value(config),
        "generation_seed": int(config["generation_seed"]),
    }


def generate_all(
    blueprints: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    scripts: list[dict[str, Any]] = []
    for blueprint in sorted(blueprints, key=lambda item: item["scenario_id"]):
        for direction_id in ("a_to_b", "b_to_a"):
            for condition in CONDITIONS:
                scripts.append(build_script(blueprint, config, direction_id, condition))
    return scripts


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    blueprints = read_jsonl(args.blueprints)
    blueprint_errors = validate_blueprints(blueprints, config)
    if blueprint_errors:
        raise SystemExit("Blueprint validation failed:\n" + "\n".join(blueprint_errors))
    scripts = generate_all(blueprints, config)
    write_jsonl(args.output, scripts)
    report = {
        "schema_version": "2.0.0",
        "blueprint_count": len(blueprints),
        "text_bundle_count": len({item["text_bundle_id"] for item in scripts}),
        "script_count": len(scripts),
        "output": portable_path(args.output),
        "output_hash": sha256_value(scripts),
        "generator_version": GENERATOR_VERSION,
        "config_hash": sha256_value(config),
        "generation_seed": config["generation_seed"],
    }
    if args.report:
        write_json(args.report, report)
    print(
        f"Generated {report['script_count']} scripts in "
        f"{report['text_bundle_count']} text bundles -> {args.output}"
    )


if __name__ == "__main__":
    main()
