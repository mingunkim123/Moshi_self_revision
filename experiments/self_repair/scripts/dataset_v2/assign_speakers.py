#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import random
from typing import Any, Iterable

from common import (
    CONDITIONS,
    DATASET_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_SCRIPTS,
    iter_duplicates,
    read_config,
    read_jsonl,
    portable_path,
    sha256_value,
    write_json,
    write_jsonl,
)
from ids import matched_audio_bundle_id, rendition_target_id, safe_component


ASSIGNMENT_VERSION = "2.0.0"
DIRECTIONS = ("a_to_b", "b_to_a")
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "assignments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic folds, speaker assignments, and recording order."
    )
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source-track",
        help="Source track in config.source_tracks; defaults to the only configured track.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--report",
        type=Path,
        help="Report path; defaults to <output-dir>/assignment_report.json.",
    )
    return parser.parse_args()


def _rng(seed: int, namespace: str) -> random.Random:
    material = f"{ASSIGNMENT_VERSION}\0{seed}\0{namespace}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(derived_seed)


def _source_track(
    config: dict[str, Any], requested_source_track: str | None
) -> tuple[str, list[dict[str, Any]]]:
    tracks = config.get("source_tracks")
    if not isinstance(tracks, dict) or not tracks:
        raise ValueError("config.source_tracks must be a non-empty object")
    if requested_source_track is None:
        if len(tracks) != 1:
            raise ValueError("--source-track is required when more than one track is configured")
        source_track_id = next(iter(tracks))
    else:
        source_track_id = requested_source_track
    safe_component(source_track_id, "source_track_id")
    if source_track_id not in tracks:
        raise ValueError(f"unknown source track: {source_track_id!r}")

    track = tracks[source_track_id]
    speakers = track.get("speakers") if isinstance(track, dict) else None
    if not isinstance(speakers, list) or not speakers:
        raise ValueError(f"source track {source_track_id!r} has no speakers")
    normalized: list[dict[str, Any]] = []
    for index, speaker in enumerate(speakers):
        if not isinstance(speaker, dict) or not isinstance(speaker.get("speaker_id"), str):
            raise ValueError(f"source track speaker {index} is missing speaker_id")
        safe_component(speaker["speaker_id"], "speaker_id")
        normalized.append(dict(speaker))
    duplicates = list(iter_duplicates(row["speaker_id"] for row in normalized))
    if duplicates:
        raise ValueError(f"duplicate speaker IDs: {duplicates}")
    declared_count = track.get("speaker_count")
    if declared_count is not None and int(declared_count) != len(normalized):
        raise ValueError(
            f"speaker_count={declared_count} but {len(normalized)} speakers are configured"
        )
    return source_track_id, normalized


def _validate_script_matrix(
    scripts: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    counts = config["counts"]
    expected_conditions = tuple(config.get("conditions", CONDITIONS))
    if expected_conditions != CONDITIONS:
        raise ValueError(
            f"config conditions must preserve the canonical order {CONDITIONS!r}"
        )
    expected_script_count = int(counts["scripts"])
    if len(scripts) != expected_script_count:
        raise ValueError(
            f"expected {expected_script_count} scripts, found {len(scripts)}"
        )

    required = {"script_id", "text_bundle_id", "scenario_id", "direction_id", "condition"}
    for index, script in enumerate(scripts):
        missing = sorted(required - script.keys())
        if missing:
            raise ValueError(f"script row {index} is missing fields: {missing}")
    duplicate_scripts = list(iter_duplicates(str(row["script_id"]) for row in scripts))
    if duplicate_scripts:
        raise ValueError(f"duplicate script IDs: {duplicate_scripts}")

    matrix: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    bundle_metadata: dict[str, tuple[str, str]] = {}
    for script in scripts:
        script_id = str(script["script_id"])
        bundle_id = str(script["text_bundle_id"])
        scenario_id = str(script["scenario_id"])
        direction_id = str(script["direction_id"])
        condition = str(script["condition"])
        safe_component(script_id, "script_id")
        safe_component(bundle_id, "text_bundle_id")
        safe_component(scenario_id, "scenario_id")
        if direction_id not in DIRECTIONS:
            raise ValueError(f"script {script_id}: invalid direction {direction_id!r}")
        if condition not in expected_conditions:
            raise ValueError(f"script {script_id}: invalid condition {condition!r}")
        if script_id != f"{bundle_id}__{condition}":
            raise ValueError(f"script {script_id}: ID does not match bundle and condition")
        if bundle_id != f"{scenario_id}__{direction_id}":
            raise ValueError(f"script {script_id}: text bundle ID does not match metadata")
        metadata = (scenario_id, direction_id)
        if bundle_id in bundle_metadata and bundle_metadata[bundle_id] != metadata:
            raise ValueError(f"text bundle {bundle_id}: inconsistent scenario/direction")
        bundle_metadata[bundle_id] = metadata
        if condition in matrix[bundle_id]:
            raise ValueError(f"text bundle {bundle_id}: duplicate condition {condition}")
        matrix[bundle_id][condition] = script

    expected_bundles = int(counts["text_bundles"])
    if len(matrix) != expected_bundles:
        raise ValueError(f"expected {expected_bundles} text bundles, found {len(matrix)}")
    condition_set = set(expected_conditions)
    for bundle_id, conditions in matrix.items():
        if set(conditions) != condition_set:
            missing = sorted(condition_set - set(conditions))
            extra = sorted(set(conditions) - condition_set)
            raise ValueError(
                f"text bundle {bundle_id}: condition mismatch; missing={missing}, extra={extra}"
            )

    scenario_directions: dict[str, set[str]] = defaultdict(set)
    for scenario_id, direction_id in bundle_metadata.values():
        scenario_directions[scenario_id].add(direction_id)
    expected_scenarios = int(counts["scenarios"])
    if len(scenario_directions) != expected_scenarios:
        raise ValueError(f"expected {expected_scenarios} scenarios, found {len(scenario_directions)}")
    for scenario_id, directions in scenario_directions.items():
        if directions != set(DIRECTIONS):
            raise ValueError(f"scenario {scenario_id}: directions are {sorted(directions)}")
    return dict(matrix)


def _analysis_folds(
    scenario_ids: Iterable[str], config: dict[str, Any], seed: int
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    scenarios = sorted(set(scenario_ids))
    fold_count = int(config["release"]["analysis_folds"])
    if fold_count != 5 or len(scenarios) != 30:
        raise ValueError(
            "v2 assignment requires exactly 30 scenarios and 5 analysis folds"
        )
    shuffled = list(scenarios)
    _rng(seed, "analysis-folds").shuffle(shuffled)
    fold_by_scenario = {
        scenario_id: index % fold_count + 1
        for index, scenario_id in enumerate(shuffled)
    }
    rows = [
        {
            "schema_version": "2.0.0",
            "assignment_version": ASSIGNMENT_VERSION,
            "scenario_id": scenario_id,
            "analysis_fold": fold_by_scenario[scenario_id],
            "inferential_role": "confirmatory_evaluation",
            "generation_seed": seed,
        }
        for scenario_id in sorted(scenarios)
    ]
    return fold_by_scenario, rows


def _pair_slots(
    speaker_ids: list[str], extra_speakers: tuple[str, str], rng: random.Random
) -> list[tuple[str, str]]:
    slots = [*speaker_ids, *extra_speakers]
    for _ in range(512):
        rng.shuffle(slots)
        pairs = list(zip(slots[::2], slots[1::2]))
        if all(first != second for first, second in pairs):
            return pairs
    raise RuntimeError("could not create distinct-speaker pairs")


def _speaker_assignments(
    matrix: dict[str, dict[str, dict[str, Any]]],
    fold_by_scenario: dict[str, int],
    speakers: list[dict[str, Any]],
    source_track_id: str,
    seed: int,
) -> list[dict[str, Any]]:
    speaker_ids = sorted(str(row["speaker_id"]) for row in speakers)
    if len(speaker_ids) != 10:
        raise ValueError("v2 assignment requires exactly 10 speakers")
    speaker_cycle = list(speaker_ids)
    _rng(seed, f"{source_track_id}:speaker-cycle").shuffle(speaker_cycle)
    profile_by_id = {str(row["speaker_id"]): row for row in speakers}

    bundles_by_cell: dict[tuple[int, str], list[str]] = defaultdict(list)
    for bundle_id, conditions in matrix.items():
        representative = next(iter(conditions.values()))
        scenario_id = str(representative["scenario_id"])
        direction_id = str(representative["direction_id"])
        bundles_by_cell[(fold_by_scenario[scenario_id], direction_id)].append(bundle_id)

    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_index = fold - 1
        extras_by_direction = {
            "a_to_b": (
                speaker_cycle[(2 * fold_index) % 10],
                speaker_cycle[(2 * fold_index + 1) % 10],
            ),
            "b_to_a": (
                speaker_cycle[(2 * fold_index + 2) % 10],
                speaker_cycle[(2 * fold_index + 3) % 10],
            ),
        }
        for direction_id in DIRECTIONS:
            bundles = sorted(bundles_by_cell[(fold, direction_id)])
            if len(bundles) != 6:
                raise ValueError(
                    f"fold {fold}/{direction_id}: expected 6 text bundles, found {len(bundles)}"
                )
            cell_rng = _rng(seed, f"{source_track_id}:fold-{fold}:{direction_id}")
            cell_rng.shuffle(bundles)
            pairs = _pair_slots(
                speaker_cycle, extras_by_direction[direction_id], cell_rng
            )
            for bundle_id, pair in zip(bundles, pairs):
                representative = next(iter(matrix[bundle_id].values()))
                for speaker_id in pair:
                    profile = profile_by_id[speaker_id]
                    matched_id = matched_audio_bundle_id(
                        bundle_id, source_track_id, speaker_id
                    )
                    script_ids = [
                        str(matrix[bundle_id][condition]["script_id"])
                        for condition in CONDITIONS
                    ]
                    target_ids = [
                        rendition_target_id(script_id, source_track_id, speaker_id)
                        for script_id in script_ids
                    ]
                    rows.append(
                        {
                            "schema_version": "2.0.0",
                            "assignment_version": ASSIGNMENT_VERSION,
                            "matched_audio_bundle_id": matched_id,
                            "text_bundle_id": bundle_id,
                            "scenario_id": str(representative["scenario_id"]),
                            "direction_id": direction_id,
                            "source_track_id": source_track_id,
                            "speaker_id": speaker_id,
                            "voice": profile.get("voice"),
                            "analysis_fold": fold,
                            "inferential_role": "confirmatory_evaluation",
                            "condition_count": len(CONDITIONS),
                            "script_ids": script_ids,
                            "rendition_target_ids": target_ids,
                            "generation_seed": seed,
                        }
                    )
    return sorted(rows, key=lambda row: row["matched_audio_bundle_id"])


def _rendition_targets(
    assignments: list[dict[str, Any]],
    matrix: dict[str, dict[str, dict[str, Any]]],
    source_track_id: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        bundle_id = str(assignment["text_bundle_id"])
        speaker_id = str(assignment["speaker_id"])
        for condition in CONDITIONS:
            script = matrix[bundle_id][condition]
            script_id = str(script["script_id"])
            rows.append(
                {
                    "schema_version": "2.0.0",
                    "assignment_version": ASSIGNMENT_VERSION,
                    "rendition_target_id": rendition_target_id(
                        script_id, source_track_id, speaker_id
                    ),
                    "script_id": script_id,
                    "text_bundle_id": bundle_id,
                    "matched_audio_bundle_id": assignment["matched_audio_bundle_id"],
                    "scenario_id": assignment["scenario_id"],
                    "direction_id": assignment["direction_id"],
                    "condition": condition,
                    "source_track_id": source_track_id,
                    "speaker_id": speaker_id,
                    "voice": assignment.get("voice"),
                    "analysis_fold": assignment["analysis_fold"],
                    "inferential_role": "confirmatory_evaluation",
                    "generation_seed": seed,
                }
            )
    return sorted(rows, key=lambda row: row["rendition_target_id"])


def _recording_order(
    targets: list[dict[str, Any]], source_track_id: str, seed: int
) -> list[dict[str, Any]]:
    by_speaker_bundle: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for target in targets:
        by_speaker_bundle[str(target["speaker_id"])][str(target["text_bundle_id"])][
            str(target["condition"])
        ] = target

    rows: list[dict[str, Any]] = []
    for speaker_id in sorted(by_speaker_bundle):
        bundle_targets = by_speaker_bundle[speaker_id]
        condition_order: dict[str, list[str]] = {}
        for bundle_id in sorted(bundle_targets):
            conditions = list(CONDITIONS)
            _rng(seed, f"{source_track_id}:{speaker_id}:{bundle_id}:conditions").shuffle(
                conditions
            )
            condition_order[bundle_id] = conditions

        previous_bundle: str | None = None
        position = 0
        for round_index in range(len(CONDITIONS)):
            bundles = sorted(bundle_targets)
            _rng(
                seed,
                f"{source_track_id}:{speaker_id}:recording-round-{round_index + 1}",
            ).shuffle(bundles)
            if previous_bundle is not None and bundles[0] == previous_bundle:
                swap_index = next(
                    index for index, bundle_id in enumerate(bundles[1:], 1)
                    if bundle_id != previous_bundle
                )
                bundles[0], bundles[swap_index] = bundles[swap_index], bundles[0]
            for bundle_id in bundles:
                condition = condition_order[bundle_id][round_index]
                target = bundle_targets[bundle_id][condition]
                position += 1
                rows.append(
                    {
                        "schema_version": "2.0.0",
                        "assignment_version": ASSIGNMENT_VERSION,
                        "recording_order_id": (
                            f"{source_track_id}__{speaker_id}__position_{position:03d}"
                        ),
                        "recording_position": position,
                        "rendition_target_id": target["rendition_target_id"],
                        "script_id": target["script_id"],
                        "text_bundle_id": bundle_id,
                        "matched_audio_bundle_id": target["matched_audio_bundle_id"],
                        "scenario_id": target["scenario_id"],
                        "direction_id": target["direction_id"],
                        "condition": condition,
                        "source_track_id": source_track_id,
                        "speaker_id": speaker_id,
                        "voice": target.get("voice"),
                        "analysis_fold": target["analysis_fold"],
                        "generation_seed": seed,
                    }
                )
                previous_bundle = bundle_id
    return rows


def validate_manifests(
    manifests: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    source_track_id: str,
) -> list[str]:
    errors: list[str] = []
    counts = config["counts"]
    assignments = manifests["speaker_bundles"]
    targets = manifests["rendition_targets"]
    folds = manifests["analysis_folds"]
    recording = manifests["recording_order"]

    configured_tracks = config.get("source_tracks")
    configured_speakers: dict[str, dict[str, Any]] = {}
    if not isinstance(configured_tracks, dict) or source_track_id not in configured_tracks:
        errors.append(f"source track {source_track_id!r} is not configured")
    else:
        raw_speakers = configured_tracks[source_track_id].get("speakers", [])
        if isinstance(raw_speakers, list):
            configured_speakers = {
                str(row.get("speaker_id")): row
                for row in raw_speakers
                if isinstance(row, dict) and row.get("speaker_id")
            }

    expected_assignment_count = int(counts["matched_audio_bundles_per_track"])
    expected_target_count = int(counts["rendition_targets_per_track"])
    if len(assignments) != expected_assignment_count:
        errors.append(
            f"matched audio bundles: expected {expected_assignment_count}, found {len(assignments)}"
        )
    if len(targets) != expected_target_count:
        errors.append(f"rendition targets: expected {expected_target_count}, found {len(targets)}")
    if len(recording) != expected_target_count:
        errors.append(f"recording rows: expected {expected_target_count}, found {len(recording)}")

    id_specs = (
        (assignments, "matched_audio_bundle_id"),
        (targets, "rendition_target_id"),
        (recording, "recording_order_id"),
    )
    for rows, field in id_specs:
        duplicates = list(iter_duplicates(str(row[field]) for row in rows))
        if duplicates:
            errors.append(f"duplicate {field}: {duplicates}")

    assignments_by_bundle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    assignments_by_speaker = Counter()
    assignments_by_id: dict[str, dict[str, Any]] = {}
    for row in assignments:
        matched_id = str(row["matched_audio_bundle_id"])
        assignments_by_id[matched_id] = row
        assignments_by_bundle[str(row["text_bundle_id"])].append(row)
        assignments_by_speaker[str(row["speaker_id"])] += 1
        if row["source_track_id"] != source_track_id:
            errors.append(f"assignment {row['matched_audio_bundle_id']}: wrong source track")
        expected_id = matched_audio_bundle_id(
            str(row["text_bundle_id"]), source_track_id, str(row["speaker_id"])
        )
        if row["matched_audio_bundle_id"] != expected_id:
            errors.append(f"assignment {row['matched_audio_bundle_id']}: non-canonical ID")
        expected_scripts = {
            f"{row['text_bundle_id']}__{condition}" for condition in CONDITIONS
        }
        script_ids = row.get("script_ids")
        if not isinstance(script_ids, list) or set(map(str, script_ids)) != expected_scripts:
            errors.append(f"assignment {row['matched_audio_bundle_id']}: script IDs are not exact")
        expected_targets = {
            rendition_target_id(script_id, source_track_id, str(row["speaker_id"]))
            for script_id in expected_scripts
        }
        target_ids_declared = row.get("rendition_target_ids")
        if not isinstance(target_ids_declared, list) or set(map(str, target_ids_declared)) != expected_targets:
            errors.append(
                f"assignment {row['matched_audio_bundle_id']}: rendition target IDs are not exact"
            )
        profile = configured_speakers.get(str(row["speaker_id"]))
        if profile is None:
            errors.append(
                f"assignment {row['matched_audio_bundle_id']}: speaker is not configured"
            )
        elif row.get("voice") != profile.get("voice"):
            errors.append(
                f"assignment {row['matched_audio_bundle_id']}: voice does not match speaker config"
            )
    for bundle_id, rows in assignments_by_bundle.items():
        speakers = {str(row["speaker_id"]) for row in rows}
        if len(rows) != 2 or len(speakers) != 2:
            errors.append(f"text bundle {bundle_id}: expected two distinct speakers")
    expected_per_speaker = expected_assignment_count // len(assignments_by_speaker or {"": 1})
    if set(assignments_by_speaker) != set(configured_speakers):
        errors.append(
            "assignment speaker set does not exactly match source-track config: "
            f"observed={sorted(assignments_by_speaker)}, configured={sorted(configured_speakers)}"
        )
    if set(assignments_by_speaker.values()) != {expected_per_speaker}:
        errors.append(f"speaker assignment imbalance: {dict(sorted(assignments_by_speaker.items()))}")

    targets_by_assignment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_ids: set[str] = set()
    for row in targets:
        target_id = str(row["rendition_target_id"])
        target_ids.add(target_id)
        targets_by_assignment[str(row["matched_audio_bundle_id"])].append(row)
        expected_id = rendition_target_id(
            str(row["script_id"]), source_track_id, str(row["speaker_id"])
        )
        if target_id != expected_id:
            errors.append(f"target {target_id}: non-canonical ID")
        assignment = assignments_by_id.get(str(row["matched_audio_bundle_id"]))
        if assignment is None:
            errors.append(f"target {target_id}: unknown matched audio bundle")
            continue
        expected_matched_id = matched_audio_bundle_id(
            str(row["text_bundle_id"]), source_track_id, str(row["speaker_id"])
        )
        if row["matched_audio_bundle_id"] != expected_matched_id:
            errors.append(f"target {target_id}: matched audio bundle ID is inconsistent")
        condition = str(row.get("condition", ""))
        if row.get("script_id") != f"{row.get('text_bundle_id')}__{condition}":
            errors.append(f"target {target_id}: script/text-bundle/condition join is inconsistent")
        for field in (
            "text_bundle_id",
            "scenario_id",
            "direction_id",
            "source_track_id",
            "speaker_id",
            "voice",
            "analysis_fold",
            "inferential_role",
        ):
            if row.get(field) != assignment.get(field):
                errors.append(
                    f"target {target_id}: {field} does not match matched audio bundle"
                )
    for matched_id, rows in targets_by_assignment.items():
        conditions = {str(row["condition"]) for row in rows}
        if len(rows) != len(CONDITIONS) or conditions != set(CONDITIONS):
            errors.append(f"matched audio bundle {matched_id}: condition set mismatch")
        assignment = assignments_by_id.get(matched_id)
        if assignment is not None:
            observed_target_ids = {str(row["rendition_target_id"]) for row in rows}
            if observed_target_ids != set(map(str, assignment.get("rendition_target_ids", []))):
                errors.append(f"matched audio bundle {matched_id}: target ID list mismatch")
    if set(targets_by_assignment) != set(assignments_by_id):
        errors.append("rendition targets do not exactly cover matched audio bundles")

    fold_counts = Counter(int(row["analysis_fold"]) for row in folds)
    if fold_counts != Counter({fold: 6 for fold in range(1, 6)}):
        errors.append(f"analysis fold scenario counts are not 6 each: {dict(fold_counts)}")
    fold_by_scenario = {str(row["scenario_id"]): int(row["analysis_fold"]) for row in folds}
    for row in assignments:
        if fold_by_scenario.get(str(row["scenario_id"])) != int(row["analysis_fold"]):
            errors.append(f"assignment {row['matched_audio_bundle_id']}: scenario fold mismatch")

    recording_ids = [str(row["rendition_target_id"]) for row in recording]
    if set(recording_ids) != target_ids or len(recording_ids) != len(target_ids):
        errors.append("recording order is not a one-to-one permutation of rendition targets")
    recording_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_by_id = {str(row["rendition_target_id"]): row for row in targets}
    for row in recording:
        target_id = str(row["rendition_target_id"])
        recording_by_speaker[str(row["speaker_id"])].append(row)
        target = target_by_id.get(target_id)
        if target is None:
            continue
        for field in (
            "script_id",
            "text_bundle_id",
            "matched_audio_bundle_id",
            "scenario_id",
            "direction_id",
            "condition",
            "source_track_id",
            "speaker_id",
            "voice",
            "analysis_fold",
        ):
            if row.get(field) != target.get(field):
                errors.append(f"recording row {row.get('recording_order_id')}: {field} mismatch")
        expected_recording_id = (
            f"{source_track_id}__{row['speaker_id']}__position_"
            f"{int(row['recording_position']):03d}"
        )
        if row.get("recording_order_id") != expected_recording_id:
            errors.append(f"recording row {row.get('recording_order_id')}: non-canonical ID")
    expected_recordings_per_speaker = expected_target_count // len(recording_by_speaker or {"": 1})
    for speaker_id, rows in recording_by_speaker.items():
        ordered = sorted(rows, key=lambda row: int(row["recording_position"]))
        positions = [int(row["recording_position"]) for row in ordered]
        if positions != list(range(1, expected_recordings_per_speaker + 1)):
            errors.append(f"speaker {speaker_id}: recording positions are not contiguous")
        for first, second in zip(ordered, ordered[1:]):
            if first["text_bundle_id"] == second["text_bundle_id"]:
                errors.append(
                    f"speaker {speaker_id}: adjacent text bundle at positions "
                    f"{first['recording_position']}/{second['recording_position']}"
                )
    return errors


def build_manifests(
    scripts: list[dict[str, Any]],
    config: dict[str, Any],
    source_track_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    seed = int(config["generation_seed"])
    resolved_source_track, speakers = _source_track(config, source_track_id)
    matrix = _validate_script_matrix(scripts, config)
    scenario_ids = {
        str(next(iter(conditions.values()))["scenario_id"])
        for conditions in matrix.values()
    }
    fold_by_scenario, folds = _analysis_folds(scenario_ids, config, seed)
    assignments = _speaker_assignments(
        matrix, fold_by_scenario, speakers, resolved_source_track, seed
    )
    targets = _rendition_targets(assignments, matrix, resolved_source_track, seed)
    recording = _recording_order(targets, resolved_source_track, seed)
    manifests = {
        "analysis_folds": folds,
        "speaker_bundles": assignments,
        "rendition_targets": targets,
        "recording_order": recording,
    }
    errors = validate_manifests(manifests, config, resolved_source_track)
    if errors:
        raise ValueError("assignment validation failed:\n" + "\n".join(errors))
    return manifests


def make_report(
    manifests: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    source_track_id: str,
    outputs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    assignments = manifests["speaker_bundles"]
    targets = manifests["rendition_targets"]
    folds = manifests["analysis_folds"]
    assignment_by_speaker = Counter(str(row["speaker_id"]) for row in assignments)
    target_by_speaker = Counter(str(row["speaker_id"]) for row in targets)
    direction_by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    fold_by_speaker: dict[str, Counter[int]] = defaultdict(Counter)
    for row in assignments:
        speaker_id = str(row["speaker_id"])
        direction_by_speaker[speaker_id][str(row["direction_id"])] += 1
        fold_by_speaker[speaker_id][int(row["analysis_fold"])] += 1
    return {
        "schema_version": "2.0.0",
        "assignment_version": ASSIGNMENT_VERSION,
        "source_track_id": source_track_id,
        "generation_seed": int(config["generation_seed"]),
        "validation": {"status": "passed", "errors": []},
        "counts": {
            "scenarios": len(folds),
            "analysis_folds": len({row["analysis_fold"] for row in folds}),
            "matched_audio_bundles": len(assignments),
            "rendition_targets": len(targets),
            "recording_order_rows": len(manifests["recording_order"]),
        },
        "distributions": {
            "matched_audio_bundles_by_speaker": dict(sorted(assignment_by_speaker.items())),
            "rendition_targets_by_speaker": dict(sorted(target_by_speaker.items())),
            "matched_audio_bundles_by_speaker_direction": {
                speaker: dict(sorted(values.items()))
                for speaker, values in sorted(direction_by_speaker.items())
            },
            "matched_audio_bundles_by_speaker_fold": {
                speaker: {str(fold): count for fold, count in sorted(values.items())}
                for speaker, values in sorted(fold_by_speaker.items())
            },
            "scenarios_by_analysis_fold": dict(
                sorted(Counter(str(row["analysis_fold"]) for row in folds).items())
            ),
        },
        "manifest_hashes": {
            name: sha256_value(rows) for name, rows in sorted(manifests.items())
        },
        "outputs": (
            {name: portable_path(path) for name, path in sorted(outputs.items())}
            if outputs is not None
            else None
        ),
    }


def write_manifests(
    output_dir: Path,
    report_path: Path,
    manifests: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    source_track_id: str,
) -> dict[str, Path]:
    outputs = {
        "analysis_folds": output_dir / "analysis_folds.jsonl",
        "speaker_bundles": output_dir / "speaker_bundles.jsonl",
        "rendition_targets": output_dir / "rendition_targets.jsonl",
        "recording_order": output_dir / "recording_order.jsonl",
    }
    for name, path in outputs.items():
        write_jsonl(path, manifests[name])
    report = make_report(manifests, config, source_track_id, outputs)
    write_json(report_path, report)
    return {**outputs, "report": report_path}


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    scripts = read_jsonl(args.scripts)
    source_track_id, _ = _source_track(config, args.source_track)
    manifests = build_manifests(scripts, config, source_track_id)
    report_path = args.report or args.output_dir / "assignment_report.json"
    outputs = write_manifests(
        args.output_dir, report_path, manifests, config, source_track_id
    )
    print(
        f"Assigned {len(manifests['speaker_bundles'])} matched audio bundles and "
        f"{len(manifests['rendition_targets'])} rendition targets -> {args.output_dir}"
    )
    print(f"Validation report -> {outputs['report']}")


if __name__ == "__main__":
    main()
