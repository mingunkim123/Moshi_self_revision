#!/usr/bin/env python3
"""Create and ingest condition-blind v2 annotation sheets.

Public sheets contain no eval-trial, accepted-audio, condition, or generation-seed
identifier.  The private blind map is therefore an access-controlled artifact and
must never be distributed with an annotator's sheet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

try:
    from .common import DATASET_ROOT, canonical_json, read_jsonl, write_csv, write_jsonl
except ImportError:  # pragma: no cover - exercised by direct CLI use.
    from common import DATASET_ROOT, canonical_json, read_jsonl, write_csv, write_jsonl


SCHEMA_VERSION = "2.0.0"
OVERALL_LABELS = {
    "target_only",
    "stale_only",
    "both",
    "recovered",
    "clarification",
    "irrelevant",
    "no_speech",
    "unintelligible",
    "no_evidence",
}
RELATION_LABELS = {
    "new_bound",
    "old_bound",
    "both",
    "unresolved",
    "not_addressed",
}
RELATIONS = ("D1", "D2", "D3")
CONDITION_NAMES = {
    "clean_final",
    "immediate_repair",
    "delayed_neutral",
    "delayed_one_dependency",
    "delayed_three_dependencies",
}
DEFAULT_ACCEPTED_MANIFEST = DATASET_ROOT / "manifests/accepted_audio.jsonl"
DEFAULT_ANSWER_KEYS = DATASET_ROOT / "answer_keys/answer_keys.jsonl"
PUBLIC_FIELDS = (
    "annotation_order",
    "blind_id",
    "annotator_id",
    "context_label",
    "target_value",
    "stale_value",
    "D1_relation_planning_constraint",
    "D2_relation_planning_constraint",
    "D3_relation_planning_constraint",
    "root_invariant_constraints",
    "safety_note",
    "response_text",
    "response_audio_file",
    "overall_label",
    "relation_D1",
    "relation_D2",
    "relation_D3",
    "final_target_correct",
    "stale_state_error",
    "assistant_started_before_repair",
    "notes",
)
FORBIDDEN_PUBLIC_FIELDS = {
    "eval_run_id",
    "eval_trial_id",
    "accepted_audio_id",
    "rendition_target_id",
    "text_bundle_id",
    "matched_audio_bundle_id",
    "scenario_id",
    "direction_id",
    "condition",
    "generation_seed",
    "answer_key_id",
}
DECISION_FIELDS = (
    "overall_label",
    "relation_labels",
    "final_target_correct",
    "stale_state_error",
    "assistant_started_before_repair",
)


def _require_two_annotators(annotator_ids: Sequence[str]) -> tuple[str, str]:
    normalized = tuple(annotator_ids)
    if len(normalized) != 2 or len(set(normalized)) != 2:
        raise ValueError("exactly two distinct primary annotator IDs are required")
    for value in normalized:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("annotator IDs must be non-empty strings")
    return normalized[0], normalized[1]


def _hash_order(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()


def _blind_id(eval_run_id: str, eval_trial_id: str, shuffle_seed: int) -> str:
    material = f"v2-blind\0{shuffle_seed}\0{eval_run_id}\0{eval_trial_id}"
    return "blind_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _response_text(response: Mapping[str, Any]) -> str:
    for field in ("transcript", "text"):
        value = response.get(field)
        if isinstance(value, str):
            return value
    return ""


def _audio_source(response: Mapping[str, Any]) -> Path | None:
    value = response.get("audio_path")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("response.audio_path must be a non-empty local path")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        raise ValueError("response.audio_path must be local, not a URI")
    return Path(value)


def _required_text(row: Mapping[str, Any], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} has an invalid {field}")
    return value


def _accepted_join_index(
    rows: Sequence[dict[str, Any]], expected_count: int
) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_count:
        raise ValueError(
            f"expected {expected_count} accepted audio rows, found {len(rows)}"
        )
    indexed: dict[str, dict[str, Any]] = {}
    required = {
        "accepted_audio_id",
        "text_bundle_id",
        "scenario_id",
        "direction_id",
        "condition",
        "source_track_id",
        "speaker_id",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"accepted audio row {index} is missing fields: {missing}")
        accepted_id = _required_text(row, "accepted_audio_id", f"accepted audio row {index}")
        if accepted_id in indexed:
            raise ValueError(f"duplicate accepted_audio_id: {accepted_id}")
        if row.get("lifecycle_status") not in ("accepted", "prepared"):
            raise ValueError(
                f"accepted audio {accepted_id}: lifecycle_status must be accepted/prepared"
            )
        direction = row.get("direction_id")
        if direction not in ("a_to_b", "b_to_a"):
            raise ValueError(f"accepted audio {accepted_id}: invalid direction_id")
        if row.get("condition") not in CONDITION_NAMES:
            raise ValueError(f"accepted audio {accepted_id}: invalid condition")
        indexed[accepted_id] = row
    return indexed


def _answer_key_contexts(
    rows: Sequence[dict[str, Any]], expected_count: int
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} answer keys, found {len(rows)}")
    contexts: dict[str, dict[str, str]] = {}
    raw_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        label = f"answer key row {index}"
        required = {
            "answer_key_id",
            "scenario_id",
            "context_label",
            "direction_id",
            "target_value",
            "stale_value",
            "dependent_relations",
            "root_invariant_constraints",
            "safety_note",
        }
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{label} is missing fields: {missing}")
        answer_key_id = _required_text(row, "answer_key_id", label)
        scenario_id = _required_text(row, "scenario_id", label)
        direction_id = _required_text(row, "direction_id", label)
        if direction_id not in ("a_to_b", "b_to_a"):
            raise ValueError(f"{label} has an invalid direction_id")
        if answer_key_id != f"{scenario_id}__{direction_id}":
            raise ValueError(f"{label} answer_key_id does not match scenario/direction")
        if answer_key_id in contexts:
            raise ValueError(f"duplicate answer_key_id: {answer_key_id}")

        dependent = row["dependent_relations"]
        if not isinstance(dependent, list) or len(dependent) != 3:
            raise ValueError(f"{label} must have exactly three dependent_relations")
        relation_context: dict[str, dict[str, str]] = {}
        for relation_row in dependent:
            if not isinstance(relation_row, dict):
                raise ValueError(f"{label} dependent_relations must contain objects")
            unit_id = relation_row.get("unit_id")
            if unit_id not in RELATIONS or unit_id in relation_context:
                raise ValueError(f"{label} has duplicate/invalid dependent unit {unit_id!r}")
            relation_context[str(unit_id)] = {
                "relation": _required_text(
                    relation_row, "relation", f"{answer_key_id}/{unit_id}"
                ),
                "planning_constraint": _required_text(
                    relation_row, "planning_constraint", f"{answer_key_id}/{unit_id}"
                ),
            }
        if set(relation_context) != set(RELATIONS):
            raise ValueError(f"{label} must cover D1, D2, and D3 exactly once")

        invariants = row["root_invariant_constraints"]
        if not isinstance(invariants, list) or not invariants:
            raise ValueError(f"{label} root_invariant_constraints must be a non-empty list")
        concise_invariants: list[dict[str, Any]] = []
        for invariant_index, invariant in enumerate(invariants):
            if not isinstance(invariant, dict):
                raise ValueError(f"{label} root-invariant entry must be an object")
            state = invariant.get("state")
            if not isinstance(state, dict) or not state:
                raise ValueError(
                    f"{answer_key_id}/invariant-{invariant_index} has an invalid state"
                )
            concise_invariants.append(
                {
                    "unit_id": _required_text(
                        invariant, "unit_id", f"{answer_key_id}/invariant-{invariant_index}"
                    ),
                    "relation": _required_text(
                        invariant, "relation", f"{answer_key_id}/invariant-{invariant_index}"
                    ),
                    "state": state,
                }
            )
        safety_note = row["safety_note"]
        if safety_note is not None and not isinstance(safety_note, str):
            raise ValueError(f"{label} safety_note must be a string or null")
        context = {
            "context_label": canonical_json(
                _required_text(row, "context_label", label)
            ),
            "target_value": canonical_json(
                _required_text(row, "target_value", label)
            ),
            "stale_value": canonical_json(
                _required_text(row, "stale_value", label)
            ),
            **{
                f"{unit_id}_relation_planning_constraint": canonical_json(
                    relation_context[unit_id]
                )
                for unit_id in RELATIONS
            },
            "root_invariant_constraints": canonical_json(concise_invariants),
            "safety_note": canonical_json(safety_note),
        }
        serialized_context = canonical_json(context).casefold()
        leaked_conditions = sorted(
            condition for condition in CONDITION_NAMES if condition in serialized_context
        )
        if leaked_conditions:
            raise ValueError(
                f"{answer_key_id}: condition identifiers leaked into public rubric context: "
                f"{leaked_conditions}"
            )
        contexts[answer_key_id] = context
        raw_by_id[answer_key_id] = row
    return contexts, raw_by_id


def build_annotation_package(
    eval_trials: Sequence[dict[str, Any]],
    accepted_audio: Sequence[dict[str, Any]],
    answer_keys: Sequence[dict[str, Any]],
    annotator_ids: Sequence[str],
    *,
    shuffle_seed: int,
    expected_accepted_count: int = 600,
    expected_answer_key_count: int = 60,
    expected_trials_per_audio: int = 5,
) -> dict[str, Any]:
    """Return two stable shuffled public sheets and one private blind map."""

    annotators = _require_two_annotators(annotator_ids)
    if not isinstance(shuffle_seed, int) or isinstance(shuffle_seed, bool):
        raise ValueError("shuffle_seed must be an integer")
    accepted_by_id = _accepted_join_index(accepted_audio, expected_accepted_count)
    context_by_key, raw_answer_keys = _answer_key_contexts(
        answer_keys, expected_answer_key_count
    )
    accepted_bundles = {
        str(row["text_bundle_id"]) for row in accepted_by_id.values()
    }
    if accepted_bundles != set(context_by_key):
        missing = sorted(accepted_bundles - set(context_by_key))
        extra = sorted(set(context_by_key) - accepted_bundles)
        raise ValueError(
            "accepted-audio/answer-key join is not exact: "
            f"missing_answer_keys={missing}, unused_answer_keys={extra}"
        )

    trial_ids: set[str] = set()
    blind_ids: set[str] = set()
    trial_counts_by_audio: dict[str, int] = {accepted_id: 0 for accepted_id in accepted_by_id}
    mapping: list[dict[str, Any]] = []
    source_by_blind: dict[str, Path] = {}
    public_base: dict[str, dict[str, Any]] = {}
    for index, trial in enumerate(eval_trials):
        trial_id = trial.get("eval_trial_id")
        run_id = trial.get("eval_run_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError(f"eval trial row {index} has no eval_trial_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"eval trial row {index} has no eval_run_id")
        accepted_id = trial.get("accepted_audio_id")
        if not isinstance(accepted_id, str) or accepted_id not in accepted_by_id:
            raise ValueError(
                f"eval trial {trial_id}: accepted_audio_id does not join exactly once"
            )
        if trial_id in trial_ids:
            raise ValueError(f"duplicate eval_trial_id: {trial_id}")
        trial_ids.add(trial_id)
        trial_counts_by_audio[accepted_id] += 1
        accepted_row = accepted_by_id[accepted_id]
        answer_key_id = str(accepted_row["text_bundle_id"])
        answer_key = raw_answer_keys[answer_key_id]
        if accepted_row["scenario_id"] != answer_key["scenario_id"]:
            raise ValueError(
                f"accepted audio {accepted_id}: scenario_id disagrees with answer key"
            )
        if accepted_row["direction_id"] != answer_key["direction_id"]:
            raise ValueError(
                f"accepted audio {accepted_id}: direction_id disagrees with answer key"
            )
        blind_id = _blind_id(run_id, trial_id, shuffle_seed)
        if blind_id in blind_ids:
            raise ValueError("blind_id collision; choose another shuffle seed")
        blind_ids.add(blind_id)
        response = trial.get("response")
        if not isinstance(response, dict):
            raise ValueError(f"eval trial {trial_id}: response must be an object")
        audio_path = _audio_source(response)
        audio_name = ""
        if audio_path is not None:
            suffix = audio_path.suffix.casefold()
            if not suffix or not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
                suffix = ".wav"
            audio_name = f"blind_media/{blind_id}{suffix}"
            source_by_blind[blind_id] = audio_path
        public_base[blind_id] = {
            "blind_id": blind_id,
            **context_by_key[answer_key_id],
            "response_text": _response_text(response),
            "response_audio_file": audio_name,
        }
        mapping.append(
            {
                "schema_version": SCHEMA_VERSION,
                "blind_id": blind_id,
                "eval_run_id": run_id,
                "eval_trial_id": trial_id,
                "accepted_audio_id": accepted_id,
                "answer_key_id": answer_key_id,
            }
        )

    bad_trial_counts = {
        accepted_id: count
        for accepted_id, count in trial_counts_by_audio.items()
        if count != expected_trials_per_audio
    }
    if bad_trial_counts:
        examples = dict(list(sorted(bad_trial_counts.items()))[:5])
        raise ValueError(
            "eval-trial/accepted-audio join is incomplete or duplicated; "
            f"expected {expected_trials_per_audio} trials per audio, examples={examples}"
        )

    sheets: dict[str, list[dict[str, Any]]] = {}
    for annotator_id in annotators:
        ordered_blind_ids = sorted(
            blind_ids,
            key=lambda value: _hash_order(
                f"v2-sheet\0{shuffle_seed}\0{annotator_id}", value
            ),
        )
        rows: list[dict[str, Any]] = []
        for order, blind_id in enumerate(ordered_blind_ids, 1):
            row = {
                "annotation_order": order,
                **public_base[blind_id],
                "annotator_id": annotator_id,
                "overall_label": "",
                "relation_D1": "",
                "relation_D2": "",
                "relation_D3": "",
                "final_target_correct": "",
                "stale_state_error": "",
                "assistant_started_before_repair": "",
                "notes": "",
            }
            if set(row) & FORBIDDEN_PUBLIC_FIELDS:
                raise AssertionError("a sensitive field entered the public annotation sheet")
            rows.append(row)
        sheets[annotator_id] = rows
    return {
        "sheets": sheets,
        "blind_map": sorted(mapping, key=lambda row: str(row["blind_id"])),
        "audio_sources": source_by_blind,
    }


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError(f"cannot construct a safe filename from {value!r}")
    return safe


def write_annotation_package(output_dir: Path, package: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "PRIVATE_blind_map.jsonl", package["blind_map"])
    for annotator_id, rows in package["sheets"].items():
        write_csv(
            output_dir / f"annotation_{_safe_filename(annotator_id)}.csv",
            rows,
            PUBLIC_FIELDS,
        )

    sources: Mapping[str, Path] = package.get("audio_sources", {})
    if sources:
        media_dir = output_dir / "blind_media"
        media_dir.mkdir(parents=True, exist_ok=True)
        for blind_id, source in sources.items():
            if not source.is_file():
                raise FileNotFoundError(f"response audio does not exist: {source}")
            suffix = source.suffix.casefold()
            if not suffix or not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
                suffix = ".wav"
            shutil.copy2(source, media_dir / f"{blind_id}{suffix}")


def parse_required_bool(value: Any, field: str) -> bool:
    """Parse a required boolean without ever interpreting a blank as False."""

    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{field} is missing; blank labels are not coerced to false")
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field} must be true or false, found {value!r}")


def parse_optional_bool(value: Any, field: str) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return parse_required_bool(value, field)


def validate_annotation(annotation: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "blind_id",
        "eval_trial_id",
        "annotator_id",
        "overall_label",
        "relation_labels",
        "final_target_correct",
        "stale_state_error",
        "adjudicator",
    }
    missing = sorted(required - annotation.keys())
    if missing:
        raise ValueError(f"annotation is missing fields: {missing}")
    if annotation["schema_version"] != SCHEMA_VERSION:
        raise ValueError("annotation has the wrong schema_version")
    for field in ("blind_id", "eval_trial_id", "annotator_id"):
        if not isinstance(annotation[field], str) or not annotation[field]:
            raise ValueError(f"annotation {field} must be a non-empty string")
    if annotation["overall_label"] not in OVERALL_LABELS:
        raise ValueError(f"invalid overall_label: {annotation['overall_label']!r}")
    relations = annotation["relation_labels"]
    if not isinstance(relations, dict) or set(relations) != set(RELATIONS):
        raise ValueError("relation_labels must contain exactly D1, D2, and D3")
    for relation, label in relations.items():
        if label not in RELATION_LABELS:
            raise ValueError(f"invalid relation label for {relation}: {label!r}")
    for field in ("final_target_correct", "stale_state_error", "adjudicator"):
        if type(annotation[field]) is not bool:  # bool only; reject None and 0/1.
            raise ValueError(f"annotation {field} must be a boolean")
    if "assistant_started_before_repair" in annotation:
        value = annotation["assistant_started_before_repair"]
        if value is not None and type(value) is not bool:
            raise ValueError("assistant_started_before_repair must be boolean or null")


def annotation_from_sheet_row(
    row: Mapping[str, Any],
    blind_map: Mapping[str, str],
    *,
    expected_annotator_id: str,
    adjudicator: bool = False,
) -> dict[str, Any]:
    blind_id = str(row.get("blind_id", ""))
    if blind_id not in blind_map:
        raise ValueError(f"unknown blind_id: {blind_id!r}")
    observed_annotator = str(row.get("annotator_id", ""))
    if observed_annotator != expected_annotator_id:
        raise ValueError(
            f"sheet annotator_id={observed_annotator!r} does not match "
            f"expected {expected_annotator_id!r}"
        )
    overall = str(row.get("overall_label", "")).strip()
    relations = {
        relation: str(row.get(f"relation_{relation}", "")).strip()
        for relation in RELATIONS
    }
    annotation = {
        "schema_version": SCHEMA_VERSION,
        "blind_id": blind_id,
        "eval_trial_id": blind_map[blind_id],
        "annotator_id": expected_annotator_id,
        "overall_label": overall,
        "relation_labels": relations,
        "final_target_correct": parse_required_bool(
            row.get("final_target_correct"), "final_target_correct"
        ),
        "stale_state_error": parse_required_bool(
            row.get("stale_state_error"), "stale_state_error"
        ),
        "assistant_started_before_repair": parse_optional_bool(
            row.get("assistant_started_before_repair"),
            "assistant_started_before_repair",
        ),
        "notes": str(row.get("notes", "")),
        "adjudicator": bool(adjudicator),
    }
    validate_annotation(annotation)
    return annotation


def load_blind_map(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    trials: set[str] = set()
    for index, row in enumerate(rows):
        blind_id = row.get("blind_id")
        trial_id = row.get("eval_trial_id")
        if not isinstance(blind_id, str) or not blind_id:
            raise ValueError(f"blind map row {index} has an invalid blind_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError(f"blind map row {index} has an invalid eval_trial_id")
        if blind_id in mapping or trial_id in trials:
            raise ValueError("blind map must be one-to-one")
        mapping[blind_id] = trial_id
        trials.add(trial_id)
    return mapping


def read_completed_sheet(
    path: Path,
    blind_map: Mapping[str, str],
    *,
    annotator_id: str,
    adjudicator: bool = False,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"annotation sheet has no header: {path}")
        for line_number, row in enumerate(reader, 2):
            try:
                annotations.append(
                    annotation_from_sheet_row(
                        row,
                        blind_map,
                        expected_annotator_id=annotator_id,
                        adjudicator=adjudicator,
                    )
                )
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return annotations


def _decision_signature(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        annotation["overall_label"],
        tuple(annotation["relation_labels"][relation] for relation in RELATIONS),
        annotation["final_target_correct"],
        annotation["stale_state_error"],
        annotation.get("assistant_started_before_repair"),
    )


def resolve_annotations(
    expected_eval_trial_ids: Iterable[str],
    annotations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enforce two independent labels and adjudicate every disagreement."""

    expected = set(expected_eval_trial_ids)
    by_trial: dict[str, list[dict[str, Any]]] = {trial_id: [] for trial_id in expected}
    for annotation in annotations:
        validate_annotation(annotation)
        trial_id = str(annotation["eval_trial_id"])
        if trial_id not in expected:
            raise ValueError(f"annotation references an unexpected eval_trial_id: {trial_id}")
        by_trial[trial_id].append(annotation)

    resolved: list[dict[str, Any]] = []
    for trial_id in sorted(expected):
        rows = by_trial[trial_id]
        primary = [row for row in rows if not row["adjudicator"]]
        adjudicators = [row for row in rows if row["adjudicator"]]
        if len(primary) != 2:
            raise ValueError(
                f"eval trial {trial_id}: expected exactly two primary annotations, "
                f"found {len(primary)}"
            )
        primary_ids = {str(row["annotator_id"]) for row in primary}
        if len(primary_ids) != 2:
            raise ValueError(f"eval trial {trial_id}: primary annotations are not independent")
        blind_ids = {str(row["blind_id"]) for row in rows}
        if len(blind_ids) != 1:
            raise ValueError(f"eval trial {trial_id}: annotation blind_id values disagree")
        disagreement = _decision_signature(primary[0]) != _decision_signature(primary[1])
        if disagreement:
            if len(adjudicators) != 1:
                raise ValueError(
                    f"eval trial {trial_id}: disagreement requires exactly one adjudication"
                )
            adjudicator = adjudicators[0]
            if adjudicator["annotator_id"] in primary_ids:
                raise ValueError(
                    f"eval trial {trial_id}: adjudicator must be independent of both annotators"
                )
            chosen = adjudicator
            method = "adjudicated_disagreement"
        else:
            if adjudicators:
                raise ValueError(
                    f"eval trial {trial_id}: agreeing labels must not receive adjudication"
                )
            chosen = primary[0]
            method = "independent_agreement"
        resolved.append(
            {
                "schema_version": SCHEMA_VERSION,
                "blind_id": chosen["blind_id"],
                "eval_trial_id": trial_id,
                "overall_label": chosen["overall_label"],
                "relation_labels": dict(chosen["relation_labels"]),
                "final_target_correct": chosen["final_target_correct"],
                "stale_state_error": chosen["stale_state_error"],
                "assistant_started_before_repair": chosen.get(
                    "assistant_started_before_repair"
                ),
                "resolution_method": method,
                "primary_annotator_ids": sorted(primary_ids),
                "adjudicator_id": (
                    str(adjudicators[0]["annotator_id"]) if disagreement else None
                ),
            }
        )
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create two blind annotation sheets")
    create.add_argument("--eval-trials", type=Path, required=True)
    create.add_argument(
        "--accepted-manifest", type=Path, default=DEFAULT_ACCEPTED_MANIFEST
    )
    create.add_argument("--answer-keys", type=Path, default=DEFAULT_ANSWER_KEYS)
    create.add_argument("--annotator", action="append", required=True)
    create.add_argument("--shuffle-seed", type=int, required=True)
    create.add_argument("--output-dir", type=Path, required=True)

    ingest = subparsers.add_parser("ingest", help="convert one completed CSV to JSONL")
    ingest.add_argument("--blind-map", type=Path, required=True)
    ingest.add_argument("--sheet", type=Path, required=True)
    ingest.add_argument("--annotator-id", required=True)
    ingest.add_argument("--adjudicator", action="store_true")
    ingest.add_argument("--output", type=Path, required=True)

    resolve = subparsers.add_parser("resolve", help="resolve combined primary/adjudicator JSONL")
    resolve.add_argument("--eval-trials", type=Path, required=True)
    resolve.add_argument("--annotations", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create":
        trials = read_jsonl(args.eval_trials)
        package = build_annotation_package(
            trials,
            read_jsonl(args.accepted_manifest),
            read_jsonl(args.answer_keys),
            args.annotator,
            shuffle_seed=args.shuffle_seed,
        )
        write_annotation_package(args.output_dir, package)
        print(
            f"Wrote {len(args.annotator)} blind sheets for {len(trials)} trials -> "
            f"{args.output_dir}"
        )
        return
    if args.command == "ingest":
        blind_map = load_blind_map(read_jsonl(args.blind_map))
        annotations = read_completed_sheet(
            args.sheet,
            blind_map,
            annotator_id=args.annotator_id,
            adjudicator=args.adjudicator,
        )
        write_jsonl(args.output, annotations)
        print(f"Ingested {len(annotations)} annotations -> {args.output}")
        return
    trials = read_jsonl(args.eval_trials)
    annotations = read_jsonl(args.annotations)
    resolved = resolve_annotations(
        (str(row["eval_trial_id"]) for row in trials), annotations
    )
    write_jsonl(args.output, resolved)
    print(f"Resolved {len(resolved)} annotations -> {args.output}")


if __name__ == "__main__":
    main()
