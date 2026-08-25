#!/usr/bin/env python3
"""Build a fail-closed, metadata-only dataset v2 release directory.

The builder intentionally supports two different artifacts:

``text-development``
    A non-release snapshot containing reviewed text, schemas, configuration, answer
    keys, and deterministic speaker assignment metadata.  It never contains audio,
    evaluation, or annotation artifacts.

``full-audio-release``
    An approved public metadata package.  It is created only after the complete
    blueprint -> script -> assignment -> audio -> evaluation -> annotation matrix
    passes and an external approval document binds every source input by SHA-256.

Audio bytes live outside this package.  Public audio manifests may contain only
relative artifact URIs and hashes; raw candidates and provider responses are never
copied.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

try:  # Direct CLI execution.
    from alignment_evidence import (
        validate_alignment_gate_contract,
        validate_downstream_alignment_evidence,
    )
    from build_eval_adapter import validate_eval_trials
    from ids import accepted_audio_id, matched_audio_bundle_id, prepared_stimulus_id, rendition_target_id
    from response_validation import resolve_content_uri, validate_trial_response
except ImportError:  # pragma: no cover - package-style imports in external callers.
    from .alignment_evidence import (
        validate_alignment_gate_contract,
        validate_downstream_alignment_evidence,
    )
    from .build_eval_adapter import validate_eval_trials
    from .ids import accepted_audio_id, matched_audio_bundle_id, prepared_stimulus_id, rendition_target_id
    from .response_validation import resolve_content_uri, validate_trial_response


FORMAT_VERSION = "1.0.0"
SCHEMA_VERSION = "2.0.0"
TEXT_DEVELOPMENT = "text-development"
FULL_AUDIO_RELEASE = "full-audio-release"
RELEASE_KINDS = (TEXT_DEVELOPMENT, FULL_AUDIO_RELEASE)

CONDITIONS = (
    "clean_final",
    "immediate_repair",
    "delayed_neutral",
    "delayed_one_dependency",
    "delayed_three_dependencies",
)
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
RELATION_LABELS = {"new_bound", "old_bound", "both", "unresolved", "not_addressed"}

DOC_FILES = (
    "README.md",
    "DATASET_CARD.md",
    "DECISIONS.md",
    "TTS_PROVIDER_REVIEW.md",
    "ANALYSIS_PROTOCOL.md",
    "ANNOTATION_GUIDE.md",
)
CONFIG_FILES = (
    "config/dataset.yaml",
    "config/eval.json",
    "config/value_evidence.json",
)
SCHEMA_FILES = (
    "schemas/blueprint.schema.json",
    "schemas/script.schema.json",
    "schemas/answer_key.schema.json",
    "schemas/audio.schema.json",
    "schemas/eval_trial.schema.json",
    "schemas/annotation.schema.json",
)
TEXT_MANIFEST_FILES = {
    "blueprints": "blueprints/scenarios.jsonl",
    "scripts": "generated/scripts.jsonl",
    "answer_keys": "answer_keys/answer_keys.jsonl",
    "analysis_folds": "assignments/analysis_folds.jsonl",
    "speaker_bundles": "assignments/speaker_bundles.jsonl",
    "rendition_targets": "assignments/rendition_targets.jsonl",
    "recording_order": "assignments/recording_order.jsonl",
}
FULL_PRIVATE_INPUTS = {
    "accepted_audio": (
        "manifests/accepted_audio.jsonl",
        "audio/accepted_manifest.jsonl",
    ),
    "prepared_stimuli": ("manifests/prepared_stimuli.jsonl",),
    "eval_trials": ("evaluation/eval_trials.jsonl",),
    "annotations": ("annotations/annotations.jsonl",),
}
FULL_PUBLIC_FILES = {
    "accepted_audio": "manifests/accepted_audio.public.jsonl",
    "prepared_stimuli": "manifests/prepared_stimuli.public.jsonl",
    "audio_inventory": "manifests/audio_artifacts.sha256.jsonl",
    "eval_trials": "evaluation/eval_trials.public.jsonl",
    "annotations": "annotations/resolved_annotations.public.jsonl",
}
FULL_EVIDENCE_FILES = {
    "selection_policy": "release_evidence/selection_policy.json",
    "timing_policy": "release_evidence/timing_policy.json",
    "alignment_report": "release_evidence/alignment_report.json",
    "audio_qc_report": "release_evidence/audio_qc_report.json",
    "double_listen_report": "release_evidence/double_listen_report.json",
    "analysis_result": "release_evidence/analysis_result.json",
    "baseline_report": "release_evidence/baseline_report.json",
}
PUBLIC_RESPONSE_EVIDENCE_FIELDS = {
    "audio_sha256",
    "audio_duration_ms",
    "audio_sample_rate",
    "audio_channels",
    "audio_sample_width_bytes",
    "timebase",
    "stream_origin_ms",
    "primary_window_start_ms",
    "requested_target_end_ms",
    "actual_target_end_ms",
    "fed_sample_count",
    "fed_frame_count",
    "output_sample_count",
    "output_frame_count",
    "appended_zero_sample_count",
    "coverage_complete",
    "eos_reached",
    "runner_source_sha256",
    "effective_generation_config_sha256",
    "mimi_frame_samples",
    "transcript_sha256",
    "stream_events_sha256",
    "evidence_sha256",
    "stream_event_count",
    "first_stream_event_ms",
    "last_stream_event_ms",
}
PUBLIC_RUN_CONTRACT_EVIDENCE_FIELDS = {
    "input_stimulus_sha256",
    "capture_contract_sha256",
    "matrix_contract_sha256",
    "execution_contract_sha256",
    "runner_source_sha256",
}
PUBLIC_EVAL_ROW_FIELDS = {
    "schema_version",
    "eval_run_id",
    "eval_trial_id",
    "accepted_audio_id",
    "model_repo",
    "resolved_revision",
    "generation_config_hash",
    "code_commit",
    "generation_seed",
    "condition",
    "prepared_stimulus_id",
    "response_status",
    "run_contract_evidence",
    "response_evidence",
}

REQUIRED_APPROVAL_GATES = (
    "license_review",
    "audio_redistribution_rights",
    "privacy_and_pii_review",
    "timing_thresholds_frozen",
    "independent_alignment_or_audited_manual_review",
    "automatic_audio_qc",
    "required_manual_audio_qc",
    "double_listen_audio_qc",
    "analysis_protocol_frozen_before_model_outputs",
    "eval_complete",
    "double_annotation_complete",
    "adjudication_complete",
    "baseline_complete",
)

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
SECRET_PATTERNS = (
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?key|secret|token|password|authorization)"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{8,}"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"),
)
EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
LOCAL_PATH_RE = re.compile(
    r"(?<![:/A-Za-z0-9])(?:/(?:Users|home|root|tmp|private|var|Volumes|opt|etc|usr|Applications)/[^\s\"'<>)]*|[A-Za-z]:[\\/][^\s\"'<>)]*)"
)
FORBIDDEN_PATH_PARTS = {
    "private_blind_map.jsonl",
    "private",
    "raw_candidates.jsonl",
    "canonical_candidates.jsonl",
    "provider_response.json",
    "provider_responses.jsonl",
    ".env",
}
FORBIDDEN_JSON_KEYS = {
    "api_key",
    "access_key",
    "secret_key",
    "password",
    "authorization",
    "credential",
    "credentials",
    "provider_token",
    "private_key",
    "email",
    "phone",
    "real_name",
    "full_name",
    "raw_candidate",
    "private_blind_map",
}
FORBIDDEN_EVIDENCE_KEYS = {
    "annotator_id",
    "annotator_ids",
    "approver_id",
    "audio_path",
    "blind_id",
    "blind_ids",
    "model_output",
    "model_outputs",
    "model_response",
    "model_responses",
    "provider_response",
    "provider_responses",
    "raw_candidate",
    "raw_candidates",
    "response",
    "responses",
    "reviewer_id",
    "reviewer_ids",
    "stream_events",
    "transcript",
    "transcripts",
}


class ReleaseError(ValueError):
    """A release contract or safety gate failed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"cannot read JSON {path}: {error}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ReleaseError(f"{path}:{line_number}: JSONL row is not an object")
                rows.append(value)
    except ReleaseError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)))
            handle.write("\n")


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ReleaseError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise ReleaseError(f"missing required {label}: {path}")
    return path


def _resolve_alternative(root: Path, label: str, relatives: Sequence[str]) -> Path:
    matches = [root / relative for relative in relatives if (root / relative).is_file()]
    if not matches:
        raise ReleaseError(
            f"missing required {label}; expected one of: {', '.join(relatives)}"
        )
    for path in matches:
        _require_regular_file(path, label)
    if len(matches) > 1:
        hashes = {sha256_file(path) for path in matches}
        if len(hashes) != 1:
            raise ReleaseError(f"ambiguous {label}: alternative manifests have different hashes")
    return matches[0]


def _dataset_version(root: Path) -> str:
    version = _require_regular_file(root / "VERSION", "VERSION").read_text(
        encoding="utf-8"
    ).strip()
    if version != SCHEMA_VERSION:
        raise ReleaseError(f"VERSION must be {SCHEMA_VERSION!r}, found {version!r}")
    return version


def _dataset_config(root: Path) -> dict[str, Any]:
    value = read_json(_require_regular_file(root / "config/dataset.yaml", "dataset config"))
    if not isinstance(value, dict):
        raise ReleaseError("dataset config must be an object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseError("dataset config schema_version does not match VERSION")
    return value


def _source_inputs(root: Path, *, full: bool) -> dict[str, Path]:
    paths: dict[str, Path] = {"version": _require_regular_file(root / "VERSION", "VERSION")}
    for relative in DOC_FILES:
        paths[f"doc:{relative}"] = _require_regular_file(root / relative, relative)
    for relative in CONFIG_FILES:
        paths[f"config:{relative}"] = _require_regular_file(root / relative, relative)
    for relative in SCHEMA_FILES:
        paths[f"schema:{relative}"] = _require_regular_file(root / relative, relative)
    for logical, relative in TEXT_MANIFEST_FILES.items():
        paths[logical] = _require_regular_file(root / relative, logical)
    if full:
        license_matches = [root / name for name in ("LICENSE", "LICENSE.md") if (root / name).is_file()]
        if len(license_matches) != 1:
            raise ReleaseError("full release requires exactly one dataset LICENSE or LICENSE.md")
        paths["license"] = _require_regular_file(license_matches[0], "dataset license")
        for logical, alternatives in FULL_PRIVATE_INPUTS.items():
            paths[logical] = _resolve_alternative(root, logical, alternatives)
        for logical, relative in FULL_EVIDENCE_FILES.items():
            paths[logical] = _require_regular_file(root / relative, logical)
    return paths


def collect_source_hashes(dataset_root: Path, *, full: bool = True) -> dict[str, str]:
    """Return the exact logical-input hash map an approval document must bind."""

    root = dataset_root.resolve()
    return {
        logical: sha256_file(path)
        for logical, path in sorted(_source_inputs(root, full=full).items())
    }


def _counts(config: Mapping[str, Any]) -> dict[str, int]:
    raw = config.get("counts")
    if not isinstance(raw, Mapping):
        raise ReleaseError("dataset config counts must be an object")
    keys = {
        "scenarios": "scenarios",
        "text_bundles": "text_bundles",
        "scripts": "scripts",
        "matched_audio_bundles": "matched_audio_bundles_per_track",
        "rendition_targets": "rendition_targets_per_track",
    }
    result: dict[str, int] = {}
    for output, source in keys.items():
        value = raw.get(source)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ReleaseError(f"config counts.{source} must be a positive integer")
        result[output] = value
    seeds = config.get("evaluation", {}).get("generation_seeds")
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds
    ):
        raise ReleaseError("config evaluation.generation_seeds must be a non-empty integer list")
    if len(set(seeds)) != len(seeds):
        raise ReleaseError("config generation seeds must be unique")
    result["generation_seeds"] = len(seeds)
    result["eval_trials"] = result["rendition_targets"] * len(seeds)
    result["primary_annotations"] = result["eval_trials"] * int(
        config.get("evaluation", {}).get("annotations_per_trial", 0)
    )
    return result


def _require_unique(rows: Sequence[Mapping[str, Any]], field: str, label: str) -> set[str]:
    values: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ReleaseError(f"{label} row {index} has an invalid {field}")
        values.append(value)
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ReleaseError(f"{label} has duplicate {field}: {sorted(duplicates)[:5]}")
    return set(values)


def _validate_blueprints(
    rows: Sequence[dict[str, Any]], config: Mapping[str, Any], counts: Mapping[str, int]
) -> dict[str, Any]:
    if len(rows) != counts["scenarios"]:
        raise ReleaseError(
            f"blueprint count must be {counts['scenarios']}, found {len(rows)}"
        )
    scenario_ids = _require_unique(rows, "scenario_id", "blueprints")
    for row in rows:
        scenario_id = row["scenario_id"]
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ReleaseError(f"blueprint {scenario_id}: wrong schema_version")
        if row.get("review_status") != "approved":
            raise ReleaseError(f"blueprint {scenario_id}: review_status is not approved")
        reviews = row.get("reviews")
        if not isinstance(reviews, list) or len(reviews) != 2:
            raise ReleaseError(f"blueprint {scenario_id}: exactly two reviews are required")
        reviewer_ids = {review.get("reviewer_id") for review in reviews if isinstance(review, dict)}
        if len(reviewer_ids) != 2 or None in reviewer_ids:
            raise ReleaseError(f"blueprint {scenario_id}: reviews are not independent")
        if any(review.get("decision") != "approved" for review in reviews):
            raise ReleaseError(f"blueprint {scenario_id}: a review is not approved")
        source = row.get("source")
        if not isinstance(source, dict) or not source.get("license"):
            raise ReleaseError(f"blueprint {scenario_id}: source license is missing")
        if row.get("language") != config.get("language") or row.get("domain") != config.get("domain"):
            raise ReleaseError(f"blueprint {scenario_id}: language/domain mismatch")
        dependent = row.get("dependent_units")
        neutral = row.get("neutral_units")
        if not isinstance(dependent, list) or not isinstance(neutral, list):
            raise ReleaseError(f"blueprint {scenario_id}: unit arrays are missing")
        unit_ids = [unit.get("unit_id") for unit in [*dependent, *neutral] if isinstance(unit, dict)]
        if len(unit_ids) != 6 or set(unit_ids) != {"D1", "D2", "D3", "N1", "N2", "N3"}:
            raise ReleaseError(f"blueprint {scenario_id}: D1-D3/N1-N3 coverage is invalid")
    return {"status": "passed", "count": len(rows), "unique_scenarios": len(scenario_ids)}


def _validate_scripts(
    rows: Sequence[dict[str, Any]],
    blueprints: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    if len(rows) != counts["scripts"]:
        raise ReleaseError(f"script count must be {counts['scripts']}, found {len(rows)}")
    _require_unique(rows, "script_id", "scripts")
    blueprint_by_id = {str(row["scenario_id"]): row for row in blueprints}
    by_bundle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    config_hash = sha256_value(config)
    for row in rows:
        script_id = str(row["script_id"])
        scenario_id = row.get("scenario_id")
        blueprint = blueprint_by_id.get(str(scenario_id))
        if blueprint is None:
            raise ReleaseError(f"script {script_id}: unknown scenario_id")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ReleaseError(f"script {script_id}: wrong schema_version")
        if row.get("blueprint_hash") != sha256_value(blueprint):
            raise ReleaseError(f"script {script_id}: blueprint hash mismatch")
        if row.get("config_hash") != config_hash:
            raise ReleaseError(f"script {script_id}: config hash mismatch")
        condition = row.get("condition")
        if condition not in CONDITIONS:
            raise ReleaseError(f"script {script_id}: invalid condition")
        bundle_id = row.get("text_bundle_id")
        if not isinstance(bundle_id, str) or script_id != f"{bundle_id}__{condition}":
            raise ReleaseError(f"script {script_id}: non-canonical ID")
        by_bundle[bundle_id].append(row)
    if len(by_bundle) != counts["text_bundles"]:
        raise ReleaseError(
            f"text bundle count must be {counts['text_bundles']}, found {len(by_bundle)}"
        )
    for bundle_id, bundle_rows in by_bundle.items():
        if len(bundle_rows) != len(CONDITIONS) or {
            str(row["condition"]) for row in bundle_rows
        } != set(CONDITIONS):
            raise ReleaseError(f"text bundle {bundle_id}: condition matrix is incomplete")
    return {
        "status": "passed",
        "count": len(rows),
        "text_bundles": len(by_bundle),
        "conditions": len(CONDITIONS),
    }


def _validate_answer_keys(
    rows: Sequence[dict[str, Any]], scripts: Sequence[dict[str, Any]], counts: Mapping[str, int]
) -> dict[str, Any]:
    if len(rows) != counts["text_bundles"]:
        raise ReleaseError(
            f"answer-key count must be {counts['text_bundles']}, found {len(rows)}"
        )
    ids = _require_unique(rows, "answer_key_id", "answer keys")
    bundle_ids = {str(row["text_bundle_id"]) for row in scripts}
    if ids != bundle_ids:
        raise ReleaseError("answer-key IDs do not match the complete text-bundle set")
    return {"status": "passed", "count": len(rows)}


def _validate_assignments(
    folds: Sequence[dict[str, Any]],
    bundles: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    recording: Sequence[dict[str, Any]],
    scripts: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    if len(folds) != counts["scenarios"]:
        raise ReleaseError("analysis-fold manifest does not cover every scenario")
    _require_unique(folds, "scenario_id", "analysis folds")
    if len(bundles) != counts["matched_audio_bundles"]:
        raise ReleaseError(
            f"matched-audio bundle count must be {counts['matched_audio_bundles']}, "
            f"found {len(bundles)}"
        )
    bundle_ids = _require_unique(bundles, "matched_audio_bundle_id", "speaker bundles")
    script_by_id = {str(row["script_id"]): row for row in scripts}
    fold_by_scenario = {str(row["scenario_id"]): row.get("analysis_fold") for row in folds}
    source_tracks = config.get("source_tracks")
    if not isinstance(source_tracks, Mapping) or not source_tracks:
        raise ReleaseError("dataset config source_tracks must be a non-empty object")
    observed_tracks = {str(row.get("source_track_id", "")) for row in bundles}
    if len(observed_tracks) != 1 or "" in observed_tracks:
        raise ReleaseError("speaker bundles must contain exactly one non-empty source track")
    source_track_id = next(iter(observed_tracks))
    track = source_tracks.get(source_track_id)
    if not isinstance(track, Mapping):
        raise ReleaseError(f"assignment source track is not configured: {source_track_id}")
    raw_speakers = track.get("speakers")
    if not isinstance(raw_speakers, list) or not raw_speakers:
        raise ReleaseError(f"source track {source_track_id}: configured speakers are missing")
    speaker_profiles = {
        str(row.get("speaker_id", "")): row
        for row in raw_speakers
        if isinstance(row, Mapping) and row.get("speaker_id")
    }
    if len(speaker_profiles) != len(raw_speakers):
        raise ReleaseError(f"source track {source_track_id}: speaker IDs are invalid or duplicated")
    by_text_bundle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bundle_by_id: dict[str, dict[str, Any]] = {}
    for row in bundles:
        matched_id = str(row["matched_audio_bundle_id"])
        text_bundle_id = str(row.get("text_bundle_id", ""))
        speaker_id = str(row.get("speaker_id", ""))
        try:
            expected_id = matched_audio_bundle_id(text_bundle_id, source_track_id, speaker_id)
        except ValueError as error:
            raise ReleaseError(f"speaker bundle {matched_id}: invalid canonical ID fields") from error
        if matched_id != expected_id:
            raise ReleaseError(f"speaker bundle {matched_id}: non-canonical ID")
        if row.get("source_track_id") != source_track_id or speaker_id not in speaker_profiles:
            raise ReleaseError(f"speaker bundle {matched_id}: source-track/speaker mismatch")
        if row.get("voice") != speaker_profiles[speaker_id].get("voice"):
            raise ReleaseError(f"speaker bundle {matched_id}: voice does not match speaker config")
        scenario_id = str(row.get("scenario_id", ""))
        direction_id = str(row.get("direction_id", ""))
        if text_bundle_id != f"{scenario_id}__{direction_id}":
            raise ReleaseError(f"speaker bundle {matched_id}: text-bundle metadata mismatch")
        if row.get("analysis_fold") != fold_by_scenario.get(scenario_id):
            raise ReleaseError(f"speaker bundle {matched_id}: analysis-fold mismatch")
        expected_scripts = {
            f"{text_bundle_id}__{condition}" for condition in CONDITIONS
        }
        if set(map(str, row.get("script_ids", []))) != expected_scripts:
            raise ReleaseError(f"speaker bundle {matched_id}: script IDs are not exact")
        expected_targets = {
            rendition_target_id(script_id, source_track_id, speaker_id)
            for script_id in expected_scripts
        }
        if set(map(str, row.get("rendition_target_ids", []))) != expected_targets:
            raise ReleaseError(f"speaker bundle {matched_id}: rendition target IDs are not exact")
        bundle_by_id[matched_id] = row
        by_text_bundle[text_bundle_id].append(row)
    if len(by_text_bundle) != counts["text_bundles"]:
        raise ReleaseError("speaker bundles do not cover every text bundle")
    for text_bundle_id, rows in by_text_bundle.items():
        speakers = {str(row.get("speaker_id", "")) for row in rows}
        if len(rows) != 2 or len(speakers) != 2 or "" in speakers:
            raise ReleaseError(f"text bundle {text_bundle_id}: requires two distinct speakers")
    if {str(row.get("speaker_id", "")) for row in bundles} != set(speaker_profiles):
        raise ReleaseError("assignment speaker set does not exactly match source-track config")
    if len(targets) != counts["rendition_targets"]:
        raise ReleaseError(
            f"rendition-target count must be {counts['rendition_targets']}, found {len(targets)}"
        )
    target_ids = _require_unique(targets, "rendition_target_id", "rendition targets")
    targets_by_bundle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_by_id: dict[str, dict[str, Any]] = {}
    for row in targets:
        target_id = str(row["rendition_target_id"])
        matched_id = str(row.get("matched_audio_bundle_id", ""))
        bundle = bundle_by_id.get(matched_id)
        if bundle is None:
            raise ReleaseError("rendition target references an unknown matched-audio bundle")
        script = script_by_id.get(str(row.get("script_id", "")))
        if script is None:
            raise ReleaseError("rendition target references an unknown script")
        for field in ("text_bundle_id", "scenario_id", "direction_id", "condition"):
            if row.get(field) != script.get(field):
                raise ReleaseError(f"rendition target {target_id}: {field} does not match script")
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
            if row.get(field) != bundle.get(field):
                raise ReleaseError(
                    f"rendition target {target_id}: {field} does not match matched-audio bundle"
                )
        try:
            expected_target_id = rendition_target_id(
                str(row["script_id"]), source_track_id, str(row["speaker_id"])
            )
            expected_matched_id = matched_audio_bundle_id(
                str(row["text_bundle_id"]), source_track_id, str(row["speaker_id"])
            )
        except ValueError as error:
            raise ReleaseError(f"rendition target {target_id}: invalid canonical ID fields") from error
        if target_id != expected_target_id or matched_id != expected_matched_id:
            raise ReleaseError(f"rendition target {target_id}: non-canonical lineage IDs")
        target_by_id[target_id] = row
        targets_by_bundle[matched_id].append(row)
    for matched_id, rows in targets_by_bundle.items():
        if len(rows) != len(CONDITIONS) or {str(row.get("condition")) for row in rows} != set(CONDITIONS):
            raise ReleaseError(f"matched-audio bundle {matched_id}: condition matrix is incomplete")
        if {str(row["rendition_target_id"]) for row in rows} != set(
            map(str, bundle_by_id[matched_id].get("rendition_target_ids", []))
        ):
            raise ReleaseError(f"matched-audio bundle {matched_id}: target ID list mismatch")
    if set(targets_by_bundle) != bundle_ids:
        raise ReleaseError("one or more matched-audio bundles have no rendition targets")
    if len(recording) != counts["rendition_targets"]:
        raise ReleaseError("recording-order count does not match rendition targets")
    recording_targets = [str(row.get("rendition_target_id", "")) for row in recording]
    if len(set(recording_targets)) != len(recording_targets) or set(recording_targets) != target_ids:
        raise ReleaseError("recording order is not a permutation of rendition targets")
    recording_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in recording:
        recording_id = str(row.get("recording_order_id", ""))
        target_id = str(row.get("rendition_target_id", ""))
        target = target_by_id[target_id]
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
                raise ReleaseError(f"recording row {recording_id}: {field} does not match target")
        position = row.get("recording_position")
        if not isinstance(position, int) or isinstance(position, bool) or position < 1:
            raise ReleaseError(f"recording row {recording_id}: invalid recording position")
        expected_recording_id = (
            f"{source_track_id}__{row['speaker_id']}__position_{position:03d}"
        )
        if recording_id != expected_recording_id:
            raise ReleaseError(f"recording row {recording_id}: non-canonical ID")
        recording_by_speaker[str(row["speaker_id"])].append(row)
    for speaker_id, rows in recording_by_speaker.items():
        ordered = sorted(rows, key=lambda row: int(row["recording_position"]))
        if [int(row["recording_position"]) for row in ordered] != list(
            range(1, len(ordered) + 1)
        ):
            raise ReleaseError(f"speaker {speaker_id}: recording positions are not contiguous")
    return {
        "status": "passed",
        "analysis_folds": len(folds),
        "matched_audio_bundles": len(bundles),
        "rendition_targets": len(targets),
        "recording_order_rows": len(recording),
    }


def _run_production_text_validators(
    blueprints: list[dict[str, Any]],
    scripts: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    speaker_bundles: list[dict[str, Any]],
    rendition_targets: list[dict[str, Any]],
    recording_order: list[dict[str, Any]],
    config: dict[str, Any],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    """Run the detailed repository validators for the immutable v2 production shape.

    Tiny contract fixtures deliberately use scaled counts, but an actual 30/300/600
    build cannot bypass the complete semantic and assignment validators.
    """

    production_shape = (
        counts["scenarios"],
        counts["text_bundles"],
        counts["scripts"],
        counts["matched_audio_bundles"],
        counts["rendition_targets"],
    ) == (30, 60, 300, 120, 600)
    if not production_shape:
        return {"status": "passed", "mode": "scaled_contract_fixture"}
    try:
        from validate_blueprints import validate_blueprints as detailed_blueprint_validation
        from validate_scripts import validate_scripts as detailed_script_validation
        from assign_speakers import validate_manifests as detailed_assignment_validation
    except ImportError as error:  # pragma: no cover - indicates a broken release runtime.
        raise ReleaseError(f"cannot import production validators: {error}") from error
    errors = [
        *detailed_blueprint_validation(blueprints, config),
        *detailed_script_validation(scripts, blueprints, config),
    ]
    source_tracks = {str(row.get("source_track_id", "")) for row in rendition_targets}
    if len(source_tracks) != 1 or "" in source_tracks:
        errors.append("production assignments must contain exactly one source track")
    else:
        errors.extend(
            detailed_assignment_validation(
                {
                    "analysis_folds": folds,
                    "speaker_bundles": speaker_bundles,
                    "rendition_targets": rendition_targets,
                    "recording_order": recording_order,
                },
                config,
                next(iter(source_tracks)),
            )
        )
    if errors:
        raise ReleaseError("detailed production text validation failed:\n" + "\n".join(errors[:50]))
    return {"status": "passed", "mode": "full_repository_validators"}


def _relative_artifact_uri(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseError(f"{label}: artifact URI must be a non-empty relative path")
    if "\\" in value or "\x00" in value or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value):
        raise ReleaseError(f"{label}: artifact URI must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ReleaseError(f"{label}: artifact URI escapes the public artifact root")
    return path.as_posix()


def _public_artifact(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseError(f"{label}: artifact metadata is missing")
    uri = _relative_artifact_uri(value.get("uri"), label)
    digest = value.get("sha256")
    if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
        raise ReleaseError(f"{label}: artifact SHA-256 is invalid")
    duration = value.get("duration_ms")
    sample_rate = value.get("sample_rate")
    channels = value.get("channels")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        raise ReleaseError(f"{label}: duration_ms must be positive")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ReleaseError(f"{label}: sample_rate must be positive")
    if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
        raise ReleaseError(f"{label}: channels must be positive")
    result = {
        "uri": uri,
        "sha256": digest,
        "duration_ms": duration,
        "sample_rate": sample_rate,
        "channels": channels,
    }
    for key in ("sample_width_bytes", "codec", "timeline", "source_canonical_sha256"):
        if key in value:
            result[key] = value[key]
    return result


def _select_fields(value: Any, fields: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {key: value[key] for key in fields if key in value}


def _public_accepted_row(row: Mapping[str, Any]) -> dict[str, Any]:
    public = _select_fields(
        row,
        (
            "schema_version",
            "accepted_audio_id",
            "selected_candidate_id",
            "rendition_target_id",
            "script_id",
            "text_bundle_id",
            "matched_audio_bundle_id",
            "scenario_id",
            "direction_id",
            "condition",
            "root_slot",
            "old_value",
            "new_value",
            "pre_repair_units",
            "post_repair_units",
            "source_track_id",
            "source_type",
            "speaker_id",
            "voice",
            "lifecycle_status",
            "analysis_fold",
            "inferential_role",
            "dataset_version",
            "code_commit",
        ),
    )
    accepted_id = str(row.get("accepted_audio_id", "<missing>"))
    public["accepted_utterance"] = _public_artifact(
        row.get("accepted_utterance"), f"accepted audio {accepted_id}"
    )
    timing = row.get("timing")
    public["timing"] = dict(timing) if isinstance(timing, Mapping) else {}
    alignment = row.get("alignment")
    public["alignment"] = dict(alignment) if isinstance(alignment, Mapping) else {}
    public["synthesis"] = _select_fields(
        row.get("synthesis"),
        (
            "provider",
            "model",
            "model_version",
            "voice",
            "rate",
            "style",
            "pause_policy",
            "prosody_control",
            "ssml_template_hash",
        ),
    )
    public["selection"] = _select_fields(
        row.get("selection"),
        (
            "selection_version",
            "status",
            "policy_version",
            "policy_hash",
            "selection_score",
            "selected_candidate_id",
            "selected_canonical_sha256",
            "alignment_gate_hash",
            "materialization_mode",
            "outcome_blind",
            "tail_policy",
        ),
    )
    qc = row.get("qc")
    public["qc"] = dict(qc) if isinstance(qc, Mapping) else {}
    public["license"] = _select_fields(
        row.get("license"),
        ("identifier", "scope", "redistribution_allowed", "attribution"),
    )
    canonical = row.get("canonical_candidate")
    canonical_sha256 = canonical.get("sha256") if isinstance(canonical, Mapping) else None
    public["selected_evidence"] = {
        "candidate_id": row.get("selected_candidate_id"),
        "canonical_audio_sha256": canonical_sha256,
        "timing_sha256": sha256_value(public["timing"]),
        "alignment_sha256": sha256_value(public["alignment"]),
        "qc_sha256": sha256_value(public["qc"]),
    }
    return public


def _public_prepared_row(row: Mapping[str, Any]) -> dict[str, Any]:
    public = _select_fields(
        row,
        (
            "schema_version",
            "prepared_stimulus_id",
            "accepted_audio_id",
            "rendition_target_id",
            "script_id",
            "text_bundle_id",
            "matched_audio_bundle_id",
            "scenario_id",
            "direction_id",
            "condition",
            "source_track_id",
            "speaker_id",
            "voice",
            "lifecycle_status",
            "preparation_hash",
            "analysis_fold",
            "inferential_role",
        ),
    )
    prepared_id = str(row.get("prepared_stimulus_id", "<missing>"))
    public["prepared_stimulus"] = _public_artifact(
        row.get("prepared_stimulus"), f"prepared stimulus {prepared_id}"
    )
    public["preparation"] = _select_fields(
        row.get("preparation"),
        (
            "sample_rate",
            "prefix_silence_ms",
            "prefix_ms_actual",
            "prefix_samples",
            "mimi_frame_samples",
            "frame_pad_samples",
            "normalization_stage",
        ),
    )
    public["prepared_timing"] = _select_fields(
        row.get("prepared_timing"),
        (
            "old_value_onset_ms",
            "old_value_offset_ms",
            "repair_cue_onset_ms",
            "repair_cue_offset_ms",
            "new_value_onset_ms",
            "new_value_offset_ms",
            "repeated_old_onset_ms",
            "repeated_old_offset_ms",
            "closing_prompt_onset_ms",
            "closing_prompt_offset_ms",
            "utterance_end_ms",
            "actual_latency_ms",
            "post_final_value_duration_ms",
            "post_repair_duration_ms",
            "post_cue_duration_ms",
        ),
    )
    return public


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must be a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseError(f"{label} has the wrong schema_version")
    return value


def _declared_policy_hash(value: Mapping[str, Any], label: str) -> str:
    declared = value.get("policy_hash")
    if not isinstance(declared, str) or not HASH_RE.fullmatch(declared):
        raise ReleaseError(f"{label}.policy_hash must be a lowercase SHA-256")
    calculated = sha256_value({key: child for key, child in value.items() if key != "policy_hash"})
    if declared != calculated:
        raise ReleaseError(f"{label}.policy_hash does not match its canonical contents")
    return declared


def _validate_policy_evidence(
    paths: Mapping[str, Path], config: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], str, str]:
    evidence = {
        logical: _read_object(paths[logical], logical)
        for logical in FULL_EVIDENCE_FILES
    }
    for logical, value in evidence.items():
        private_paths = [
            value_path
            for value_path, _ in _json_values(value)
            if value_path.rsplit(".", 1)[-1].casefold() in FORBIDDEN_EVIDENCE_KEYS
        ]
        if private_paths:
            raise ReleaseError(
                f"{logical} contains forbidden private response/reviewer fields: "
                + ", ".join(private_paths[:5])
            )
    selection = evidence["selection_policy"]
    if selection.get("status") != "frozen" or not selection.get("policy_version"):
        raise ReleaseError("selection policy must have status=frozen and a policy_version")
    selection_hash = _declared_policy_hash(selection, "selection policy")
    alignment_gate_errors = validate_alignment_gate_contract(selection.get("alignment_gate"))
    if alignment_gate_errors:
        raise ReleaseError(
            "selection policy has an invalid alignment_gate: "
            + "; ".join(alignment_gate_errors)
        )

    timing = evidence["timing_policy"]
    if timing.get("status") != "frozen" or not timing.get("policy_version"):
        raise ReleaseError("timing policy must have status=frozen and a policy_version")
    timing_hash = _declared_policy_hash(timing, "timing policy")
    if timing.get("dataset_config_canonical_sha256") != sha256_value(config):
        raise ReleaseError("timing policy is not bound to the canonical dataset config")
    if timing.get("selection_policy_hash") != selection_hash:
        raise ReleaseError("timing policy is not bound to the frozen selection policy")
    return evidence, selection_hash, timing_hash


def _require_evidence_bindings(
    report: Mapping[str, Any], label: str, expected: Mapping[str, Any]
) -> None:
    for field, value in expected.items():
        if report.get(field) != value:
            raise ReleaseError(f"{label}.{field} does not match its approved input")


def _validate_gate_reports(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    selection_policy_hash: str,
    timing_policy_hash: str,
    accepted_manifest_sha256: str,
    eval_manifest_sha256: str,
    annotation_manifest_sha256: str,
    accepted_count: int,
    eval_trial_count: int,
    eval_run_id: str,
) -> dict[str, Any]:
    common = {
        "dataset_config_canonical_sha256": sha256_value(config),
        "selection_policy_hash": selection_policy_hash,
        "timing_policy_hash": timing_policy_hash,
    }
    alignment = evidence["alignment_report"]
    _require_evidence_bindings(
        alignment,
        "alignment report",
        {
            **common,
            "accepted_manifest_sha256": accepted_manifest_sha256,
            "accepted_audio_count": accepted_count,
            "eligible_alignment_count": accepted_count,
            "status": "passed",
        },
    )
    audio_qc = evidence["audio_qc_report"]
    _require_evidence_bindings(
        audio_qc,
        "audio QC report",
        {
            **common,
            "accepted_manifest_sha256": accepted_manifest_sha256,
            "accepted_audio_count": accepted_count,
            "automatic_pass_count": accepted_count,
            "unresolved_count": 0,
            "status": "passed",
        },
    )
    double_listen = evidence["double_listen_report"]
    _require_evidence_bindings(
        double_listen,
        "double-listen report",
        {
            **common,
            "accepted_manifest_sha256": accepted_manifest_sha256,
            "accepted_audio_count": accepted_count,
            "unresolved_count": 0,
            "status": "passed",
        },
    )
    reviewed_count = double_listen.get("reviewed_count")
    minimum_reviewed = math.ceil(accepted_count * 0.20)
    if (
        not isinstance(reviewed_count, int)
        or isinstance(reviewed_count, bool)
        or reviewed_count < minimum_reviewed
        or reviewed_count > accepted_count
    ):
        raise ReleaseError(
            f"double-listen report must bind at least 20% ({minimum_reviewed}) of accepted audio"
        )

    analysis_bindings = {
        **common,
        "accepted_manifest_sha256": accepted_manifest_sha256,
        "eval_manifest_sha256": eval_manifest_sha256,
        "annotation_manifest_sha256": annotation_manifest_sha256,
        "eval_trial_count": eval_trial_count,
        "eval_run_id": eval_run_id,
        "status": "completed",
    }
    _require_evidence_bindings(
        evidence["analysis_result"], "analysis result", analysis_bindings
    )
    _require_evidence_bindings(
        evidence["baseline_report"], "baseline report", analysis_bindings
    )
    return {
        "status": "passed",
        "selection_policy_hash": selection_policy_hash,
        "timing_policy_hash": timing_policy_hash,
        "double_listen_reviewed_count": reviewed_count,
        "minimum_double_listen_count": minimum_reviewed,
        "analysis_result_status": "completed",
        "baseline_status": "completed",
    }


def _validate_audio(
    accepted: Sequence[dict[str, Any]],
    prepared: Sequence[dict[str, Any]],
    target_by_id: Mapping[str, Mapping[str, Any]],
    script_by_id: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    counts: Mapping[str, int],
    selection_policy_hash: str,
    selection_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if config.get("timing", {}).get("status") != "frozen":
        raise ReleaseError("full release requires timing.status=frozen")
    source_tracks = config.get("source_tracks")
    if not isinstance(source_tracks, Mapping) or not source_tracks:
        raise ReleaseError("full release requires a configured source track")
    pending_markers = ("pending", "tbd", "unconfirmed", "not_approved")
    for source_track_id, track in source_tracks.items():
        status = str(track.get("status", "")).casefold() if isinstance(track, Mapping) else ""
        if not status or any(marker in status for marker in pending_markers):
            raise ReleaseError(f"source track {source_track_id}: release status is not approved")

    if len(accepted) != counts["rendition_targets"]:
        raise ReleaseError(
            f"accepted-audio count must be {counts['rendition_targets']}, found {len(accepted)}"
        )
    accepted_ids = _require_unique(accepted, "accepted_audio_id", "accepted audio")
    observed_targets = _require_unique(accepted, "rendition_target_id", "accepted audio")
    if observed_targets != set(target_by_id):
        raise ReleaseError("accepted audio is not one-to-one with rendition targets")
    public_accepted: list[dict[str, Any]] = []
    matched_ids: set[str] = set()
    accepted_by_id: dict[str, dict[str, Any]] = {}
    accepted_artifact_uris: set[str] = set()
    accepted_artifact_hashes: set[str] = set()
    for row in accepted:
        accepted_id = str(row["accepted_audio_id"])
        target_id = str(row["rendition_target_id"])
        target = target_by_id[target_id]
        try:
            expected_accepted_id = accepted_audio_id(target_id)
        except ValueError as error:
            raise ReleaseError(f"accepted audio {accepted_id}: invalid target ID") from error
        if accepted_id != expected_accepted_id:
            raise ReleaseError(f"accepted audio {accepted_id}: non-canonical accepted ID")
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
            "inferential_role",
        ):
            if row.get(field) != target.get(field):
                raise ReleaseError(f"accepted audio {accepted_id}: {field} does not match target")
        if row.get("schema_version") != SCHEMA_VERSION or row.get("lifecycle_status") != "accepted":
            raise ReleaseError(f"accepted audio {accepted_id}: lifecycle/schema gate failed")
        selected_candidate_id = row.get("selected_candidate_id")
        if not isinstance(selected_candidate_id, str) or not re.fullmatch(
            re.escape(target_id) + r"__cand\d{2}", selected_candidate_id
        ):
            raise ReleaseError(f"accepted audio {accepted_id}: selected candidate lineage is missing")
        matched_ids.add(str(row.get("matched_audio_bundle_id", "")))
        timing = row.get("timing")
        if not isinstance(timing, Mapping):
            raise ReleaseError(f"accepted audio {accepted_id}: timing is missing")
        for field in ("new_value_onset_ms", "new_value_offset_ms", "utterance_end_ms"):
            if not isinstance(timing.get(field), (int, float)) or isinstance(timing.get(field), bool):
                raise ReleaseError(f"accepted audio {accepted_id}: timing.{field} is missing")
        if row.get("condition") != "clean_final" and not isinstance(
            timing.get("repair_cue_onset_ms"), (int, float)
        ):
            raise ReleaseError(f"accepted audio {accepted_id}: repair cue timing is missing")
        if str(row.get("condition", "")).startswith("delayed_") and not isinstance(
            timing.get("actual_latency_ms"), (int, float)
        ):
            raise ReleaseError(f"accepted audio {accepted_id}: delayed actual latency is missing")
        alignment = row.get("alignment")
        if not isinstance(alignment, Mapping):
            raise ReleaseError(f"accepted audio {accepted_id}: alignment is missing")
        qc = row.get("qc")
        if not isinstance(qc, Mapping) or qc.get("automatic_status", qc.get("status")) != "passed":
            raise ReleaseError(f"accepted audio {accepted_id}: automatic QC did not pass")
        if qc.get("outcome_blind") is not True:
            raise ReleaseError(f"accepted audio {accepted_id}: selected QC is not outcome-blind")
        license_value = row.get("license")
        if (
            not isinstance(license_value, Mapping)
            or license_value.get("redistribution_allowed") is not True
            or not isinstance(license_value.get("identifier"), str)
            or not str(license_value.get("identifier")).strip()
            or not isinstance(license_value.get("scope"), str)
            or not str(license_value.get("scope")).strip()
        ):
            raise ReleaseError(f"accepted audio {accepted_id}: redistribution license is incomplete")
        selection = row.get("selection")
        if not isinstance(selection, Mapping):
            raise ReleaseError(f"accepted audio {accepted_id}: selection metadata is missing")
        if selection.get("policy_hash") != selection_policy_hash:
            raise ReleaseError(f"accepted audio {accepted_id}: selection policy hash mismatch")
        if selection.get("selected_candidate_id") != selected_candidate_id:
            raise ReleaseError(f"accepted audio {accepted_id}: selected candidate lineage mismatch")
        alignment_gate = selection_policy.get("alignment_gate")
        if selection.get("alignment_gate_hash") != sha256_value(alignment_gate):
            raise ReleaseError(f"accepted audio {accepted_id}: alignment gate hash mismatch")
        canonical = row.get("canonical_candidate")
        canonical_digest = canonical.get("sha256") if isinstance(canonical, Mapping) else None
        if (
            not isinstance(canonical_digest, str)
            or not HASH_RE.fullmatch(canonical_digest)
            or selection.get("selected_canonical_sha256") != canonical_digest
        ):
            raise ReleaseError(f"accepted audio {accepted_id}: selected canonical hash mismatch")
        artifact = row.get("accepted_utterance")
        public_artifact = _public_artifact(artifact, f"accepted audio {accepted_id}")
        if public_artifact.get("source_canonical_sha256") != canonical_digest:
            raise ReleaseError(f"accepted audio {accepted_id}: materialized artifact source hash mismatch")
        alignment_errors = validate_downstream_alignment_evidence(
            dict(row),
            dict(script_by_id[str(row["script_id"])]),
            dict(alignment_gate),
            actual_canonical_audio_sha256=canonical_digest,
        )
        if alignment_errors:
            raise ReleaseError(
                f"accepted audio {accepted_id}: selected alignment evidence is invalid: "
                + "; ".join(alignment_errors)
            )
        if (
            public_artifact["uri"] in accepted_artifact_uris
            or public_artifact["sha256"] in accepted_artifact_hashes
        ):
            raise ReleaseError("accepted audio artifacts must be one-to-one with rendition targets")
        accepted_artifact_uris.add(public_artifact["uri"])
        accepted_artifact_hashes.add(public_artifact["sha256"])
        accepted_by_id[accepted_id] = row
        public_accepted.append(_public_accepted_row(row))
    if len(matched_ids) != counts["matched_audio_bundles"]:
        raise ReleaseError("accepted audio does not cover the exact matched-audio bundle count")

    if len(prepared) != counts["rendition_targets"]:
        raise ReleaseError(
            f"prepared-stimulus count must be {counts['rendition_targets']}, found {len(prepared)}"
        )
    _require_unique(prepared, "prepared_stimulus_id", "prepared stimuli")
    prepared_accepted = _require_unique(prepared, "accepted_audio_id", "prepared stimuli")
    if prepared_accepted != accepted_ids:
        raise ReleaseError("prepared stimuli are not one-to-one with accepted audio")
    public_prepared: list[dict[str, Any]] = []
    prepared_artifact_uris: set[str] = set()
    prepared_artifact_hashes: set[str] = set()
    for row in prepared:
        prepared_id = str(row["prepared_stimulus_id"])
        if row.get("schema_version") != SCHEMA_VERSION or row.get("lifecycle_status") != "prepared":
            raise ReleaseError(f"prepared stimulus {prepared_id}: lifecycle/schema gate failed")
        if not isinstance(row.get("preparation_hash"), str) or not HASH_RE.fullmatch(
            str(row["preparation_hash"])
        ):
            raise ReleaseError(f"prepared stimulus {prepared_id}: preparation hash is invalid")
        accepted_id = str(row["accepted_audio_id"])
        accepted_row = accepted_by_id[accepted_id]
        try:
            expected_prepared_id = prepared_stimulus_id(
                accepted_id, str(row["preparation_hash"])
            )
        except ValueError as error:
            raise ReleaseError(f"prepared stimulus {prepared_id}: invalid canonical ID fields") from error
        if prepared_id != expected_prepared_id:
            raise ReleaseError(f"prepared stimulus {prepared_id}: non-canonical prepared ID")
        for field in (
            "rendition_target_id",
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
            "inferential_role",
        ):
            if row.get(field) != accepted_row.get(field):
                raise ReleaseError(f"prepared stimulus {prepared_id}: {field} does not match accepted audio")
        artifact = _public_artifact(
            row.get("prepared_stimulus"), f"prepared stimulus {prepared_id}"
        )
        if (
            artifact["uri"] in prepared_artifact_uris
            or artifact["sha256"] in prepared_artifact_hashes
        ):
            raise ReleaseError("prepared artifacts must be one-to-one with accepted audio")
        prepared_artifact_uris.add(artifact["uri"])
        prepared_artifact_hashes.add(artifact["sha256"])
        public_prepared.append(_public_prepared_row(row))
    public_accepted.sort(key=lambda row: str(row["accepted_audio_id"]))
    public_prepared.sort(key=lambda row: str(row["prepared_stimulus_id"]))
    return (
        {
            "status": "passed",
            "accepted_audio": len(accepted),
            "prepared_stimuli": len(prepared),
            "matched_audio_bundles": len(matched_ids),
            "relative_artifact_uris_only": True,
        },
        public_accepted,
        public_prepared,
    )


def _validate_eval(
    trials: Sequence[dict[str, Any]],
    accepted_ids: set[str],
    prepared_by_accepted: Mapping[str, Mapping[str, Any]],
    response_root: Path,
    eval_config: Mapping[str, Any],
    config: Mapping[str, Any],
    counts: Mapping[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(trials) != counts["eval_trials"]:
        raise ReleaseError(
            f"eval-trial count must be {counts['eval_trials']}, found {len(trials)}"
        )
    seeds = tuple(config["evaluation"]["generation_seeds"])
    try:
        validate_eval_trials(
            trials,
            expected_audio_ids=accepted_ids,
            expected_seeds=seeds,
        )
    except ValueError as error:
        raise ReleaseError(f"eval manifest violates the frozen runner contract: {error}") from error
    _require_unique(trials, "eval_trial_id", "eval trials")
    if set(prepared_by_accepted) != accepted_ids:
        raise ReleaseError("prepared/eval accepted-audio sets do not match")
    expected_generation_config_hash = sha256_value(eval_config)
    observed: set[tuple[str, int]] = set()
    run_ids: set[str] = set()
    identities: set[tuple[str, str, str, str]] = set()
    matrix_hashes: set[str] = set()
    execution_hashes: set[str] = set()
    public: list[dict[str, Any]] = []
    for row in trials:
        trial_id = str(row["eval_trial_id"])
        accepted_id = row.get("accepted_audio_id")
        seed = row.get("generation_seed")
        if accepted_id not in accepted_ids or seed not in seeds:
            raise ReleaseError(f"eval trial {trial_id}: invalid accepted-audio/seed cell")
        cell = (str(accepted_id), int(seed))
        if cell in observed:
            raise ReleaseError(f"eval trial {trial_id}: duplicate audio/seed cell")
        observed.add(cell)
        prepared = prepared_by_accepted[str(accepted_id)]
        if row.get("condition") != prepared.get("condition"):
            raise ReleaseError(f"eval trial {trial_id}: condition does not match prepared input")
        input_stimulus = row.get("input_stimulus")
        prepared_artifact = prepared.get("prepared_stimulus")
        preparation = prepared.get("preparation")
        if (
            not isinstance(input_stimulus, Mapping)
            or not isinstance(prepared_artifact, Mapping)
            or not isinstance(preparation, Mapping)
        ):
            raise ReleaseError(f"eval trial {trial_id}: prepared input lineage is incomplete")
        expected_input = {
            "prepared_stimulus_id": prepared.get("prepared_stimulus_id"),
            "preparation_hash": prepared.get("preparation_hash"),
            "uri": prepared_artifact.get("uri"),
            "sha256": prepared_artifact.get("sha256"),
            "duration_ms": prepared_artifact.get("duration_ms"),
            "sample_rate": prepared_artifact.get("sample_rate"),
            "channels": prepared_artifact.get("channels"),
            "sample_width_bytes": prepared_artifact.get("sample_width_bytes"),
            "timeline": prepared_artifact.get("timeline"),
            "mimi_frame_samples": preparation.get("mimi_frame_samples"),
        }
        if dict(input_stimulus) != expected_input:
            raise ReleaseError(f"eval trial {trial_id}: input stimulus does not match prepared manifest")
        capture = row.get("capture_contract")
        if (
            not isinstance(capture, Mapping)
            or capture.get("prepared_timing") != prepared.get("prepared_timing")
        ):
            raise ReleaseError(f"eval trial {trial_id}: capture timing does not match prepared manifest")
        response = row.get("response")
        if not isinstance(response, Mapping) or response.get("status") != "completed":
            raise ReleaseError(f"eval trial {trial_id}: model response is not completed")
        try:
            validate_trial_response(
                row,
                verify_audio=True,
                response_root=response_root,
            )
        except (FileNotFoundError, ValueError) as error:
            raise ReleaseError(f"eval trial {trial_id}: invalid completed response evidence: {error}") from error
        audio_path = resolve_content_uri(
            response["audio_path"], response_root, label=f"{trial_id}.response.audio_path"
        )
        if audio_path.is_symlink():
            raise ReleaseError(f"eval trial {trial_id}: response audio must not be a symlink")
        stream_events = row["stream_events"]
        run_ids.add(str(row.get("eval_run_id", "")))
        identity = tuple(
            str(row.get(field, ""))
            for field in (
                "model_repo",
                "resolved_revision",
                "generation_config_hash",
                "code_commit",
            )
        )
        if not all(identity) or not HASH_RE.fullmatch(identity[2]):
            raise ReleaseError(f"eval trial {trial_id}: run identity is incomplete")
        if identity[2] != expected_generation_config_hash:
            raise ReleaseError(f"eval trial {trial_id}: generation config hash is stale")
        identities.add(identity)
        matrix_hashes.add(sha256_value(row["matrix_contract"]))
        execution_hashes.add(sha256_value(row["execution_contract"]))
        public.append(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_run_id": row["eval_run_id"],
                "eval_trial_id": trial_id,
                "accepted_audio_id": accepted_id,
                "model_repo": row["model_repo"],
                "resolved_revision": row["resolved_revision"],
                "generation_config_hash": row["generation_config_hash"],
                "code_commit": row["code_commit"],
                "generation_seed": seed,
                "condition": row["condition"],
                "prepared_stimulus_id": input_stimulus["prepared_stimulus_id"],
                "response_status": "completed",
                "run_contract_evidence": {
                    "input_stimulus_sha256": sha256_value(input_stimulus),
                    "capture_contract_sha256": sha256_value(capture),
                    "matrix_contract_sha256": sha256_value(row["matrix_contract"]),
                    "execution_contract_sha256": sha256_value(row["execution_contract"]),
                    "runner_source_sha256": row["execution_contract"][
                        "runner_source_sha256"
                    ],
                },
                "response_evidence": {
                    "audio_sha256": response["audio_sha256"],
                    "audio_duration_ms": float(response["audio_duration_ms"]),
                    "audio_sample_rate": response["audio_sample_rate"],
                    "audio_channels": response["audio_channels"],
                    "audio_sample_width_bytes": response["audio_sample_width_bytes"],
                    "timebase": response["timebase"],
                    "stream_origin_ms": response["stream_origin_ms"],
                    "primary_window_start_ms": response["primary_window_start_ms"],
                    "requested_target_end_ms": response["requested_target_end_ms"],
                    "actual_target_end_ms": response["actual_target_end_ms"],
                    "fed_sample_count": response["fed_sample_count"],
                    "fed_frame_count": response["fed_frame_count"],
                    "output_sample_count": response["output_sample_count"],
                    "output_frame_count": response["output_frame_count"],
                    "appended_zero_sample_count": response["appended_zero_sample_count"],
                    "coverage_complete": response["coverage_complete"],
                    "eos_reached": response["eos_reached"],
                    "runner_source_sha256": response["runner_source_sha256"],
                    "effective_generation_config_sha256": response[
                        "effective_generation_config_sha256"
                    ],
                    "mimi_frame_samples": row["execution_contract"]["mimi_frame_samples"],
                    "transcript_sha256": response["transcript_sha256"],
                    "stream_events_sha256": response["stream_events_sha256"],
                    "evidence_sha256": response["evidence_sha256"],
                    "stream_event_count": len(stream_events),
                    "first_stream_event_ms": float(stream_events[0]["time_ms"]),
                    "last_stream_event_ms": float(stream_events[-1]["time_ms"]),
                },
            }
        )
    expected = {(accepted_id, seed) for accepted_id in accepted_ids for seed in seeds}
    if observed != expected:
        raise ReleaseError("eval trials are not the complete accepted-audio x seed matrix")
    if (
        len(run_ids) != 1
        or "" in run_ids
        or len(identities) != 1
        or len(matrix_hashes) != 1
        or len(execution_hashes) != 1
    ):
        raise ReleaseError("eval manifest mixes run identities")
    public.sort(key=lambda row: str(row["eval_trial_id"]))
    return (
        {
            "status": "passed",
            "eval_trials": len(trials),
            "generation_seeds": len(seeds),
            "eval_run_id": next(iter(run_ids)),
            "responses_completed": len(trials),
            "response_payloads_packaged": False,
            "matrix_contract_sha256": next(iter(matrix_hashes)),
            "execution_contract_sha256": next(iter(execution_hashes)),
        },
        public,
    )


def _decision_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    relation_labels = row.get("relation_labels")
    relations = tuple(
        relation_labels.get(relation) if isinstance(relation_labels, Mapping) else None
        for relation in ("D1", "D2", "D3")
    )
    return (
        row.get("overall_label"),
        relations,
        row.get("final_target_correct"),
        row.get("stale_state_error"),
        row.get("assistant_started_before_repair"),
    )


def _validate_annotations(
    rows: Sequence[dict[str, Any]], trial_ids: set[str], counts: Mapping[str, int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        trial_id = row.get("eval_trial_id")
        if trial_id not in trial_ids:
            raise ReleaseError(f"annotation row {index}: unexpected eval_trial_id")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ReleaseError(f"annotation row {index}: wrong schema_version")
        if not isinstance(row.get("annotator_id"), str) or not row["annotator_id"]:
            raise ReleaseError(f"annotation row {index}: annotator ID is missing")
        if not isinstance(row.get("final_target_correct"), bool) or not isinstance(
            row.get("stale_state_error"), bool
        ):
            raise ReleaseError(f"annotation row {index}: boolean decisions are incomplete")
        overall_label = row.get("overall_label")
        if overall_label not in OVERALL_LABELS:
            raise ReleaseError(f"annotation row {index}: overall_label is invalid")
        relations = row.get("relation_labels")
        if not isinstance(relations, Mapping) or set(relations) != {"D1", "D2", "D3"}:
            raise ReleaseError(f"annotation row {index}: relation labels are incomplete")
        if any(value not in RELATION_LABELS for value in relations.values()):
            raise ReleaseError(f"annotation row {index}: relation label value is invalid")
        expected_final = overall_label in {"target_only", "recovered"}
        if row["final_target_correct"] is not expected_final:
            raise ReleaseError(
                f"annotation row {index}: final_target_correct contradicts overall_label"
            )
        expected_stale = any(value in {"old_bound", "both"} for value in relations.values())
        if row["stale_state_error"] is not expected_stale:
            raise ReleaseError(
                f"annotation row {index}: stale_state_error contradicts relation labels"
            )
        early = row.get("assistant_started_before_repair")
        if early is not None and not isinstance(early, bool):
            raise ReleaseError(
                f"annotation row {index}: assistant_started_before_repair is invalid"
            )
        by_trial[str(trial_id)].append(row)
    if set(by_trial) != trial_ids:
        raise ReleaseError("annotations do not cover every eval trial")

    resolved: list[dict[str, Any]] = []
    primary_count = 0
    adjudication_count = 0
    disagreements = 0
    for trial_id in sorted(trial_ids):
        trial_rows = by_trial[trial_id]
        primary = [row for row in trial_rows if row.get("adjudicator") is False]
        adjudicators = [row for row in trial_rows if row.get("adjudicator") is True]
        if len(primary) != 2 or len({row["annotator_id"] for row in primary}) != 2:
            raise ReleaseError(
                f"eval trial {trial_id}: exactly two independent primary annotations are required"
            )
        primary_count += 2
        disagreement = _decision_signature(primary[0]) != _decision_signature(primary[1])
        if disagreement:
            disagreements += 1
            if len(adjudicators) != 1:
                raise ReleaseError(
                    f"eval trial {trial_id}: disagreement requires exactly one adjudication"
                )
            if adjudicators[0]["annotator_id"] in {row["annotator_id"] for row in primary}:
                raise ReleaseError(f"eval trial {trial_id}: adjudicator is not independent")
            selected = adjudicators[0]
            resolution_method = "adjudicated_disagreement"
            adjudication_count += 1
        else:
            if adjudicators:
                raise ReleaseError(
                    f"eval trial {trial_id}: agreeing annotations must not be adjudicated"
                )
            selected = primary[0]
            resolution_method = "independent_agreement"
        resolved.append(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_trial_id": trial_id,
                "overall_label": selected.get("overall_label"),
                "relation_labels": dict(selected["relation_labels"]),
                "final_target_correct": selected["final_target_correct"],
                "stale_state_error": selected["stale_state_error"],
                "assistant_started_before_repair": selected.get(
                    "assistant_started_before_repair"
                ),
                "resolution_method": resolution_method,
            }
        )
    if primary_count != counts["primary_annotations"]:
        raise ReleaseError(
            f"primary annotation count must be {counts['primary_annotations']}, found {primary_count}"
        )
    if len(rows) != primary_count + adjudication_count:
        raise ReleaseError("annotation manifest contains unexpected extra rows")
    return (
        {
            "status": "passed",
            "primary_annotations": primary_count,
            "disagreements": disagreements,
            "adjudications": adjudication_count,
            "resolved_annotations": len(resolved),
            "annotator_identifiers_packaged": False,
            "private_blind_map_packaged": False,
        },
        resolved,
    )


def _approval_summary(
    approval_path: Path,
    source_hashes: Mapping[str, str],
    dataset_version: str,
    git_commit: str,
) -> dict[str, Any]:
    approval = read_json(_require_regular_file(approval_path, "release approval"))
    if not isinstance(approval, Mapping):
        raise ReleaseError("release approval must be an object")
    if approval.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseError("release approval has the wrong schema_version")
    if approval.get("dataset_version") != dataset_version:
        raise ReleaseError("release approval is for another dataset version")
    if approval.get("release_kind") != FULL_AUDIO_RELEASE:
        raise ReleaseError("release approval is not for a full audio release")
    if approval.get("status") != "approved" or approval.get("public_release") is not True:
        raise ReleaseError("public release approval is not explicit")
    if approval.get("approved_git_commit") != git_commit:
        raise ReleaseError("release approval git commit does not match the requested release")
    approved_hashes = approval.get("approved_source_hashes")
    if approved_hashes != dict(source_hashes):
        raise ReleaseError("release approval source hashes do not match the exact current inputs")
    gates = approval.get("gates")
    if not isinstance(gates, Mapping):
        raise ReleaseError("release approval gates are missing")
    for gate in REQUIRED_APPROVAL_GATES:
        if gates.get(gate) not in ("passed", "approved"):
            raise ReleaseError(f"release approval gate did not pass: {gate}")
    license_value = approval.get("license")
    if not isinstance(license_value, Mapping):
        raise ReleaseError("release approval license scope is missing")
    if license_value.get("redistribution_allowed") is not True:
        raise ReleaseError("release approval does not allow redistribution")
    for field in ("identifier", "scope"):
        if not isinstance(license_value.get(field), str) or not license_value[field].strip():
            raise ReleaseError(f"release approval license.{field} is missing")
    return {
        "approval_sha256": sha256_file(approval_path),
        "status": "approved",
        "public_release": True,
        "approved_git_commit": git_commit,
        "approved_source_hashes_sha256": sha256_value(source_hashes),
        "gates": {gate: "passed" for gate in REQUIRED_APPROVAL_GATES},
        "license": {
            "identifier": license_value["identifier"],
            "scope": license_value["scope"],
            "redistribution_allowed": True,
        },
    }


def _json_values(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield child_path, child
            yield from _json_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            yield child_path, child
            yield from _json_values(child, child_path)


def _scan_json_value(value: Any, label: str, *, check_keys: bool = True) -> list[str]:
    issues: list[str] = []
    for value_path, child in _json_values(value):
        key = value_path.rsplit(".", 1)[-1].casefold()
        if check_keys and key in FORBIDDEN_JSON_KEYS:
            issues.append(f"{label}:{value_path}: forbidden private/credential field")
        if isinstance(child, str):
            if child.startswith(("/", "~/", "file://")) or re.match(r"^[A-Za-z]:[\\/]", child):
                issues.append(f"{label}:{value_path}: local absolute path")
            if EMAIL_RE.search(child):
                issues.append(f"{label}:{value_path}: possible email/PII")
            for pattern in SECRET_PATTERNS:
                if pattern.search(child):
                    issues.append(f"{label}:{value_path}: possible secret")
                    break
    return issues


def _scan_output_tree(root: Path) -> None:
    issues: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            issues.append(f"{relative}: symlinks are forbidden")
            continue
        if path.is_dir():
            continue
        lower_parts = {part.casefold() for part in PurePosixPath(relative).parts}
        if lower_parts & FORBIDDEN_PATH_PARTS:
            issues.append(f"{relative}: forbidden private/raw path")
        if path.suffix.casefold() in {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".wav", ".mp3"}:
            issues.append(f"{relative}: binary audio/model artifact is forbidden")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"{relative}: release package must contain UTF-8 metadata only")
            continue
        if LOCAL_PATH_RE.search(text):
            issues.append(f"{relative}: possible local absolute path")
        if EMAIL_RE.search(text):
            issues.append(f"{relative}: possible email/PII")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                issues.append(f"{relative}: possible secret")
                break
        try:
            if path.suffix == ".json":
                issues.extend(
                    _scan_json_value(
                        json.loads(text),
                        relative,
                        check_keys=not relative.startswith("schemas/"),
                    )
                )
            elif path.suffix == ".jsonl":
                for line_number, line in enumerate(text.splitlines(), 1):
                    if line.strip():
                        issues.extend(
                            _scan_json_value(json.loads(line), f"{relative}:{line_number}")
                        )
        except json.JSONDecodeError as error:
            issues.append(f"{relative}: invalid JSON in output: {error}")
    if issues:
        raise ReleaseError("release safety scan failed:\n" + "\n".join(issues[:50]))


def _copy_text(path: Path, destination: Path) -> None:
    _require_regular_file(path, path.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)


def _copy_canonical_jsonl(path: Path, destination: Path, sort_field: str) -> None:
    rows = read_jsonl(path)
    try:
        rows.sort(key=lambda row: str(row[sort_field]))
    except KeyError as error:
        raise ReleaseError(f"{path}: missing sort key {sort_field}") from error
    _write_jsonl(destination, rows)


def _audio_inventory(
    accepted: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for row in accepted:
        artifact = row["accepted_utterance"]
        inventory.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_role": "accepted_utterance",
                "owner_id": row["accepted_audio_id"],
                "uri": artifact["uri"],
                "sha256": artifact["sha256"],
                "duration_ms": artifact["duration_ms"],
                "sample_rate": artifact["sample_rate"],
                "channels": artifact["channels"],
            }
        )
    for row in prepared:
        artifact = row["prepared_stimulus"]
        inventory.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_role": "prepared_stimulus",
                "owner_id": row["prepared_stimulus_id"],
                "uri": artifact["uri"],
                "sha256": artifact["sha256"],
                "duration_ms": artifact["duration_ms"],
                "sample_rate": artifact["sample_rate"],
                "channels": artifact["channels"],
            }
        )
    return sorted(inventory, key=lambda row: (str(row["artifact_role"]), str(row["owner_id"])))


def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleaseError("cannot determine git commit; pass --git-commit explicitly") from error
    return result.stdout.strip().casefold()


def _validate_commit(value: str) -> str:
    normalized = value.casefold()
    if not COMMIT_RE.fullmatch(normalized):
        raise ReleaseError("git commit must be 7-64 lowercase hexadecimal characters")
    return normalized


def _payload_metadata(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in ("RELEASE_MANIFEST.json", "CHECKSUMS.sha256"):
            continue
        relative = path.relative_to(root).as_posix()
        result[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return result


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_release(
    dataset_root: Path,
    output_dir: Path,
    *,
    kind: str,
    git_commit: str,
    approval_path: Path | None = None,
) -> dict[str, Any]:
    """Validate inputs and atomically create a deterministic release directory."""

    if kind not in RELEASE_KINDS:
        raise ReleaseError(f"unknown release kind: {kind!r}")
    root = dataset_root.resolve()
    if output_dir.exists():
        raise ReleaseError(f"output path already exists; refusing to overwrite: {output_dir}")
    output_parent = output_dir.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    commit = _validate_commit(git_commit)
    version = _dataset_version(root)
    config = _dataset_config(root)
    counts = _counts(config)
    full = kind == FULL_AUDIO_RELEASE
    inputs = _source_inputs(root, full=full)
    source_hashes = {logical: sha256_file(path) for logical, path in sorted(inputs.items())}

    blueprints = read_jsonl(inputs["blueprints"])
    scripts = read_jsonl(inputs["scripts"])
    answer_keys = read_jsonl(inputs["answer_keys"])
    folds = read_jsonl(inputs["analysis_folds"])
    speaker_bundles = read_jsonl(inputs["speaker_bundles"])
    rendition_targets = read_jsonl(inputs["rendition_targets"])
    recording_order = read_jsonl(inputs["recording_order"])
    gate_results = {
        "blueprints": _validate_blueprints(blueprints, config, counts),
        "scripts": _validate_scripts(scripts, blueprints, config, counts),
        "answer_keys": _validate_answer_keys(answer_keys, scripts, counts),
        "assignments": _validate_assignments(
            folds,
            speaker_bundles,
            rendition_targets,
            recording_order,
            scripts,
            config,
            counts,
        ),
    }
    gate_results["detailed_text_validation"] = _run_production_text_validators(
        blueprints,
        scripts,
        folds,
        speaker_bundles,
        rendition_targets,
        recording_order,
        config,
        counts,
    )

    approval_summary: dict[str, Any] | None = None
    public_accepted: list[dict[str, Any]] = []
    public_prepared: list[dict[str, Any]] = []
    public_eval: list[dict[str, Any]] = []
    public_annotations: list[dict[str, Any]] = []
    if full:
        if approval_path is None:
            raise ReleaseError("full audio release requires --release-approval")
        evidence, selection_policy_hash, timing_policy_hash = _validate_policy_evidence(
            inputs, config
        )
        accepted = read_jsonl(inputs["accepted_audio"])
        prepared = read_jsonl(inputs["prepared_stimuli"])
        targets = {str(row["rendition_target_id"]): row for row in rendition_targets}
        audio_gate, public_accepted, public_prepared = _validate_audio(
            accepted,
            prepared,
            targets,
            {str(row["script_id"]): row for row in scripts},
            config,
            counts,
            selection_policy_hash,
            evidence["selection_policy"],
        )
        gate_results["audio"] = audio_gate
        accepted_ids = {str(row["accepted_audio_id"]) for row in accepted}
        trials = read_jsonl(inputs["eval_trials"])
        eval_config = read_json(inputs["config:config/eval.json"])
        if not isinstance(eval_config, Mapping):
            raise ReleaseError("eval config must be a JSON object")
        eval_gate, public_eval = _validate_eval(
            trials,
            accepted_ids,
            {str(row["accepted_audio_id"]): row for row in prepared},
            root / "evaluation/response_artifacts",
            eval_config,
            config,
            counts,
        )
        gate_results["evaluation"] = eval_gate
        annotations = read_jsonl(inputs["annotations"])
        trial_ids = {str(row["eval_trial_id"]) for row in trials}
        annotation_gate, public_annotations = _validate_annotations(
            annotations, trial_ids, counts
        )
        gate_results["annotations"] = annotation_gate
        gate_results["release_evidence"] = _validate_gate_reports(
            evidence,
            config=config,
            selection_policy_hash=selection_policy_hash,
            timing_policy_hash=timing_policy_hash,
            accepted_manifest_sha256=source_hashes["accepted_audio"],
            eval_manifest_sha256=source_hashes["eval_trials"],
            annotation_manifest_sha256=source_hashes["annotations"],
            accepted_count=len(accepted),
            eval_trial_count=len(trials),
            eval_run_id=str(eval_gate["eval_run_id"]),
        )
        approval_summary = _approval_summary(
            approval_path.resolve(), source_hashes, version, commit
        )

    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_parent)
    )
    try:
        _copy_text(inputs["version"], staging_path / "VERSION")
        for relative in DOC_FILES:
            _copy_text(inputs[f"doc:{relative}"], staging_path / relative)
        for relative in CONFIG_FILES:
            _copy_text(inputs[f"config:{relative}"], staging_path / relative)
        for relative in SCHEMA_FILES:
            _copy_text(inputs[f"schema:{relative}"], staging_path / relative)
        _copy_canonical_jsonl(inputs["blueprints"], staging_path / TEXT_MANIFEST_FILES["blueprints"], "scenario_id")
        _copy_canonical_jsonl(inputs["scripts"], staging_path / TEXT_MANIFEST_FILES["scripts"], "script_id")
        _copy_canonical_jsonl(inputs["answer_keys"], staging_path / TEXT_MANIFEST_FILES["answer_keys"], "answer_key_id")
        _copy_canonical_jsonl(inputs["analysis_folds"], staging_path / TEXT_MANIFEST_FILES["analysis_folds"], "scenario_id")
        _copy_canonical_jsonl(inputs["speaker_bundles"], staging_path / TEXT_MANIFEST_FILES["speaker_bundles"], "matched_audio_bundle_id")
        _copy_canonical_jsonl(inputs["rendition_targets"], staging_path / TEXT_MANIFEST_FILES["rendition_targets"], "rendition_target_id")
        _copy_canonical_jsonl(inputs["recording_order"], staging_path / TEXT_MANIFEST_FILES["recording_order"], "recording_order_id")

        if full:
            _copy_text(inputs["license"], staging_path / "LICENSE")
            for logical, relative in FULL_EVIDENCE_FILES.items():
                _copy_text(inputs[logical], staging_path / relative)
            _write_jsonl(staging_path / FULL_PUBLIC_FILES["accepted_audio"], public_accepted)
            _write_jsonl(staging_path / FULL_PUBLIC_FILES["prepared_stimuli"], public_prepared)
            _write_jsonl(
                staging_path / FULL_PUBLIC_FILES["audio_inventory"],
                _audio_inventory(public_accepted, public_prepared),
            )
            _write_jsonl(staging_path / FULL_PUBLIC_FILES["eval_trials"], public_eval)
            _write_jsonl(staging_path / FULL_PUBLIC_FILES["annotations"], public_annotations)
        else:
            (staging_path / "DEVELOPMENT_SNAPSHOT_NOTICE.md").write_text(
                "# Text-only development snapshot\n\n"
                "This artifact is not a public dataset release. It contains no audio, "
                "model responses, or annotations. Audio redistribution, timing, alignment, "
                "evaluation, annotation, privacy, and license gates have not been asserted "
                "by this snapshot.\n",
                encoding="utf-8",
            )

        _scan_output_tree(staging_path)
        payload = _payload_metadata(staging_path)
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "dataset_version": version,
            "dataset_id": config.get("dataset_id"),
            "release_kind": kind,
            "release_eligible": full,
            "status": (
                "approved_full_audio_release"
                if full
                else "development_snapshot_not_public_release"
            ),
            "git_commit": commit,
            "counts": {
                **counts,
                "adjudications": gate_results.get("annotations", {}).get("adjudications", 0),
            },
            "config_hashes": {
                "dataset_config_file_sha256": source_hashes["config:config/dataset.yaml"],
                "dataset_config_canonical_sha256": sha256_value(config),
                "eval_config_file_sha256": source_hashes["config:config/eval.json"],
                "value_evidence_file_sha256": source_hashes[
                    "config:config/value_evidence.json"
                ],
            },
            "source_lineage": {
                logical: {"sha256": digest} for logical, digest in sorted(source_hashes.items())
            },
            "source_lineage_sha256": sha256_value(source_hashes),
            "gate_results": gate_results,
            "approval": approval_summary,
            "privacy": {
                "raw_candidates_packaged": False,
                "provider_responses_packaged": False,
                "model_response_payloads_packaged": False,
                "private_blind_maps_packaged": False,
                "annotator_identifiers_packaged": False,
                "local_absolute_paths_allowed": False,
                "secrets_allowed": False,
                "model_weights_packaged": False,
            },
            "audio_delivery": (
                {
                    "mode": "metadata_only_relative_artifact_uris",
                    "hash_algorithm": "sha256",
                    "audio_bytes_packaged": False,
                }
                if full
                else None
            ),
            "payload_files": payload,
        }
        _write_json(staging_path / "RELEASE_MANIFEST.json", manifest)
        _write_checksums(staging_path)
        _scan_output_tree(staging_path)
        os.replace(staging_path, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2] / "dataset_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=RELEASE_KINDS, required=True)
    parser.add_argument("--git-commit")
    parser.add_argument("--release-approval", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    commit = args.git_commit or _git_commit(repo_root)
    manifest = build_release(
        args.dataset_root,
        args.output,
        kind=args.kind,
        git_commit=commit,
        approval_path=args.release_approval,
    )
    print(
        f"Built {manifest['release_kind']} dataset {manifest['dataset_version']} "
        f"metadata directory -> {args.output}"
    )


if __name__ == "__main__":
    main()
