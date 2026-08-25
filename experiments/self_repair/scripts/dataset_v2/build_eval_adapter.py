#!/usr/bin/env python3
"""Build the immutable v2 Moshi evaluation trial manifest.

This module deliberately does not import or run Moshi. It turns the prepared-
stimulus manifest and a resolved generation configuration into the exact jobs that
a model runner must execute. Keeping this boundary small lets us freeze all 3,000
trial identities before any model output is inspected.
"""

from __future__ import annotations

import argparse
import math
from pathlib import PurePosixPath
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # Support both direct script execution and package-style imports in tests.
    from .common import (
        DATASET_ROOT,
        DEFAULT_CONFIG,
        iter_duplicates,
        read_config,
        read_jsonl,
        sha256_file,
        sha256_value,
        write_json,
        write_jsonl,
    )
    from .ids import prepared_stimulus_id
    from .response_validation import (
        validate_capture_contract,
        validate_execution_contract,
        validate_input_stimulus,
        validate_matrix_contract,
        validate_relative_content_uri,
        validate_trial_response,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI use.
    from common import (
        DATASET_ROOT,
        DEFAULT_CONFIG,
        iter_duplicates,
        read_config,
        read_jsonl,
        sha256_file,
        sha256_value,
        write_json,
        write_jsonl,
    )
    from ids import prepared_stimulus_id
    from response_validation import (
        validate_capture_contract,
        validate_execution_contract,
        validate_input_stimulus,
        validate_matrix_contract,
        validate_relative_content_uri,
        validate_trial_response,
    )


SCHEMA_VERSION = "2.0.0"
ADAPTER_VERSION = "2.2.0"
RUNNER_VERSION = "2.2.0"
PRODUCTION_AUDIO_COUNT = 600
PRODUCTION_SEED_COUNT = 5
PRODUCTION_TRIAL_COUNT = 3000
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
GENERATION_PARAMETER_KEYS = {
    "use_sampling",
    "temp",
    "temp_text",
    "top_k",
    "top_k_text",
    "check",
    "support_out_of_sync",
    "cfg_is_masked_until",
    "cfg_is_no_text",
}
PREPARATION_HASH_FIELDS = (
    "sample_rate",
    "prefix_silence_ms",
    "mimi_frame_samples",
    "normalization_stage",
)
IDENTITY_FIELDS = (
    "model_repo",
    "resolved_revision",
    "generation_config_hash",
    "code_commit",
)
DEFAULT_PREPARED_MANIFEST = DATASET_ROOT / "manifests/prepared_stimuli.jsonl"
# Kept as an import alias for callers that used the early v2 adapter constant.
DEFAULT_ACCEPTED_MANIFEST = DEFAULT_PREPARED_MANIFEST
DEFAULT_OUTPUT = DATASET_ROOT / "evaluation/eval_trials.jsonl"
RUNNER_SOURCE = Path(__file__).with_name("run_eval_v2.py")


def _slug(value: str, *, maximum: int = 48) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "-", value.casefold()).strip("-")
    if not normalized:
        normalized = "value"
    return normalized[:maximum]


def generation_config_hash(generation_config: dict[str, Any]) -> str:
    """Hash the parsed config, so insignificant JSON formatting cannot change identity."""

    if not isinstance(generation_config, dict) or not generation_config:
        raise ValueError("generation_config must be a non-empty object")
    generation_parameters(generation_config)
    return sha256_value(generation_config)


def generation_parameters(generation_config: dict[str, Any]) -> dict[str, Any]:
    """Return frozen LMGen parameters and fail closed on unknown generation keys."""

    if not isinstance(generation_config, dict) or not generation_config:
        raise ValueError("generation_config must be a non-empty object")
    if "generation" in generation_config:
        parameters = generation_config["generation"]
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("generation_config.generation must be a non-empty object")
    else:
        parameters = generation_config
    unknown = sorted(set(parameters) - GENERATION_PARAMETER_KEYS)
    if unknown:
        raise ValueError(f"unknown Moshi generation parameters: {unknown}")
    return dict(parameters)


def make_eval_run_id(
    model_repo: str,
    resolved_revision: str,
    config_hash: str,
    code_commit: str,
) -> str:
    """Return a readable ID whose digest covers every run-defining field."""

    values = {
        "model_repo": model_repo,
        "resolved_revision": resolved_revision,
        "generation_config_hash": config_hash,
        "code_commit": code_commit,
    }
    for field, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
    if not re.fullmatch(r"[0-9a-f]{64}", config_hash):
        raise ValueError("generation_config_hash must be a lowercase SHA-256")
    if not FULL_SHA_RE.fullmatch(resolved_revision):
        raise ValueError("resolved_revision must be a full lowercase 40-hex commit SHA")
    if not FULL_SHA_RE.fullmatch(code_commit):
        raise ValueError("code_commit must be a full lowercase 40-hex commit SHA")

    identity_digest = sha256_value(values)
    return (
        f"eval__repo_{_slug(model_repo)}"
        f"__rev_{_slug(resolved_revision)}"
        f"__cfg_{config_hash}"
        f"__code_{_slug(code_commit)}"
        f"__id_{identity_digest[:16]}"
    )


def make_eval_trial_id(accepted_audio_id: str, eval_run_id: str, seed: int) -> str:
    if not isinstance(accepted_audio_id, str) or not accepted_audio_id:
        raise ValueError("accepted_audio_id must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("generation seed must be a non-negative integer")
    # The internal trial ID is intentionally descriptive.  It is never exposed in
    # the condition-blind annotation sheets; those use an opaque blind_id.
    return f"{accepted_audio_id}__{eval_run_id}__seed_{seed}"


def _prepared_audio_rows(
    rows: Iterable[dict[str, Any]],
    expected_audio_count: int,
    *,
    artifact_root: Path | None,
    streaming: dict[str, Any],
) -> list[dict[str, Any]]:
    prepared = list(rows)
    if len(prepared) != expected_audio_count:
        raise ValueError(
            f"expected {expected_audio_count} prepared stimulus rows, found {len(prepared)}"
        )
    required = {
        "accepted_audio_id",
        "rendition_target_id",
        "text_bundle_id",
        "matched_audio_bundle_id",
        "scenario_id",
        "direction_id",
        "condition",
        "source_track_id",
        "speaker_id",
        "prepared_stimulus_id",
        "preparation_hash",
        "prepared_stimulus",
        "preparation",
        "prepared_timing",
    }
    for index, row in enumerate(prepared):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"prepared stimulus row {index} is missing fields: {missing}")
        status = row.get("lifecycle_status")
        if status != "prepared":
            raise ValueError(
                f"prepared stimulus row {index} has lifecycle_status={status!r}; "
                "only prepared rows may enter evaluation"
            )
        if not isinstance(row["accepted_audio_id"], str) or not row["accepted_audio_id"]:
            raise ValueError(f"prepared stimulus row {index} has an invalid accepted_audio_id")
        _input_stimulus(
            row,
            index=index,
            artifact_root=artifact_root,
            streaming=streaming,
        )
    duplicates = list(
        iter_duplicates(str(row["accepted_audio_id"]) for row in prepared)
    )
    if duplicates:
        raise ValueError(f"duplicate accepted_audio_id values: {duplicates[:5]}")
    prepared_duplicates = list(
        iter_duplicates(str(row["prepared_stimulus_id"]) for row in prepared)
    )
    if prepared_duplicates:
        raise ValueError(
            f"duplicate prepared_stimulus_id values: {prepared_duplicates[:5]}"
        )
    return sorted(prepared, key=lambda row: str(row["accepted_audio_id"]))


def _portable_artifact_uri(value: Any, artifact_root: Path | None, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if path.is_absolute():
        if artifact_root is None:
            raise ValueError(f"{label} is absolute but no artifact_root was supplied")
        resolved_root = artifact_root.expanduser().resolve()
        try:
            relative = path.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"{label} is outside artifact_root") from error
        portable = PurePosixPath(*relative.parts).as_posix()
    else:
        portable = PurePosixPath(value).as_posix()
    return validate_relative_content_uri(portable, label=label)


def _input_stimulus(
    row: dict[str, Any],
    *,
    artifact_root: Path | None,
    streaming: dict[str, Any],
    index: int | None = None,
) -> dict[str, Any]:
    label = (
        f"prepared stimulus row {index}"
        if index is not None
        else str(row.get("accepted_audio_id", "prepared stimulus"))
    )
    artifact = row.get("prepared_stimulus")
    preparation = row.get("preparation")
    if not isinstance(artifact, dict):
        raise ValueError(f"{label} prepared_stimulus must be an object")
    if not isinstance(preparation, dict):
        raise ValueError(f"{label} preparation must be an object")
    missing_preparation = sorted(set(PREPARATION_HASH_FIELDS) - set(preparation))
    if missing_preparation:
        raise ValueError(f"{label} preparation is missing fields: {missing_preparation}")
    preparation_basis = {
        field: preparation[field] for field in PREPARATION_HASH_FIELDS
    }
    expected_preparation_hash = sha256_value(preparation_basis)
    if row.get("preparation_hash") != expected_preparation_hash:
        raise ValueError(f"{label} preparation_hash does not match its canonical formula")
    accepted_id = str(row.get("accepted_audio_id", ""))
    expected_prepared_id = prepared_stimulus_id(
        accepted_id, expected_preparation_hash
    )
    if row.get("prepared_stimulus_id") != expected_prepared_id:
        raise ValueError(f"{label} prepared_stimulus_id is not canonical")
    exact_stream_matches = {
        "sample_rate": "input_sample_rate",
        "mimi_frame_samples": "mimi_frame_samples",
        "prefix_silence_ms": "prefix_silence_ms",
    }
    for preparation_field, streaming_field in exact_stream_matches.items():
        if preparation[preparation_field] != streaming[streaming_field]:
            raise ValueError(
                f"{label} preparation.{preparation_field} disagrees with frozen streaming config"
            )
    value = {
        "prepared_stimulus_id": row.get("prepared_stimulus_id"),
        "preparation_hash": row.get("preparation_hash"),
        "uri": _portable_artifact_uri(
            artifact.get("uri"), artifact_root, label=f"{label}.prepared_stimulus.uri"
        ),
        "sha256": artifact.get("sha256"),
        "duration_ms": artifact.get("duration_ms"),
        "sample_rate": artifact.get("sample_rate"),
        "channels": artifact.get("channels"),
        "sample_width_bytes": artifact.get("sample_width_bytes"),
        "timeline": artifact.get("timeline"),
        "mimi_frame_samples": preparation.get("mimi_frame_samples"),
    }
    try:
        validate_input_stimulus(value, label=f"{label}.input_stimulus")
    except ValueError as error:
        raise ValueError(str(error)) from error
    return value


def _streaming_contract(
    generation_config: dict[str, Any], generation_seeds: tuple[int, ...]
) -> dict[str, Any]:
    declared_seeds = generation_config.get("generation_seeds")
    if declared_seeds != list(generation_seeds):
        raise ValueError(
            "eval config generation_seeds must exactly equal the ordered dataset seeds"
        )
    streaming = generation_config.get("streaming")
    if not isinstance(streaming, dict):
        raise ValueError("generation_config.streaming must be an object")
    required = {
        "input_sample_rate",
        "mimi_frame_samples",
        "prefix_silence_ms",
        "response_capture_ms",
        "reset_model_stream_between_trials",
        "reset_rng_for_each_trial_seed",
    }
    if set(streaming) != required:
        raise ValueError(
            f"generation_config.streaming must contain exactly {sorted(required)}"
        )
    for field in ("input_sample_rate", "mimi_frame_samples"):
        value = streaming[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"streaming.{field} must be a positive integer")
    for field in ("prefix_silence_ms", "response_capture_ms"):
        value = streaming[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"streaming.{field} must be finite and positive")
    if streaming["reset_model_stream_between_trials"] is not True:
        raise ValueError("streaming config must reset model state between trials")
    if streaming["reset_rng_for_each_trial_seed"] is not True:
        raise ValueError("streaming config must reset RNG for every trial seed")
    return dict(streaming)


def _capture_contract(
    row: dict[str, Any], input_stimulus: dict[str, Any], streaming: dict[str, Any]
) -> dict[str, Any]:
    timing = row.get("prepared_timing")
    if not isinstance(timing, dict):
        raise ValueError(f"{row.get('accepted_audio_id')}: prepared_timing must be an object")
    utterance_end = timing.get("utterance_end_ms")
    primary_start = timing.get("closing_prompt_offset_ms")
    if not isinstance(utterance_end, (int, float)) or isinstance(utterance_end, bool):
        raise ValueError(f"{row.get('accepted_audio_id')}: invalid utterance_end_ms")
    if not isinstance(primary_start, (int, float)) or isinstance(primary_start, bool):
        raise ValueError(f"{row.get('accepted_audio_id')}: invalid closing_prompt_offset_ms")
    response_capture_ms = float(streaming["response_capture_ms"])
    requested_end_ms = float(utterance_end) + response_capture_ms
    sample_rate = int(input_stimulus["sample_rate"])
    frame_samples = int(input_stimulus["mimi_frame_samples"])
    target_frames = math.ceil(
        requested_end_ms * sample_rate / (1000.0 * frame_samples) - 1e-12
    )
    target_samples = target_frames * frame_samples
    contract = {
        "condition": row.get("condition"),
        "timebase": "prepared_stream_relative",
        "stream_origin_ms": 0,
        "prepared_timing": dict(timing),
        "prepared_timing_sha256": sha256_value(timing),
        "primary_window_start_ms": primary_start,
        "utterance_end_ms": utterance_end,
        "response_capture_ms": response_capture_ms,
        "requested_target_end_ms": requested_end_ms,
        "target_end_sample_count": target_samples,
        "target_end_frame_count": target_frames,
        "actual_target_end_ms": target_samples * 1000.0 / sample_rate,
    }
    validate_capture_contract(contract, input_stimulus)
    return contract


def _execution_contract(streaming: dict[str, Any]) -> dict[str, Any]:
    if not RUNNER_SOURCE.is_file():
        raise FileNotFoundError(f"evaluation runner source is missing: {RUNNER_SOURCE}")
    contract = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": sha256_file(RUNNER_SOURCE),
        **streaming,
        "required_model_type": "moshi",
        "required_max_lm_delay": 1,
    }
    validate_execution_contract(contract)
    return contract


def _generation_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(seeds)
    if not normalized:
        raise ValueError("at least one generation seed is required")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in normalized):
        raise ValueError("generation seeds must be non-negative integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("generation seeds must be unique")
    return normalized


def _matrix_contract(
    frozen_audio: Sequence[dict[str, Any]],
    seeds: tuple[int, ...],
    *,
    production_matrix: bool,
) -> dict[str, Any]:
    accepted_ids = sorted(str(row["accepted_audio_id"]) for row in frozen_audio)
    prepared_ids = sorted(
        str(row["input_stimulus"]["prepared_stimulus_id"]) for row in frozen_audio
    )
    content_projection = [
        {
            "accepted_audio_id": row["accepted_audio_id"],
            "condition": row["condition"],
            "input_stimulus": row["input_stimulus"],
            "capture_contract": row["capture_contract"],
        }
        for row in sorted(frozen_audio, key=lambda item: str(item["accepted_audio_id"]))
    ]
    contract = {
        "production_matrix": production_matrix,
        "accepted_audio_count": len(accepted_ids),
        "generation_seeds": list(seeds),
        "eval_trial_count": len(accepted_ids) * len(seeds),
        "accepted_audio_set_sha256": sha256_value(accepted_ids),
        "prepared_stimulus_set_sha256": sha256_value(prepared_ids),
        "prepared_manifest_content_sha256": sha256_value(content_projection),
    }
    validate_matrix_contract(contract)
    return contract


def build_eval_trials(
    accepted_audio: Iterable[dict[str, Any]],
    *,
    model_repo: str,
    resolved_revision: str,
    generation_config: dict[str, Any],
    code_commit: str,
    generation_seeds: Sequence[int],
    expected_audio_count: int = 600,
    artifact_root: Path | None = None,
    allow_nonproduction_matrix: bool = False,
) -> list[dict[str, Any]]:
    """Create one pending eval trial for every accepted-audio × seed cell."""

    seeds = _generation_seeds(generation_seeds)
    if not allow_nonproduction_matrix and (
        expected_audio_count != PRODUCTION_AUDIO_COUNT
        or len(seeds) != PRODUCTION_SEED_COUNT
    ):
        raise ValueError("production evaluation must be exactly 600 audio x 5 seeds")
    streaming = _streaming_contract(generation_config, seeds)
    accepted = _prepared_audio_rows(
        accepted_audio,
        expected_audio_count,
        artifact_root=artifact_root,
        streaming=streaming,
    )
    config_hash = generation_config_hash(generation_config)
    run_id = make_eval_run_id(
        model_repo, resolved_revision, config_hash, code_commit
    )
    execution_contract = _execution_contract(streaming)
    frozen_audio: list[dict[str, Any]] = []
    for audio in accepted:
        accepted_id = str(audio["accepted_audio_id"])
        input_stimulus = _input_stimulus(
            audio, artifact_root=artifact_root, streaming=streaming
        )
        frozen_audio.append(
            {
                "accepted_audio_id": accepted_id,
                "condition": audio["condition"],
                "input_stimulus": input_stimulus,
                "capture_contract": _capture_contract(
                    audio, input_stimulus, streaming
                ),
            }
        )
    matrix_contract = _matrix_contract(
        frozen_audio, seeds, production_matrix=not allow_nonproduction_matrix
    )
    trials: list[dict[str, Any]] = []
    for frozen in frozen_audio:
        accepted_id = str(frozen["accepted_audio_id"])
        for seed in seeds:
            trials.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "eval_run_id": run_id,
                    "eval_trial_id": make_eval_trial_id(accepted_id, run_id, seed),
                    "accepted_audio_id": accepted_id,
                    "model_repo": model_repo,
                    "resolved_revision": resolved_revision,
                    "generation_config_hash": config_hash,
                    "code_commit": code_commit,
                    "generation_seed": seed,
                    "condition": frozen["condition"],
                    "input_stimulus": frozen["input_stimulus"],
                    "capture_contract": frozen["capture_contract"],
                    "matrix_contract": matrix_contract,
                    "execution_contract": execution_contract,
                    "response": {"status": "pending"},
                }
            )
    validate_eval_trials(
        trials,
        expected_audio_ids={str(row["accepted_audio_id"]) for row in accepted},
        expected_seeds=seeds,
    )
    return trials


def eval_identity(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    if not rows:
        raise ValueError("eval trial manifest is empty")
    identities = {
        tuple(str(row.get(field, "")) for field in IDENTITY_FIELDS) for row in rows
    }
    run_ids = {str(row.get("eval_run_id", "")) for row in rows}
    if len(identities) != 1 or len(run_ids) != 1:
        raise ValueError("eval trial manifest mixes multiple run identities")
    values = next(iter(identities))
    identity = dict(zip(IDENTITY_FIELDS, values))
    identity["eval_run_id"] = next(iter(run_ids))
    expected_run_id = make_eval_run_id(
        identity["model_repo"],
        identity["resolved_revision"],
        identity["generation_config_hash"],
        identity["code_commit"],
    )
    if identity["eval_run_id"] != expected_run_id:
        raise ValueError("eval_run_id does not match its full run identity")
    return identity


def validate_eval_trials(
    trials: Sequence[dict[str, Any]],
    *,
    expected_audio_ids: set[str] | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> None:
    """Validate uniqueness and the complete audio × seed Cartesian product."""

    identity = eval_identity(trials)
    trial_ids: list[str] = []
    observed: set[tuple[str, int]] = set()
    audio_ids: set[str] = set()
    seeds: set[int] = set()
    observed_order: list[tuple[str, int]] = []
    input_by_audio: dict[str, dict[str, Any]] = {}
    capture_by_audio: dict[str, dict[str, Any]] = {}
    condition_by_audio: dict[str, str] = {}
    prepared_owner: dict[str, str] = {}
    matrix_contracts: dict[str, dict[str, Any]] = {}
    execution_contracts: dict[str, dict[str, Any]] = {}
    required = {
        "schema_version",
        "eval_run_id",
        "eval_trial_id",
        "accepted_audio_id",
        *IDENTITY_FIELDS,
        "generation_seed",
        "condition",
        "input_stimulus",
        "capture_contract",
        "matrix_contract",
        "execution_contract",
        "response",
    }
    for index, row in enumerate(trials):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"eval trial row {index} is missing fields: {missing}")
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"eval trial row {index} has the wrong schema_version")
        accepted_id = row["accepted_audio_id"]
        seed = row["generation_seed"]
        if not isinstance(accepted_id, str) or not accepted_id:
            raise ValueError(f"eval trial row {index} has an invalid accepted_audio_id")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"eval trial row {index} has an invalid generation_seed")
        if not isinstance(row["response"], dict):
            raise ValueError(f"eval trial row {index} response must be an object")
        validate_input_stimulus(
            row["input_stimulus"], label=f"eval trial row {index}.input_stimulus"
        )
        if row["condition"] not in (
            "clean_final",
            "immediate_repair",
            "delayed_neutral",
            "delayed_one_dependency",
            "delayed_three_dependencies",
        ):
            raise ValueError(f"eval trial row {index} has an invalid condition")
        validate_capture_contract(
            row["capture_contract"],
            row["input_stimulus"],
            label=f"eval trial row {index}.capture_contract",
        )
        if row["capture_contract"]["condition"] != row["condition"]:
            raise ValueError(f"eval trial row {index} condition/capture mismatch")
        validate_matrix_contract(
            row["matrix_contract"], label=f"eval trial row {index}.matrix_contract"
        )
        validate_execution_contract(
            row["execution_contract"],
            label=f"eval trial row {index}.execution_contract",
        )
        validate_trial_response(row)
        frozen_input = dict(row["input_stimulus"])
        prior_input = input_by_audio.get(accepted_id)
        if prior_input is not None and prior_input != frozen_input:
            raise ValueError(
                f"eval trial row {index} changes prepared input across seeds for {accepted_id}"
            )
        input_by_audio[accepted_id] = frozen_input
        frozen_capture = dict(row["capture_contract"])
        prior_capture = capture_by_audio.get(accepted_id)
        if prior_capture is not None and prior_capture != frozen_capture:
            raise ValueError(
                f"eval trial row {index} changes capture contract across seeds"
            )
        capture_by_audio[accepted_id] = frozen_capture
        prior_condition = condition_by_audio.get(accepted_id)
        if prior_condition is not None and prior_condition != row["condition"]:
            raise ValueError(f"eval trial row {index} changes condition across seeds")
        condition_by_audio[accepted_id] = str(row["condition"])
        matrix_contracts[sha256_value(row["matrix_contract"])] = dict(
            row["matrix_contract"]
        )
        execution_contracts[sha256_value(row["execution_contract"])] = dict(
            row["execution_contract"]
        )
        prepared_id = str(frozen_input["prepared_stimulus_id"])
        prior_owner = prepared_owner.get(prepared_id)
        if prior_owner is not None and prior_owner != accepted_id:
            raise ValueError(
                f"prepared_stimulus_id {prepared_id!r} is reused by multiple accepted audio IDs"
            )
        prepared_owner[prepared_id] = accepted_id
        expected_trial_id = make_eval_trial_id(
            accepted_id, identity["eval_run_id"], seed
        )
        if row["eval_trial_id"] != expected_trial_id:
            raise ValueError(f"eval trial row {index} has a non-canonical eval_trial_id")
        trial_ids.append(str(row["eval_trial_id"]))
        cell = (accepted_id, seed)
        if cell in observed:
            raise ValueError(f"duplicate accepted-audio/seed cell: {cell}")
        observed.add(cell)
        observed_order.append(cell)
        audio_ids.add(accepted_id)
        seeds.add(seed)
    duplicate_ids = list(iter_duplicates(trial_ids))
    if duplicate_ids:
        raise ValueError(f"duplicate eval_trial_id values: {duplicate_ids[:5]}")

    if len(matrix_contracts) != 1:
        raise ValueError("eval trial manifest mixes matrix contracts")
    if len(execution_contracts) != 1:
        raise ValueError("eval trial manifest mixes execution contracts")
    matrix_contract = next(iter(matrix_contracts.values()))
    execution_contract = next(iter(execution_contracts.values()))
    if matrix_contract["accepted_audio_count"] != len(audio_ids):
        raise ValueError("matrix contract accepted-audio count mismatch")
    if matrix_contract["eval_trial_count"] != len(trials):
        raise ValueError("matrix contract eval-trial count mismatch")
    if execution_contract["input_sample_rate"] != next(
        iter(input_by_audio.values())
    )["sample_rate"]:
        raise ValueError("execution contract sample rate disagrees with inputs")
    if any(
        item["sample_rate"] != execution_contract["input_sample_rate"]
        or item["mimi_frame_samples"] != execution_contract["mimi_frame_samples"]
        for item in input_by_audio.values()
    ):
        raise ValueError("prepared inputs disagree with execution streaming contract")
    accepted_ids_sorted = sorted(audio_ids)
    prepared_ids_sorted = sorted(
        str(value["prepared_stimulus_id"]) for value in input_by_audio.values()
    )
    content_projection = [
        {
            "accepted_audio_id": accepted_id,
            "condition": condition_by_audio[accepted_id],
            "input_stimulus": input_by_audio[accepted_id],
            "capture_contract": capture_by_audio[accepted_id],
        }
        for accepted_id in accepted_ids_sorted
    ]
    expected_hashes = {
        "accepted_audio_set_sha256": sha256_value(accepted_ids_sorted),
        "prepared_stimulus_set_sha256": sha256_value(prepared_ids_sorted),
        "prepared_manifest_content_sha256": sha256_value(content_projection),
    }
    for field, expected_hash in expected_hashes.items():
        if matrix_contract[field] != expected_hash:
            raise ValueError(f"matrix contract {field} mismatch")

    wanted_audio = expected_audio_ids if expected_audio_ids is not None else audio_ids
    contract_seeds = tuple(matrix_contract["generation_seeds"])
    wanted_seed_order = (
        _generation_seeds(expected_seeds)
        if expected_seeds is not None
        else _generation_seeds(contract_seeds)
    )
    if tuple(contract_seeds) != wanted_seed_order:
        raise ValueError("matrix contract ordered seeds mismatch")
    wanted_seeds = set(wanted_seed_order)
    expected_cells = {(audio_id, seed) for audio_id in wanted_audio for seed in wanted_seeds}
    if observed != expected_cells:
        missing = len(expected_cells - observed)
        extra = len(observed - expected_cells)
        raise ValueError(
            f"eval trials are not a complete audio × seed matrix: missing={missing}, extra={extra}"
        )
    expected_order = [
        (audio_id, seed)
        for audio_id in sorted(wanted_audio)
        for seed in wanted_seed_order
    ]
    if observed_order != expected_order:
        raise ValueError("eval trials are not in canonical accepted-audio x ordered-seed order")


def guard_output_reuse(output_path: Path, new_trials: Sequence[dict[str, Any]]) -> None:
    """Reject an existing path if its full run identity differs in any field."""

    if not output_path.exists() or output_path.stat().st_size == 0:
        return
    existing = read_jsonl(output_path)
    old_identity = eval_identity(existing)
    new_identity = eval_identity(new_trials)
    if old_identity != new_identity:
        changed = [
            field
            for field in (*IDENTITY_FIELDS, "eval_run_id")
            if old_identity.get(field) != new_identity.get(field)
        ]
        raise RuntimeError(
            "refusing to reuse eval output with a different run identity; "
            f"changed fields: {', '.join(changed)}"
        )
    immutable_fields = {
        str(row["eval_trial_id"]): {
            key: value
            for key, value in row.items()
            if key not in {"response", "stream_events"}
        }
        for row in existing
    }
    new_immutable_fields = {
        str(row["eval_trial_id"]): {
            key: value
            for key, value in row.items()
            if key not in {"response", "stream_events"}
        }
        for row in new_trials
    }
    if immutable_fields != new_immutable_fields:
        raise RuntimeError(
            "refusing to reuse eval output because trial IDs, seeds, or prepared "
            "stimulus evidence differ"
        )
    if any(row.get("response", {}).get("status") == "completed" for row in existing):
        if list(existing) != list(new_trials):
            raise RuntimeError(
                "refusing to overwrite completed evaluation progress with a pending manifest"
            )


def write_eval_manifest(
    output_path: Path,
    trials: Sequence[dict[str, Any]],
    *,
    report_path: Path | None = None,
) -> None:
    validate_eval_trials(trials)
    guard_output_reuse(output_path, trials)
    write_jsonl(output_path, trials)
    if report_path is not None:
        identity = eval_identity(trials)
        write_json(
            report_path,
            {
                "schema_version": SCHEMA_VERSION,
                "adapter_version": ADAPTER_VERSION,
                "identity": identity,
                "counts": {
                    "accepted_audio": len({row["accepted_audio_id"] for row in trials}),
                    "generation_seeds": len({row["generation_seed"] for row in trials}),
                    "eval_trials": len(trials),
                },
                "manifest_hash": sha256_value(list(trials)),
                "status": "pending_model_execution",
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-manifest",
        "--accepted-manifest",
        dest="prepared_manifest",
        type=Path,
        default=DEFAULT_PREPARED_MANIFEST,
    )
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root used to freeze prepared stimulus URIs as portable relative paths.",
    )
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--resolved-revision", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = read_config(args.dataset_config)
    generation_config = read_config(args.generation_config)
    accepted = read_jsonl(args.prepared_manifest)
    seeds = dataset_config["evaluation"]["generation_seeds"]
    expected_audio = int(dataset_config["counts"]["rendition_targets_per_track"])
    trials = build_eval_trials(
        accepted,
        model_repo=args.model_repo,
        resolved_revision=args.resolved_revision,
        generation_config=generation_config,
        code_commit=args.code_commit,
        generation_seeds=seeds,
        expected_audio_count=expected_audio,
        artifact_root=args.artifact_root,
    )
    report_path = args.report or args.output.with_suffix(".report.json")
    write_eval_manifest(args.output, trials, report_path=report_path)
    print(f"Wrote {len(trials)} pending eval trials -> {args.output}")
    print(f"Run identity -> {eval_identity(trials)['eval_run_id']}")


if __name__ == "__main__":
    main()
