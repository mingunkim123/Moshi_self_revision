from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import wave

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.self_repair.mechanistic import HARNESS_VERSION
from experiments.self_repair.mechanistic.conversation import (
    DATASET_V2_CONTRACT_SOURCE,
    NATURAL_START_STATUS,
    REVIEWED_MULTIVALUE_CONTRACT_SOURCE,
    REQUIRED_EXPERIMENTAL_STARTUP_MODES,
    RESPONSE_CAPTURE_FRAMES,
    STARTUP_MODE_NATURAL,
    TAIL_GUARD_FRAMES,
    ConversationContract,
    ConversationContractError,
    diagnose_response_boundaries,
)
from experiments.self_repair.mechanistic.audio_activity import (
    diagnose_audio_tail,
    frame_rms_dbfs,
)
from experiments.self_repair.mechanistic.blinding import BlindAssignmentStore
from experiments.self_repair.mechanistic.causal_scan import (
    CausalCellPlan,
    DonorAssignment,
    PathSpecification,
    active_arms,
    exact_anchor_frame,
    intervention_sites,
    materialize_cell_grid,
    materialize_donor_assignments,
    parse_path_specification,
    repair_recipients,
    trial_metadata,
)
from experiments.self_repair.mechanistic.core import (
    AtomicCellStore,
    ContractError,
    FRAME_MS,
    FRAME_SAMPLES,
    MODEL_REPO,
    MODEL_REVISION,
    PatchCell,
    REQUIRED_SITES,
    SAMPLE_RATE,
    anchor_rows,
    apply_probe,
    bootstrap_mean_ci,
    build_run_identity,
    canonical_json,
    deterministic_derangement,
    fit_ridge_probe,
    freeze_selection,
    holm_adjust,
    package_tree,
    read_json,
    read_jsonl,
    require_relative_uri,
    sha256_file,
    sha256_value,
    validate_runtime_environment,
    validate_sha256,
    verify_archive,
    write_csv,
    write_json,
    write_jsonl,
)
from experiments.self_repair.mechanistic.response_window import primary_response_window
from experiments.self_repair.mechanistic.runtime import (
    FROZEN_AUDIO_ACTIVITY_THRESHOLD_DBFS,
    FROZEN_GREETING_MAX_FRAMES,
    FROZEN_GREETING_QUIET_FRAMES,
    FROZEN_PREPARED_LEADIN_FRAMES,
    GeneratedSequence,
    MoshiBackend,
    PairedGeneration,
    SyntheticBackend,
)
from experiments.self_repair.mechanistic.probes import (
    PROBE_APPLICATION_ROLES,
    PROBE_TRAINING_ROLES,
    apply_frozen_probe,
    fit_grouped_ridge_probe,
    freeze_probe_report,
    validate_frozen_probe_artifact,
)
from experiments.self_repair.mechanistic.analysis_protocol import (
    analyze_frozen_contrasts,
    freeze_analysis_artifacts,
    load_frozen_analysis_inputs,
)
from experiments.self_repair.mechanistic.readiness import (
    FROZEN_AUDIO_ACTIVITY_POLICY,
    FROZEN_AUDIO_ACTIVITY_POLICY_SHA256,
    ReadinessError,
    estimate_workload,
    target_binding_sha256,
    verify_authorization_artifact,
)
from experiments.self_repair.mechanistic.scripts.readiness_cli import (
    build_target_binding_from_files,
    validate_scan_execution,
)
from experiments.self_repair.mechanistic.verification import (
    package_checksum_manifest,
    verify_analysis_provenance,
    verify_artifact_manifest,
    verify_or_create_artifact_manifest,
    verify_package_checksums,
    verify_patch_artifacts,
)


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--synthetic", action="store_true", help="Use analytic fixtures; never empirical evidence.")
    return parser


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ints(value: str) -> list[int]:
    if ":" in value and "," not in value:
        start, stop = (int(item) for item in value.split(":"))
        return list(range(start, stop))
    return [int(item) for item in _csv(value)]


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()


def _rows_or_empty(path: Path | None) -> list[dict[str, Any]]:
    return read_jsonl(path) if path is not None and path.exists() else []


def _infer_run_file(output_root: Path, relative: str) -> Path | None:
    for root in (output_root, *output_root.parents):
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _code_array(
    value: Any, *, label: str, frames: int, codebooks: int = 8,
) -> np.ndarray:
    array = _numpy(value)
    if array.ndim != 3 or tuple(array.shape[:2]) != (1, codebooks):
        raise ContractError(
            f"{label} must have shape [1, {codebooks}, {frames}], got {list(array.shape)}"
        )
    if int(array.shape[-1]) != frames:
        raise ContractError(
            f"{label} does not exactly cover [0, {frames}): {array.shape[-1]} frames"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ContractError(f"{label} must contain integer Mimi codes")
    if bool((array < 0).any()):
        raise ContractError(f"{label} contains a negative Mimi code")
    return np.ascontiguousarray(array)


def _atomic_savez(path: Path, **arrays: Any) -> None:
    """Atomically commit a compressed NumPy archive without suffix rewriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _archive_identity(archive: Any, path: Path) -> str:
    if "artifact_identity_sha256" not in archive.files:
        raise ContractError(f"encoded archive has no embedded identity: {path}")
    value = np.asarray(archive["artifact_identity_sha256"])
    if value.shape != () or not isinstance(value.item(), str):
        raise ContractError(f"encoded archive identity is malformed: {path}")
    return validate_sha256(value.item(), f"embedded artifact identity for {path}")


def _safe_artifact_uri(path: Path, manifest_path: Path) -> str:
    try:
        relative = path.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
    except ValueError as error:
        raise ContractError(
            "artifact output root must be contained by the output-manifest directory"
        ) from error
    return require_relative_uri(relative)


def _exact_int_field(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ContractError(f"{label} must be an integer")
    return int(value)


def _frozen_model_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model = config.get("model")
    audio = config.get("audio")
    if not isinstance(model, Mapping) or not isinstance(audio, Mapping):
        raise ContractError("config must contain model and audio objects")
    if model.get("repo") != MODEL_REPO or model.get("revision") != MODEL_REVISION:
        raise ContractError("config model identity differs from the frozen Moshiko checkpoint")
    expected_audio = {
        "sample_rate": SAMPLE_RATE,
        "mimi_frame_samples": FRAME_SAMPLES,
        "frame_ms": FRAME_MS,
    }
    if any(audio.get(key) != value for key, value in expected_audio.items()):
        raise ContractError("config audio identity differs from the frozen Mimi timebase")
    return model


def _anchor_lookup(path: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows = read_jsonl(path)
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        trial_id, anchor = str(row.get("trial_id", "")), str(row.get("anchor", ""))
        if not trial_id or not anchor:
            raise ContractError("anchor map contains a missing trial_id or anchor")
        key = (trial_id, anchor)
        if key in lookup:
            raise ContractError(f"anchor map has a duplicate binding: {trial_id}:{anchor}")
        frame = _exact_int_field(row.get("frame"), f"{trial_id}:{anchor} frame")
        if frame < 0:
            raise ContractError(f"{trial_id}:{anchor} frame must be non-negative")
        lookup[key] = dict(row)
    if not lookup:
        raise ContractError("anchor map is empty")
    return rows, lookup


def _logmeanexp(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ContractError("readout aggregation requires finite schedule scores")
    maximum = float(array.max())
    return maximum + math.log(float(np.exp(array - maximum).mean()))


def _trial_values(row: Mapping[str, Any]) -> tuple[str, str]:
    if row.get("old_value") and row.get("new_value"):
        return str(row["old_value"]), str(row["new_value"])
    direction = str(row.get("direction_id", "a_to_b"))
    return ("Boston", "Seattle") if direction == "a_to_b" else ("Seattle", "Boston")


def _conversation_contract(
    source_row: Mapping[str, Any], *, trial_id: str, prepared_frame_count: int,
) -> dict[str, Any] | None:
    """Preserve the frozen v2 stream/capture contract without reinterpretation."""
    capture = source_row.get("capture_contract")
    execution = source_row.get("execution_contract")
    if capture is None and execution is None:
        return None
    if not isinstance(capture, Mapping) or not isinstance(execution, Mapping):
        raise ContractError("source trial has only one half of the conversation contract")
    contract_source = source_row.get(
        "conversation_contract_source", DATASET_V2_CONTRACT_SOURCE
    )
    if contract_source not in {
        DATASET_V2_CONTRACT_SOURCE,
        REVIEWED_MULTIVALUE_CONTRACT_SOURCE,
    }:
        raise ContractError("source trial has an unknown conversation contract provenance")
    required_capture = {
        "timebase", "stream_origin_ms", "prepared_timing", "prepared_timing_sha256",
        "utterance_end_ms", "primary_window_start_ms", "response_capture_ms",
        "requested_target_end_ms", "target_end_sample_count", "target_end_frame_count",
        "actual_target_end_ms",
    }
    required_execution = {
        "input_sample_rate", "mimi_frame_samples", "prefix_silence_ms",
        "response_capture_ms",
        "required_max_lm_delay", "required_model_type",
        "reset_model_stream_between_trials", "reset_rng_for_each_trial_seed",
    }
    missing = sorted(required_capture - set(capture)) + sorted(required_execution - set(execution))
    if missing:
        raise ContractError(f"source conversation contract is missing fields: {missing}")
    if int(execution["input_sample_rate"]) != 24000 or int(execution["mimi_frame_samples"]) != FRAME_SAMPLES:
        raise ContractError("source conversation contract is not mono 24 kHz / 1,920-sample Mimi")
    if execution["required_model_type"] != "moshi" or int(execution["required_max_lm_delay"]) != 1:
        raise ContractError("source conversation contract is not the pinned Moshiko streaming contract")
    if execution["reset_model_stream_between_trials"] is not True or execution["reset_rng_for_each_trial_seed"] is not True:
        raise ContractError("source conversation contract must reset stream and RNG per trial")
    if capture["timebase"] != "prepared_stream_relative" or capture["stream_origin_ms"] != 0:
        raise ContractError("source capture contract has an unsupported timebase")
    if sha256_value(capture["prepared_timing"]) != capture["prepared_timing_sha256"]:
        raise ContractError("source prepared timing hash mismatch")
    target_samples = int(capture["target_end_sample_count"])
    target_frames = int(capture["target_end_frame_count"])
    if target_samples != target_frames * FRAME_SAMPLES or target_frames < prepared_frame_count:
        raise ContractError("source capture target frame/sample counts are inconsistent")
    utterance_end_ms = float(capture["utterance_end_ms"])
    query_end_exclusive = int(math.ceil(utterance_end_ms / FRAME_MS - 1e-12))
    if query_end_exclusive <= 0 or query_end_exclusive > prepared_frame_count:
        raise ContractError("source query end is outside the prepared audio")
    prefix_ms = float(execution["prefix_silence_ms"])
    prefix_frames = int(round(prefix_ms / FRAME_MS))
    if abs(prefix_frames * FRAME_MS - prefix_ms) > 1e-6:
        raise ContractError("prefix silence is not Mimi-frame aligned")
    if prefix_frames >= query_end_exclusive:
        raise ContractError("source user prefix does not precede the semantic query end")
    response_capture_ms = float(capture["response_capture_ms"])
    if abs(float(execution["response_capture_ms"]) - response_capture_ms) > 1e-6:
        raise ContractError("source capture and execution response horizons disagree")
    response_capture_frames_float = response_capture_ms / FRAME_MS
    response_capture_frames = int(round(response_capture_frames_float))
    if abs(response_capture_frames_float - response_capture_frames) > 1e-9:
        raise ContractError("source response horizon is not Mimi-frame aligned")
    if response_capture_frames != RESPONSE_CAPTURE_FRAMES:
        raise ContractError("source response horizon must be exactly 40 seconds / 500 frames")
    requested_end = utterance_end_ms + response_capture_ms
    if abs(float(capture["primary_window_start_ms"]) - utterance_end_ms) > 1e-6:
        raise ContractError("source primary window does not start at the semantic query end")
    if abs(float(capture["requested_target_end_ms"]) - requested_end) > 1e-6:
        raise ContractError("source requested capture end is inconsistent")
    if target_frames != query_end_exclusive + response_capture_frames:
        raise ContractError("source capture frame horizon does not preserve the full response window")
    if abs(float(capture["actual_target_end_ms"]) - target_frames * FRAME_MS) > 1e-6:
        raise ContractError("source actual capture end disagrees with its frame count")
    return {
        "schema_version": "1.0.0",
        "source": contract_source,
        "trial_id": trial_id,
        "startup_mode": STARTUP_MODE_NATURAL,
        "startup_status": NATURAL_START_STATUS,
        "required_startup_modes": list(REQUIRED_EXPERIMENTAL_STARTUP_MODES),
        "file_replay_startup": "prime_once_then_consume_first_mimi_frame",
        "assistant_output_origin_frame": 0,
        "sample_rate": 24000,
        "frame_samples": FRAME_SAMPLES,
        "prefix_silence_ms": prefix_ms,
        "user_start_frame": prefix_frames,
        "query_end_ms": utterance_end_ms,
        "query_end_frame": query_end_exclusive,
        "user_end_frame": query_end_exclusive,
        "user_frame_count": prepared_frame_count,
        "user_sample_count": prepared_frame_count * FRAME_SAMPLES,
        "response_capture_ms": response_capture_ms,
        "response_capture_frames": response_capture_frames,
        "tail_guard_frames": TAIL_GUARD_FRAMES,
        "target_end_frame_count": target_frames,
        "target_end_sample_count": target_samples,
        "appended_zero_frame_count": target_frames - prepared_frame_count,
        "source_capture_contract_sha256": sha256_value(capture),
        "source_execution_contract_sha256": sha256_value(execution),
    }


def build_mech_manifest(argv: Sequence[str]) -> int:
    parser = _parser("Build a portable, hash-bound mechanistic trial manifest.")
    parser.add_argument("--source-eval-manifest", type=Path)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--analysis-folds", type=Path)
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--seeds", help="Optional comma-separated generation seed allowlist.")
    parser.add_argument("--data-status", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prepared_manifest_sha256 = sha256_file(args.prepared_manifest)
    analysis_folds_manifest_sha256 = (
        sha256_file(args.analysis_folds) if args.analysis_folds is not None else None
    )
    role_manifest_sha256 = (
        sha256_file(args.role_manifest) if args.role_manifest is not None else None
    )
    source_eval_manifest_sha256 = (
        sha256_file(args.source_eval_manifest)
        if args.source_eval_manifest is not None
        else None
    )
    prepared = read_jsonl(args.prepared_manifest)
    by_prepared = {str(row["prepared_stimulus_id"]): row for row in prepared}
    if len(by_prepared) != len(prepared):
        raise ContractError("prepared manifest contains duplicate prepared_stimulus_id values")
    fold_rows = _rows_or_empty(args.analysis_folds)
    folds = {str(row["scenario_id"]): int(row["analysis_fold"]) for row in fold_rows}
    if len(folds) != len(fold_rows):
        raise ContractError("analysis-fold manifest contains duplicate scenario IDs")
    role_rows = _rows_or_empty(args.role_manifest)
    roles: dict[str, dict[str, Any]] = {}
    for role_row in role_rows:
        role_key = str(role_row.get("prepared_stimulus_id", role_row.get("trial_id", "")))
        if not role_key or role_key in roles:
            raise ContractError("role manifest has a missing or duplicate trial binding")
        roles[role_key] = role_row
    source = _rows_or_empty(args.source_eval_manifest) or prepared
    selected_seeds = set(_ints(args.seeds)) if args.seeds else None
    audio_by_name: dict[str, list[Path]] = defaultdict(list)
    if args.audio_root is not None:
        for path in args.audio_root.rglob("*.wav"):
            if path.is_file():
                audio_by_name[path.name].append(path)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source:
        row_seed = row.get("generation_seed")
        if selected_seeds is not None and row_seed is not None and int(row_seed) not in selected_seeds:
            continue
        input_stimulus = row.get("input_stimulus", {})
        if input_stimulus and not isinstance(input_stimulus, Mapping):
            raise ContractError("source input_stimulus must be an object")
        prepared_id = str(row.get("prepared_stimulus_id", input_stimulus.get("prepared_stimulus_id", "")))
        item = by_prepared.get(prepared_id)
        if item is None:
            raise ContractError(f"prepared stimulus missing for {prepared_id}")
        prepared_audio = item.get("prepared_stimulus", input_stimulus)
        if not isinstance(prepared_audio, Mapping):
            raise ContractError(f"{prepared_id}: prepared_stimulus must be an object")
        if input_stimulus:
            redundant_fields = (
                "sha256", "duration_ms", "sample_rate", "channels", "sample_width_bytes"
            )
            disagreements = [
                field for field in redundant_fields
                if field not in input_stimulus
                or field not in prepared_audio
                or input_stimulus[field] != prepared_audio[field]
            ]
            if input_stimulus.get("prepared_stimulus_id") != prepared_id or disagreements:
                raise ContractError(
                    f"{prepared_id}: source/prepared audio evidence disagrees: {disagreements}"
                )
            if int(input_stimulus.get("mimi_frame_samples", -1)) != FRAME_SAMPLES:
                raise ContractError(f"{prepared_id}: source Mimi frame size is not 1,920 samples")
        uri = str(prepared_audio.get("uri", ""))
        basename = Path(uri).name
        relative_uri = require_relative_uri(f"audio/{basename}")
        audio_sha = str(prepared_audio.get("sha256", ""))
        if args.audio_root is not None:
            matches = audio_by_name.get(basename, [])
            if len(matches) != 1:
                raise ContractError(f"expected one runtime WAV named {basename}, found {len(matches)}")
            if sha256_file(matches[0]) != audio_sha:
                raise ContractError(f"runtime WAV hash mismatch for {basename}")
            relative_uri = matches[0].relative_to(args.audio_root).as_posix()
        trial_id = str(row.get("eval_trial_id", row.get("trial_id", prepared_id)))
        if trial_id in seen:
            raise ContractError(f"duplicate trial_id: {trial_id}")
        seen.add(trial_id)
        old_value, new_value = _trial_values(item)
        direction_id = item.get("direction_id")
        if not isinstance(direction_id, str) or not direction_id.strip():
            # Reviewed multivalue controls are defined by an ordered old/new
            # pair even when their source schema predates direction_id.  Bind a
            # deterministic opaque identifier instead of emitting JSON null,
            # which would make matched canary selection impossible.
            direction_id = f"ordered_value_pair_{sha256_value([old_value, new_value])[:16]}"
        role_entry = roles.get(trial_id, roles.get(prepared_id))
        if args.role_manifest is not None and role_entry is None:
            raise ContractError(f"{trial_id}: immutable role-manifest binding is missing")
        sample_rate = int(prepared_audio.get("sample_rate", 0))
        if (
            sample_rate != 24000
            or int(prepared_audio.get("channels", 0)) != 1
            or int(prepared_audio.get("sample_width_bytes", 0)) != 2
        ):
            raise ContractError(f"{trial_id}: prepared audio is not mono 24 kHz PCM16")
        exact_sample_count = float(prepared_audio.get("duration_ms", 0)) * sample_rate / 1000.0
        sample_count = int(round(exact_sample_count))
        if abs(exact_sample_count - sample_count) > 1e-6:
            raise ContractError(f"{trial_id}: prepared duration is not an integer sample count")
        if sample_count <= 0 or sample_count % FRAME_SAMPLES:
            raise ContractError(f"{trial_id}: prepared audio is not Mimi-frame aligned")
        frame_count = sample_count // FRAME_SAMPLES
        conversation = _conversation_contract(
            row, trial_id=trial_id, prepared_frame_count=frame_count
        )
        scenario_id = str(item.get("scenario_id", ""))
        analysis_fold = folds.get(scenario_id, item.get("analysis_fold"))
        if scenario_id in folds and item.get("analysis_fold") is not None:
            if int(item["analysis_fold"]) != int(analysis_fold):
                raise ContractError(f"{trial_id}: analysis-fold evidence disagrees")
        if role_entry is not None:
            role = role_entry.get("role", role_entry.get("inferential_role"))
            if not isinstance(role, str) or not role:
                raise ContractError(f"{trial_id}: role manifest entry has no role")
            role_policy = "immutable_role_manifest"
        elif analysis_fold is not None and int(analysis_fold) in {1, 2, 3, 4, 5}:
            role = "discovery" if int(analysis_fold) <= 3 else "internal_validation"
            role_policy = "frozen_v2_folds_1_3_discovery_4_5_internal_validation"
        else:
            role = item.get("inferential_role")
            role_policy = "prepared_manifest_explicit_role"
        if not isinstance(role, str) or not role:
            raise ContractError(f"{trial_id}: no immutable analysis role can be derived")
        output_row = {
            "schema_version": "1.0.0", "trial_id": trial_id, "prepared_stimulus_id": prepared_id,
            "scenario_id": item.get("scenario_id"), "condition": item.get("condition"),
            "direction_id": direction_id, "speaker_id": item.get("speaker_id"),
            "generation_seed": row.get("generation_seed", item.get("generation_seed")),
            "analysis_fold": analysis_fold,
            "role": role,
            "role_policy": role_policy,
            "source_inferential_role": item.get("inferential_role"),
            "data_status": args.data_status, "old_value": old_value, "new_value": new_value,
            "audio_uri": relative_uri, "audio_sha256": audio_sha,
            "sample_rate": sample_rate,
            "sample_count": sample_count,
            "frame_count": frame_count,
            "prepared_manifest_sha256": prepared_manifest_sha256,
        }
        if analysis_folds_manifest_sha256 is not None:
            output_row["analysis_folds_manifest_sha256"] = analysis_folds_manifest_sha256
        if role_manifest_sha256 is not None:
            output_row["role_manifest_sha256"] = role_manifest_sha256
            output_row["role_binding_sha256"] = sha256_value(role_entry)
        if conversation is not None:
            output_row.update({
                "input_stimulus": dict(input_stimulus),
                "capture_contract": dict(row["capture_contract"]),
                "execution_contract": dict(row["execution_contract"]),
                "source_row_sha256": sha256_value(row),
                "model_repo": row.get("model_repo", MODEL_REPO),
                "resolved_revision": row.get("resolved_revision", MODEL_REVISION),
                "generation_config_hash": row.get("generation_config_hash"),
            })
            if source_eval_manifest_sha256 is not None:
                output_row["source_eval_manifest_sha256"] = source_eval_manifest_sha256
            output_row["conversation_contract"] = conversation
            try:
                ConversationContract.from_manifest_row(output_row)
            except ConversationContractError as error:
                raise ContractError(f"{trial_id}: {error}") from error
        output.append(output_row)
    write_jsonl(args.output, output)
    print(f"wrote {len(output)} portable mechanistic trials -> {args.output}")
    return 0


def build_anchor_map(argv: Sequence[str]) -> int:
    parser = _parser("Map semantic timings to 80 ms Mimi/LM frames.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-trace-output", type=Path, required=True)
    args = parser.parse_args(argv)
    anchors, trace = anchor_rows(read_jsonl(args.manifest), read_jsonl(args.prepared_manifest))
    write_jsonl(args.output, anchors)
    write_jsonl(args.frame_trace_output, trace)
    print(f"mapped {len(anchors)} anchors across {len(set(r['trial_id'] for r in anchors))} trials")
    return 0


def simulate_multivalue_power(argv: Sequence[str]) -> int:
    parser = _parser("Simulate scenario-cluster power before multivalue audio production.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--city-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=10000)
    args = parser.parse_args(argv)
    config, cities = read_json(args.config), read_json(args.city_config)
    design = cities["design"]
    clusters = int(design["formal_scenario_clusters"])
    sesoi = float(config["statistics"]["sesoi_nats_per_token"])
    rng = np.random.default_rng(int(config["statistics"]["seed"]))
    effects = np.linspace(sesoi / 2, sesoi * 2, 7)
    rows = []
    for effect in effects:
        estimates = rng.normal(effect, float(design["scenario_sd"]) / math.sqrt(clusters), args.simulations)
        rejected = (estimates - 1.96 * float(design["scenario_sd"]) / math.sqrt(clusters)) > 0
        rows.append({"effect": float(effect), "power": float(rejected.mean())})
    at_sesoi = min(rows, key=lambda row: abs(row["effect"] - sesoi))
    report = {"schema_version": "1.0.0", "status": "design_sensitivity_not_observed_data",
              "simulations": args.simulations, "scenario_clusters": clusters, "sesoi": sesoi,
              "power_at_sesoi": at_sesoi["power"], "passes_target": at_sesoi["power"] >= float(design["target_power"]),
              "effect_grid": rows, "config_sha256": sha256_file(args.config),
              "city_config_sha256": sha256_file(args.city_config)}
    write_json(args.output, report)
    if not report["passes_target"]:
        raise ContractError("multivalue design does not reach target power; increase independent scenarios")
    print(f"power gate passed ({report['power_at_sesoi']:.3f}) -> {args.output}")
    return 0


def build_multivalue_controls(argv: Sequence[str]) -> int:
    parser = _parser("Create frozen multivalue scripts, roles, and review templates.")
    parser.add_argument("--city-config", type=Path, required=True)
    parser.add_argument("--scenario-blueprints", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    city_config = read_json(args.city_config)
    city_rows = city_config.get("cities") if isinstance(city_config, Mapping) else None
    if not isinstance(city_rows, list) or any(not isinstance(item, Mapping) for item in city_rows):
        raise ContractError("city config must contain a cities object array")
    eligible_rows = [item for item in city_rows if item.get("eligible") is True]
    if any(
        item.get("screen_status") != "passed_on_excluded_calibration_set"
        for item in eligible_rows
    ):
        raise ContractError(
            "every eligible city must pass the excluded clean-recognition calibration screen"
        )
    cities = [str(item.get("value", "")).strip() for item in eligible_rows]
    if len(cities) < 4:
        raise ContractError("at least four frozen eligible cities are required")
    if any(not city for city in cities) or len({city.casefold() for city in cities}) != len(cities):
        raise ContractError("eligible city values must be non-empty and case-insensitively unique")
    scenarios = read_jsonl(args.scenario_blueprints)
    pairs = [(old, new) for old in cities for new in cities if old != new]
    design = city_config["design"]
    speakers = [str(value) for value in design["speaker_ids"]]
    conditions = [str(value) for value in design["conditions"]]
    if len(set(speakers)) != len(speakers) or any(not value for value in speakers):
        raise ContractError("multivalue speaker IDs must be non-empty and unique")
    if len(speakers) < int(design["minimum_speakers"]):
        raise ContractError("multivalue speaker inventory is below the frozen minimum")
    required_conditions = {"clean_current", "repair_immediate", "repair_delayed_640"}
    if set(conditions) != required_conditions or len(conditions) != len(required_conditions):
        raise ContractError(
            "multivalue design must contain exactly clean_current, repair_immediate, "
            "and repair_delayed_640"
        )
    scenario_ids = [str(scenario.get("scenario_id", "")).strip() for scenario in scenarios]
    if any(not value for value in scenario_ids) or len(set(scenario_ids)) != len(scenario_ids):
        raise ContractError("scenario blueprint IDs must be non-empty and unique")
    scenario_role = {
        str(scenario["scenario_id"]): ("multivalue_calibration" if index < int(design["calibration_scenario_clusters"])
                                       else "formal_confirmation")
        for index, scenario in enumerate(scenarios)
    }
    if Counter(scenario_role.values())["formal_confirmation"] != int(design["formal_scenario_clusters"]):
        raise ContractError("scenario blueprint count does not match the frozen role design")
    city_index = {city: index for index, city in enumerate(cities)}
    pair_role = {(old, new): ("multivalue_calibration" if (city_index[old] + city_index[new]) % 2 == 0
                              else "formal_confirmation") for old, new in pairs}
    derangement: dict[str, str] = {}
    for role_index, role_name in enumerate(("multivalue_calibration", "formal_confirmation")):
        role_pairs = [f"{old}->{new}" for old, new in pairs if pair_role[(old, new)] == role_name]
        derangement.update(deterministic_derangement(role_pairs, int(city_config["split_seed"]) + role_index))
    scripts, roles, reviews = [], [], []
    for scenario in scenarios:
        for old, new in pairs:
            pair_id = f"{old}->{new}"
            role = pair_role[(old, new)]
            if scenario_role[str(scenario["scenario_id"])] != role:
                continue
            root_old = str(scenario["root_template"]).format(value=old)
            root_new = str(scenario["root_template"]).format(value=new)
            repair = str(scenario["repair_template"]).format(new=new, old=old)
            dependencies = [str(unit["text"]) for unit in scenario.get("dependent_units", [])]
            closing = str(scenario["closing_prompt"])
            for speaker in speakers:
                for condition in conditions:
                    root = root_new if condition == "clean_current" else f"{root_old}. {repair}"
                    text = ". ".join([root, *dependencies, closing])
                    trial_id = (f"mv__{scenario['scenario_id']}__{old.lower()}_to_{new.lower()}__"
                                f"{condition}__{speaker}")
                    scripts.append({"schema_version": "1.0.0", "trial_id": trial_id,
                                    "scenario_id": scenario["scenario_id"], "speaker_id": speaker,
                                    "old_value": old, "new_value": new, "condition": condition, "text": text,
                                    "requested_repair_pause_ms": 640 if condition == "repair_delayed_640" else 0,
                                    "audio_status": "awaiting_reviewed_audio"})
                    roles.append({"schema_version": "1.0.0", "trial_id": trial_id, "ordered_pair": pair_id,
                                  "scenario_id": scenario["scenario_id"], "role": role,
                                  "deranged_control_pair": derangement[pair_id]})
                    reviews.append({"trial_id": trial_id, "wav_sha256": None,
                                    "alignment_reviewer": None,
                                    "listener_1": None, "listener_1_decision": None,
                                    "listener_2": None, "listener_2_decision": None,
                                    "adjudicator": None, "adjudication_decision": None,
                                    "reviewed_at": None, "status": "pending"})
    root = args.output_root
    write_jsonl(root / "source_scripts.jsonl", scripts)
    write_jsonl(root / "role_manifest.jsonl", roles)
    write_jsonl(root / "review_template.jsonl", reviews)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    write_csv(root / "recording_targets.csv", [
        {"trial_id": row["trial_id"], "speaker_id": row["speaker_id"], "condition": row["condition"],
         "text": row["text"], "target_wav": f"audio/{row['trial_id']}.wav"} for row in scripts])
    timing_path, reviews_path = root / "timing.jsonl", root / "reviews.jsonl"
    prepared_rows: list[dict[str, Any]] = []
    if timing_path.exists() and reviews_path.exists():
        timing_by_id = {str(row["trial_id"]): row for row in read_jsonl(timing_path)}
        reviews_by_id = {str(row["trial_id"]): row for row in read_jsonl(reviews_path)}
        for row in scripts:
            trial_id = str(row["trial_id"])
            wav = root / "audio" / f"{trial_id}.wav"
            timing = timing_by_id.get(trial_id)
            review = reviews_by_id.get(trial_id)
            if not wav.is_file() or timing is None or review is None or review.get("status") != "passed":
                raise ContractError(f"{trial_id}: reviewed audio materialization is incomplete")
            with wave.open(str(wav), "rb") as handle:
                channels = handle.getnchannels()
                sample_rate = handle.getframerate()
                sample_width = handle.getsampwidth()
                sample_count = handle.getnframes()
            if (
                channels != 1
                or sample_rate != 24000
                or sample_width != 2
                or sample_count % FRAME_SAMPLES
            ):
                raise ContractError(
                    f"{trial_id}: WAV must be mono 24 kHz PCM16 and Mimi-frame aligned"
                )
            digest = sha256_file(wav)
            if review.get("wav_sha256") != digest:
                raise ContractError(f"{trial_id}: reviewed WAV hash mismatch")
            if not isinstance(timing, Mapping) or "utterance_end_ms" not in timing:
                raise ContractError(f"{trial_id}: timing must record utterance_end_ms")
            if timing.get("timebase", "prepared_stream_relative") != "prepared_stream_relative":
                raise ContractError(f"{trial_id}: timing must use prepared_stream_relative")
            utterance_end_ms = float(timing["utterance_end_ms"])
            utterance_end_samples_float = utterance_end_ms * sample_rate / 1000.0
            utterance_end_samples = int(round(utterance_end_samples_float))
            if abs(utterance_end_samples_float - utterance_end_samples) > 1e-6:
                raise ContractError(f"{trial_id}: utterance end is not sample-exact")
            user_end_frame = int(math.ceil(utterance_end_samples / FRAME_SAMPLES))
            user_frame_count = sample_count // FRAME_SAMPLES
            if user_end_frame <= 0 or user_end_frame > user_frame_count:
                raise ContractError(f"{trial_id}: utterance end is outside the reviewed WAV")
            prefix_silence_ms = float(timing.get("prefix_silence_ms", 0))
            prefix_frames_float = prefix_silence_ms / FRAME_MS
            prefix_frames = int(round(prefix_frames_float))
            if (
                prefix_silence_ms < 0
                or abs(prefix_frames_float - prefix_frames) > 1e-9
                or prefix_frames >= user_end_frame
            ):
                raise ContractError(f"{trial_id}: prefix silence is not a valid Mimi-frame boundary")
            target_end_frames = user_end_frame + RESPONSE_CAPTURE_FRAMES
            target_end_samples = target_end_frames * FRAME_SAMPLES
            duration_ms = sample_count * 1000.0 / sample_rate
            prepared_timing = dict(timing)
            prepared_timing["utterance_end_ms"] = utterance_end_ms
            capture_contract = {
                "condition": row["condition"],
                "timebase": "prepared_stream_relative",
                "stream_origin_ms": 0,
                "prepared_timing": prepared_timing,
                "prepared_timing_sha256": sha256_value(prepared_timing),
                "primary_window_start_ms": utterance_end_ms,
                "utterance_end_ms": utterance_end_ms,
                "response_capture_ms": 40_000,
                "requested_target_end_ms": utterance_end_ms + 40_000,
                "target_end_sample_count": target_end_samples,
                "target_end_frame_count": target_end_frames,
                "actual_target_end_ms": target_end_frames * FRAME_MS,
            }
            execution_contract = {
                "runner_version": HARNESS_VERSION,
                "runner_source_sha256": sha256_file(Path(__file__)),
                "input_sample_rate": sample_rate,
                "mimi_frame_samples": FRAME_SAMPLES,
                "prefix_silence_ms": prefix_silence_ms,
                "response_capture_ms": 40_000,
                "reset_model_stream_between_trials": True,
                "reset_rng_for_each_trial_seed": True,
                "required_model_type": "moshi",
                "required_max_lm_delay": 1,
            }
            input_stimulus = {
                "prepared_stimulus_id": trial_id,
                "uri": f"audio/{trial_id}.wav",
                "sha256": digest,
                "duration_ms": duration_ms,
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width_bytes": sample_width,
                "timeline": "prepared_stream_relative",
                "mimi_frame_samples": FRAME_SAMPLES,
            }
            prepared_rows.append({
                **row,
                "prepared_stimulus_id": trial_id,
                "prepared_stimulus": dict(input_stimulus),
                "input_stimulus": input_stimulus,
                "prepared_timing": prepared_timing,
                "preparation": {
                    "sample_rate": sample_rate,
                    "mimi_frame_samples": FRAME_SAMPLES,
                    "prefix_silence_ms": prefix_silence_ms,
                    "prefix_ms_actual": prefix_silence_ms,
                    "prefix_samples": prefix_frames * FRAME_SAMPLES,
                },
                "capture_contract": capture_contract,
                "execution_contract": execution_contract,
                "conversation_contract_source": REVIEWED_MULTIVALUE_CONTRACT_SOURCE,
                "model_repo": MODEL_REPO,
                "resolved_revision": MODEL_REVISION,
                "alignment": {
                    "unit_spans": timing.get("unit_spans", []),
                    "independent_forced_alignment": True,
                },
                "data_status": "reviewed_multivalue",
            })
        write_jsonl(root / "prepared_stimuli.jsonl", prepared_rows)
    status = "reviewed_audio_materialized" if prepared_rows else "awaiting_audio_alignment_and_human_review"
    write_json(root / "BUILD_STATUS.json", {"status": status, "trial_count": len(scripts),
                                             "prepared_count": len(prepared_rows),
                                             "city_config_sha256": sha256_file(args.city_config)})
    print(f"created {len(scripts)} frozen scripts; status={status}")
    return 0


def validate_multivalue_controls(argv: Sequence[str]) -> int:
    parser = _parser("Fail-closed multivalue coverage, audio, alignment, and review validator.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--mechanistic-manifest", type=Path)
    parser.add_argument("--require-independent-alignment", action="store_true")
    parser.add_argument("--require-double-listen-review", action="store_true")
    args = parser.parse_args(argv)
    scripts = read_jsonl(args.input_root / "source_scripts.jsonl")
    roles = read_jsonl(args.input_root / "role_manifest.jsonl")
    reviews_path = args.input_root / "reviews.jsonl"
    if not reviews_path.exists():
        raise ContractError("reviews.jsonl is missing; review_template.jsonl is not evidence")
    reviews = {str(row["trial_id"]): row for row in read_jsonl(reviews_path)}
    prepared_path = args.input_root / "prepared_stimuli.jsonl"
    if not prepared_path.exists():
        raise ContractError("prepared_stimuli.jsonl is missing; rerun the builder after audio/alignment/review")
    prepared = {str(row["trial_id"]): row for row in read_jsonl(prepared_path)}
    if {row["trial_id"] for row in scripts} != {row["trial_id"] for row in roles}:
        raise ContractError("script and role trial sets differ")
    role_pairs: dict[str, set[str]] = defaultdict(set)
    role_scenarios: dict[str, set[str]] = defaultdict(set)
    role_old: dict[str, set[str]] = defaultdict(set)
    role_new: dict[str, set[str]] = defaultdict(set)
    script_by_id = {str(row["trial_id"]): row for row in scripts}
    for row in roles:
        role = str(row["role"])
        role_pairs[role].add(str(row["ordered_pair"]))
        role_scenarios[role].add(str(row["scenario_id"]))
        script = script_by_id[str(row["trial_id"])]
        role_old[role].add(str(script["old_value"]))
        role_new[role].add(str(script["new_value"]))
    overlap = role_pairs.get("multivalue_calibration", set()) & role_pairs.get("formal_confirmation", set())
    if overlap:
        raise ContractError(f"ordered-pair leakage across roles: {sorted(overlap)[:3]}")
    scenario_overlap = role_scenarios.get("multivalue_calibration", set()) & role_scenarios.get("formal_confirmation", set())
    if scenario_overlap:
        raise ContractError(f"scenario-template leakage across roles: {sorted(scenario_overlap)[:3]}")
    all_cities = {str(row["old_value"]) for row in scripts} | {str(row["new_value"]) for row in scripts}
    for role in ("multivalue_calibration", "formal_confirmation"):
        if role_old[role] != all_cities or role_new[role] != all_cities:
            raise ContractError(f"{role}: every city must appear in both old and new roles")
    required_conditions = {"clean_current", "repair_immediate", "repair_delayed_640"}
    coverage: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in scripts:
        coverage[(str(row["scenario_id"]), str(row["old_value"]), str(row["new_value"]))].add(str(row["condition"]))
    if any(conditions != required_conditions for conditions in coverage.values()):
        raise ContractError("condition coverage is incomplete for at least one scenario/pair")
    for script in scripts:
        trial_id = str(script["trial_id"])
        review = reviews.get(trial_id)
        if review is None or review.get("status") != "passed":
            raise ContractError(f"{trial_id}: human review is not passed")
        wav = args.input_root / "audio" / f"{trial_id}.wav"
        if not wav.is_file() or sha256_file(wav) != review.get("wav_sha256"):
            raise ContractError(f"{trial_id}: missing WAV or review hash mismatch")
        prepared_row = prepared.get(trial_id)
        if prepared_row is None or prepared_row.get("data_status") != "reviewed_multivalue":
            raise ContractError(f"{trial_id}: reviewed prepared stimulus evidence missing")
        alignment_reviewer = review.get("alignment_reviewer")
        if args.require_independent_alignment and not (
            isinstance(alignment_reviewer, str) and alignment_reviewer.strip()
        ):
            raise ContractError(f"{trial_id}: independent alignment evidence missing")
        if args.require_double_listen_review:
            listener_1 = review.get("listener_1")
            listener_2 = review.get("listener_2")
            decisions = (review.get("listener_1_decision"), review.get("listener_2_decision"))
            if not all(isinstance(value, str) and value.strip() for value in (listener_1, listener_2)):
                raise ContractError(f"{trial_id}: double-listen reviewer identities are missing")
            if str(listener_1).strip() == str(listener_2).strip():
                raise ContractError(f"{trial_id}: double-listen reviewers must be distinct")
            if any(value not in {"passed", "failed"} for value in decisions):
                raise ContractError(f"{trial_id}: double-listen decisions are missing or invalid")
            if decisions[0] != decisions[1]:
                if not (
                    isinstance(review.get("adjudicator"), str)
                    and str(review["adjudicator"]).strip()
                    and review.get("adjudication_decision") in {"passed", "failed"}
                ):
                    raise ContractError(f"{trial_id}: listener disagreement lacks adjudication")
                final_decision = review["adjudication_decision"]
            else:
                final_decision = decisions[0]
            if final_decision != "passed" or review.get("status") != "passed":
                raise ContractError(f"{trial_id}: human review final decision is not passed")
            reviewed_at = review.get("reviewed_at")
            if not isinstance(reviewed_at, str) or not reviewed_at.strip():
                raise ContractError(f"{trial_id}: review timestamp is missing")
            try:
                parsed = datetime.datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise ContractError(f"{trial_id}: review timestamp is not ISO-8601") from error
            if parsed.tzinfo is None:
                raise ContractError(f"{trial_id}: review timestamp must include a timezone")
    if args.mechanistic_manifest:
        manifest_ids = {row["trial_id"] for row in read_jsonl(args.mechanistic_manifest)}
        if manifest_ids != {row["trial_id"] for row in scripts}:
            raise ContractError("mechanistic manifest does not bind the reviewed trial set")
    print(f"multivalue confirmation gate passed for {len(scripts)} reviewed trials")
    return 0


def _encoding_frame_contract(row: Mapping[str, Any]) -> dict[str, int]:
    trial_id = str(row.get("trial_id", ""))
    if not trial_id:
        raise ContractError("encoding manifest row has no trial_id")
    user_frames = _exact_int_field(row.get("frame_count"), f"{trial_id} frame_count")
    if user_frames < 1:
        raise ContractError(f"{trial_id}: frame_count must be positive")
    sample_count = _exact_int_field(row.get("sample_count"), f"{trial_id} sample_count")
    if sample_count != user_frames * FRAME_SAMPLES:
        raise ContractError(f"{trial_id}: sample/frame coverage is not exact")
    target_frames = user_frames
    query_end = user_frames
    user_start = 0
    if isinstance(row.get("conversation_contract"), Mapping):
        try:
            conversation = ConversationContract.from_manifest_row(row)
        except ConversationContractError as error:
            raise ContractError(f"{trial_id}: {error}") from error
        if conversation.user_frame_count != user_frames:
            raise ContractError(f"{trial_id}: conversation/user frame coverage disagrees")
        target_frames = conversation.target_end_frame_count
        query_end = conversation.query_end_frame
        user_start = conversation.user_start_frame
    if not 0 <= user_start < query_end <= user_frames <= target_frames:
        raise ContractError(f"{trial_id}: invalid half-open conversation frame coverage")
    return {
        "user_start_frame": user_start,
        "query_end_frame_exclusive": query_end,
        "user_frame_count": user_frames,
        "target_frame_count": target_frames,
    }


def _validate_source_wav(root: Path, row: Mapping[str, Any], frames: int) -> tuple[Path, str]:
    trial_id = str(row["trial_id"])
    declared = validate_sha256(str(row.get("audio_sha256", "")), f"{trial_id} source WAV")
    relative = require_relative_uri(str(row.get("audio_uri", "")))
    root_resolved = root.resolve()
    try:
        wav = (root_resolved / relative).resolve(strict=True)
        wav.relative_to(root_resolved)
    except (FileNotFoundError, ValueError) as error:
        raise ContractError(f"{trial_id}: source WAV is missing or escapes its artifact root") from error
    if not wav.is_file() or sha256_file(wav) != declared:
        raise ContractError(f"{trial_id}: source WAV hash mismatch")
    try:
        with wave.open(str(wav), "rb") as handle:
            observed = (
                handle.getnchannels(), handle.getframerate(), handle.getsampwidth(),
                handle.getnframes(), handle.getcomptype(),
            )
    except (wave.Error, EOFError) as error:
        raise ContractError(f"{trial_id}: source WAV is not readable PCM") from error
    expected = (1, SAMPLE_RATE, 2, frames * FRAME_SAMPLES, "NONE")
    if observed != expected:
        raise ContractError(f"{trial_id}: source WAV coverage/format mismatch: {observed}")
    return wav, declared


def _validate_encoded_archive(
    path: Path, *, identity_sha256: str, source_audio_sha256: str,
    model_identity_sha256: str, user_frames: int, target_frames: int,
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if _archive_identity(archive, path) != identity_sha256:
                raise ContractError(f"encoded archive identity mismatch: {path}")
            for key, expected in (
                ("source_audio_sha256", source_audio_sha256),
                ("model_identity_sha256", model_identity_sha256),
            ):
                if key not in archive.files or np.asarray(archive[key]).shape != ():
                    raise ContractError(f"encoded archive has no valid {key}: {path}")
                if np.asarray(archive[key]).item() != expected:
                    raise ContractError(f"encoded archive {key} mismatch: {path}")
            user = _code_array(
                archive["user_codes"], label="user_codes", frames=user_frames)
            conversation = _code_array(
                archive["conversation_codes"], label="conversation_codes", frames=target_frames)
            silence = _code_array(
                archive["assistant_silence_codes"], label="assistant_silence_codes",
                frames=target_frames)
            if "codes" not in archive.files:
                raise ContractError(f"encoded archive has no compatibility user codes: {path}")
            codes = _code_array(archive["codes"], label="codes", frames=user_frames)
    except (KeyError, OSError, ValueError) as error:
        raise ContractError(f"cannot validate encoded archive {path}: {error}") from error
    if not np.array_equal(codes, user):
        raise ContractError(f"encoded archive codes/user_codes disagree: {path}")
    if not np.array_equal(conversation[..., :user_frames], user):
        raise ContractError(f"conversation codes do not preserve the exact user prefix: {path}")
    return {
        "user_codes": user,
        "conversation_codes": conversation,
        "assistant_silence_codes": silence,
    }


def _encoded_manifest_row(
    source: Mapping[str, Any], *, frames: Mapping[str, int], destination: Path,
    output_manifest: Path, arrays: Mapping[str, np.ndarray], archive_sha256: str,
    manifest_sha256: str, source_audio_sha256: str, model_identity_sha256: str,
    artifact_identity_sha256: str, code_commit: str, repeat_checked: bool,
    synthetic: bool,
) -> dict[str, Any]:
    user = arrays["user_codes"]
    conversation = arrays["conversation_codes"]
    silence = arrays["assistant_silence_codes"]
    row = {
        "schema_version": "1.1.0",
        "trial_id": str(source["trial_id"]),
        "scenario_id": source.get("scenario_id"),
        "condition": source.get("condition"),
        "old_value": source.get("old_value"),
        "new_value": source.get("new_value"),
        "analysis_fold": source.get("analysis_fold"),
        "role": source.get("role"),
        "source_manifest_sha256": manifest_sha256,
        "source_row_sha256": sha256_value(source),
        "source_audio_sha256": source_audio_sha256,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_identity_sha256": model_identity_sha256,
        "code_commit": code_commit,
        "harness_version": HARNESS_VERSION,
        "artifact_identity_sha256": artifact_identity_sha256,
        "codes_uri": _safe_artifact_uri(destination, output_manifest),
        "archive_sha256": archive_sha256,
        # Compatibility fields are the exact prepared-user tensor, never the
        # appended conversation horizon.
        "shape": list(user.shape),
        "dtype": str(user.dtype),
        "codes_sha256": _array_sha256(user),
        "user_codes_shape": list(user.shape),
        "user_codes_dtype": str(user.dtype),
        "user_codes_sha256": _array_sha256(user),
        "conversation_codes_shape": list(conversation.shape),
        "conversation_codes_dtype": str(conversation.dtype),
        "conversation_codes_sha256": _array_sha256(conversation),
        "assistant_silence_codes_shape": list(silence.shape),
        "assistant_silence_codes_dtype": str(silence.dtype),
        "assistant_silence_codes_sha256": _array_sha256(silence),
        "user_frame_start": 0,
        "user_frame_end_exclusive": frames["user_frame_count"],
        "user_frame_count": frames["user_frame_count"],
        "user_sample_start": 0,
        "user_sample_end_exclusive": frames["user_frame_count"] * FRAME_SAMPLES,
        "conversation_frame_start": 0,
        "conversation_frame_end_exclusive": frames["target_frame_count"],
        "target_frame_count": frames["target_frame_count"],
        "conversation_sample_start": 0,
        "conversation_sample_end_exclusive": frames["target_frame_count"] * FRAME_SAMPLES,
        "assistant_silence_frame_start": 0,
        "assistant_silence_frame_end_exclusive": frames["target_frame_count"],
        "user_start_frame": frames["user_start_frame"],
        "query_end_frame_exclusive": frames["query_end_frame_exclusive"],
        "repeat_encode_check": "passed" if repeat_checked else "not_selected_bounded_check",
        "synthetic": synthetic,
    }
    if isinstance(source.get("conversation_contract"), Mapping):
        row["conversation_contract_sha256"] = sha256_value(source["conversation_contract"])
    return row


def encode_user_audio(argv: Sequence[str]) -> int:
    parser = _parser("Encode exact user, conversation, and assistant-silence Mimi streams.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-artifact-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.model_revision != MODEL_REVISION:
        raise ContractError("encode requested a non-frozen model revision")
    rows = read_jsonl(args.manifest)
    trial_ids = [str(row.get("trial_id", "")) for row in rows]
    if not rows or any(not trial_id for trial_id in trial_ids) or len(set(trial_ids)) != len(rows):
        raise ContractError("encoding manifest is empty or has missing/duplicate trial IDs")
    if args.output_manifest.exists() and not args.resume:
        raise ContractError("encoded output manifest already exists; use --resume after identity verification")
    existing_rows = read_jsonl(args.output_manifest) if args.output_manifest.exists() else []
    existing_by_id = {str(row.get("trial_id", "")): row for row in existing_rows}
    if len(existing_by_id) != len(existing_rows) or not set(existing_by_id) <= set(trial_ids):
        raise ContractError("resume manifest has duplicate or out-of-scope trial IDs")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = sha256_file(args.manifest)
    code_commit = _git_commit()
    model_identity_sha256 = sha256_value({
        "model_repo": MODEL_REPO,
        "model_revision": args.model_revision,
        "mimi_sample_rate": SAMPLE_RATE,
        "mimi_frame_samples": FRAME_SAMPLES,
        "user_codebooks": 8,
        "assistant_codebooks": 8,
    })
    repeat_trial_ids = {min(trial_ids), max(trial_ids)}
    jobs: list[dict[str, Any]] = []
    for source in rows:
        trial_id = str(source["trial_id"])
        frames = _encoding_frame_contract(source)
        if args.synthetic:
            declared = source.get("audio_sha256")
            source_audio_sha256 = (
                validate_sha256(str(declared), f"{trial_id} synthetic source")
                if declared is not None
                else sha256_value({"synthetic_trial_id": trial_id, **frames})
            )
            wav = None
        else:
            wav, source_audio_sha256 = _validate_source_wav(
                args.input_artifact_root, source, frames["user_frame_count"])
        identity = {
            "schema_version": "1.0.0",
            "operation": "encode_user_audio",
            "harness_version": HARNESS_VERSION,
            "code_commit": code_commit,
            "source_manifest_sha256": manifest_sha256,
            "source_row_sha256": sha256_value(source),
            "source_audio_sha256": source_audio_sha256,
            "model_identity_sha256": model_identity_sha256,
            "frame_contract": frames,
            "synthetic": bool(args.synthetic),
        }
        identity_sha256 = sha256_value(identity)
        destination = args.output_root / f"{identity_sha256}.npz"
        old = existing_by_id.get(trial_id)
        if old is not None and old.get("artifact_identity_sha256") != identity_sha256:
            raise ContractError(f"{trial_id}: resume identity differs from the completed encoding")
        if old is not None and not destination.is_file():
            raise ContractError(f"{trial_id}: resume manifest points to a missing encoding artifact")
        if destination.exists() and not args.resume:
            raise ContractError(f"{trial_id}: encoding artifact already exists; use --resume")
        jobs.append({
            "source": source, "frames": frames, "wav": wav,
            "source_audio_sha256": source_audio_sha256, "identity_sha256": identity_sha256,
            "destination": destination, "old": old,
        })

    backend = None
    output: list[dict[str, Any]] = []
    for job in jobs:
        source = job["source"]
        trial_id = str(source["trial_id"])
        frames = job["frames"]
        destination = job["destination"]
        repeat_checked = trial_id in repeat_trial_ids
        prior_repeat_evidence = (
            job["old"] is not None and job["old"].get("repeat_encode_check") == "passed"
        )
        if destination.exists():
            arrays = _validate_encoded_archive(
                destination,
                identity_sha256=job["identity_sha256"],
                source_audio_sha256=job["source_audio_sha256"],
                model_identity_sha256=model_identity_sha256,
                user_frames=frames["user_frame_count"],
                target_frames=frames["target_frame_count"],
            )
        elif args.synthetic:
            arrays = {
                "user_codes": np.zeros(
                    (1, 8, frames["user_frame_count"]), dtype=np.int64),
                "conversation_codes": np.zeros(
                    (1, 8, frames["target_frame_count"]), dtype=np.int64),
                "assistant_silence_codes": np.zeros(
                    (1, 8, frames["target_frame_count"]), dtype=np.int64),
            }
            _atomic_savez(
                destination, codes=arrays["user_codes"], **arrays,
                artifact_identity_sha256=np.asarray(job["identity_sha256"]),
                source_audio_sha256=np.asarray(job["source_audio_sha256"]),
                model_identity_sha256=np.asarray(model_identity_sha256),
            )
        else:
            if backend is None:
                # No checkpoint is constructed when every artifact was identity-verified on resume.
                backend = MoshiBackend(
                    model_repo=MODEL_REPO, model_revision=args.model_revision, use_sampling=False)
            encoded = backend.encode_conversation_file(
                job["wav"], target_frame_count=frames["target_frame_count"])
            if (
                int(encoded.user_frame_count) != frames["user_frame_count"]
                or int(encoded.target_frame_count) != frames["target_frame_count"]
            ):
                raise ContractError(f"{trial_id}: backend-reported encode coverage mismatch")
            arrays = {
                "user_codes": _code_array(
                    encoded.user_codes, label=f"{trial_id} user_codes",
                    frames=frames["user_frame_count"]),
                "conversation_codes": _code_array(
                    encoded.conversation_codes, label=f"{trial_id} conversation_codes",
                    frames=frames["target_frame_count"]),
                "assistant_silence_codes": _code_array(
                    encoded.assistant_silence_codes, label=f"{trial_id} assistant_silence_codes",
                    frames=frames["target_frame_count"]),
            }
            if not np.array_equal(
                arrays["conversation_codes"][..., :frames["user_frame_count"]],
                arrays["user_codes"],
            ):
                raise ContractError(f"{trial_id}: continuous conversation encode changed the user prefix")
            _atomic_savez(
                destination, codes=arrays["user_codes"], **arrays,
                artifact_identity_sha256=np.asarray(job["identity_sha256"]),
                source_audio_sha256=np.asarray(job["source_audio_sha256"]),
                model_identity_sha256=np.asarray(model_identity_sha256),
            )
        if repeat_checked and not prior_repeat_evidence:
            if args.synthetic:
                repeated_arrays = {
                    "user_codes": np.zeros_like(arrays["user_codes"]),
                    "conversation_codes": np.zeros_like(arrays["conversation_codes"]),
                    "assistant_silence_codes": np.zeros_like(arrays["assistant_silence_codes"]),
                }
            else:
                if backend is None:
                    backend = MoshiBackend(
                        model_repo=MODEL_REPO, model_revision=args.model_revision,
                        use_sampling=False)
                repeated = backend.encode_conversation_file(
                    job["wav"], target_frame_count=frames["target_frame_count"])
                repeated_arrays = {
                    "user_codes": _code_array(
                        repeated.user_codes, label="repeated user_codes",
                        frames=frames["user_frame_count"]),
                    "conversation_codes": _code_array(
                        repeated.conversation_codes, label="repeated conversation_codes",
                        frames=frames["target_frame_count"]),
                    "assistant_silence_codes": _code_array(
                        repeated.assistant_silence_codes,
                        label="repeated assistant_silence_codes",
                        frames=frames["target_frame_count"]),
                }
            if any(not np.array_equal(arrays[key], repeated_arrays[key]) for key in arrays):
                raise ContractError(f"{trial_id}: repeated Mimi encoding is not byte-identical")
        arrays = _validate_encoded_archive(
            destination,
            identity_sha256=job["identity_sha256"],
            source_audio_sha256=job["source_audio_sha256"],
            model_identity_sha256=model_identity_sha256,
            user_frames=frames["user_frame_count"], target_frames=frames["target_frame_count"],
        )
        output_row = _encoded_manifest_row(
            source, frames=frames, destination=destination, output_manifest=args.output_manifest,
            arrays=arrays, archive_sha256=sha256_file(destination),
            manifest_sha256=manifest_sha256, source_audio_sha256=job["source_audio_sha256"],
            model_identity_sha256=model_identity_sha256,
            artifact_identity_sha256=job["identity_sha256"], code_commit=code_commit,
            repeat_checked=repeat_checked, synthetic=bool(args.synthetic),
        )
        if job["old"] is not None and job["old"] != output_row:
            raise ContractError(f"{trial_id}: completed encoding manifest row failed resume verification")
        output.append(output_row)
    write_jsonl(args.output_manifest, output)
    observed_ids = [str(row["trial_id"]) for row in output]
    summary = {
        "schema_version": "1.0.0", "operation": "encode_user_audio",
        "source_manifest_sha256": manifest_sha256,
        "encoded_manifest_sha256": sha256_file(args.output_manifest),
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "model_identity_sha256": model_identity_sha256,
        "code_commit": code_commit, "harness_version": HARNESS_VERSION,
        "expected_trial_count": len(rows), "encoded_trial_count": len(output),
        "expected_coverage_fraction": 1.0,
        "observed_coverage_fraction": len(output) / len(rows),
        "duplicate_source_trial_count": len(trial_ids) - len(set(trial_ids)),
        "duplicate_encoded_trial_count": len(observed_ids) - len(set(observed_ids)),
        "missing_trial_count": len(set(trial_ids) - set(observed_ids)),
        "extra_trial_count": len(set(observed_ids) - set(trial_ids)),
        "source_hash_mismatch_count": 0,
        "artifact_hash_mismatch_count": 0,
        "exact_half_open_coverage_count": sum(
            row["user_frame_end_exclusive"] == row["user_codes_shape"][-1]
            and row["conversation_frame_end_exclusive"] == row["conversation_codes_shape"][-1]
            and row["assistant_silence_frame_end_exclusive"]
            == row["assistant_silence_codes_shape"][-1]
            for row in output
        ),
        "repeated_encode_trial_ids": sorted(repeat_trial_ids),
        "repeated_encode_check_count": sum(
            row["repeat_encode_check"] == "passed" for row in output),
        "repeated_encode_mismatch_count": 0,
        "synthetic": bool(args.synthetic),
    }
    summary["passed"] = all((
        summary["encoded_trial_count"] == summary["expected_trial_count"],
        summary["observed_coverage_fraction"] == 1.0,
        summary["duplicate_source_trial_count"] == 0,
        summary["duplicate_encoded_trial_count"] == 0,
        summary["missing_trial_count"] == 0,
        summary["extra_trial_count"] == 0,
        summary["source_hash_mismatch_count"] == 0,
        summary["artifact_hash_mismatch_count"] == 0,
        summary["exact_half_open_coverage_count"] == len(rows),
        summary["repeated_encode_check_count"] == len(repeat_trial_ids),
        summary["repeated_encode_mismatch_count"] == 0,
    ))
    summary["summary_identity_sha256"] = sha256_value(summary)
    write_json(args.output_manifest.with_suffix(args.output_manifest.suffix + ".summary.json"), summary)
    if not summary["passed"]:
        raise ContractError("encoded coverage/repeat summary failed")
    print(f"encoded {len(output)} exact trial streams -> {args.output_manifest}")
    return 0


def validate_mechanistic_contract(argv: Sequence[str]) -> int:
    parser = _parser("Validate input, identity, environment, and model contracts.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config, rows = read_json(args.config), read_jsonl(args.manifest)
    model_config = _frozen_model_config(config)
    expected_model_contract = {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "layers": 32,
        "heads": 32,
        "hidden_size": 4096,
        "head_dim": 128,
        "max_lm_delay": 1,
    }
    if any(model_config.get(key) != value for key, value in expected_model_contract.items()):
        raise ContractError("config does not contain the exact frozen Moshiko shape/dtype contract")
    try:
        import jsonschema
        trial_schema = read_json(SCRIPT_DIR.parent / "schemas/trial.schema.json")
        for index, row in enumerate(rows):
            try:
                jsonschema.validate(row, trial_schema)
            except jsonschema.ValidationError as error:
                raise ContractError(f"manifest row {index} violates trial schema: {error.message}") from error
    except ImportError as error:
        raise ContractError("jsonschema is required for contract validation") from error
    if not rows or len({row["trial_id"] for row in rows}) != len(rows):
        raise ContractError("manifest is empty or has duplicate trial IDs")
    data_statuses = {row.get("data_status") for row in rows}
    if len(data_statuses) != 1:
        raise ContractError("manifest mixes data-status classes")
    if args.model_repo != MODEL_REPO or args.model_revision != MODEL_REVISION:
        raise ContractError("model repository/revision differs from the frozen Moshiko identity")
    if rows[0].get("data_status") == "exploratory_provisional":
        expected = int(config.get("manifest", {}).get("expected_v2_discovery_trials", 0))
        if expected and len(rows) != expected:
            raise ContractError(f"expected {expected} frozen v2 discovery trials, found {len(rows)}")
    environment = validate_runtime_environment(require_cuda=not (args.dry_run or args.synthetic))
    mismatches = []
    conversation_config = config.get("conversation", {})
    if not isinstance(conversation_config, Mapping):
        raise ContractError("config.conversation must be an object")
    if tuple(conversation_config.get("required_modes", ())) != REQUIRED_EXPERIMENTAL_STARTUP_MODES:
        raise ContractError("config must require common-handshake and greeting-suppressed runs")
    if (
        conversation_config.get("legacy_fixed_start_mode") != STARTUP_MODE_NATURAL
        or conversation_config.get("legacy_fixed_start_status") != NATURAL_START_STATUS
    ):
        raise ContractError("config must mark natural_model_start as greeting-confounded diagnostic only")
    if float(conversation_config.get("response", {}).get("tail_guard_ms", -1)) != (
        TAIL_GUARD_FRAMES * FRAME_MS
    ):
        raise ContractError("config tail guard must be exactly 2 seconds / 25 frames")
    conversation_required = bool(conversation_config.get("response", {}).get(
        "source_capture_contract_required", False))
    configured_capture_ms = float(conversation_config.get("response", {}).get(
        "post_user_max_ms", 0))
    for row in rows:
        sample_count = int(row.get("sample_count", 0))
        frame_count = int(row.get("frame_count", 0))
        if sample_count != frame_count * FRAME_SAMPLES:
            raise ContractError(f"{row['trial_id']}: manifest frame/sample counts disagree")
        conversation = row.get("conversation_contract")
        if conversation_required and not isinstance(conversation, Mapping):
            raise ContractError(f"{row['trial_id']}: frozen conversation capture contract is missing")
        if isinstance(conversation, Mapping):
            try:
                parsed = ConversationContract.from_manifest_row(row)
            except ConversationContractError as error:
                raise ContractError(f"{row['trial_id']}: {error}") from error
            if parsed.user_frame_count != frame_count:
                raise ContractError(f"{row['trial_id']}: conversation user frame count mismatch")
            if parsed.appended_zero_frame_count != parsed.target_end_frame_count - frame_count:
                raise ContractError(f"{row['trial_id']}: invalid exact-zero continuation length")
            if configured_capture_ms and abs(parsed.response_capture_ms - configured_capture_ms) > 1e-6:
                raise ContractError(f"{row['trial_id']}: response capture differs from frozen config")
            if row.get("model_repo") != MODEL_REPO or row.get("resolved_revision") != MODEL_REVISION:
                raise ContractError(f"{row['trial_id']}: frozen source model identity differs")
        path = args.input_artifact_root / require_relative_uri(str(row["audio_uri"]))
        if path.exists():
            if sha256_file(path) != row["audio_sha256"]:
                mismatches.append(str(row["trial_id"]))
                continue
            try:
                with wave.open(str(path), "rb") as handle:
                    wav_contract = (
                        handle.getnchannels(), handle.getframerate(), handle.getsampwidth(), handle.getnframes())
            except (wave.Error, EOFError) as error:
                raise ContractError(f"{row['trial_id']}: unreadable PCM WAV") from error
            if wav_contract != (1, 24000, 2, sample_count):
                raise ContractError(
                    f"{row['trial_id']}: WAV metadata/count differs from manifest: {wav_contract}")
        elif not path.exists() and not args.synthetic:
            mismatches.append(str(row["trial_id"]))
    if mismatches:
        raise ContractError(f"input artifact failures: {mismatches[:5]}")
    identity = build_run_identity(code_commit=_git_commit(), config=config, manifest_path=args.manifest,
                                  data_status=str(rows[0].get("data_status", "unknown")),
                                  model_repo=args.model_repo, model_revision=args.model_revision)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "run_identity.json", {**identity.__dict__, "run_identity_sha256": identity.sha256,
                                                        "validation_mode": "synthetic/local" if args.synthetic else "dry_run" if args.dry_run else "gpu"})
    write_json(args.output_root / "environment.json", {**environment, "harness_version": HARNESS_VERSION})
    write_jsonl(args.output_root / "input_hash_manifest.jsonl", [
        {"trial_id": row["trial_id"], "audio_uri": row["audio_uri"], "sha256": row["audio_sha256"]} for row in rows])
    if not (args.dry_run or args.synthetic):
        model = MoshiBackend(
            model_repo=args.model_repo,
            model_revision=args.model_revision,
            use_sampling=False,
        )
        first_row = rows[0]
        first_wav = args.input_artifact_root / require_relative_uri(str(first_row["audio_uri"]))
        contract = ConversationContract.from_manifest_row(first_row)
        encoded = model.encode_conversation_file(
            first_wav, target_frame_count=contract.target_end_frame_count
        )
        expected_user_shape = (1, 8, contract.user_frame_count)
        expected_conversation_shape = (1, 8, contract.target_end_frame_count)
        if (
            tuple(encoded.user_codes.shape) != expected_user_shape
            or tuple(encoded.conversation_codes.shape) != expected_conversation_shape
            or tuple(encoded.assistant_silence_codes.shape) != expected_conversation_shape
        ):
            raise ContractError("Mimi model contract produced an unexpected encoded stream shape")
        # The expensive identity smoke is deliberately bounded.  Exact full-horizon
        # replay is exercised by the <=8-row open-loop canary before a paid scan.
        smoke_frames = min(8, contract.query_end_frame)
        codes = encoded.conversation_codes
        hook_off = model.replay_codes(
            codes, hook_enabled=False, end_frame_exclusive=smoke_frames
        )
        first_replay = model.replay_codes(
            codes,
            sites=["resid_post"],
            capture_layers=[0],
            capture_frames=[0],
            end_frame_exclusive=smoke_frames,
        )
        second_replay = model.replay_codes(
            codes,
            sites=["resid_post"],
            capture_layers=[0],
            capture_frames=[0],
            end_frame_exclusive=smoke_frames,
        )
        deterministic = np.array_equal(first_replay.logits, second_replay.logits)
        hook_off_identity = np.array_equal(hook_off.logits, first_replay.logits)
        event_key = next((key for key in first_replay.event_tensors if key[0] == "resid_post"), None)
        if event_key is None:
            raise ContractError("model contract could not observe resid_post")
        identity_replay = model.replay_codes(
            codes,
            sites=["resid_post"],
            replacement={event_key: first_replay.event_tensors[event_key]},
            capture_layers=[0],
            capture_frames=[0],
            end_frame_exclusive=smoke_frames,
        )
        identity_noop = np.array_equal(first_replay.logits, identity_replay.logits)
        metadata = dict(model.metadata)
        configured_model = config.get("model", {})
        shape_contract = all((
            metadata.get("layers") == configured_model.get("layers"),
            metadata.get("heads") == configured_model.get("heads"),
            metadata.get("hidden_size") == configured_model.get("hidden_size"),
            metadata.get("head_dim") == configured_model.get("head_dim"),
        ))
        mimi_contract = all((
            metadata.get("sample_rate") == SAMPLE_RATE,
            metadata.get("frame_samples") == FRAME_SAMPLES,
            metadata.get("mimi_frame_rate") == 12.5,
            metadata.get("mimi_channels") == 1,
            metadata.get("mimi_num_codebooks") == 8,
            metadata.get("mimi_cardinality") == 2048,
            metadata.get("user_codebooks") == 8,
            metadata.get("assistant_codebooks") == 8,
            metadata.get("num_codebooks") == 17,
            metadata.get("dep_q") == 8,
            metadata.get("card") == 2048,
            metadata.get("delays") == [0, 0, *([1] * 7), 0, *([1] * 7)],
        ))
        dtype_contract = str(metadata.get("dtype", "")).removeprefix("torch.") == str(
            configured_model.get("dtype")
        )
        device_contract = str(metadata.get("device", "")).startswith("cuda")
        checks = {
            "exact_model_revision": (
                metadata.get("model_repo") == MODEL_REPO
                and metadata.get("model_revision") == MODEL_REVISION
            ),
            "model_type_moshi": metadata.get("model_type") == "moshi",
            "shape_contract": shape_contract,
            "mimi_contract": mimi_contract,
            "dtype_contract": dtype_contract,
            "device_contract": device_contract,
            "hook_off_identity": hook_off_identity,
            "identity_patch_noop": identity_noop,
        }
        diagnostics = {
            "deterministic_reset_replay": deterministic,
            "feedback_byte_identical": (
                hook_off.feedback_sha256
                == first_replay.feedback_sha256
                == second_replay.feedback_sha256
                == identity_replay.feedback_sha256
            ),
            "bounded_smoke_frames": smoke_frames,
            "user_codes_shape": list(encoded.user_codes.shape),
            "conversation_codes_shape": list(encoded.conversation_codes.shape),
            "assistant_silence_codes_shape": list(encoded.assistant_silence_codes.shape),
        }
        if not all(checks.values()) or not all(
            value for key, value in diagnostics.items() if key in {
                "deterministic_reset_replay", "feedback_byte_identical"
            }
        ):
            raise ContractError(
                f"loaded-model mechanistic smoke failed: checks={checks}, diagnostics={diagnostics}"
            )
        write_json(args.output_root / "model_contract.json", {
            "schema_version": "1.0.0",
            **metadata,
            "passed": True,
            "checks": checks,
            "diagnostics": diagnostics,
            "run_identity_sha256": identity.sha256,
            "code_commit": identity.code_commit,
            "config_sha256": sha256_file(args.config),
            "manifest_sha256": sha256_file(args.manifest),
        })
        readout_source = read_json(args.config.parent / "readouts.json")
        bound_readouts = []
        for readout in readout_source["readouts"]:
            token_ids = list(model.tokenizer.encode(str(readout["prefix"]), out_type=int))
            if not token_ids:
                raise ContractError(f"readout prefix tokenized to an empty sequence: {readout['id']}")
            bound_readouts.append({**readout, "prefix_token_ids": token_ids})
        values = sorted({str(row["old_value"]) for row in rows} | {str(row["new_value"]) for row in rows})
        candidate_token_ids = {
            value: list(model.tokenizer.encode(value, out_type=int)) for value in values
        }
        if any(not token_ids for token_ids in candidate_token_ids.values()):
            raise ContractError("a candidate verbalizer tokenized to an empty sequence")
        bound = {**readout_source, "readouts": bound_readouts,
                 "candidate_token_ids": candidate_token_ids,
                 "model_repo": args.model_repo,
                 "model_revision": args.model_revision,
                 "run_identity_sha256": identity.sha256,
                 "config_sha256": sha256_file(args.config),
                 "manifest_sha256": sha256_file(args.manifest)}
        bound["bound_readout_sha256"] = sha256_value(bound)
        write_json(args.output_root / "readouts.bound.json", bound)
    print(f"mechanistic contract passed for {len(rows)} trials -> {args.output_root}")
    return 0


def _load_trials(
    manifest: Path | None,
    role: str,
    folds: list[int] | None = None,
    role_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    if manifest is None or not manifest.exists():
        if role in {"smoke", "local_validation"}:
            return [
                {"trial_id": "syn-repair", "scenario_id": "syn-1", "condition": "repair", "old_value": "Boston",
                 "new_value": "Seattle", "frame_count": 12, "analysis_fold": 1},
                {"trial_id": "syn-clean", "scenario_id": "syn-1", "condition": "clean_current", "old_value": "Boston",
                 "new_value": "Seattle", "frame_count": 12, "analysis_fold": 1},
            ]
        raise ContractError("a mechanistic manifest is required")
    rows = read_jsonl(manifest)
    if not rows:
        raise ContractError("mechanistic manifest is empty")
    trial_ids = [str(row.get("trial_id", "")) for row in rows]
    if any(not trial_id for trial_id in trial_ids) or len(set(trial_ids)) != len(trial_ids):
        raise ContractError("mechanistic manifest has a missing or duplicate trial ID")

    if role_manifest is not None:
        role_rows = read_jsonl(role_manifest)
        role_by_id: dict[str, dict[str, Any]] = {}
        for role_row in role_rows:
            keys = {
                str(role_row[key])
                for key in ("trial_id", "prepared_stimulus_id")
                if role_row.get(key)
            }
            if not keys:
                raise ContractError("role manifest row has no immutable trial identifier")
            for key in keys:
                if key in role_by_id:
                    raise ContractError(f"role manifest has duplicate binding: {key}")
                role_by_id[key] = role_row
        role_hash = sha256_file(role_manifest)
        rebound: list[dict[str, Any]] = []
        for row in rows:
            candidates = {
                id(role_by_id[key]): role_by_id[key]
                for key in (str(row["trial_id"]), str(row.get("prepared_stimulus_id", "")))
                if key in role_by_id
            }
            if len(candidates) != 1:
                raise ContractError(f"{row['trial_id']}: role-manifest binding is missing or ambiguous")
            role_row = next(iter(candidates.values()))
            bound_role = role_row.get("role", role_row.get("inferential_role"))
            if not isinstance(bound_role, str) or not bound_role:
                raise ContractError(f"{row['trial_id']}: role-manifest entry has no role")
            expected_manifest_hash = row.get("role_manifest_sha256")
            expected_binding_hash = row.get("role_binding_sha256")
            if expected_manifest_hash != role_hash or expected_binding_hash != sha256_value(role_row):
                raise ContractError(f"{row['trial_id']}: immutable role-manifest hash mismatch")
            if row.get("role") != bound_role:
                raise ContractError(f"{row['trial_id']}: manifest role disagrees with role manifest")
            rebound.append(row)
        rows = rebound

    requested_folds = None
    if folds is not None:
        if not folds or len(set(folds)) != len(folds):
            raise ContractError("--folds must contain unique fold IDs")
        requested_folds = set(folds)

    v2_fold_roles = {
        "discovery": {1, 2, 3},
        "internal_validation": {4, 5},
        "smoke": {1, 2, 3},
    }
    if role in v2_fold_roles:
        allowed_folds = v2_fold_roles[role]
        if requested_folds is not None and not requested_folds <= allowed_folds:
            raise ContractError(
                f"role {role} cannot use folds {sorted(requested_folds - allowed_folds)}"
            )
        selected_folds = requested_folds or allowed_folds
        selected: list[dict[str, Any]] = []
        for row in rows:
            fold_value = row.get("analysis_fold")
            if isinstance(fold_value, bool) or not isinstance(fold_value, int) or fold_value not in range(1, 6):
                raise ContractError(f"{row['trial_id']}: v2 role requires a frozen fold in 1..5")
            expected_role = "discovery" if fold_value <= 3 else "internal_validation"
            if row.get("role") != expected_role:
                raise ContractError(
                    f"{row['trial_id']}: role disagrees with frozen v2 fold policy"
                )
            if fold_value in selected_folds:
                selected.append(row)
    else:
        if requested_folds is not None:
            selected = [row for row in rows if row.get("analysis_fold") in requested_folds]
        else:
            selected = list(rows)
        selected = [row for row in selected if row.get("role") == role]

    if not selected:
        raise ContractError(f"role {role!r} selects no trials")
    return selected


def _encoded_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    trial_ids = [str(row.get("trial_id", "")) for row in rows]
    by_id = {trial_id: row for trial_id, row in zip(trial_ids, rows, strict=True)}
    if not rows or any(not trial_id for trial_id in trial_ids) or len(by_id) != len(rows):
        raise ContractError("encoded manifest is empty or has missing/duplicate trial IDs")
    return rows, by_id


def _load_encoded_array(
    encoded_manifest: Path, row: Mapping[str, Any], key: str,
    *, require_current_contract: bool = False,
) -> np.ndarray:
    if key not in {"user_codes", "conversation_codes", "assistant_silence_codes"}:
        raise ContractError(f"unsupported encoded array: {key}")
    relative = require_relative_uri(str(row.get("codes_uri", "")))
    root = encoded_manifest.parent.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ContractError(f"encoded tensor escapes its manifest root: {relative}") from error
    if not path.is_file():
        raise ContractError(f"encoded tensor is missing: {path}")
    archive_sha = row.get("archive_sha256")
    if archive_sha is not None and sha256_file(path) != validate_sha256(
        str(archive_sha), f"{row.get('trial_id')} encoded archive"):
        raise ContractError(f"encoded archive hash mismatch: {row.get('trial_id')}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            archive_key = key
            if key == "user_codes" and key not in archive.files and not require_current_contract:
                archive_key = "codes"
            if archive_key not in archive.files:
                raise ContractError(f"encoded archive is missing {key}: {row.get('trial_id')}")
            value = np.ascontiguousarray(np.asarray(archive[archive_key]))
            embedded = (
                _archive_identity(archive, path)
                if "artifact_identity_sha256" in archive.files else None
            )
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot load encoded tensor {path}: {error}") from error
    identity = row.get("artifact_identity_sha256")
    if require_current_contract and not isinstance(identity, str):
        raise ContractError(f"{row.get('trial_id')}: encoded row lacks an artifact identity")
    if identity is not None:
        identity = validate_sha256(str(identity), f"{row.get('trial_id')} artifact identity")
        if embedded != identity:
            raise ContractError(f"encoded artifact identity mismatch: {row.get('trial_id')}")
    prefix = "user_codes" if key == "user_codes" else key
    hash_field = f"{prefix}_sha256"
    shape_field = f"{prefix}_shape"
    dtype_field = f"{prefix}_dtype"
    if key == "user_codes" and hash_field not in row and not require_current_contract:
        hash_field, shape_field, dtype_field = "codes_sha256", "shape", "dtype"
    expected_hash = row.get(hash_field)
    if not isinstance(expected_hash, str) or _array_sha256(value) != validate_sha256(
        expected_hash, f"{row.get('trial_id')} {key}"):
        raise ContractError(f"encoded tensor hash mismatch: {row.get('trial_id')}:{key}")
    if list(value.shape) != row.get(shape_field) or str(value.dtype) != row.get(dtype_field):
        raise ContractError(f"encoded tensor shape/dtype mismatch: {row.get('trial_id')}:{key}")
    if value.ndim != 3 or tuple(value.shape[:2]) != (1, 8) or not np.issubdtype(
        value.dtype, np.integer):
        raise ContractError(f"invalid encoded Mimi tensor: {row.get('trial_id')}:{key}")
    frame_end_field = {
        "user_codes": "user_frame_end_exclusive",
        "conversation_codes": "conversation_frame_end_exclusive",
        "assistant_silence_codes": "assistant_silence_frame_end_exclusive",
    }[key]
    if require_current_contract:
        end = _exact_int_field(row.get(frame_end_field), f"{row.get('trial_id')} {frame_end_field}")
        if end != int(value.shape[-1]):
            raise ContractError(f"encoded half-open coverage mismatch: {row.get('trial_id')}:{key}")
    return value


def _load_codes(encoded_manifest: Path, row: Mapping[str, Any]) -> np.ndarray:
    """Compatibility loader for prepared-user codes used by legacy scan paths."""
    return _load_encoded_array(encoded_manifest, row, "user_codes")


def _validate_open_loop_policy(config: Mapping[str, Any]) -> None:
    policy = config.get("open_loop_policy")
    expected = {
        "primary": "zero_text_and_audio_tokens",
        "text_feedback": "model_zero_token",
        "audio_feedback": "model_zero_token",
        "sampled_tokens_enter_feedback": False,
    }
    if not isinstance(policy, Mapping) or any(policy.get(key) != value for key, value in expected.items()):
        raise ContractError("config does not freeze the required zero-feedback open-loop policy")


def _expected_moshi_feedback_sha256(backend: Any, lm_steps: int) -> str:
    model = backend.lm_gen.lm_model
    zero = int(model.zero_token_id)
    text = np.full((1,), zero, dtype=np.int64)
    audio = np.full((1, int(model.dep_q)), zero, dtype=np.int64)
    digest = hashlib.sha256()
    for _ in range(lm_steps):
        digest.update(text.tobytes(order="C"))
        digest.update(audio.tobytes(order="C"))
    return digest.hexdigest()


def _replay_equal(left: Any, right: Any) -> bool:
    if left.feedback_sha256 != right.feedback_sha256:
        return False
    if not np.array_equal(left.logits, right.logits):
        return False
    if set(left.event_tensors) != set(right.event_tensors):
        return False
    if not all(np.array_equal(left.event_tensors[key], right.event_tensors[key])
               for key in left.event_tensors):
        return False
    if set(left.activations) != set(right.activations):
        return False
    return all(np.array_equal(left.activations[key], right.activations[key])
               for key in left.activations)


def _basic_readout_plan(source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    readouts = source.get("readouts")
    schedules = source.get("emission_schedules")
    if not isinstance(readouts, list) or not readouts:
        raise ContractError("readout config must contain at least one readout")
    if not isinstance(schedules, list) or not schedules:
        raise ContractError("readout config must contain at least one emission schedule")
    normalized_readouts: list[dict[str, Any]] = []
    seen_readouts: set[str] = set()
    for item in readouts:
        if not isinstance(item, Mapping):
            raise ContractError("readout entries must be objects")
        readout_id = str(item.get("id", ""))
        prefix, anchor = item.get("prefix"), item.get("anchor")
        if not readout_id or readout_id in seen_readouts or not isinstance(prefix, str) or not prefix:
            raise ContractError("readout IDs must be unique and prefixes must be non-empty")
        if not isinstance(anchor, str) or not anchor:
            raise ContractError(f"readout {readout_id} has no semantic anchor")
        seen_readouts.add(readout_id)
        normalized_readouts.append(dict(item))
    normalized_schedules: list[dict[str, Any]] = []
    seen_schedules: set[str] = set()
    for item in schedules:
        if not isinstance(item, Mapping):
            raise ContractError("emission schedules must be objects")
        schedule_id = str(item.get("id", ""))
        if not schedule_id or schedule_id in seen_schedules:
            raise ContractError("emission schedule IDs must be present and unique")
        offset = _exact_int_field(
            item.get("prefix_start_offset_frames"),
            f"schedule {schedule_id} prefix_start_offset_frames")
        padding = _exact_int_field(
            item.get("pad_frames_between_tokens"),
            f"schedule {schedule_id} pad_frames_between_tokens")
        if offset < 0 or padding < 0:
            raise ContractError(f"schedule {schedule_id} has a negative frame offset")
        seen_schedules.add(schedule_id)
        normalized_schedules.append({
            **dict(item),
            "prefix_start_offset_frames": offset,
            "pad_frames_between_tokens": padding,
        })
    return normalized_readouts, normalized_schedules


def _synthetic_scores(
    backend: SyntheticBackend, trial: Mapping[str, Any], candidates: Mapping[str, str],
    *, anchor_end_exclusive: int, prefix: str, schedule: Mapping[str, Any],
) -> dict[str, float]:
    replay_trial = {**dict(trial), "frame_count": anchor_end_exclusive}
    margin = float(backend.replay(replay_trial, ["resid_post"]).logits[0])
    target = str(trial.get("new_value"))
    schedule_penalty = 1e-4 * (
        int(schedule["prefix_start_offset_frames"])
        + int(schedule["pad_frames_between_tokens"])
        + len(prefix)
    )
    return {
        name: (margin / 2.0 if text == target else -margin / 2.0) - schedule_penalty
        for name, text in candidates.items()
    }


def validate_open_loop(argv: Sequence[str]) -> int:
    parser = _parser("Validate every claimed strict open-loop property from executed evidence.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--encoded-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gpu-trials", type=int, default=8)
    args = parser.parse_args(argv)
    config = read_json(args.config)
    if not isinstance(config, Mapping):
        raise ContractError("mechanistic config must be an object")
    model_config = _frozen_model_config(config)
    _validate_open_loop_policy(config)
    rows, _ = _encoded_rows(args.encoded_manifest)
    if not args.synthetic and not 2 <= len(rows) <= args.max_gpu_trials <= 8:
        raise ContractError(
            "real open-loop validation is a bounded 2..8-trial GPU canary; "
            "select a matched canary manifest first"
        )
    validated_inputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for row in rows:
        trial_id = str(row["trial_id"])
        if not args.synthetic:
            if row.get("synthetic") is not False:
                raise ContractError(f"{trial_id}: real open-loop validation rejects synthetic codes")
            if row.get("model_repo") != MODEL_REPO or row.get("model_revision") != MODEL_REVISION:
                raise ContractError(f"{trial_id}: encoded model identity mismatch")
        user = _load_encoded_array(
            args.encoded_manifest, row, "user_codes", require_current_contract=True)
        conversation = _load_encoded_array(
            args.encoded_manifest, row, "conversation_codes", require_current_contract=True)
        silence = _load_encoded_array(
            args.encoded_manifest, row, "assistant_silence_codes", require_current_contract=True)
        if not np.array_equal(conversation[..., :user.shape[-1]], user):
            raise ContractError(f"{trial_id}: encoded conversation/user prefix mismatch")
        if int(silence.shape[-1]) != int(conversation.shape[-1]):
            raise ContractError(f"{trial_id}: assistant silence coverage mismatch")
        frames = _exact_int_field(
            row.get("query_end_frame_exclusive"), f"{trial_id} query end")
        if not 1 <= frames <= int(user.shape[-1]):
            raise ContractError(f"{trial_id}: query end is outside exact user-code coverage")
        validated_inputs[trial_id] = (user, conversation, silence, frames)

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row.get("scenario_id", ""))].append(row)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for scenario_rows in by_scenario.values():
        clean = next((row for row in scenario_rows
                      if str(row.get("condition", "")).startswith("clean")), None)
        repair = next((row for row in scenario_rows
                       if not str(row.get("condition", "")).startswith("clean")), None)
        if clean is not None and repair is not None:
            pairs.append((clean, repair))
    if not pairs:
        raise ContractError("open-loop validation requires a scenario-matched clean/repair pair")

    readout_source = read_json(args.config.parent / "readouts.json")
    readouts, schedules = _basic_readout_plan(readout_source)
    query_readout = next((item for item in readouts if item["anchor"] == "query_end"), readouts[0])
    backend = SyntheticBackend() if args.synthetic else MoshiBackend(
        model_repo=MODEL_REPO, model_revision=MODEL_REVISION, use_sampling=False)
    tolerance = 1e-6
    trial_evidence: list[dict[str, Any]] = []
    deterministic_results: list[bool] = []
    feedback_absent_results: list[bool] = []
    delay_results: list[bool] = []
    replay_inputs: dict[str, tuple[Any, int]] = {}
    for row in rows:
        trial_id = str(row["trial_id"])
        _, conversation, _, frames = validated_inputs[trial_id]
        if args.synthetic:
            replay_trial = {**row, "frame_count": frames}
            first = backend.replay(replay_trial, ["resid_post"])
            second = backend.replay(replay_trial, ["resid_post"])
            expected_feedback = hashlib.sha256(
                np.zeros((frames, 9), dtype=np.int64).tobytes(order="C")).hexdigest()
            observed = np.asarray(first.activations.get("resid_post"))
            event_frames = list(range(int(observed.shape[1]))) if observed.ndim >= 2 else []
            replay_inputs[trial_id] = (replay_trial, frames)
        else:
            codes = backend.torch.as_tensor(conversation, device=backend.device)
            capture_frames = sorted({0, frames - 1})
            kwargs = {
                "sites": ["resid_post"], "capture_layers": [0],
                "capture_frames": capture_frames, "end_frame_exclusive": frames,
            }
            first = backend.replay_codes(codes, **kwargs)
            second = backend.replay_codes(codes, **kwargs)
            expected_feedback = _expected_moshi_feedback_sha256(backend, frames + 1)
            event_frames = sorted({key[2] for key in first.event_tensors
                                   if key[0] == "resid_post" and key[1] == 0})
            replay_inputs[trial_id] = (codes, frames)
        deterministic = _replay_equal(first, second)
        feedback_absent = first.feedback_sha256 == expected_feedback
        expected_event_frames = sorted({0, frames - 1}) if not args.synthetic else list(range(frames))
        delay_valid = (
            int(first.frame_count) == frames
            and int(first.lm_step_count) == frames + 1
            and event_frames == expected_event_frames
        )
        if not args.synthetic:
            delay_valid = delay_valid and max(backend.lm_gen.lm_model.delays) == int(
                model_config.get("max_lm_delay")) == 1
        deterministic_results.append(deterministic)
        feedback_absent_results.append(feedback_absent)
        delay_results.append(delay_valid)
        trial_evidence.append({
            "trial_id": trial_id,
            "end_frame_exclusive": frames,
            "expected_lm_step_count": frames + 1,
            "observed_lm_step_count": int(first.lm_step_count),
            "captured_event_frames": event_frames,
            "logits_sha256": _array_sha256(np.asarray(first.logits)),
            "repeat_logits_sha256": _array_sha256(np.asarray(second.logits)),
            "feedback_sha256": first.feedback_sha256,
            "expected_forced_feedback_sha256": expected_feedback,
            "deterministic": deterministic,
            "sampled_feedback_absent": feedback_absent,
            "delay_mapping_valid": delay_valid,
        })

    paired_evidence: list[dict[str, Any]] = []
    for clean, repair in pairs:
        clean_input, clean_end = replay_inputs[str(clean["trial_id"])]
        repair_input, repair_end = replay_inputs[str(repair["trial_id"])]
        shared_end = min(clean_end, repair_end)
        if args.synthetic:
            clean_replay = backend.replay({**clean_input, "frame_count": shared_end}, ["resid_post"])
            repair_replay = backend.replay({**repair_input, "frame_count": shared_end}, ["resid_post"])
        else:
            pair_kwargs = {"sites": [], "hook_enabled": False,
                           "end_frame_exclusive": shared_end}
            clean_replay = backend.replay_codes(clean_input, **pair_kwargs)
            repair_replay = backend.replay_codes(repair_input, **pair_kwargs)
        identical = clean_replay.feedback_sha256 == repair_replay.feedback_sha256
        paired_evidence.append({
            "scenario_id": clean.get("scenario_id"),
            "clean_trial_id": clean["trial_id"], "repair_trial_id": repair["trial_id"],
            "compared_frame_span": [0, shared_end],
            "clean_feedback_sha256": clean_replay.feedback_sha256,
            "repair_feedback_sha256": repair_replay.feedback_sha256,
            "identical": identical,
        })

    identity_row = pairs[0][1]
    identity_input, identity_end = replay_inputs[str(identity_row["trial_id"])]
    identity_frame = identity_end - 1
    if args.synthetic:
        identity_metric = backend.patch(
            identity_input, identity_input, component="resid_post", layer=0,
            head=None, anchor_frame=identity_frame)
        identity_delta = float(identity_metric["delta_M"])
        identity_noop = math.isfinite(identity_delta) and abs(identity_delta) <= tolerance
        identity_evidence = {
            "kind": "synthetic_receiver_self_patch", "frame": identity_frame,
            "delta_M": identity_delta, "tolerance": tolerance,
        }
    else:
        capture_kwargs = {
            "sites": ["resid_post"], "capture_layers": [0],
            "capture_frames": [identity_frame], "end_frame_exclusive": identity_end,
        }
        baseline = backend.replay_codes(identity_input, **capture_kwargs)
        event_key = ("resid_post", 0, identity_frame)
        tensor = baseline.event_tensors.get(event_key)
        if tensor is None:
            identity_noop = False
            identity_logits_sha = None
            identity_feedback_sha = None
        else:
            identity = backend.replay_codes(
                identity_input, replacement={event_key: tensor}, **capture_kwargs)
            identity_noop = (
                np.array_equal(baseline.logits, identity.logits)
                and baseline.feedback_sha256 == identity.feedback_sha256)
            identity_logits_sha = _array_sha256(np.asarray(identity.logits))
            identity_feedback_sha = identity.feedback_sha256
        identity_evidence = {
            "kind": "captured_tensor_identity_hook", "site": "resid_post", "layer": 0,
            "frame": identity_frame, "baseline_logits_sha256": _array_sha256(np.asarray(baseline.logits)),
            "identity_logits_sha256": identity_logits_sha,
            "baseline_feedback_sha256": baseline.feedback_sha256,
            "identity_feedback_sha256": identity_feedback_sha,
            "exact_noop": identity_noop,
        }

    target, stale = str(identity_row.get("new_value", "")), str(identity_row.get("old_value", ""))
    if not target or not stale or target == stale:
        raise ContractError("candidate-order validation requires distinct non-empty target/stale values")
    candidate_evidence: list[dict[str, Any]] = []
    candidate_checks: list[bool] = []
    if args.synthetic:
        for schedule in schedules:
            forward = _synthetic_scores(
                backend, identity_input, {"target": target, "stale": stale},
                anchor_end_exclusive=identity_end, prefix=str(query_readout["prefix"]),
                schedule=schedule)
            reverse = _synthetic_scores(
                backend, identity_input, {"stale": stale, "target": target},
                anchor_end_exclusive=identity_end, prefix=str(query_readout["prefix"]),
                schedule=schedule)
            deltas = {name: forward[name] - reverse[name] for name in ("target", "stale")}
            invariant = all(math.isfinite(value) and abs(value) <= tolerance for value in deltas.values())
            candidate_checks.append(invariant)
            candidate_evidence.append({
                "readout_id": query_readout["id"], "schedule_id": schedule["id"],
                "forward": forward, "reverse": reverse, "score_deltas": deltas,
                "invariant": invariant,
            })
    else:
        backend.replay_codes(
            identity_input, sites=[], hook_enabled=False,
            end_frame_exclusive=identity_end)
        snapshot = backend.lm_gen.snapshot_streaming_state()
        for schedule in schedules:
            score_kwargs = {
                "prefix": str(query_readout["prefix"]),
                "prefix_start_offset_frames": int(schedule["prefix_start_offset_frames"]),
                "pad_frames_between_tokens": int(schedule["pad_frames_between_tokens"]),
            }
            forward = backend.score_candidates(
                snapshot, {"target": target, "stale": stale}, **score_kwargs)
            reverse = backend.score_candidates(
                snapshot, {"stale": stale, "target": target}, **score_kwargs)
            deltas = {name: float(forward[name]) - float(reverse[name])
                      for name in ("target", "stale")}
            invariant = all(math.isfinite(value) and abs(value) <= tolerance for value in deltas.values())
            candidate_checks.append(invariant)
            candidate_evidence.append({
                "readout_id": query_readout["id"], "schedule_id": schedule["id"],
                "forward": {key: float(value) for key, value in forward.items()},
                "reverse": {key: float(value) for key, value in reverse.items()},
                "score_deltas": deltas, "invariant": invariant,
            })

    checks = {
        "paired_feedback_identical": bool(paired_evidence) and all(
            item["identical"] for item in paired_evidence),
        "sampled_feedback_absent": bool(feedback_absent_results) and all(feedback_absent_results),
        "deterministic_replay": bool(deterministic_results) and all(deterministic_results),
        "identity_patch_noop": bool(identity_noop),
        "candidate_order_invariant": bool(candidate_checks) and all(candidate_checks),
        "delay_mapping_valid": bool(delay_results) and all(delay_results),
    }
    report = {
        "schema_version": "1.1.0",
        "analysis_status": "synthetic_local_validation" if args.synthetic else "empirical_gpu_canary",
        "trial_count": len(rows), "paired_comparison_count": len(paired_evidence),
        "code_commit": _git_commit(), "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "config_sha256": sha256_file(args.config),
        "encoded_manifest_sha256": sha256_file(args.encoded_manifest),
        "readouts_sha256": sha256_file(args.config.parent / "readouts.json"),
        "numeric_tolerance": tolerance,
        "evidence": {
            "trials": trial_evidence, "paired_feedback": paired_evidence,
            "identity_patch": identity_evidence, "candidate_order": candidate_evidence,
        },
        "checks": checks, "passed": all(checks.values()),
        "limitations": (["Synthetic execution is not evidence about Moshiko."] if args.synthetic else []),
    }
    write_json(args.output, report)
    if not report["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ContractError(f"open-loop validation failed executed checks: {failed}")
    print(f"open-loop execution evidence passed -> {args.output}")
    return 0


def _capture_archive(
    path: Path, *, identity_sha256: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if _archive_identity(archive, path) != identity_sha256:
                raise ContractError(f"capture archive identity mismatch: {path}")
            if "capture_metadata_json" not in archive.files or "features" not in archive.files:
                raise ContractError(f"capture archive metadata/features are missing: {path}")
            raw_metadata = np.asarray(archive["capture_metadata_json"])
            if raw_metadata.shape != () or not isinstance(raw_metadata.item(), str):
                raise ContractError(f"capture archive metadata is malformed: {path}")
            metadata = json.loads(raw_metadata.item())
            if not isinstance(metadata, dict) or not isinstance(metadata.get("tensors"), list):
                raise ContractError(f"capture archive metadata is not an object: {path}")
            descriptors = metadata["tensors"]
            tensor_keys = {str(item.get("key", "")) for item in descriptors
                           if isinstance(item, Mapping)}
            expected_keys = tensor_keys | {
                "artifact_identity_sha256", "capture_metadata_json", "features"}
            if not tensor_keys or set(archive.files) != expected_keys:
                raise ContractError(f"capture archive tensor set differs from its metadata: {path}")
            for item in descriptors:
                key = str(item["key"])
                value = np.asarray(archive[key])
                if (
                    list(value.shape) != item.get("shape")
                    or str(value.dtype) != item.get("dtype")
                    or _array_sha256(value) != item.get("sha256")
                ):
                    raise ContractError(f"capture tensor failed hash/shape/dtype verification: {key}")
            features = np.ascontiguousarray(np.asarray(archive["features"]))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot validate capture archive {path}: {error}") from error
    if features.ndim != 2 or features.shape[0] != 1 or not np.isfinite(features).all():
        raise ContractError(f"capture probe feature is invalid: {path}")
    if (
        list(features.shape) != metadata.get("feature_shape")
        or str(features.dtype) != metadata.get("feature_dtype")
        or _array_sha256(features) != metadata.get("feature_tensor_sha256")
    ):
        raise ContractError(f"capture probe feature failed verification: {path}")
    return features, descriptors, metadata


def _capture_probe_feature(
    captured: Mapping[str, np.ndarray], descriptors: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, str]:
    preferred = [
        np.asarray(captured[str(item["key"])], dtype=np.float32).reshape(-1)
        for item in descriptors if item.get("site") == "resid_post"
    ]
    non_logits = [
        np.asarray(captured[str(item["key"])], dtype=np.float32).reshape(-1)
        for item in descriptors if item.get("site") != "logits"
    ]
    vectors = preferred or non_logits
    if not vectors:
        logit_vectors = [
            np.asarray(captured[str(item["key"])], dtype=np.float32).reshape(-1)
            for item in descriptors
        ]
        feature = np.asarray(
            [[float(vector.mean()) for vector in logit_vectors]], dtype=np.float32)
        policy = "requested_logit_tensor_scalar_means"
    elif len({vector.size for vector in vectors}) == 1:
        feature = np.mean(np.stack(vectors, axis=0), axis=0, dtype=np.float32)[None]
        policy = "mean_requested_resid_post_vectors" if preferred else "mean_requested_site_vectors"
    else:
        feature = np.asarray([[float(vector.mean()) for vector in vectors]], dtype=np.float32)
        policy = "requested_tensor_scalar_means"
    if feature.size == 0 or not np.isfinite(feature).all():
        raise ContractError("requested captures cannot produce a finite probe feature")
    return np.ascontiguousarray(feature), policy


def capture_activations(argv: Sequence[str]) -> int:
    parser = _parser("Capture only requested activation sites/layers/semantic frames.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--encoded-manifest", type=Path)
    parser.add_argument("--anchor-map", type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--sites", default="logits,resid_post")
    parser.add_argument("--layers", required=True,
                        help="Explicit unique layer IDs, e.g. 0,8,16,24,31.")
    parser.add_argument("--anchors", default="query_end")
    parser.add_argument("--max-capture-tensors-per-trial", type=int, default=512)
    parser.add_argument("--max-bytes-per-trial", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    if not isinstance(config, Mapping):
        raise ContractError("mechanistic config must be an object")
    model_config = _frozen_model_config(config)
    if args.manifest is None:
        args.manifest = _infer_run_file(args.output_root, "manifests/mechanistic_trials.jsonl")
    if args.manifest is None:
        raise ContractError("activation capture requires --manifest")
    if args.anchor_map is None:
        args.anchor_map = _infer_run_file(args.output_root, "anchor_map.jsonl")
    if args.anchor_map is None:
        raise ContractError("activation capture requires --anchor-map")
    if not args.synthetic and args.encoded_manifest is None:
        args.encoded_manifest = _infer_run_file(args.output_root, "encoded_user_manifest.jsonl")
    if not args.synthetic and args.encoded_manifest is None:
        raise ContractError("Moshiko activation capture requires --encoded-manifest")
    trials = _load_trials(args.manifest, args.role)
    sites = _csv(args.sites)
    layers = _ints(args.layers)
    anchors = _csv(args.anchors)
    if (
        not sites or len(set(sites)) != len(sites)
        or not layers or len(set(layers)) != len(layers)
        or not anchors or len(set(anchors)) != len(anchors)
    ):
        raise ContractError("capture sites, layers, and anchors must be non-empty and unique")
    unknown_sites = set(sites) - (REQUIRED_SITES | {"logits"})
    if unknown_sites:
        raise ContractError(f"unsupported capture sites: {sorted(unknown_sites)}")
    max_layers = 6 if args.synthetic else _exact_int_field(model_config.get("layers"), "model layers")
    if any(layer < 0 or layer >= max_layers for layer in layers):
        raise ContractError(f"capture layers must lie in [0, {max_layers})")
    if args.max_capture_tensors_per_trial < 1 or args.max_bytes_per_trial < 1:
        raise ContractError("capture tensor/byte limits must be positive")
    requested_tensor_count = len(anchors) * (
        len(layers) * len([site for site in sites if site != "logits"])
        + int("logits" in sites)
    )
    if requested_tensor_count > args.max_capture_tensors_per_trial:
        raise ContractError(
            f"requested {requested_tensor_count} tensors/trial exceeds the explicit safety cap "
            f"{args.max_capture_tensors_per_trial}"
        )
    _, anchor_by_key = _anchor_lookup(args.anchor_map)
    encoded_by_id: dict[str, dict[str, Any]] = {}
    if not args.synthetic:
        assert args.encoded_manifest is not None
        _, encoded_by_id = _encoded_rows(args.encoded_manifest)
    config_sha = sha256_file(args.config)
    manifest_sha = sha256_file(args.manifest)
    anchor_sha = sha256_file(args.anchor_map)
    encoded_sha = sha256_file(args.encoded_manifest) if args.encoded_manifest else None
    code_commit = _git_commit()
    feature_root = args.output_root / "features"
    capture_manifest = args.output_root / "capture_manifest.jsonl"
    if capture_manifest.exists() and not args.resume:
        raise ContractError("capture manifest already exists; use --resume after identity verification")
    existing_rows = read_jsonl(capture_manifest) if capture_manifest.exists() else []
    existing_by_id = {str(row.get("trial_id", "")): row for row in existing_rows}
    selected_ids = {str(trial["trial_id"]) for trial in trials}
    if len(existing_by_id) != len(existing_rows) or not set(existing_by_id) <= selected_ids:
        raise ContractError("capture resume manifest has duplicate or out-of-scope trial IDs")
    jobs: list[dict[str, Any]] = []
    for trial in trials:
        trial_id = str(trial["trial_id"])
        anchor_rows_for_trial = []
        for anchor in anchors:
            anchor_row = anchor_by_key.get((trial_id, anchor))
            if anchor_row is None:
                raise ContractError(f"{trial_id}: requested anchor is missing: {anchor}")
            anchor_rows_for_trial.append({
                "anchor": anchor, "frame": int(anchor_row["frame"]),
                "time_ms": anchor_row.get("time_ms"), "timebase": anchor_row.get("timebase"),
            })
        frames = sorted({int(row["frame"]) for row in anchor_rows_for_trial})
        if args.synthetic:
            available_frames = _exact_int_field(trial.get("frame_count"), f"{trial_id} frame_count")
            encoded_row = None
        else:
            encoded_row = encoded_by_id.get(trial_id)
            if encoded_row is None:
                raise ContractError(f"missing encoded row for {trial_id}")
            if encoded_row.get("synthetic") is not False:
                raise ContractError(f"{trial_id}: activation capture rejects synthetic encoded data")
            available_frames = _exact_int_field(
                encoded_row.get("conversation_frame_end_exclusive"),
                f"{trial_id} conversation frame end")
            validated_conversation = _load_encoded_array(
                args.encoded_manifest, encoded_row, "conversation_codes",
                require_current_contract=True)
            if int(validated_conversation.shape[-1]) != available_frames:
                raise ContractError(f"{trial_id}: encoded capture coverage mismatch")
        if any(frame < 0 or frame >= available_frames for frame in frames):
            raise ContractError(f"{trial_id}: requested capture frame is outside encoded coverage")
        identity = {
            "schema_version": "1.0.0", "operation": "capture_activations",
            "harness_version": HARNESS_VERSION, "code_commit": code_commit,
            "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
            "config_sha256": config_sha, "manifest_sha256": manifest_sha,
            "encoded_manifest_sha256": encoded_sha, "anchor_map_sha256": anchor_sha,
            "trial_id": trial_id, "trial_sha256": sha256_value(trial),
            "encoded_artifact_identity_sha256": (
                encoded_row.get("artifact_identity_sha256") if encoded_row else None),
            "role": args.role, "sites": sites, "layers": layers,
            "anchors": anchor_rows_for_trial, "replay_frame_span": [0, max(frames) + 1],
            "synthetic": bool(args.synthetic),
        }
        identity_sha = sha256_value(identity)
        destination = feature_root / f"{identity_sha}.npz"
        old = existing_by_id.get(trial_id)
        if old is not None and old.get("capture_identity_sha256") != identity_sha:
            raise ContractError(f"{trial_id}: capture resume identity mismatch")
        if old is not None and not destination.is_file():
            raise ContractError(f"{trial_id}: capture resume artifact is missing")
        if destination.exists() and not args.resume:
            raise ContractError(f"{trial_id}: capture artifact already exists; use --resume")
        jobs.append({
            "trial": trial, "encoded": encoded_row, "anchors": anchor_rows_for_trial,
            "frames": frames, "identity": identity, "identity_sha": identity_sha,
            "destination": destination, "old": old,
        })
    expected_paths = {job["destination"].resolve() for job in jobs}
    if feature_root.exists():
        extras = [path for path in feature_root.glob("*.npz") if path.resolve() not in expected_paths]
        if extras:
            raise ContractError("capture output contains artifacts from a different run identity")

    pending = [job for job in jobs if not job["destination"].exists()]
    backend = None
    if pending:
        backend = SyntheticBackend() if args.synthetic else MoshiBackend(
            model_repo=MODEL_REPO, model_revision=MODEL_REVISION, use_sampling=False)
    output: list[dict[str, Any]] = []
    for job in jobs:
        trial = job["trial"]
        trial_id = str(trial["trial_id"])
        destination = job["destination"]
        if not destination.exists():
            assert backend is not None
            frames = job["frames"]
            end_frame_exclusive = max(frames) + 1
            instrumentation_sites = [site for site in sites if site != "logits"]
            replay_sites = instrumentation_sites or ["resid_post"]
            if args.synthetic:
                result = backend.replay(
                    {**dict(trial), "frame_count": end_frame_exclusive}, replay_sites)
                event_tensors = {
                    (site, layer, frame): np.asarray(result.activations[site][layer, frame])
                    for site in instrumentation_sites for layer in layers for frame in frames
                }
            else:
                assert args.encoded_manifest is not None
                conversation = _load_encoded_array(
                    args.encoded_manifest, job["encoded"], "conversation_codes",
                    require_current_contract=True)
                codes = backend.torch.as_tensor(conversation, device=backend.device)
                if instrumentation_sites:
                    result = backend.replay_codes(
                        codes, sites=replay_sites, capture_layers=layers,
                        capture_frames=frames, end_frame_exclusive=end_frame_exclusive)
                else:
                    result = backend.replay_codes(
                        codes, sites=[], hook_enabled=False,
                        end_frame_exclusive=end_frame_exclusive)
                event_tensors = result.event_tensors
            expected_events = {
                (site, layer, frame)
                for site in instrumentation_sites for layer in layers for frame in frames
            }
            if set(event_tensors) != expected_events:
                missing = sorted(expected_events - set(event_tensors))[:5]
                extra = sorted(set(event_tensors) - expected_events)[:5]
                raise ContractError(
                    f"{trial_id}: requested activation event coverage mismatch; "
                    f"missing={missing}, extra={extra}"
                )
            captured: dict[str, np.ndarray] = {}
            descriptors: list[dict[str, Any]] = []
            anchors_by_frame = defaultdict(list)
            for anchor_row in job["anchors"]:
                anchors_by_frame[int(anchor_row["frame"])].append(str(anchor_row["anchor"]))
            for site, layer, frame in sorted(expected_events):
                key = f"activation__{site}__L{layer:03d}__F{frame:08d}"
                value = np.ascontiguousarray(_numpy(event_tensors[(site, layer, frame)]))
                if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                    raise ContractError(f"{trial_id}: non-finite/non-numeric activation at {site}/L{layer}/F{frame}")
                captured[key] = value
                descriptors.append({
                    "key": key, "site": site, "layer": layer, "frame": frame,
                    "anchors": sorted(anchors_by_frame[frame]), "shape": list(value.shape),
                    "dtype": str(value.dtype), "sha256": _array_sha256(value),
                    "nbytes": int(value.nbytes),
                })
            if "logits" in sites:
                logits = np.asarray(result.logits)
                if args.synthetic:
                    if not np.isfinite(logits).all():
                        raise ContractError(f"{trial_id}: synthetic logits are non-finite")
                    for frame in frames:
                        key = f"logits__F{frame:08d}"
                        value = np.ascontiguousarray(logits)
                        captured[key] = value
                        descriptors.append({
                            "key": key, "site": "logits", "layer": None, "frame": frame,
                            "anchors": sorted(anchors_by_frame[frame]), "shape": list(value.shape),
                            "dtype": str(value.dtype), "sha256": _array_sha256(value),
                            "nbytes": int(value.nbytes),
                        })
                else:
                    if logits.ndim < 2 or int(logits.shape[-2]) != end_frame_exclusive:
                        raise ContractError(f"{trial_id}: logits do not cover replay span [0, {end_frame_exclusive})")
                    for frame in frames:
                        key = f"logits__F{frame:08d}"
                        value = np.ascontiguousarray(logits[..., frame, :])
                        if not np.isfinite(value).all():
                            raise ContractError(f"{trial_id}: logits are non-finite at frame {frame}")
                        captured[key] = value
                        descriptors.append({
                            "key": key, "site": "logits", "layer": None, "frame": frame,
                            "anchors": sorted(anchors_by_frame[frame]), "shape": list(value.shape),
                            "dtype": str(value.dtype), "sha256": _array_sha256(value),
                            "nbytes": int(value.nbytes),
                        })
            captured_bytes = sum(int(value.nbytes) for value in captured.values())
            if captured_bytes > args.max_bytes_per_trial:
                raise ContractError(
                    f"{trial_id}: selected capture is {captured_bytes} bytes, above safety cap "
                    f"{args.max_bytes_per_trial}"
                )
            features, feature_policy = _capture_probe_feature(captured, descriptors)
            metadata = {
                "tensors": descriptors, "captured_tensor_count": len(descriptors),
                "captured_tensor_bytes": captured_bytes,
                "feature_policy": feature_policy, "feature_shape": list(features.shape),
                "feature_dtype": str(features.dtype),
                "feature_tensor_sha256": _array_sha256(features),
                "feedback_sha256": result.feedback_sha256,
                "observed_frame_count": int(result.frame_count),
                "observed_lm_step_count": int(result.lm_step_count),
            }
            _atomic_savez(
                destination, **captured, features=features,
                artifact_identity_sha256=np.asarray(job["identity_sha"]),
                capture_metadata_json=np.asarray(canonical_json(metadata)),
            )
        features, descriptors, metadata = _capture_archive(
            destination, identity_sha256=job["identity_sha"])
        output_row = {
            "schema_version": "1.1.0", "trial_id": trial_id,
            "scenario_id": trial.get("scenario_id"), "label": trial.get("new_value"),
            "role": args.role, "capture_identity_sha256": job["identity_sha"],
            "feature_uri": destination.relative_to(args.output_root).as_posix(),
            "feature_sha256": sha256_file(destination),
            "feature_tensor_sha256": _array_sha256(features),
            "feature_shape": list(features.shape), "feature_dtype": str(features.dtype),
            "feature_policy": metadata["feature_policy"],
            "sites": sites, "layers": layers, "anchors": job["anchors"],
            "replay_frame_span": [0, max(job["frames"]) + 1],
            "captured_tensor_count": int(metadata["captured_tensor_count"]),
            "captured_tensor_bytes": int(metadata["captured_tensor_bytes"]),
            "tensors": descriptors, "feedback_sha256": metadata["feedback_sha256"],
            "observed_frame_count": int(metadata["observed_frame_count"]),
            "observed_lm_step_count": int(metadata["observed_lm_step_count"]),
            "provenance": {
                "code_commit": code_commit, "harness_version": HARNESS_VERSION,
                "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                "config_sha256": config_sha, "manifest_sha256": manifest_sha,
                "encoded_manifest_sha256": encoded_sha, "anchor_map_sha256": anchor_sha,
            },
            "synthetic": bool(args.synthetic),
        }
        if job["old"] is not None and job["old"] != output_row:
            raise ContractError(f"{trial_id}: completed capture row failed resume verification")
        output.append(output_row)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(capture_manifest, output)
    write_json(args.output_root / "capture_summary.json", {
        "schema_version": "1.0.0", "trial_count": len(output),
        "capture_identity_sha256": sha256_value([row["capture_identity_sha256"] for row in output]),
        "captured_tensor_count": sum(row["captured_tensor_count"] for row in output),
        "captured_tensor_bytes": sum(row["captured_tensor_bytes"] for row in output),
        "resumed_trial_count": sum(job["old"] is not None for job in jobs),
        "config_sha256": config_sha, "manifest_sha256": manifest_sha,
        "encoded_manifest_sha256": encoded_sha, "anchor_map_sha256": anchor_sha,
        "synthetic": bool(args.synthetic),
    })
    print(f"captured bounded activations for {len(output)} trials -> {args.output_root}")
    return 0


def _token_id_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{label} must be a non-empty token-ID array")
    output = []
    for token in value:
        token_id = _exact_int_field(token, label)
        if token_id < 0:
            raise ContractError(f"{label} contains a negative token ID")
        output.append(token_id)
    return output


def _validate_readout_contract(
    source: Mapping[str, Any], readouts: Sequence[Mapping[str, Any]],
    *, backend: Any | None, candidates: Sequence[str], require_bound: bool,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    expected = {
        "candidate_scoring": "mean_log_probability_per_token",
        "candidate_branching": "restore_identical_query_snapshot_before_each_candidate",
        "schedule_aggregation": "logmeanexp_over_all_preregistered_schedules",
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise ContractError("readout config changes the frozen scoring/branching/aggregation contract")
    bound_hash = source.get("bound_readout_sha256")
    if bound_hash is not None:
        observed = sha256_value({key: value for key, value in source.items()
                                 if key != "bound_readout_sha256"})
        if observed != validate_sha256(str(bound_hash), "bound readout"):
            raise ContractError("bound readout self-hash mismatch")
    if require_bound and (
        source.get("model_revision") != MODEL_REVISION or bound_hash is None
    ):
        raise ContractError(
            "real scoring requires the model-preflight readouts.bound.json with exact token IDs"
        )
    prefix_ids: dict[str, list[int]] = {}
    for readout in readouts:
        value = readout.get("prefix_token_ids")
        if value is None and not require_bound:
            if backend is not None:
                value = list(backend.tokenizer.encode(str(readout["prefix"]), out_type=int))
            else:
                value = [ord(character) for character in str(readout["prefix"])]
        prefix_ids[str(readout["id"])] = _token_id_list(
            value, f"readout {readout['id']} prefix_token_ids")
        if backend is not None:
            observed = list(backend.tokenizer.encode(str(readout["prefix"]), out_type=int))
            if observed != prefix_ids[str(readout["id"])]:
                raise ContractError(f"readout {readout['id']} prefix token IDs mismatch the model")
    raw_candidates = source.get("candidate_token_ids")
    if require_bound and not isinstance(raw_candidates, Mapping):
        raise ContractError("bound readout config has no candidate_token_ids object")
    candidate_ids: dict[str, list[int]] = {}
    for text in sorted(set(candidates)):
        value = raw_candidates.get(text) if isinstance(raw_candidates, Mapping) else None
        if value is None and not require_bound:
            if backend is not None:
                value = list(backend.tokenizer.encode(text, out_type=int))
            else:
                value = [ord(character) for character in text]
        candidate_ids[text] = _token_id_list(value, f"candidate {text!r} token IDs")
        if backend is not None:
            observed = list(backend.tokenizer.encode(text, out_type=int))
            if observed != candidate_ids[text]:
                raise ContractError(f"candidate {text!r} token IDs mismatch the model")
    return prefix_ids, candidate_ids


def score_readouts(argv: Sequence[str]) -> int:
    parser = _parser("Score every frozen readout at its exact semantic anchor and schedule.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--encoded-manifest", type=Path)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--folds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = read_json(args.config)
    if not isinstance(config, Mapping):
        raise ContractError("mechanistic config must be an object")
    _frozen_model_config(config)
    trials = _load_trials(
        args.manifest, args.role, _ints(args.folds) if args.folds else None,
        args.role_manifest)
    readout_source = read_json(args.readouts)
    if not isinstance(readout_source, Mapping):
        raise ContractError("readout config must be an object")
    readouts, schedules = _basic_readout_plan(readout_source)
    _, anchor_by_key = _anchor_lookup(args.anchors)
    candidates = []
    jobs: list[dict[str, Any]] = []
    encoded_by_id: dict[str, dict[str, Any]] = {}
    conversation_by_id: dict[str, np.ndarray] = {}
    if not args.synthetic:
        if args.encoded_manifest is None:
            candidate = args.manifest.parent.parent / "encoded_user_manifest.jsonl"
            if not candidate.exists():
                raise ContractError("Moshiko readout scoring requires --encoded-manifest")
            args.encoded_manifest = candidate
        _, encoded_by_id = _encoded_rows(args.encoded_manifest)
    for trial in trials:
        trial_id = str(trial["trial_id"])
        target, stale = str(trial.get("new_value", "")), str(trial.get("old_value", ""))
        if not target or not stale or target == stale:
            raise ContractError(f"{trial_id}: target and stale candidates must be distinct and non-empty")
        candidates.extend((target, stale))
        encoded = encoded_by_id.get(trial_id) if not args.synthetic else None
        if not args.synthetic:
            if encoded is None:
                raise ContractError(f"missing encoded row for {trial_id}")
            if (
                encoded.get("synthetic") is not False
                or encoded.get("model_repo") != MODEL_REPO
                or encoded.get("model_revision") != MODEL_REVISION
            ):
                raise ContractError(f"{trial_id}: encoded data is synthetic or has wrong model identity")
            available_frames = _exact_int_field(
                encoded.get("conversation_frame_end_exclusive"),
                f"{trial_id} conversation frame end")
            conversation = _load_encoded_array(
                args.encoded_manifest, encoded, "conversation_codes",
                require_current_contract=True)
            if int(conversation.shape[-1]) != available_frames:
                raise ContractError(f"{trial_id}: encoded readout coverage mismatch")
            conversation_by_id[trial_id] = conversation
        else:
            available_frames = _exact_int_field(trial.get("frame_count"), f"{trial_id} frame_count")
        for readout in readouts:
            anchor = str(readout["anchor"])
            anchor_row = anchor_by_key.get((trial_id, anchor))
            if anchor_row is None:
                raise ContractError(f"{trial_id}: readout {readout['id']} anchor {anchor!r} is missing")
            frame = int(anchor_row["frame"])
            if frame < 0 or frame >= available_frames:
                raise ContractError(f"{trial_id}: readout anchor {anchor!r} is outside encoded coverage")
            jobs.append({
                "trial": trial, "encoded": encoded, "readout": readout,
                "anchor_row": anchor_row, "anchor_frame": frame,
                "end_frame_exclusive": frame + 1, "target": target, "stale": stale,
            })

    prefix_ids, candidate_ids = _validate_readout_contract(
        readout_source, readouts, backend=None,
        candidates=candidates, require_bound=not args.synthetic)
    backend = SyntheticBackend() if args.synthetic else MoshiBackend(
        model_repo=MODEL_REPO, model_revision=MODEL_REVISION, use_sampling=False)
    if not args.synthetic:
        prefix_ids, candidate_ids = _validate_readout_contract(
            readout_source, readouts, backend=backend, candidates=candidates,
            require_bound=True)
    if any(candidate_ids[target] == candidate_ids[stale]
           for target, stale in ((job["target"], job["stale"]) for job in jobs)):
        raise ContractError("distinct target/stale candidates map to identical frozen token IDs")
    config_sha = sha256_file(args.config)
    manifest_sha = sha256_file(args.manifest)
    readouts_sha = sha256_file(args.readouts)
    anchors_sha = sha256_file(args.anchors)
    encoded_sha = sha256_file(args.encoded_manifest) if args.encoded_manifest else None
    tolerance = 1e-6
    output: list[dict[str, Any]] = []
    for job in jobs:
        trial = job["trial"]
        trial_id = str(trial["trial_id"])
        end_frame_exclusive = int(job["end_frame_exclusive"])
        if args.synthetic:
            replay = backend.replay(
                {**dict(trial), "frame_count": end_frame_exclusive}, ["resid_post"])
            snapshot = None
        else:
            assert args.encoded_manifest is not None
            conversation = conversation_by_id[trial_id]
            codes = backend.torch.as_tensor(conversation, device=backend.device)
            replay = backend.replay_codes(
                codes, sites=[], hook_enabled=False,
                end_frame_exclusive=end_frame_exclusive)
            if (
                int(replay.frame_count) != end_frame_exclusive
                or int(replay.lm_step_count) != end_frame_exclusive + 1
            ):
                raise ContractError(f"{trial_id}: replay did not end at the requested readout anchor")
            snapshot = backend.lm_gen.snapshot_streaming_state()
        schedule_rows = []
        target_schedule_scores: list[float] = []
        stale_schedule_scores: list[float] = []
        order_deltas: list[float] = []
        for schedule in schedules:
            candidate_mapping = {"target": job["target"], "stale": job["stale"]}
            if args.synthetic:
                forward = _synthetic_scores(
                    backend, trial, candidate_mapping,
                    anchor_end_exclusive=end_frame_exclusive,
                    prefix=str(job["readout"]["prefix"]), schedule=schedule)
                reverse = _synthetic_scores(
                    backend, trial, {"stale": job["stale"], "target": job["target"]},
                    anchor_end_exclusive=end_frame_exclusive,
                    prefix=str(job["readout"]["prefix"]), schedule=schedule)
            else:
                score_kwargs = {
                    "prefix": str(job["readout"]["prefix"]),
                    "prefix_start_offset_frames": int(schedule["prefix_start_offset_frames"]),
                    "pad_frames_between_tokens": int(schedule["pad_frames_between_tokens"]),
                }
                forward = backend.score_candidates(snapshot, candidate_mapping, **score_kwargs)
                reverse = backend.score_candidates(
                    snapshot, {"stale": job["stale"], "target": job["target"]},
                    **score_kwargs)
            if set(forward) != {"target", "stale"} or set(reverse) != {"target", "stale"}:
                raise ContractError(f"{trial_id}: candidate scorer returned an invalid key set")
            forward_values = {name: float(forward[name]) for name in ("target", "stale")}
            reverse_values = {name: float(reverse[name]) for name in ("target", "stale")}
            if not all(math.isfinite(value) for value in (*forward_values.values(), *reverse_values.values())):
                raise ContractError(f"{trial_id}: candidate scorer returned a non-finite value")
            deltas = {name: forward_values[name] - reverse_values[name]
                      for name in ("target", "stale")}
            maximum_delta = max(abs(value) for value in deltas.values())
            if maximum_delta > tolerance:
                raise ContractError(
                    f"{trial_id}:{job['readout']['id']}:{schedule['id']} candidate order changed scores"
                )
            target_schedule_scores.append(forward_values["target"])
            stale_schedule_scores.append(forward_values["stale"])
            order_deltas.append(maximum_delta)
            schedule_rows.append({
                "schedule_id": schedule["id"],
                "prefix_start_offset_frames": schedule["prefix_start_offset_frames"],
                "pad_frames_between_tokens": schedule["pad_frames_between_tokens"],
                "forward_order": ["target", "stale"], "reverse_order": ["stale", "target"],
                "forward_scores": forward_values, "reverse_scores": reverse_values,
                "order_score_deltas": deltas, "max_abs_order_score_delta": maximum_delta,
            })
        target_logprob = _logmeanexp(target_schedule_scores)
        stale_logprob = _logmeanexp(stale_schedule_scores)
        margin = target_logprob - stale_logprob
        output.append({
            "schema_version": "1.1.0", "trial_id": trial_id,
            "scenario_id": trial.get("scenario_id"), "condition": trial.get("condition"),
            "role": args.role, "readout_id": job["readout"]["id"],
            "prefix": job["readout"]["prefix"],
            "prefix_token_ids": prefix_ids[str(job["readout"]["id"])],
            "anchor": job["readout"]["anchor"], "anchor_frame": job["anchor_frame"],
            "anchor_end_frame_exclusive": end_frame_exclusive,
            "anchor_time_ms": job["anchor_row"].get("time_ms"),
            "anchor_timebase": job["anchor_row"].get("timebase"),
            "target": job["target"], "stale": job["stale"],
            "target_token_ids": candidate_ids[job["target"]],
            "stale_token_ids": candidate_ids[job["stale"]],
            "target_logprob": target_logprob, "stale_logprob": stale_logprob,
            "margin_M": margin,
            "candidate_order_delta": max(order_deltas),
            "candidate_order_tolerance": tolerance,
            "schedule_aggregation": "logmeanexp_over_all_preregistered_schedules",
            "schedules": schedule_rows,
            "replay_feedback_sha256": replay.feedback_sha256,
            "provenance": {
                "code_commit": _git_commit(), "harness_version": HARNESS_VERSION,
                "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                "config_sha256": config_sha, "manifest_sha256": manifest_sha,
                "encoded_manifest_sha256": encoded_sha,
                "readouts_sha256": readouts_sha, "anchor_map_sha256": anchors_sha,
            },
            "synthetic": bool(args.synthetic),
        })
    expected_rows = len(trials) * len(readouts)
    if len(output) != expected_rows:
        raise ContractError(f"readout output coverage mismatch: {len(output)} != {expected_rows}")
    write_jsonl(args.output, output)
    print(f"scored {len(output)} anchored readouts -> {args.output}")
    return 0


def _probe_capture_contract(
    captures: Sequence[Mapping[str, Any]], *, feature_dimension: int,
) -> dict[str, Any]:
    contracts: list[dict[str, Any]] = []
    for row in captures:
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ContractError(f"{row.get('trial_id')}: probe capture provenance is missing")
        raw_anchors = row.get("anchors")
        if not isinstance(raw_anchors, list):
            raise ContractError(f"{row.get('trial_id')}: probe capture anchors are missing")
        anchor_names = []
        for anchor in raw_anchors:
            if not isinstance(anchor, Mapping) or not str(anchor.get("anchor", "")):
                raise ContractError(f"{row.get('trial_id')}: probe capture anchor is malformed")
            anchor_names.append(str(anchor["anchor"]))
        contract = {
            "schema_version": "1.0.0",
            "feature_dimension": feature_dimension,
            "feature_policy": row.get("feature_policy"),
            "sites": row.get("sites"),
            "layers": row.get("layers"),
            "anchors": anchor_names,
            "model_repo": provenance.get("model_repo"),
            "model_revision": provenance.get("model_revision"),
            "config_sha256": provenance.get("config_sha256"),
            "code_commit": provenance.get("code_commit"),
            "harness_version": provenance.get("harness_version"),
            "synthetic": row.get("synthetic"),
        }
        if (
            not isinstance(contract["feature_policy"], str)
            or not contract["feature_policy"]
            or not isinstance(contract["sites"], list)
            or not isinstance(contract["layers"], list)
            or not contract["anchors"]
            or contract["synthetic"] not in {True, False}
        ):
            raise ContractError(f"{row.get('trial_id')}: probe capture contract is incomplete")
        if (
            contract["model_repo"] != MODEL_REPO
            or contract["model_revision"] != MODEL_REVISION
            or contract["harness_version"] != HARNESS_VERSION
            or not isinstance(contract["code_commit"], str)
            or not contract["code_commit"]
        ):
            raise ContractError(f"{row.get('trial_id')}: probe capture model/code provenance differs")
        contracts.append(contract)
    if not contracts or any(item != contracts[0] for item in contracts[1:]):
        raise ContractError("probe captures do not share one immutable site/feature provenance")
    return contracts[0]


def _load_probe_capture_dataset(
    *,
    capture_root: Path,
    role: str,
    manifest: Path,
    role_manifest: Path | None,
    config_sha256: str,
    synthetic: bool,
) -> tuple[np.ndarray, list[str], list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
    capture_manifest = capture_root / "capture_manifest.jsonl"
    if not capture_manifest.is_file():
        raise ContractError("probe operation requires an existing activation capture manifest")
    captures = read_jsonl(capture_manifest)
    if not captures or any(row.get("role") != role for row in captures):
        raise ContractError("probe captures are empty or contain a different analysis role")
    if role in {"multivalue_calibration", "formal_confirmation"} and role_manifest is None:
        raise ContractError(
            f"{role} probe operation requires the immutable --role-manifest"
        )
    expected = _load_trials(manifest, role, role_manifest=role_manifest)
    expected_by_id = {str(row["trial_id"]): row for row in expected}
    observed_ids = [str(row.get("trial_id", "")) for row in captures]
    if (
        any(not trial_id for trial_id in observed_ids)
        or len(set(observed_ids)) != len(observed_ids)
        or set(observed_ids) != set(expected_by_id)
    ):
        raise ContractError("probe capture coverage differs from the immutable role manifest")

    expected_manifest_sha = sha256_file(manifest)
    xs: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    row_ids: list[str] = []
    for row in captures:
        trial_id = str(row["trial_id"])
        trial = expected_by_id[trial_id]
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ContractError(f"{trial_id}: probe capture provenance is missing")
        if provenance.get("manifest_sha256") != expected_manifest_sha:
            raise ContractError(f"{trial_id}: probe capture source-manifest hash mismatch")
        if provenance.get("config_sha256") != config_sha256:
            raise ContractError(f"{trial_id}: probe capture config hash mismatch")
        if row.get("synthetic") is not synthetic:
            raise ContractError(f"{trial_id}: probe capture synthetic/empirical status mismatch")
        label = str(row.get("label", ""))
        group = str(row.get("scenario_id", ""))
        if label != str(trial.get("new_value", "")) or group != str(trial.get("scenario_id", "")):
            raise ContractError(f"{trial_id}: probe capture label/scenario differs from its manifest")
        relative = require_relative_uri(str(row.get("feature_uri", "")))
        feature_path = (capture_root / relative).resolve()
        try:
            feature_path.relative_to(capture_root.resolve())
        except ValueError as error:
            raise ContractError("probe feature URI escapes its capture root") from error
        if not feature_path.is_file() or sha256_file(feature_path) != row.get("feature_sha256"):
            raise ContractError(f"probe feature hash mismatch: {trial_id}")
        feature, _, _ = _capture_archive(
            feature_path, identity_sha256=str(row.get("capture_identity_sha256", ""))
        )
        if (
            row.get("feature_shape") != list(feature.shape)
            or row.get("feature_dtype") != str(feature.dtype)
            or row.get("feature_tensor_sha256") != _array_sha256(feature)
        ):
            raise ContractError(f"{trial_id}: probe feature manifest metadata mismatch")
        xs.append(feature[0])
        labels.append(label)
        groups.append(group)
        row_ids.append(trial_id)
    features = np.asarray(xs, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(captures):
        raise ContractError("probe capture features do not form one finite feature matrix")
    contract = _probe_capture_contract(captures, feature_dimension=int(features.shape[1]))
    return features, labels, groups, row_ids, captures, contract


def _probe_grid_coordinate(*, site: str, layer: int, anchor: str) -> dict[str, Any]:
    """Return the canonical identity of one diagnostic-probe feature.

    A grid cell is intentionally narrower than an activation-capture job.  It
    names one stored tensor at one semantic anchor, so neither fitting nor
    frozen application can silently average information across layers or
    timepoints.
    """

    if not site or site == "logits" or site not in REQUIRED_SITES:
        raise ContractError(f"probe-grid site must be one captured transformer site: {site!r}")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ContractError("probe-grid layer must be a non-negative integer")
    if not anchor:
        raise ContractError("probe-grid anchor must be non-empty")
    return {"site": site, "layer": layer, "anchor": anchor}


def _exact_probe_capture_contract(
    source_contract: Mapping[str, Any], *, coordinate: Mapping[str, Any],
    feature_dimension: int,
) -> dict[str, Any]:
    """Bind a probe to one exact stored tensor while retaining capture provenance."""

    contract = {
        "schema_version": "1.1.0",
        "feature_dimension": int(feature_dimension),
        "feature_policy": "flatten_one_exact_captured_tensor",
        "probe_coordinate": dict(coordinate),
        "source_capture_contract_sha256": sha256_value(source_contract),
        "source_sites": list(source_contract.get("sites", [])),
        "source_layers": list(source_contract.get("layers", [])),
        "source_anchors": list(source_contract.get("anchors", [])),
        "model_repo": source_contract.get("model_repo"),
        "model_revision": source_contract.get("model_revision"),
        "config_sha256": source_contract.get("config_sha256"),
        "code_commit": source_contract.get("code_commit"),
        "harness_version": source_contract.get("harness_version"),
        "synthetic": source_contract.get("synthetic"),
    }
    return contract


def _load_exact_probe_feature_dataset(
    *, capture_root: Path, captures: Sequence[Mapping[str, Any]],
    coordinate: Mapping[str, Any],
) -> tuple[
    np.ndarray, list[str], list[str], list[str], list[dict[str, Any]], str,
]:
    """Load one site/layer/anchor vector per row from verified capture NPZs."""

    site = str(coordinate["site"])
    layer = int(coordinate["layer"])
    anchor = str(coordinate["anchor"])
    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    row_ids: list[str] = []
    tensor_identities: list[dict[str, Any]] = []
    # Stable fitting order is part of the contract.  Capture-manifest row order
    # must not change folds, predictions, or floating-point solve order.
    ordered = sorted(captures, key=lambda row: str(row.get("trial_id", "")))
    for row in ordered:
        trial_id = str(row.get("trial_id", ""))
        relative = require_relative_uri(str(row.get("feature_uri", "")))
        feature_path = (capture_root / relative).resolve()
        try:
            feature_path.relative_to(capture_root.resolve())
        except ValueError as error:
            raise ContractError("probe feature URI escapes its capture root") from error
        _, descriptors, _ = _capture_archive(
            feature_path, identity_sha256=str(row.get("capture_identity_sha256", ""))
        )
        matches = [
            item for item in descriptors
            if (
                item.get("site") == site
                and item.get("layer") == layer
                and anchor in item.get("anchors", [])
            )
        ]
        if len(matches) != 1:
            raise ContractError(
                f"{trial_id}: exact probe tensor coverage for "
                f"{site}/L{layer}/{anchor} is {len(matches)}, expected 1"
            )
        descriptor = matches[0]
        key = str(descriptor["key"])
        try:
            with np.load(feature_path, allow_pickle=False) as archive:
                value = np.ascontiguousarray(np.asarray(archive[key]))
        except (KeyError, OSError, ValueError) as error:
            raise ContractError(f"cannot load exact probe tensor {key}: {error}") from error
        if (
            value.size < 1
            or not np.issubdtype(value.dtype, np.number)
            or not np.isfinite(value).all()
            or _array_sha256(value) != descriptor.get("sha256")
        ):
            raise ContractError(f"{trial_id}: exact probe tensor failed verification: {key}")
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        feature_rows.append(vector)
        labels.append(str(row.get("label", "")))
        groups.append(str(row.get("scenario_id", "")))
        row_ids.append(trial_id)
        tensor_identities.append({
            "row_id": trial_id,
            "capture_identity_sha256": row.get("capture_identity_sha256"),
            "tensor_key": key,
            "tensor_sha256": descriptor.get("sha256"),
            "tensor_shape": descriptor.get("shape"),
            "tensor_dtype": descriptor.get("dtype"),
        })
    dimensions = {int(vector.size) for vector in feature_rows}
    if len(dimensions) != 1:
        raise ContractError(
            f"probe-grid tensors at {site}/L{layer}/{anchor} have inconsistent dimensions"
        )
    features = np.asarray(feature_rows, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(ordered):
        raise ContractError("exact probe captures do not form a finite feature matrix")
    dataset_sha256 = sha256_value(tensor_identities)
    return features, labels, groups, row_ids, tensor_identities, dataset_sha256


def _probe_coordinate_from_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    component = str(selection.get("component", ""))
    layer = selection.get("layer")
    anchor = str(selection.get("anchor", ""))
    coordinate = _probe_grid_coordinate(site=component, layer=layer, anchor=anchor)
    if selection.get("head") is not None:
        raise ContractError(
            "a head-specific selection cannot freeze a site/layer/anchor probe-grid cell"
        )
    return coordinate


def _validate_probe_site_selection(
    selection: Mapping[str, Any] | None, capture_contract: Mapping[str, Any],
) -> None:
    if selection is None:
        return
    layer = selection.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ContractError("probe site selection has no exact layer")
    if layer not in capture_contract.get("layers", []):
        raise ContractError("probe capture does not include the frozen selected layer")
    anchor = str(selection.get("anchor", ""))
    if not anchor or anchor not in capture_contract.get("anchors", []):
        raise ContractError("probe capture does not include the frozen selected anchor")
    component = str(selection.get("component", ""))
    directly_captured = REQUIRED_SITES
    if component in directly_captured and component not in capture_contract.get("sites", []):
        raise ContractError("probe capture does not include the frozen selected component")
    if component == "kv_cache" and not {
        "k_pre_rope", "v_pre_rope"
    } <= set(capture_contract.get("sites", [])):
        raise ContractError("KV-selected probe capture must include joint pre-RoPE K and V")


def _requested_probe_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sites is None or args.layers is None or args.anchors is None:
        raise ContractError(
            "--probe-grid requires explicit --sites, --layers, and --anchors"
        )
    sites = _csv(args.sites)
    layers = _ints(args.layers)
    anchors = _csv(args.anchors)
    if (
        not sites or len(set(sites)) != len(sites)
        or not layers or len(set(layers)) != len(layers)
        or not anchors or len(set(anchors)) != len(anchors)
    ):
        raise ContractError("probe-grid sites, layers, and anchors must be non-empty and unique")
    return [
        _probe_grid_coordinate(site=site, layer=layer, anchor=anchor)
        for site in sorted(sites) for layer in sorted(layers) for anchor in sorted(anchors)
    ]


def _fit_exact_probe_grid(
    *, args: argparse.Namespace, capture_root: Path, manifest_path: Path,
    config_sha: str,
) -> int:
    if args.manifest is None or not manifest_path.is_file():
        raise ContractError(
            "probe-grid fitting requires --manifest and an existing activation capture manifest"
        )
    # This call performs the full manifest/role/archive/provenance validation.
    # Its historical aggregate feature is deliberately ignored in grid mode.
    _, _, _, _, captures, source_contract = _load_probe_capture_dataset(
        capture_root=capture_root,
        role=args.role,
        manifest=args.manifest,
        role_manifest=args.role_manifest,
        config_sha256=config_sha,
        synthetic=bool(args.synthetic),
    )
    coordinates = _requested_probe_grid(args)
    source_sites = set(source_contract.get("sites", []))
    source_layers = set(source_contract.get("layers", []))
    source_anchors = set(source_contract.get("anchors", []))
    for coordinate in coordinates:
        if coordinate["site"] not in source_sites:
            raise ContractError(f"probe-grid site was not captured: {coordinate['site']}")
        if coordinate["layer"] not in source_layers:
            raise ContractError(f"probe-grid layer was not captured: {coordinate['layer']}")
        if coordinate["anchor"] not in source_anchors:
            raise ContractError(f"probe-grid anchor was not captured: {coordinate['anchor']}")

    selection, selection_file_sha = _validated_selection(
        args.site_selection, required=args.freeze_output is not None
    )
    selected_coordinate = None
    if selection is not None:
        if selection.get("status") != "frozen_discovery_selection":
            raise ContractError("probe-grid freezing requires a frozen discovery selection")
        if selection.get("config_sha256") != config_sha:
            raise ContractError("probe-grid selection was frozen under a different config")
        _validate_probe_site_selection(selection, source_contract)
        selected_coordinate = _probe_coordinate_from_selection(selection)
        if selected_coordinate not in coordinates:
            raise ContractError("frozen probe selection is outside the requested probe grid")

    analysis_status = (
        "synthetic_local_validation" if args.synthetic else "empirical_diagnostic"
    )
    capture_manifest_file_sha = sha256_file(manifest_path)
    capture_manifest_identity_sha = sha256_value(
        sorted((dict(row) for row in captures), key=lambda row: str(row.get("trial_id", "")))
    )
    source_manifest_sha = sha256_file(args.manifest)
    role_manifest_sha = sha256_file(args.role_manifest) if args.role_manifest else None
    reports: list[dict[str, Any]] = []
    for coordinate in coordinates:
        features, labels, groups, row_ids, tensor_rows, dataset_sha = (
            _load_exact_probe_feature_dataset(
                capture_root=capture_root, captures=captures, coordinate=coordinate
            )
        )
        grouped = fit_grouped_ridge_probe(
            features,
            labels,
            groups,
            alpha=args.alpha,
            folds=args.folds,
            seed=args.seed,
            row_ids=row_ids,
        )
        capture_contract = _exact_probe_capture_contract(
            source_contract,
            coordinate=coordinate,
            feature_dimension=int(features.shape[1]),
        )
        report = {
            **grouped,
            "analysis_status": analysis_status,
            "role": args.role,
            "training_role": args.role,
            "probe_mode": "exact_site_layer_semantic_anchor_grid",
            "probe_coordinate": coordinate,
            "probe_feature_rows": tensor_rows,
            "probe_feature_dataset_sha256": dataset_sha,
            "capture_manifest_sha256": capture_manifest_file_sha,
            "capture_manifest_identity_sha256": capture_manifest_identity_sha,
            "source_manifest_sha256": source_manifest_sha,
            "role_manifest_sha256": role_manifest_sha,
            "site_selection_sha256": selection_file_sha,
            "site_selection_identity_sha256": (
                selection.get("selection_sha256") if selection else None
            ),
            "capture_contract": capture_contract,
            "capture_contract_sha256": sha256_value(capture_contract),
            "code_commit": _git_commit(),
            "harness_version": HARNESS_VERSION,
            "causal_use_prohibited": True,
            "probe_grid_cell_identity_policy": (
                "canonical_report_excluding_capture_manifest_file_sha256"
            ),
        }
        report["probe_grid_cell_sha256"] = sha256_value({
            key: value for key, value in report.items()
            if key != "capture_manifest_sha256"
        })
        reports.append(report)

    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_root / "probe_grid_metrics.jsonl"
    write_jsonl(metrics_path, reports)
    manifest = {
        "schema_version": "1.0.0",
        "status": "completed_exact_probe_grid",
        "analysis_status": analysis_status,
        "role": args.role,
        "probe_mode": "exact_site_layer_semantic_anchor_grid",
        "expected_cell_count": len(coordinates),
        "completed_cell_count": len(reports),
        "coordinates": coordinates,
        "cell_sha256": [report["probe_grid_cell_sha256"] for report in reports],
        "metrics_sha256": sha256_file(metrics_path),
        "capture_manifest_sha256": capture_manifest_file_sha,
        "capture_manifest_identity_sha256": capture_manifest_identity_sha,
        "source_manifest_sha256": source_manifest_sha,
        "role_manifest_sha256": role_manifest_sha,
        "source_capture_contract_sha256": sha256_value(source_contract),
        "site_selection_sha256": selection_file_sha,
        "site_selection_identity_sha256": (
            selection.get("selection_sha256") if selection else None
        ),
        "selected_coordinate": selected_coordinate,
        "diagnostic_only": True,
        "causal_use_prohibited": True,
        "code_commit": _git_commit(),
        "harness_version": HARNESS_VERSION,
        "limitations": [
            "Probe-grid scores measure diagnostic decodability, not causal mediation."
        ],
    }
    manifest["probe_grid_semantic_identity_sha256"] = sha256_value({
        "role": args.role,
        "coordinates": coordinates,
        "cell_sha256": manifest["cell_sha256"],
        "capture_manifest_identity_sha256": capture_manifest_identity_sha,
        "source_manifest_sha256": source_manifest_sha,
        "role_manifest_sha256": role_manifest_sha,
        "source_capture_contract_sha256": manifest["source_capture_contract_sha256"],
        "site_selection_identity_sha256": manifest["site_selection_identity_sha256"],
    })
    manifest["probe_grid_identity_sha256"] = sha256_value(manifest)
    grid_manifest_path = args.output_root / "probe_grid_manifest.json"
    write_json(grid_manifest_path, manifest)
    summary = {
        **manifest,
        "probe_grid_manifest_sha256": sha256_file(grid_manifest_path),
        "cells": [
            {
                "probe_coordinate": report["probe_coordinate"],
                "cv_accuracy": report["cv_accuracy"],
                "feature_dimension": report["feature_dimension"],
                "probe_feature_dataset_sha256": report["probe_feature_dataset_sha256"],
                "probe_grid_cell_sha256": report["probe_grid_cell_sha256"],
            }
            for report in reports
        ],
    }
    write_json(args.output_root / "probe_metrics.json", summary)
    if args.freeze_output is not None:
        assert selected_coordinate is not None
        selected = next(
            report for report in reports
            if report["probe_coordinate"] == selected_coordinate
        )
        write_json(args.freeze_output, freeze_probe_report(selected))
    print(
        f"fit {len(reports)} exact site/layer/anchor grouped diagnostic probes "
        f"-> {args.output_root}"
    )
    return 0


def fit_probes(argv: Sequence[str]) -> int:
    parser = _parser(
        "Fit a scenario-grouped K-class diagnostic ridge probe without using it for causal claims."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--group-by", default="scenario_id")
    parser.add_argument("--site-selection", type=Path)
    parser.add_argument(
        "--probe-grid", action="store_true",
        help="Fit one diagnostic probe per exact requested site/layer/semantic anchor.",
    )
    parser.add_argument("--sites", help="Comma-separated exact sites for --probe-grid.")
    parser.add_argument("--layers", help="Comma/range-separated exact layers for --probe-grid.")
    parser.add_argument("--anchors", help="Comma-separated exact semantic anchors for --probe-grid.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-output", type=Path)
    mode.add_argument("--frozen-probe", type=Path)
    parser.add_argument("--expected-training-rows-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args(argv)
    if args.group_by != "scenario_id":
        raise ContractError("diagnostic probe grouping is frozen to scenario_id")
    if args.frozen_probe is None and args.expected_training_rows_sha256 is not None:
        raise ContractError(
            "--expected-training-rows-sha256 is only valid with --frozen-probe"
        )
    if not args.probe_grid and any(
        value is not None for value in (args.sites, args.layers, args.anchors)
    ):
        raise ContractError("--sites/--layers/--anchors require --probe-grid")
    config = read_json(args.config)
    if not isinstance(config, Mapping):
        raise ContractError("mechanistic config must be an object")
    config_sha = sha256_file(args.config)
    capture_root = args.capture_root or args.output_root.parent / "discovery_baseline"
    manifest_path = capture_root / "capture_manifest.jsonl"
    if args.frozen_probe is not None:
        if args.role not in PROBE_APPLICATION_ROLES:
            raise ContractError(
                "--frozen-probe may only be applied to internal_validation or formal_confirmation"
            )
        if args.manifest is None:
            raise ContractError("frozen probe application requires --manifest")
        frozen = read_json(args.frozen_probe)
        if not isinstance(frozen, Mapping):
            raise ContractError("frozen probe artifact must be a JSON object")
        expected_training_hash = (
            args.expected_training_rows_sha256
            or str(frozen.get("training_rows_sha256", ""))
        )
        feature_dimension, classes = validate_frozen_probe_artifact(
            frozen, expected_training_rows_sha256=expected_training_hash
        )
        aggregate_features, labels, groups, row_ids, captures, source_capture_contract = (
            _load_probe_capture_dataset(
                capture_root=capture_root,
                role=args.role,
                manifest=args.manifest,
                role_manifest=args.role_manifest,
                config_sha256=config_sha,
                synthetic=bool(args.synthetic),
            )
        )
        frozen_coordinate = frozen.get("probe_coordinate")
        tensor_identity_by_id: dict[str, dict[str, Any]] = {}
        application_dataset_sha: str | None = None
        if frozen_coordinate is not None:
            if not isinstance(frozen_coordinate, Mapping):
                raise ContractError("frozen probe coordinate is malformed")
            coordinate = _probe_grid_coordinate(
                site=str(frozen_coordinate.get("site", "")),
                layer=frozen_coordinate.get("layer"),
                anchor=str(frozen_coordinate.get("anchor", "")),
            )
            if dict(frozen_coordinate) != coordinate:
                raise ContractError("frozen probe coordinate is not canonical")
            if args.probe_grid:
                requested = _requested_probe_grid(args)
                if requested != [coordinate]:
                    raise ContractError(
                        "frozen probe application accepts exactly its one bound grid coordinate"
                    )
            features, labels, groups, row_ids, tensor_rows, application_dataset_sha = (
                _load_exact_probe_feature_dataset(
                    capture_root=capture_root, captures=captures, coordinate=coordinate
                )
            )
            tensor_identity_by_id = {str(row["row_id"]): row for row in tensor_rows}
            capture_contract = _exact_probe_capture_contract(
                source_capture_contract,
                coordinate=coordinate,
                feature_dimension=int(features.shape[1]),
            )
        else:
            if args.probe_grid:
                raise ContractError("legacy averaged frozen probe has no exact grid coordinate")
            features = aggregate_features
            capture_contract = source_capture_contract
        if features.shape[1] != feature_dimension:
            raise ContractError(
                f"frozen probe feature dimension mismatch: expected {feature_dimension}, "
                f"observed {features.shape[1]}"
            )
        if frozen.get("capture_contract", {}).get("config_sha256") != config_sha:
            raise ContractError("frozen probe was fit under a different config")
        frozen_role_manifest_sha = frozen.get("role_manifest_sha256")
        application_role_manifest_sha = (
            sha256_file(args.role_manifest) if args.role_manifest else None
        )
        if (
            frozen_role_manifest_sha is not None
            and frozen_role_manifest_sha != application_role_manifest_sha
        ):
            raise ContractError(
                "application role manifest differs from the probe training role manifest"
            )
        expected_capture_contract = frozen.get("capture_contract")
        if not isinstance(expected_capture_contract, Mapping):
            raise ContractError("frozen probe has no capture provenance contract")
        if sha256_value(expected_capture_contract) != frozen.get("capture_contract_sha256"):
            raise ContractError("frozen probe capture provenance hash mismatch")
        if dict(expected_capture_contract) != capture_contract:
            raise ContractError("application capture differs from the frozen site/feature provenance")
        selection, selection_file_sha = _validated_selection(
            args.site_selection, required=frozen.get("site_selection_sha256") is not None
        )
        if selection_file_sha != frozen.get("site_selection_sha256"):
            raise ContractError("application site-selection file differs from the frozen probe")
        expected_selection_identity = frozen.get("site_selection_identity_sha256")
        if (selection or {}).get("selection_sha256") != expected_selection_identity:
            raise ContractError("application site-selection identity differs from the frozen probe")
        _validate_probe_site_selection(selection, source_capture_contract)
        if frozen_coordinate is not None and selection is not None:
            if _probe_coordinate_from_selection(selection) != dict(frozen_coordinate):
                raise ContractError("site selection differs from the frozen probe coordinate")
        training_ids = {
            str(row.get("row_id", "")) for row in frozen.get("training_rows", [])
            if isinstance(row, Mapping)
        }
        overlap = sorted(training_ids & set(row_ids))
        if overlap:
            raise ContractError(
                f"frozen probe application overlaps its training rows: {overlap[:5]}"
            )
        predictions = apply_frozen_probe(
            frozen, features, expected_training_rows_sha256=expected_training_hash
        )
        prediction_rows = []
        frozen_file_sha = sha256_file(args.frozen_probe)
        capture_by_id = {str(row["trial_id"]): row for row in captures}
        for row_id, group, label, prediction in zip(
            row_ids, groups, labels, predictions, strict=True
        ):
            capture = capture_by_id[row_id]
            tensor_identity = tensor_identity_by_id.get(row_id)
            prediction_rows.append({
                "schema_version": "1.0.0",
                "trial_id": row_id,
                "scenario_id": group,
                "role": args.role,
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
                "diagnostic_only": True,
                "causal_use_prohibited": True,
                "capture_identity_sha256": capture.get("capture_identity_sha256"),
                "feature_sha256": capture.get("feature_sha256"),
                "probe_coordinate": dict(frozen_coordinate) if frozen_coordinate else None,
                "probe_tensor_sha256": (
                    tensor_identity.get("tensor_sha256") if tensor_identity else None
                ),
                "frozen_probe_sha256": frozen["frozen_probe_sha256"],
                "frozen_probe_file_sha256": frozen_file_sha,
            })
        args.output_root.mkdir(parents=True, exist_ok=True)
        predictions_path = args.output_root / "probe_predictions.jsonl"
        write_jsonl(predictions_path, prediction_rows)
        accuracy = float(np.mean([row["correct"] for row in prediction_rows]))
        report = {
            "schema_version": "1.0.0",
            "status": "completed_frozen_probe_application",
            "mode": "apply_without_refit",
            "analysis_status": (
                "synthetic_local_validation" if args.synthetic else "empirical_diagnostic"
            ),
            "role": args.role,
            "training_role": frozen["training_role"],
            "n_rows": len(prediction_rows),
            "classes": classes,
            "accuracy": accuracy,
            "diagnostic_only": True,
            "causal_use_prohibited": True,
            "training_rows_sha256": frozen["training_rows_sha256"],
            "frozen_probe_sha256": frozen["frozen_probe_sha256"],
            "frozen_probe_file_sha256": frozen_file_sha,
            "capture_manifest_sha256": sha256_file(manifest_path),
            "source_manifest_sha256": sha256_file(args.manifest),
            "role_manifest_sha256": (
                sha256_file(args.role_manifest) if args.role_manifest else None
            ),
            "site_selection_sha256": selection_file_sha,
            "probe_coordinate": dict(frozen_coordinate) if frozen_coordinate else None,
            "probe_feature_dataset_sha256": application_dataset_sha,
            "capture_contract_sha256": sha256_value(capture_contract),
            "predictions_sha256": sha256_file(predictions_path),
            "code_commit": _git_commit(),
            "harness_version": HARNESS_VERSION,
            "limitations": [
                "Probe predictions are diagnostic decodability evidence, not a causal intervention."
            ],
        }
        write_json(args.output_root / "probe_metrics.json", report)
        print(
            f"applied frozen {len(classes)}-class diagnostic probe without refit "
            f"({len(prediction_rows)} rows, accuracy {accuracy:.3f}) -> {args.output_root}"
        )
        return 0

    if args.role not in PROBE_TRAINING_ROLES and not (
        args.synthetic and args.role in {"smoke", "local_validation"}
    ):
        raise ContractError(
            f"probe role {args.role!r} may not train; use --frozen-probe for validation roles"
        )
    if args.probe_grid:
        return _fit_exact_probe_grid(
            args=args,
            capture_root=capture_root,
            manifest_path=manifest_path,
            config_sha=config_sha,
        )
    if args.manifest is not None and manifest_path.exists():
        features, labels, groups, row_ids, captures, capture_contract = (
            _load_probe_capture_dataset(
                capture_root=capture_root,
                role=args.role,
                manifest=args.manifest,
                role_manifest=args.role_manifest,
                config_sha256=config_sha,
                synthetic=bool(args.synthetic),
            )
        )
    else:
        if not args.synthetic:
            raise ContractError("probe fitting requires --manifest and an activation capture manifest")
        labels = [city for scenario in range(8) for city in (
            "Boston", "Seattle", "Chicago", "Denver"
        )]
        groups = [f"synthetic-scenario-{scenario}" for scenario in range(8) for _ in range(4)]
        row_ids = [f"{group}:{label}" for group, label in zip(groups, labels)]
        rng = np.random.default_rng(17)
        features = rng.normal(scale=0.05, size=(len(labels), 8))
        for index, label in enumerate(labels):
            features[index, sorted(set(labels)).index(label)] += 4.0
        captures = []
        capture_contract = {
            "schema_version": "1.0.0", "feature_dimension": 8,
            "feature_policy": "analytic_fixture", "sites": ["analytic_fixture"],
            "layers": [], "anchors": ["synthetic_anchor"],
            "model_repo": "analytic_fixture", "model_revision": "analytic_fixture",
            "config_sha256": config_sha, "code_commit": _git_commit(),
            "harness_version": HARNESS_VERSION, "synthetic": True,
        }
    grouped = fit_grouped_ridge_probe(
        features,
        labels,
        groups,
        alpha=args.alpha,
        folds=args.folds,
        seed=args.seed,
        row_ids=row_ids,
    )
    selection, site_selection_sha = _validated_selection(
        args.site_selection, required=False
    )
    _validate_probe_site_selection(selection, capture_contract)
    report = {
        **grouped,
        "analysis_status": (
            "synthetic_local_validation" if args.synthetic else "empirical_diagnostic"
        ),
        "role": args.role,
        "training_role": args.role,
        "capture_manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "source_manifest_sha256": sha256_file(args.manifest) if args.manifest else None,
        "role_manifest_sha256": sha256_file(args.role_manifest) if args.role_manifest else None,
        "site_selection_sha256": site_selection_sha,
        "site_selection_identity_sha256": (
            selection.get("selection_sha256") if selection else None
        ),
        "capture_contract": capture_contract,
        "capture_contract_sha256": sha256_value(capture_contract),
        "code_commit": _git_commit(),
        "harness_version": HARNESS_VERSION,
        "causal_use_prohibited": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "probe_metrics.json", report)
    if args.freeze_output:
        write_json(args.freeze_output, freeze_probe_report(report))
    print(
        f"fit {len(grouped['classes'])}-class grouped diagnostic probe "
        f"(CV accuracy {grouped['cv_accuracy']:.3f}) -> {args.output_root}"
    )
    return 0


def _validated_selection(path: Path | None, *, required: bool) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        if required:
            raise ContractError("this causal scan requires a frozen --selection")
        return None, None
    source = read_json(path)
    if not isinstance(source, Mapping):
        raise ContractError("frozen selection must be a JSON object")
    selection = dict(source)
    expected = selection.get("selection_sha256")
    observed = sha256_value({key: value for key, value in selection.items()
                             if key != "selection_sha256"})
    if expected != observed:
        raise ContractError("frozen selection hash mismatch")
    return selection, sha256_file(path)


def _scan_readout_path(
    config_path: Path, output_root: Path, explicit: Path | None, *, synthetic: bool,
) -> Path:
    if explicit is not None:
        path = explicit
    elif not synthetic:
        path = (
            _infer_run_file(output_root, "preflight/readouts.bound.json")
            or _infer_run_file(output_root, "readouts.bound.json")
        )
        if path is None:
            raise ContractError(
                "real causal scans require --readouts or preflight/readouts.bound.json"
            )
    else:
        path = config_path.parent / "readouts.json"
    if not path.is_file():
        raise ContractError(f"causal-scan readout config is missing: {path}")
    return path


def _query_readout_plan(
    path: Path, *, backend: Any | None, candidates: Sequence[str], require_bound: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[int]], dict[str, list[int]]]:
    source = read_json(path)
    if not isinstance(source, Mapping):
        raise ContractError("causal-scan readout config must be an object")
    readouts, schedules = _basic_readout_plan(source)
    root = [row for row in readouts if row["id"] == "root" and row["anchor"] == "query_end"]
    if len(root) != 1:
        raise ContractError("causal scans require exactly one root readout at query_end")
    prefix_ids, candidate_ids = _validate_readout_contract(
        source, root, backend=backend, candidates=candidates, require_bound=require_bound)
    return root[0], schedules, prefix_ids, candidate_ids


def _score_query_snapshot(
    backend: Any,
    snapshot: Any,
    *,
    target: str,
    stale: str,
    readout: Mapping[str, Any],
    schedules: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_scores: list[float] = []
    stale_scores: list[float] = []
    schedule_rows: list[dict[str, Any]] = []
    maximum_order_delta = 0.0
    for schedule in schedules:
        kwargs = {
            "prefix": str(readout["prefix"]),
            "prefix_start_offset_frames": int(schedule["prefix_start_offset_frames"]),
            "pad_frames_between_tokens": int(schedule["pad_frames_between_tokens"]),
        }
        forward = backend.score_candidates(
            snapshot, {"target": target, "stale": stale}, **kwargs)
        reverse = backend.score_candidates(
            snapshot, {"stale": stale, "target": target}, **kwargs)
        if set(forward) != {"target", "stale"} or set(reverse) != {"target", "stale"}:
            raise ContractError("causal-scan candidate scorer returned an invalid key set")
        values = [float(forward[name]) for name in ("target", "stale")]
        reverse_values = [float(reverse[name]) for name in ("target", "stale")]
        if not all(math.isfinite(value) for value in (*values, *reverse_values)):
            raise ContractError("causal-scan candidate scorer returned NaN or infinity")
        order_delta = max(abs(values[index] - reverse_values[index]) for index in (0, 1))
        if order_delta > 1e-6:
            raise ContractError("causal-scan candidate scores depend on candidate order")
        maximum_order_delta = max(maximum_order_delta, order_delta)
        target_scores.append(values[0])
        stale_scores.append(values[1])
        schedule_rows.append({
            "schedule_id": schedule["id"],
            "target_logprob": values[0],
            "stale_logprob": values[1],
            "candidate_order_delta": order_delta,
        })
    target_score = _logmeanexp(target_scores)
    stale_score = _logmeanexp(stale_scores)
    return {
        "target_logprob": target_score,
        "stale_logprob": stale_score,
        "margin_M": target_score - stale_score,
        "candidate_order_delta": maximum_order_delta,
        "schedules": schedule_rows,
    }


def _tensor_replacement(value: np.ndarray, head: int | None) -> Any:
    if head is None:
        return value
    if value.ndim != 4 or head >= int(value.shape[1]):
        raise ContractError(
            f"selected head {head} is outside donor tensor shape {list(value.shape)}"
        )
    return {"head": head, "tensor": value[:, head]}


def _real_patch_metric(
    backend: Any,
    *,
    donor_codes: np.ndarray,
    recipient_codes: np.ndarray,
    plan: CausalCellPlan,
    readout: Mapping[str, Any],
    schedules: Sequence[Mapping[str, Any]],
    target: str,
    stale: str,
    path: PathSpecification | None,
    anchor_rows_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    if path is None:
        sites = intervention_sites(plan.component)
        donor_replay = backend.replay_codes(
            backend.torch.as_tensor(donor_codes, device=backend.device),
            sites=sites,
            capture_layers=[plan.layer],
            capture_frames=[plan.source_frame],
            end_frame_exclusive=plan.source_frame + 1,
        )
        replacements: dict[tuple[str, int, int], Any] = {}
        for site in sites:
            donor_value = donor_replay.event_tensors.get((site, plan.layer, plan.source_frame))
            if donor_value is None:
                raise ContractError(
                    f"donor activation missing at {site}/L{plan.layer}/F{plan.source_frame}"
                )
            replacements[(site, plan.layer, plan.target_frame)] = _tensor_replacement(
                donor_value, plan.head)
        path_evidence = None
    else:
        writer = path.writer
        mediator = path.mediator
        donor_source = exact_anchor_frame(
            anchor_rows_by_key, plan.donor_trial_id, writer.anchor,
            available_frames=int(donor_codes.shape[-1]))
        writer_target = exact_anchor_frame(
            anchor_rows_by_key, plan.recipient_trial_id, writer.anchor,
            available_frames=int(recipient_codes.shape[-1]))
        mediator_target = exact_anchor_frame(
            anchor_rows_by_key, plan.recipient_trial_id, mediator.anchor,
            available_frames=int(recipient_codes.shape[-1]))
        if writer_target > mediator_target:
            raise ContractError("path writer frame occurs after its mediator frame")
        if mediator_target >= plan.query_end_frame_exclusive:
            raise ContractError("path mediator occurs after the query readout")
        donor_replay = backend.replay_codes(
            backend.torch.as_tensor(donor_codes, device=backend.device),
            sites=[writer.site], capture_layers=[writer.layer],
            capture_frames=[donor_source], end_frame_exclusive=donor_source + 1,
        )
        donor_writer = donor_replay.event_tensors.get(
            (writer.site, writer.layer, donor_source))
        if donor_writer is None:
            raise ContractError("path donor writer activation is absent")
        writer_replacement = {
            (writer.site, writer.layer, writer_target): _tensor_replacement(
                donor_writer, writer.head)
        }
        stage_one = backend.replay_codes(
            backend.torch.as_tensor(recipient_codes, device=backend.device),
            sites=[mediator.site], replacement=writer_replacement,
            capture_layers=[mediator.layer], capture_frames=[mediator_target],
            end_frame_exclusive=mediator_target + 1,
        )
        mediator_value = stage_one.event_tensors.get(
            (mediator.site, mediator.layer, mediator_target))
        if mediator_value is None:
            raise ContractError("path counterfactual mediator activation is absent")
        replacements = {
            (mediator.site, mediator.layer, mediator_target): _tensor_replacement(
                mediator_value, mediator.head)
        }
        path_evidence = {
            "algorithm": "two_stage_writer_to_mediator_path_patch_v1",
            "writer": {
                **path.identity["writer"],
                "donor_source_frame": donor_source,
                "recipient_target_frame": writer_target,
            },
            "mediator": {**path.identity["mediator"], "target_frame": mediator_target},
            "stage_one_feedback_sha256": stage_one.feedback_sha256,
        }

    recipient_tensor = backend.torch.as_tensor(recipient_codes, device=backend.device)
    baseline_replay = backend.replay_codes(
        recipient_tensor, sites=[], hook_enabled=False,
        end_frame_exclusive=plan.query_end_frame_exclusive)
    baseline_snapshot = backend.lm_gen.snapshot_streaming_state()
    baseline = _score_query_snapshot(
        backend, baseline_snapshot, target=target, stale=stale,
        readout=readout, schedules=schedules)
    patched_replay = backend.replay_codes(
        recipient_tensor, sites=[], replacement=replacements,
        capture_layers=[], capture_frames=[],
        end_frame_exclusive=plan.query_end_frame_exclusive)
    patched_snapshot = backend.lm_gen.snapshot_streaming_state()
    patched = _score_query_snapshot(
        backend, patched_snapshot, target=target, stale=stale,
        readout=readout, schedules=schedules)
    if baseline_replay.feedback_sha256 != patched_replay.feedback_sha256:
        raise ContractError("paired causal patch arms used different open-loop feedback")
    delta = float(patched["margin_M"] - baseline["margin_M"])
    if not math.isfinite(delta):
        raise ContractError("causal patch produced a non-finite effect")
    return {
        "baseline_M": baseline["margin_M"],
        "patched_M": patched["margin_M"],
        "delta_M": delta,
        "baseline_readout": baseline,
        "patched_readout": patched,
        "feedback_sha256": patched_replay.feedback_sha256,
        "donor_capture_frame_count": donor_replay.frame_count,
        "recipient_replay_frame_count": patched_replay.frame_count,
        "readout_end_frame_exclusive": plan.query_end_frame_exclusive,
        "path_evidence": path_evidence,
    }


def _synthetic_patch_metric(
    backend: SyntheticBackend,
    *,
    recipient: Mapping[str, Any],
    donor: Mapping[str, Any],
    plan: CausalCellPlan,
    path: PathSpecification | None,
    mediator_frame: int | None = None,
) -> dict[str, Any]:
    if path is None:
        metric = backend.patch(
            recipient, donor, component=plan.component, layer=plan.layer,
            head=plan.head, anchor_frame=plan.target_frame)
        return {**metric, "path_evidence": None}
    if mediator_frame is None:
        raise ContractError("analytic path fixture requires its exact mediator frame")
    recipient_run = backend.replay(recipient, [path.writer.site, path.mediator.site])
    donor_run = backend.replay(donor, [path.writer.site])
    writer_source = donor_run.activations[path.writer.site][
        path.writer.layer % backend.layers, plan.source_frame]
    writer_baseline = recipient_run.activations[path.writer.site][
        path.writer.layer % backend.layers, plan.target_frame]
    mediator = recipient_run.activations[path.mediator.site][
        path.mediator.layer % backend.layers, mediator_frame]
    # Stage one propagates the isolated writer difference into a frozen
    # counterfactual mediator; stage two injects only that mediator message.
    counterfactual_mediator = mediator + 0.5 * (writer_source - writer_baseline)
    direction = backend._vector(str(recipient["new_value"])) - backend._vector(
        str(recipient["old_value"]))
    baseline_margin = float(recipient_run.logits[0])
    delta = 0.5 * float(np.dot(counterfactual_mediator - mediator, direction))
    return {
        "baseline_M": baseline_margin,
        "patched_M": baseline_margin + delta,
        "delta_M": delta,
        "feedback_sha256": recipient_run.feedback_sha256,
        "path_evidence": {
            "algorithm": "two_stage_writer_to_mediator_path_patch_v1",
            "writer": path.identity["writer"],
            "mediator": path.identity["mediator"],
            "analytic_fixture": True,
        },
    }


def _scan(argv: Sequence[str], kind: str) -> int:
    parser = _parser(f"Run resumable {kind} causal patches.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--encoded-manifest", type=Path)
    parser.add_argument("--anchor-map", type=Path)
    parser.add_argument("--readouts", type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--layers")
    parser.add_argument("--anchors")
    parser.add_argument("--donors", default="clean_current,self,shuffled")
    parser.add_argument("--controls", default="self,current,wrong,shuffled")
    parser.add_argument("--components", default="attn_out,mlp_out,head_z")
    parser.add_argument("--modes", default="k_only,v_only,kv")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--limit-scenarios", type=int)
    parser.add_argument("--scan-spec", type=Path,
                        help="Exact static workload/execution contract; mandatory for real scans.")
    parser.add_argument("--readiness-go", type=Path,
                        help="Hash-bound GO authorization; mandatory for real scans.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Materialize and authenticate the exact cell grid without constructing a model.",
    )
    args = parser.parse_args(argv)
    if args.plan_only and args.resume:
        raise ContractError("--plan-only and --resume are mutually exclusive")
    selection, selection_file_sha = _validated_selection(
        args.selection, required=kind == "path")
    path_spec = parse_path_specification(selection or {}) if kind == "path" else None
    default_layers = (
        str(path_spec.writer.layer)
        if path_spec is not None
        else str(selection["layer"])
        if selection is not None and kind in {"component", "kv"}
        and isinstance(selection.get("layer"), int)
        else "0,3,5"
    )
    default_anchors = (
        path_spec.writer.anchor
        if path_spec is not None
        else str(selection["anchor"])
        if selection is not None and kind in {"component", "kv"}
        and isinstance(selection.get("anchor"), str)
        else "new_end,query_end"
    )
    layers = _ints(args.layers or default_layers)
    anchors = _csv(args.anchors or default_anchors)
    if not layers or len(layers) != len(set(layers)) or any(layer < 0 for layer in layers):
        raise ContractError("--layers must contain unique non-negative layer indices")
    if not anchors or len(anchors) != len(set(anchors)):
        raise ContractError("--anchors must contain unique semantic anchor names")
    if path_spec is not None and (
        layers != [path_spec.writer.layer] or anchors != [path_spec.writer.anchor]
    ):
        raise ContractError(
            "path scan grid must exactly match the frozen writer layer and anchor")
    if args.manifest is None:
        args.manifest = _infer_run_file(args.output_root, "manifests/mechanistic_trials.jsonl")
    if not args.synthetic and args.encoded_manifest is None:
        args.encoded_manifest = _infer_run_file(args.output_root, "encoded_user_manifest.jsonl")
    if args.anchor_map is None:
        args.anchor_map = _infer_run_file(args.output_root, "anchor_map.jsonl")
    if kind == "residual":
        components = ["resid_post"]
    elif kind == "component":
        components = _csv(args.components)
    elif kind == "kv":
        components = _csv(args.modes)
    else:
        components = ["path"]
    donors = _csv(args.donors)
    controls = _csv(args.controls)
    arms = active_arms(kind, donors, controls)
    authorization_binding = None
    scan_spec = None
    config = read_json(args.config)
    if not isinstance(config, Mapping):
        raise ContractError("mechanistic config must be an object")
    if not args.synthetic:
        if args.manifest is None or args.encoded_manifest is None:
            raise ContractError("Moshiko paid scan requires manifest and encoded manifest before GPU setup")
        if args.scan_spec is None or args.readiness_go is None:
            raise ContractError(
                "Moshiko paid scan requires --scan-spec and --readiness-go; run readiness assessment first")
        scan_spec = read_json(args.scan_spec)
        validate_scan_execution(scan_spec, {
            "kind": kind,
            "role": args.role,
            "layers": layers,
            "anchors": anchors,
            "donors": donors,
            "controls": controls,
            "components": components,
            "limit_scenarios": args.limit_scenarios,
            "selection_sha256": selection_file_sha,
        })
        authorization_binding = build_target_binding_from_files(
            config_path=args.config,
            manifest_path=args.manifest,
            encoded_manifest_path=args.encoded_manifest,
            scan_spec_path=args.scan_spec,
        )
        verify_authorization_artifact(read_json(args.readiness_go), authorization_binding)
    if args.synthetic and args.manifest is None:
        trials = []
        for scenario in ("synthetic-1", "synthetic-2"):
            for direction, old, new in (
                ("boston_to_seattle", "Boston", "Seattle"),
                ("seattle_to_boston", "Seattle", "Boston"),
            ):
                shared = {
                    "scenario_id": scenario, "direction_id": direction,
                    "speaker_id": "synthetic-speaker", "old_value": old,
                    "new_value": new, "frame_count": 12, "analysis_fold": 1,
                    "role": args.role,
                }
                trials.extend((
                    {**shared, "trial_id": f"{scenario}-{direction}-clean",
                     "condition": "clean_current"},
                    {**shared, "trial_id": f"{scenario}-{direction}-repair",
                     "condition": "repair"},
                ))
    else:
        trials = _load_trials(args.manifest, args.role)
    if args.limit_scenarios:
        allowed = sorted({str(row.get("scenario_id")) for row in trials})[:args.limit_scenarios]
        trials = [row for row in trials if str(row.get("scenario_id")) in allowed]
    # Validate all donor-matching metadata and materialize the arm dimension
    # before model construction.  Any missing control stops the paid run.
    repairs = repair_recipients(trials)
    assignments = materialize_donor_assignments(trials, repairs, arms)
    encoded_by_id = (
        _encoded_rows(args.encoded_manifest)[1]
        if not args.synthetic and args.encoded_manifest else {}
    )
    if not args.synthetic and not encoded_by_id:
        raise ContractError("Moshiko patch scans require --encoded-manifest")
    if args.synthetic and args.anchor_map is None:
        frozen_frames = {
            "old_end": 0, "cue_end": 1, "new_end": 2, "D1_end": 4,
            "D2_end": 6, "D3_end": 8, "query_end": 11,
        }
        anchor_rows_by_key = {
            (str(row["trial_id"]), anchor): {
                "trial_id": row["trial_id"], "anchor": anchor, "frame": frame,
                "timebase": "analytic_fixture",
            }
            for row in trials for anchor, frame in frozen_frames.items()
        }
        anchor_map_sha = sha256_value(sorted(anchor_rows_by_key.values(), key=lambda row: (
            str(row["trial_id"]), str(row["anchor"]))))
    else:
        if args.anchor_map is None or not args.anchor_map.is_file():
            raise ContractError("causal scans require an exact --anchor-map")
        _, anchor_rows_by_key = _anchor_lookup(args.anchor_map)
        anchor_map_sha = sha256_file(args.anchor_map)
    available_frames: dict[str, int] = {}
    for row in trials:
        trial_id = str(row["trial_id"])
        if args.synthetic:
            available_frames[trial_id] = _exact_int_field(
                row.get("frame_count"), f"{trial_id} frame_count")
        else:
            encoded = encoded_by_id.get(trial_id)
            if encoded is None:
                raise ContractError(f"encoded manifest has no causal-scan trial {trial_id}")
            if encoded.get("synthetic") is not False:
                raise ContractError(f"{trial_id}: paid scan cannot use synthetic encoded data")
            available_frames[trial_id] = _exact_int_field(
                encoded.get("conversation_frame_end_exclusive"),
                f"{trial_id} conversation frame end")
    store = AtomicCellStore(args.output_root)
    config_hash = sha256_file(args.config)
    run_hash = (target_binding_sha256(authorization_binding) if authorization_binding is not None else
                sha256_value({
                    "config": config_hash, "role": args.role, "kind": kind,
                    "trials": trials, "layers": layers, "anchors": anchors,
                    "arms": arms, "components": components,
                    "selection": selection, "synthetic": bool(args.synthetic),
                }))
    readout_path = _scan_readout_path(
        args.config, args.output_root, args.readouts, synthetic=bool(args.synthetic))
    readout_hash = sha256_file(readout_path)
    candidate_values = [
        value for row in repairs
        for value in (str(row["old_value"]), str(row["new_value"]))
    ]
    readout, schedules, _, _ = _query_readout_plan(
        readout_path, backend=None, candidates=candidate_values,
        require_bound=not args.synthetic)
    configured_head_count = int(config.get("model", {}).get("heads", 0))
    if configured_head_count <= 0:
        raise ContractError("config.model.heads must be positive")
    planned_head_count = 4 if args.synthetic else configured_head_count
    component_heads: dict[str, int | None] = {}
    if kind == "kv" and selection is not None:
        selected_head = selection.get("kv_head", selection.get("head"))
        for component in components:
            component_heads[component] = selected_head
    if path_spec is None:
        logical_plans = materialize_cell_grid(
            trials, repairs, assignments, arms=arms, components=components,
            layers=layers, anchors=anchors, anchor_rows=anchor_rows_by_key,
            available_frames=available_frames, head_count=planned_head_count,
            component_heads=component_heads)
    else:
        logical_plans = []
        for recipient in repairs:
            recipient_id = str(recipient["trial_id"])
            query_end = exact_anchor_frame(
                anchor_rows_by_key, recipient_id, "query_end",
                available_frames=available_frames[recipient_id]) + 1
            for arm in arms:
                assignment = assignments[(recipient_id, arm)]
                source_frame = exact_anchor_frame(
                    anchor_rows_by_key, assignment.donor_trial_id, path_spec.writer.anchor,
                    available_frames=available_frames[assignment.donor_trial_id])
                target_frame = exact_anchor_frame(
                    anchor_rows_by_key, recipient_id, path_spec.writer.anchor,
                    available_frames=available_frames[recipient_id])
                exact_anchor_frame(
                    anchor_rows_by_key, recipient_id, path_spec.mediator.anchor,
                    available_frames=available_frames[recipient_id])
                logical_plans.append(CausalCellPlan(
                    recipient_trial_id=recipient_id,
                    donor_trial_id=assignment.donor_trial_id,
                    requested_arm=arm, relation=assignment.relation,
                    component="path", layer=path_spec.writer.layer,
                    anchor=path_spec.writer.anchor, source_frame=source_frame,
                    target_frame=target_frame, query_end_frame_exclusive=query_end,
                    head=path_spec.writer.head,
                ))
    rows_by_id = {str(row["trial_id"]): row for row in trials}
    planned_cells: list[tuple[CausalCellPlan, DonorAssignment, PatchCell]] = []
    for plan in logical_plans:
        path_frames = ()
        if path_spec is not None:
            mediator_frame = exact_anchor_frame(
                anchor_rows_by_key, plan.recipient_trial_id, path_spec.mediator.anchor,
                available_frames=available_frames[plan.recipient_trial_id])
            path_frames = (mediator_frame,)
        cell = PatchCell(
            run_hash, plan.donor_trial_id, plan.recipient_trial_id, plan.component,
            plan.layer, plan.head, (plan.source_frame,),
            (plan.target_frame, *path_frames), readout_hash)
        planned_cells.append((plan, assignments[(plan.recipient_trial_id, plan.requested_arm)], cell))
    if not args.synthetic:
        declared = estimate_workload(read_jsonl(args.manifest), config, scan_spec)
        if declared.cell_count != len(planned_cells):
            raise ContractError(
                "authorized workload cell count does not match implemented grid "
                f"({declared.cell_count} declared vs {len(planned_cells)} actual); "
                "unmaterialized donor/control arms cannot be billed")
    provenance = {
        "code_commit": _git_commit(), "harness_version": HARNESS_VERSION,
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "config_sha256": config_hash,
        "manifest_sha256": sha256_file(args.manifest) if args.manifest else sha256_value(trials),
        "encoded_manifest_sha256": (
            sha256_file(args.encoded_manifest) if args.encoded_manifest else None),
        "anchor_map_sha256": anchor_map_sha,
        "readout_sha256": readout_hash,
        "scan_spec_sha256": sha256_file(args.scan_spec) if args.scan_spec else None,
        "selection_file_sha256": selection_file_sha,
        "data_sha256": authorization_binding.get("data_sha256") if authorization_binding else None,
        "run_identity_sha256": run_hash,
    }
    plan_rows = [json.loads(canonical_json({
        "schema_version": "1.0.0", "cell_id": cell.cell_id,
        **cell.__dict__, "donor_arm": plan.requested_arm,
        "relation": plan.relation, "anchor": plan.anchor,
        "query_end_frame_exclusive": plan.query_end_frame_exclusive,
        "donor_assignment": assignment.to_dict(), "path": path_spec.identity if path_spec else None,
    })) for plan, assignment, cell in planned_cells]
    plan_hash = sha256_value(plan_rows)
    plan_path = args.output_root / "planned_cells.jsonl"
    if plan_path.exists() and read_jsonl(plan_path) != plan_rows:
        raise ContractError("existing causal scan plan differs from the current exact grid")
    write_jsonl(plan_path, plan_rows)
    write_json(args.output_root / "scan_plan.json", {
        "schema_version": "1.0.0", "kind": kind, "role": args.role,
        "planned_cell_count": len(plan_rows), "planned_cells_sha256": plan_hash,
        "result_uri": f"{kind}_patch_results.jsonl",
        "active_arm_source": "controls" if kind == "component" else "donors",
        "active_arms": list(arms), "components": components,
        "layers": layers, "anchors": anchors, "provenance": provenance,
    })
    if args.plan_only:
        if (
            (args.output_root / f"{kind}_patch_results.jsonl").exists()
            or store.rows()
            or store.failure_rows()
        ):
            raise ContractError("--plan-only requires a pristine scan root with no result cells")
        print(f"planned {len(plan_rows)} {kind} cells without constructing a model")
        return 0
    pending_cells: list[tuple[CausalCellPlan, DonorAssignment, PatchCell]] = []
    skipped = 0
    for plan in planned_cells:
        cell = plan[-1]
        if store.contains(cell):
            if args.resume:
                skipped += 1
                continue
            raise ContractError(
                f"patch cell already exists; use --resume instead of replaying it: {cell.cell_id}")
        pending_cells.append(plan)
    if not pending_cells:
        rows = store.merge(args.output_root / f"{kind}_patch_results.jsonl")
        failures = store.merge_failures(args.output_root / "failures.jsonl")
        write_json(args.output_root / "resume_summary.json", {
            "planned_cells": len(planned_cells), "completed_cells": len(store.rows()),
            "unresolved_failed_cells": sum(row["status"] == "failed" for row in rows),
            "failure_attempts": len(failures), "duplicate_cells": 0,
            "skipped_existing_cells": len(planned_cells), "run_identity_sha256": run_hash,
            "planned_cells_sha256": plan_hash,
        })
        print(f"{kind} scan resumed with 0 GPU replays; {len(rows)} cells already complete")
        return 0
    # The model/checkpoint is deliberately constructed only after the GO artifact and
    # exact static grid have both been verified.
    backend = SyntheticBackend() if args.synthetic else MoshiBackend(
        model_repo=MODEL_REPO, model_revision=MODEL_REVISION, use_sampling=False)
    if not args.synthetic:
        _query_readout_plan(
            readout_path, backend=backend, candidates=candidate_values, require_bound=True)
    abort_oom: Exception | None = None
    prior_attempts = Counter(row["cell_id"] for row in store.failure_rows())
    for plan, assignment, cell in pending_cells:
        recipient = rows_by_id[plan.recipient_trial_id]
        donor = rows_by_id[plan.donor_trial_id]
        common = {
            "anchor": plan.anchor, "role": args.role,
            "scenario_id": recipient.get("scenario_id"),
            "direction_id": recipient.get("direction_id"),
            "speaker_id": recipient.get("speaker_id"),
            "old_value": recipient.get("old_value"), "new_value": recipient.get("new_value"),
            "donor_arm": plan.requested_arm, "relation": plan.relation,
            "donor_assignment": assignment.to_dict(),
            "source_frame": plan.source_frame, "target_frame": plan.target_frame,
            "query_end_frame_exclusive": plan.query_end_frame_exclusive,
            "readout_id": readout["id"], "readout_sha256": readout_hash,
            "path": path_spec.identity if path_spec is not None else None,
            "path_evidence": ({
                "algorithm": "two_stage_writer_to_mediator_path_patch_v1",
                **path_spec.identity,
            } if path_spec is not None else None),
            "attempt_index": prior_attempts[cell.cell_id] + 1,
            "provenance": provenance, "synthetic": bool(args.synthetic),
        }
        caught_error: Exception | None = None
        try:
            if args.synthetic:
                metric = _synthetic_patch_metric(
                    backend, recipient=recipient, donor=donor, plan=plan,
                    path=path_spec,
                    mediator_frame=(cell.target_frames[1] if path_spec is not None else None))
            else:
                assert args.encoded_manifest is not None
                donor_codes = _load_encoded_array(
                    args.encoded_manifest, encoded_by_id[plan.donor_trial_id],
                    "conversation_codes", require_current_contract=True)
                recipient_codes = _load_encoded_array(
                    args.encoded_manifest, encoded_by_id[plan.recipient_trial_id],
                    "conversation_codes", require_current_contract=True)
                metric = _real_patch_metric(
                    backend, donor_codes=donor_codes, recipient_codes=recipient_codes,
                    plan=plan, readout=readout, schedules=schedules,
                    target=str(recipient["new_value"]), stale=str(recipient["old_value"]),
                    path=path_spec, anchor_rows_by_key=anchor_rows_by_key)
            if not all(math.isfinite(float(metric[name]))
                       for name in ("baseline_M", "patched_M", "delta_M")):
                raise ContractError("causal patch metric contains NaN or infinity")
            if plan.relation == "self":
                tolerance = float(config.get("gates", {}).get("self_patch_abs_delta_max", 1e-5))
                if abs(float(metric["delta_M"])) > tolerance:
                    raise ContractError("self patch violated the frozen no-op tolerance")
            payload = {"status": "completed", **common, **metric}
        except Exception as error:
            caught_error = error
            payload = {"status": "failed", **common,
                       "failure_type": type(error).__name__,
                       "failure_message": str(error)}
        store.record(cell, payload)
        if payload["status"] == "failed":
            prior_attempts[cell.cell_id] += 1
        if payload["status"] == "failed" and (
            "outofmemory" in str(payload["failure_type"]).lower()
            or "out of memory" in str(payload["failure_message"]).lower()
        ):
            assert caught_error is not None
            abort_oom = caught_error
            try:
                backend.torch.cuda.empty_cache()
            except Exception:
                pass
            break
    rows = store.merge(args.output_root / f"{kind}_patch_results.jsonl")
    failures = store.merge_failures(args.output_root / "failures.jsonl")
    completed = store.rows()
    write_json(args.output_root / "resume_summary.json", {
        "planned_cells": len(planned_cells), "completed_cells": len(completed),
        "unresolved_failed_cells": sum(row["status"] == "failed" for row in rows),
        "failure_attempts": len(failures), "duplicate_cells": 0,
        "skipped_existing_cells": skipped, "run_identity_sha256": run_hash,
        "planned_cells_sha256": plan_hash,
    })
    if abort_oom is not None:
        raise ContractError(
            "GPU OOM was recorded atomically; scan stopped to prevent repeated paid failures"
        ) from abort_oom
    print(f"{kind} scan contains {len(rows)} atomic cells -> {args.output_root}")
    return 0


def freeze_mechanistic_selection(argv: Sequence[str]) -> int:
    parser = _parser("Freeze the highest-ranked discovery site before opening validation data.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument(
        "--components",
        help=(
            "Optional comma-separated eligible component names; use k_only,v_only "
            "when freezing a single-site KV writer for path discovery."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    for path in args.discovery_root.rglob("*_patch_results.jsonl"):
        rows.extend(read_jsonl(path))
    selection = freeze_selection(
        rows, sha256_file(args.config),
        components=_csv(args.components) if args.components else None,
    )
    write_json(args.output, selection)
    print(f"froze {selection['component']} layer {selection['layer']} -> {args.output}")
    return 0


def run_confirmatory(argv: Sequence[str]) -> int:
    parser = _parser("Apply one frozen selection to an internal or formal validation role.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--encoded-manifest", type=Path)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--readouts", type=Path)
    parser.add_argument("--baseline-readout", type=Path)
    parser.add_argument("--scan-spec", type=Path)
    parser.add_argument("--readiness-go", type=Path)
    parser.add_argument(
        "--control-arms",
        help=("Comma-separated preregistered control donor arms, e.g. "
              "clean_stale,self,same_value_random,shuffled."),
    )
    parser.add_argument("--role", required=True)
    parser.add_argument("--folds")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Materialize and authenticate confirmation cells without constructing a model.",
    )
    args = parser.parse_args(argv)
    if args.plan_only and args.resume:
        raise ContractError("--plan-only and --resume are mutually exclusive")
    if args.manifest is None:
        args.manifest = _infer_run_file(args.output_root, "manifests/mechanistic_trials.jsonl")
    if not args.synthetic and args.encoded_manifest is None:
        args.encoded_manifest = _infer_run_file(args.output_root, "encoded_user_manifest.jsonl")
    if args.anchors is None:
        args.anchors = _infer_run_file(args.output_root, "anchor_map.jsonl")
    selection, selection_file_sha = _validated_selection(args.selection, required=True)
    assert selection is not None and selection_file_sha is not None
    config = read_json(args.config)
    if not isinstance(config, Mapping):
        raise ContractError("mechanistic config must be an object")
    config_sha = sha256_file(args.config)
    if selection.get("config_sha256") != config_sha:
        raise ContractError("frozen selection targets a different config")
    if args.role == "formal_confirmation" and args.role_manifest is None:
        raise ContractError("formal confirmation requires an immutable role manifest")
    if args.synthetic and args.manifest is None:
        trials = []
        for scenario in ("synthetic-confirm-1", "synthetic-confirm-2"):
            for direction, old, new in (
                ("boston_to_seattle", "Boston", "Seattle"),
                ("seattle_to_boston", "Seattle", "Boston"),
            ):
                shared = {
                    "scenario_id": scenario, "direction_id": direction,
                    "speaker_id": "synthetic-speaker", "old_value": old,
                    "new_value": new, "frame_count": 12, "analysis_fold": 1,
                    "role": args.role,
                }
                trials.extend((
                    {**shared, "trial_id": f"{scenario}-{direction}-clean",
                     "condition": "clean_current"},
                    {**shared, "trial_id": f"{scenario}-{direction}-repair",
                     "condition": "repair"},
                ))
    else:
        trials = _load_trials(
            args.manifest, args.role, _ints(args.folds) if args.folds else None,
            args.role_manifest)
    recipients = repair_recipients(trials)
    donor_arm = selection.get("donor_arm")
    if not isinstance(donor_arm, str) or not donor_arm:
        raise ContractError("frozen selection has no donor_arm")
    component = str(selection.get("component", ""))
    kind = (
        "residual" if component == "resid_post"
        else "component" if component in {"attn_out", "mlp_out", "head_z"}
        else "kv" if component in {"k_only", "v_only", "kv"}
        else "path" if component == "path"
        else ""
    )
    if not kind:
        raise ContractError(f"frozen selection has unsupported component: {component!r}")
    requested_controls = _csv(args.control_arms) if args.control_arms else []
    if donor_arm in requested_controls or len(set(requested_controls)) != len(requested_controls):
        raise ContractError("confirmation control arms must be unique and exclude the primary arm")
    requested_arms = [donor_arm, *requested_controls]
    arms = active_arms(kind, requested_arms, requested_arms)
    assignments = materialize_donor_assignments(trials, recipients, arms)
    path_spec = parse_path_specification(selection) if kind == "path" else None
    layer = _exact_int_field(selection.get("layer"), "frozen selection layer")
    head = selection.get("kv_head", selection.get("head"))
    if head is not None:
        head = _exact_int_field(head, "frozen selection head")
    anchor_name = str(selection.get("anchor", ""))
    if not anchor_name:
        raise ContractError("frozen selection has no semantic anchor")

    authorization_binding = None
    scan_spec = None
    if not args.synthetic:
        if args.manifest is None or args.encoded_manifest is None:
            raise ContractError("Moshiko confirmation requires manifest and encoded manifest")
        if args.scan_spec is None or args.readiness_go is None:
            raise ContractError(
                "real confirmation requires hash-bound --scan-spec and --readiness-go")
        scan_spec = read_json(args.scan_spec)
        execution = scan_spec.get("execution") if isinstance(scan_spec, Mapping) else None
        if not isinstance(execution, Mapping):
            raise ContractError("confirmatory scan spec has no execution object")
        actual = dict(execution)
        actual.update({
            "kind": kind, "role": args.role, "layers": [layer],
            "anchors": [anchor_name], "components": [component],
            "limit_scenarios": None, "selection_sha256": selection_file_sha,
        })
        validate_scan_execution(scan_spec, actual)
        active_field = "controls" if kind == "component" else "donors"
        if list(execution.get(active_field, [])) != arms:
            raise ContractError(
                "confirmatory scan spec active arms must equal the frozen primary plus controls")
        authorization_binding = build_target_binding_from_files(
            config_path=args.config, manifest_path=args.manifest,
            encoded_manifest_path=args.encoded_manifest, scan_spec_path=args.scan_spec)
        verify_authorization_artifact(read_json(args.readiness_go), authorization_binding)

    encoded_by_id = (
        _encoded_rows(args.encoded_manifest)[1]
        if not args.synthetic and args.encoded_manifest else {})
    if not args.synthetic and not encoded_by_id:
        raise ContractError("Moshiko confirmation requires an encoded manifest")
    if args.synthetic and args.anchors is None:
        frozen_frames = {
            "old_end": 0, "cue_end": 1, "new_end": 2, "D1_end": 4,
            "D2_end": 6, "D3_end": 8, "query_end": 11,
        }
        anchor_rows_by_key = {
            (str(row["trial_id"]), anchor): {
                "trial_id": row["trial_id"], "anchor": anchor, "frame": frame,
                "timebase": "analytic_fixture",
            }
            for row in trials for anchor, frame in frozen_frames.items()
        }
        anchors_sha = sha256_value(sorted(anchor_rows_by_key.values(), key=lambda row: (
            str(row["trial_id"]), str(row["anchor"]))))
    else:
        if args.anchors is None or not args.anchors.is_file():
            raise ContractError("confirmation requires an exact semantic anchor map")
        _, anchor_rows_by_key = _anchor_lookup(args.anchors)
        anchors_sha = sha256_file(args.anchors)
    available_frames: dict[str, int] = {}
    for row in trials:
        trial_id = str(row["trial_id"])
        if args.synthetic:
            available_frames[trial_id] = _exact_int_field(
                row.get("frame_count"), f"{trial_id} frame_count")
        else:
            encoded = encoded_by_id.get(trial_id)
            if encoded is None or encoded.get("synthetic") is not False:
                raise ContractError(f"{trial_id}: confirmation encoded row is absent or synthetic")
            available_frames[trial_id] = _exact_int_field(
                encoded.get("conversation_frame_end_exclusive"),
                f"{trial_id} conversation frame end")

    readout_path = _scan_readout_path(
        args.config, args.output_root, args.readouts, synthetic=bool(args.synthetic))
    readout_sha = sha256_file(readout_path)
    if selection.get("readout_sha256") != readout_sha:
        raise ContractError("frozen selection targets a different readout contract")
    candidate_values = [
        value for row in recipients
        for value in (str(row["old_value"]), str(row["new_value"]))
    ]
    readout, schedules, _, _ = _query_readout_plan(
        readout_path, backend=None, candidates=candidate_values,
        require_bound=not args.synthetic)

    logical_plans: list[CausalCellPlan] = []
    for recipient in recipients:
        recipient_id = str(recipient["trial_id"])
        for requested_arm in arms:
            assignment = assignments[(recipient_id, requested_arm)]
            endpoint = path_spec.writer if path_spec is not None else None
            source_anchor = endpoint.anchor if endpoint is not None else anchor_name
            source = exact_anchor_frame(
                anchor_rows_by_key, assignment.donor_trial_id, source_anchor,
                available_frames=available_frames[assignment.donor_trial_id])
            target = exact_anchor_frame(
                anchor_rows_by_key, recipient_id, source_anchor,
                available_frames=available_frames[recipient_id])
            query_end = exact_anchor_frame(
                anchor_rows_by_key, recipient_id, "query_end",
                available_frames=available_frames[recipient_id]) + 1
            if target >= query_end:
                raise ContractError("frozen intervention is after the query readout")
            if path_spec is not None:
                exact_anchor_frame(
                    anchor_rows_by_key, recipient_id, path_spec.mediator.anchor,
                    available_frames=available_frames[recipient_id])
            logical_plans.append(CausalCellPlan(
                recipient_trial_id=recipient_id, donor_trial_id=assignment.donor_trial_id,
                requested_arm=requested_arm, relation=assignment.relation,
                component=component, layer=layer, anchor=anchor_name,
                source_frame=source, target_frame=target,
                query_end_frame_exclusive=query_end, head=head))

    store = AtomicCellStore(args.output_root)
    run_hash = (
        target_binding_sha256(authorization_binding)
        if authorization_binding is not None
        else sha256_value({"selection": selection, "trials": trials, "role": args.role,
                           "synthetic": True}))
    rows_by_id = {str(row["trial_id"]): row for row in trials}
    planned: list[tuple[CausalCellPlan, DonorAssignment, PatchCell]] = []
    for plan in logical_plans:
        extra_target = ()
        if path_spec is not None:
            extra_target = (exact_anchor_frame(
                anchor_rows_by_key, plan.recipient_trial_id, path_spec.mediator.anchor,
                available_frames=available_frames[plan.recipient_trial_id]),)
        cell = PatchCell(
            run_hash, plan.donor_trial_id, plan.recipient_trial_id, component,
            layer, head, (plan.source_frame,), (plan.target_frame, *extra_target), readout_sha)
        planned.append((
            plan,
            assignments[(plan.recipient_trial_id, plan.requested_arm)],
            cell,
        ))
    provenance = {
        "code_commit": _git_commit(), "harness_version": HARNESS_VERSION,
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "config_sha256": config_sha,
        "manifest_sha256": sha256_file(args.manifest) if args.manifest else sha256_value(trials),
        "role_manifest_sha256": sha256_file(args.role_manifest) if args.role_manifest else None,
        "encoded_manifest_sha256": sha256_file(args.encoded_manifest) if args.encoded_manifest else None,
        "anchor_map_sha256": anchors_sha, "readout_sha256": readout_sha,
        "baseline_readout_sha256": (
            sha256_file(args.baseline_readout) if args.baseline_readout else None),
        "selection_file_sha256": selection_file_sha,
        "scan_spec_sha256": sha256_file(args.scan_spec) if args.scan_spec else None,
        "data_sha256": authorization_binding.get("data_sha256") if authorization_binding else None,
        "run_identity_sha256": run_hash,
    }
    plan_rows = [json.loads(canonical_json({
        "schema_version": "1.0.0", "cell_id": cell.cell_id, **cell.__dict__,
        "donor_arm": plan.requested_arm, "relation": plan.relation,
        "anchor": plan.anchor, "query_end_frame_exclusive": plan.query_end_frame_exclusive,
        "donor_assignment": assignment.to_dict(), "path": path_spec.identity if path_spec else None,
        "selection_sha256": selection["selection_sha256"],
    })) for plan, assignment, cell in planned]
    plan_hash = sha256_value(plan_rows)
    plan_path = args.output_root / "planned_cells.jsonl"
    if plan_path.exists() and read_jsonl(plan_path) != plan_rows:
        raise ContractError("existing confirmatory plan differs from the frozen identities")
    write_jsonl(plan_path, plan_rows)
    write_json(args.output_root / "scan_plan.json", {
        "schema_version": "1.0.0", "kind": kind, "role": args.role,
        "planned_cell_count": len(planned), "planned_cells_sha256": plan_hash,
        "result_uri": "patch_results.jsonl", "confirmation": True,
        "selection_sha256": selection["selection_sha256"], "provenance": provenance,
    })
    if args.plan_only:
        if (
            (args.output_root / "patch_results.jsonl").exists()
            or store.rows()
            or store.failure_rows()
        ):
            raise ContractError(
                "--plan-only requires a pristine confirmation root with no result cells"
            )
        print(f"planned {len(plan_rows)} confirmatory cells without constructing a model")
        return 0
    pending: list[tuple[CausalCellPlan, DonorAssignment, PatchCell]] = []
    skipped = 0
    for row in planned:
        if store.contains(row[-1]):
            if not args.resume:
                raise ContractError(
                    f"confirmatory cell already exists; use --resume: {row[-1].cell_id}")
            skipped += 1
        else:
            pending.append(row)
    backend = None
    if pending:
        backend = SyntheticBackend() if args.synthetic else MoshiBackend(
            model_repo=MODEL_REPO, model_revision=MODEL_REVISION, use_sampling=False)
        if not args.synthetic:
            _query_readout_plan(
                readout_path, backend=backend, candidates=candidate_values, require_bound=True)
    abort_oom: Exception | None = None
    prior_attempts = Counter(row["cell_id"] for row in store.failure_rows())
    for plan, assignment, cell in pending:
        assert backend is not None
        recipient = rows_by_id[plan.recipient_trial_id]
        donor = rows_by_id[plan.donor_trial_id]
        common = {
            "role": args.role, "scenario_id": recipient.get("scenario_id"),
            "direction_id": recipient.get("direction_id"), "speaker_id": recipient.get("speaker_id"),
            "old_value": recipient.get("old_value"), "new_value": recipient.get("new_value"),
            "selection_sha256": selection["selection_sha256"],
            "donor_arm": plan.requested_arm, "relation": assignment.relation,
            "donor_assignment": assignment.to_dict(), "anchor": anchor_name,
            "source_frame": plan.source_frame, "target_frame": plan.target_frame,
            "query_end_frame_exclusive": plan.query_end_frame_exclusive,
            "readout_id": readout["id"], "readout_sha256": readout_sha,
            "path": path_spec.identity if path_spec is not None else None,
            "path_evidence": ({
                "algorithm": "two_stage_writer_to_mediator_path_patch_v1",
                **path_spec.identity,
            } if path_spec is not None else None),
            "attempt_index": prior_attempts[cell.cell_id] + 1,
            "provenance": provenance, "synthetic": bool(args.synthetic),
        }
        caught_error: Exception | None = None
        try:
            if args.synthetic:
                metric = _synthetic_patch_metric(
                    backend, recipient=recipient, donor=donor, plan=plan,
                    path=path_spec,
                    mediator_frame=(cell.target_frames[1] if path_spec is not None else None))
            else:
                assert args.encoded_manifest is not None
                donor_codes = _load_encoded_array(
                    args.encoded_manifest, encoded_by_id[plan.donor_trial_id],
                    "conversation_codes", require_current_contract=True)
                recipient_codes = _load_encoded_array(
                    args.encoded_manifest, encoded_by_id[plan.recipient_trial_id],
                    "conversation_codes", require_current_contract=True)
                metric = _real_patch_metric(
                    backend, donor_codes=donor_codes, recipient_codes=recipient_codes,
                    plan=plan, readout=readout, schedules=schedules,
                    target=str(recipient["new_value"]), stale=str(recipient["old_value"]),
                    path=path_spec, anchor_rows_by_key=anchor_rows_by_key)
            if not all(math.isfinite(float(metric[name]))
                       for name in ("baseline_M", "patched_M", "delta_M")):
                raise ContractError("confirmatory metric contains NaN or infinity")
            if plan.relation == "self":
                tolerance = float(config.get("gates", {}).get("self_patch_abs_delta_max", 1e-5))
                if abs(float(metric["delta_M"])) > tolerance:
                    raise ContractError("confirmation self patch violated the frozen no-op tolerance")
            payload = {"status": "completed", **common, **metric}
        except Exception as error:
            caught_error = error
            payload = {"status": "failed", **common,
                       "failure_type": type(error).__name__, "failure_message": str(error)}
        store.record(cell, payload)
        if payload["status"] == "failed":
            prior_attempts[cell.cell_id] += 1
        if payload["status"] == "failed" and (
            "outofmemory" in str(payload["failure_type"]).lower()
            or "out of memory" in str(payload["failure_message"]).lower()
        ):
            assert caught_error is not None
            abort_oom = caught_error
            try:
                backend.torch.cuda.empty_cache()
            except Exception:
                pass
            break
    rows = store.merge(args.output_root / "patch_results.jsonl")
    failures = store.merge_failures(args.output_root / "failures.jsonl")
    completed = store.rows()
    write_json(args.output_root / "resume_summary.json", {
        "planned_cells": len(planned), "completed_cells": len(completed),
        "unresolved_failed_cells": sum(row["status"] == "failed" for row in rows),
        "failure_attempts": len(failures), "skipped_existing_cells": skipped,
        "duplicate_cells": 0, "planned_cells_sha256": plan_hash,
        "run_identity_sha256": run_hash,
    })
    if rows and all(row.get("status") == "completed" for row in rows):
        write_json(args.output_root / "metrics.json", _metrics(rows, 2000, 20260826))
    if abort_oom is not None:
        raise ContractError(
            "GPU OOM was recorded atomically; confirmation stopped before further paid work"
        ) from abort_oom
    print(f"applied frozen selection to {len(rows)} {args.role} cells")
    return 0


_FULL_DUPLEX_REVIEW_FIELDS = (
    "natural_flow",
    "primary_response_scorable",
    "final_target_correct",
    "stale_state_error",
    "d1_binding_correct",
    "d2_binding_correct",
    "d3_binding_correct",
)


def _full_duplex_audio_policy(config: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    conversation = config.get("conversation")
    if not isinstance(conversation, Mapping):
        raise ContractError("config.conversation must be an object")
    if tuple(conversation.get("required_modes", ())) != REQUIRED_EXPERIMENTAL_STARTUP_MODES:
        raise ContractError("full-duplex config must freeze both required startup modes in order")
    if conversation.get("capture_from_stream_start") is not True:
        raise ContractError("full-duplex capture must begin at model frame zero")
    startup = conversation.get("startup")
    response = conversation.get("response")
    review = conversation.get("review")
    if not all(isinstance(value, Mapping) for value in (startup, response, review)):
        raise ContractError("conversation startup, response, and review policies are required")
    expected_startup = {
        "natural_max_ms": FROZEN_GREETING_MAX_FRAMES * FRAME_MS,
        "greeting_quiet_ms": FROZEN_GREETING_QUIET_FRAMES * FRAME_MS,
        "post_greeting_gap_ms": FROZEN_PREPARED_LEADIN_FRAMES * FRAME_MS,
        "require_greeting": True,
    }
    if any(startup.get(name) != value for name, value in expected_startup.items()):
        raise ContractError("conversation greeting/lead-in policy differs from the frozen runtime")
    if (
        response.get("post_user_max_ms") != RESPONSE_CAPTURE_FRAMES * FRAME_MS
        or response.get("trailing_text_quiet_ms") != TAIL_GUARD_FRAMES * FRAME_MS
        or response.get("tail_guard_ms") != TAIL_GUARD_FRAMES * FRAME_MS
    ):
        raise ContractError("conversation response and quiet horizons must remain 40 s / 2 s")
    expected_review = {
        "primary_window": "query_end_to_first_complete_assistant_turn",
        "save_full_stream": True,
        "save_primary_only_blind_clip": True,
        "opaque_arm_filenames": True,
    }
    if any(review.get(name) != value for name, value in expected_review.items()):
        raise ContractError("conversation review policy is not the frozen blind-review contract")
    policy = response.get("audio_activity")
    if not isinstance(policy, Mapping):
        raise ContractError("conversation audio activity policy is missing")
    expected_fields = set(FROZEN_AUDIO_ACTIVITY_POLICY) | {"policy_sha256"}
    if set(policy) != expected_fields:
        raise ContractError("conversation audio activity policy has missing or unknown fields")
    if any(policy.get(name) != value for name, value in FROZEN_AUDIO_ACTIVITY_POLICY.items()):
        raise ContractError("conversation audio activity policy differs from the frozen policy")
    body = {name: value for name, value in policy.items() if name != "policy_sha256"}
    if (
        sha256_value(body) != policy.get("policy_sha256")
        or policy.get("policy_sha256") != FROZEN_AUDIO_ACTIVITY_POLICY_SHA256
    ):
        raise ContractError("conversation audio activity policy hash mismatch")
    return dict(policy), TAIL_GUARD_FRAMES


def _full_duplex_source_identity(
    manifest: Path, rows: Sequence[Mapping[str, Any]], manifest_sha256: str,
) -> tuple[str, str | None]:
    """Return the full source-manifest hash behind a bounded canary subset."""
    sidecar = manifest.with_suffix(manifest.suffix + ".selection.json")
    if not sidecar.exists():
        return manifest_sha256, None
    payload = read_json(sidecar)
    if not isinstance(payload, Mapping):
        raise ContractError("canary selection sidecar must be an object")
    trial_ids = [str(row.get("trial_id", "")) for row in rows]
    if payload.get("trial_ids") != trial_ids or payload.get("trial_count") != len(rows):
        raise ContractError("canary selection sidecar does not bind the exact manifest rows")
    if payload.get("canary_manifest_sha256") != manifest_sha256:
        raise ContractError("canary selection sidecar manifest hash mismatch")
    source_sha = validate_sha256(
        str(payload.get("source_manifest_sha256", "")), "canary source manifest")
    return source_sha, sha256_file(sidecar)


def _full_duplex_selection(
    selection_path: Path | None, *, primary_intervention: str, config_sha256: str,
    identity_anchor: str,
) -> tuple[dict[str, Any], str, str | None]:
    if primary_intervention == "identity_noop":
        if selection_path is not None:
            raise ContractError("identity-noop flow canary must not consume a discovery selection")
        if not identity_anchor:
            raise ContractError("identity-noop flow canary requires a named semantic anchor")
        selection = {
            "schema_version": "1.0.0",
            "selection_kind": "pre_selection_identity_noop",
            "anchor": identity_anchor,
            "component": "identity_noop",
            "layer": None,
            "head": None,
        }
        return selection, sha256_value(selection), None
    if primary_intervention != "within_repair_erasure":
        raise ContractError(
            "primary intervention must be identity_noop or within_repair_erasure")
    if selection_path is None or not selection_path.is_file():
        raise ContractError("within-repair erasure requires an existing frozen selection")
    selection = read_json(selection_path)
    if not isinstance(selection, Mapping):
        raise ContractError("frozen selection must be an object")
    declared = validate_sha256(
        str(selection.get("selection_sha256", "")), "frozen selection")
    computed = sha256_value({
        name: value for name, value in selection.items() if name != "selection_sha256"
    })
    if computed != declared:
        raise ContractError("frozen selection content hash mismatch")
    if selection.get("config_sha256") != config_sha256:
        raise ContractError("frozen selection was created from a different config")
    if not isinstance(selection.get("anchor"), str) or not selection["anchor"]:
        raise ContractError("frozen selection has no semantic anchor")
    return dict(selection), declared, sha256_file(selection_path)


def _full_duplex_interventions(
    selection: Mapping[str, Any], *, frame: int, primary_intervention: str,
) -> tuple[tuple[str, int, int, int | None], ...]:
    if primary_intervention == "identity_noop":
        return ()
    component = str(selection.get("component", ""))
    layer_raw = selection.get("layer")
    head_raw = selection.get("head")
    if isinstance(layer_raw, bool) or not isinstance(layer_raw, int) or layer_raw < 0:
        raise ContractError("frozen full-duplex selection has an invalid layer")
    if head_raw is not None and (
        isinstance(head_raw, bool) or not isinstance(head_raw, int) or head_raw < 0
    ):
        raise ContractError("frozen full-duplex selection has an invalid head")
    head = None if head_raw is None else int(head_raw)
    if component == "kv":
        return (
            ("k_pre_rope", int(layer_raw), frame, head),
            ("v_pre_rope", int(layer_raw), frame, head),
        )
    mapped = {"k_only": "k_pre_rope", "v_only": "v_pre_rope"}.get(
        component, component)
    if mapped == "path":
        explicit = selection.get("full_duplex_interventions")
        if not isinstance(explicit, list) or not explicit:
            raise ContractError(
                "path selection requires explicit full_duplex_interventions; "
                "a residual zeroing cannot be relabeled as path patching")
        result: list[tuple[str, int, int, int | None]] = []
        for index, item in enumerate(explicit):
            if not isinstance(item, Mapping):
                raise ContractError(f"full_duplex_interventions[{index}] must be an object")
            site = str(item.get("site", ""))
            layer = item.get("layer")
            selected_head = item.get("head")
            if site not in REQUIRED_SITES or isinstance(layer, bool) or not isinstance(layer, int):
                raise ContractError(f"full_duplex_interventions[{index}] is invalid")
            if selected_head is not None and (
                isinstance(selected_head, bool) or not isinstance(selected_head, int)
            ):
                raise ContractError(f"full_duplex_interventions[{index}].head is invalid")
            result.append((site, layer, frame, selected_head))
        return tuple(result)
    if mapped not in REQUIRED_SITES:
        raise ContractError(f"unsupported full-duplex erasure component: {component}")
    return ((mapped, int(layer_raw), frame, head),)


def _atomic_pcm16_wav(path: Path, pcm: np.ndarray) -> None:
    array = np.asarray(pcm, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ContractError("blind-review PCM must be a finite non-empty mono timeline")
    clipped = np.clip(array, -1.0, 1.0)
    samples = np.rint(clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".wav", dir=path.parent)
    os.close(descriptor)
    try:
        with wave.open(temporary, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(samples.tobytes(order="C"))
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _synthetic_full_duplex_pair(
    contract: ConversationContract, *, startup_mode: str, seed: int, branch_frame: int,
) -> PairedGeneration:
    """Small deterministic orchestration fixture; never empirical evidence."""
    del seed
    startup_frames = (
        2 + FROZEN_GREETING_QUIET_FRAMES
        if startup_mode == REQUIRED_EXPERIMENTAL_STARTUP_MODES[0] else 0
    )
    total = startup_frames + contract.target_end_frame_count
    tokens = np.zeros((1, 9, total), dtype=np.int64)
    tokens[:, 0] = 3
    feedback = np.zeros_like(tokens)
    pieces = [""] * total
    if startup_mode == REQUIRED_EXPERIMENTAL_STARTUP_MODES[0]:
        tokens[0, 0, 0], pieces[0] = 5, " Hello"
        tokens[0, 0, 1], pieces[1] = 7, "."
    elif startup_mode == STARTUP_MODE_NATURAL:
        tokens[0, 0, 0], pieces[0] = 5, " Hello"
        tokens[0, 0, 1], pieces[1] = 7, "."
    response_start = startup_frames + contract.query_end_frame
    tokens[0, 0, response_start], pieces[response_start] = 5, " Seattle"
    tokens[0, 0, response_start + 1], pieces[response_start + 1] = 7, "."
    pcm = np.zeros(total * FRAME_SAMPLES, dtype=np.float32)
    if startup_frames:
        pcm[: 2 * FRAME_SAMPLES] = 0.02
    elif startup_mode == STARTUP_MODE_NATURAL:
        pcm[: 2 * FRAME_SAMPLES] = 0.02
    pcm[response_start * FRAME_SAMPLES : (response_start + 2) * FRAME_SAMPLES] = 0.02
    sequence = GeneratedSequence(
        tokens=tokens,
        feedback_tokens=feedback,
        text_token_ids=[int(value) for value in tokens[0, 0]],
        text_pieces=pieces,
        pcm=pcm,
        frame_count=total,
        conversation_frame_count=contract.target_end_frame_count,
        conversation_start_frame=startup_frames,
        frame_samples=FRAME_SAMPLES,
        pcm_sample_count=int(pcm.size),
    )
    absolute_branch = startup_frames + branch_frame
    prefix = tokens[..., :absolute_branch]
    prefix_feedback = feedback[..., :absolute_branch]
    return PairedGeneration(
        baseline=sequence,
        patched=GeneratedSequence(
            tokens=tokens.copy(), feedback_tokens=feedback.copy(),
            text_token_ids=list(sequence.text_token_ids), text_pieces=list(pieces),
            pcm=pcm.copy(), frame_count=total,
            conversation_frame_count=contract.target_end_frame_count,
            conversation_start_frame=startup_frames, frame_samples=FRAME_SAMPLES,
            pcm_sample_count=int(pcm.size),
        ),
        branch_frame=branch_frame,
        shared_prefix_frames=absolute_branch,
        shared_prefix_sha256=_array_sha256(prefix),
        shared_feedback_sha256=_array_sha256(prefix_feedback),
        first_feedback_divergence_frame=None,
        first_output_divergence_frame=None,
        pre_intervention_identical=True,
        startup_mode=startup_mode,
        startup_frame_count=startup_frames,
        handshake_terminal_frame=1 if startup_frames else None,
        handshake_terminal_piece="." if startup_frames else None,
        handshake_completion_signal=(
            "terminal_punctuation_plus_text_audio_quiet" if startup_frames else None),
        target_frame_count=contract.target_end_frame_count,
        lm_step_count=1 + absolute_branch + 2 * (contract.target_end_frame_count - branch_frame),
        handshake_probe_lm_step_count=(1 + startup_frames if startup_frames else 0),
        handshake_replay_identical=True if startup_frames else None,
        continuous_mimi_input_verified=True if startup_frames else None,
    )


def _full_duplex_sequence_diagnostics(
    sequence: Any, contract: ConversationContract, *, startup_mode: str,
    threshold_dbfs: float, threshold_source: str, quiet_frames: int,
    result: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    expected_full_frames = int(result.startup_frame_count) + contract.target_end_frame_count
    tokens = _numpy(sequence.tokens)
    feedback = _numpy(sequence.feedback_tokens)
    pcm = np.asarray(sequence.pcm, dtype=np.float32)
    if (
        sequence.frame_count != expected_full_frames
        or tokens.ndim != 3 or int(tokens.shape[-1]) != expected_full_frames
        or feedback.shape != tokens.shape
        or pcm.ndim != 1 or pcm.size != expected_full_frames * FRAME_SAMPLES
        or sequence.pcm_sample_count != pcm.size
        or sequence.conversation_frame_count != contract.target_end_frame_count
        or sequence.conversation_start_frame != int(result.startup_frame_count)
    ):
        raise ContractError("generated full stream does not have exact frame-zero coverage")
    start = int(result.startup_frame_count)
    end = start + contract.target_end_frame_count
    ids = [int(value) for value in sequence.text_token_ids[start:end]]
    pieces = list(sequence.text_pieces[start:end])
    conversation_pcm = np.asarray(sequence.conversation_pcm, dtype=np.float32)
    if len(ids) != contract.target_end_frame_count or len(pieces) != len(ids):
        raise ContractError("generated conversation text does not cover the frozen target horizon")
    text = diagnose_response_boundaries(contract, ids, pieces)
    audio = diagnose_audio_tail(
        conversation_pcm,
        sample_rate=SAMPLE_RATE,
        frame_samples=FRAME_SAMPLES,
        expected_frame_count=contract.target_end_frame_count,
        tail_guard_frames=contract.tail_guard_frames,
        threshold_dbfs=threshold_dbfs,
        threshold_source=threshold_source,
    )
    audio_levels = frame_rms_dbfs(conversation_pcm, frame_samples=FRAME_SAMPLES)
    activity = [
        bool(piece.strip()) or bool(audio_levels[index] >= threshold_dbfs)
        for index, piece in enumerate(pieces)
    ]
    primary = primary_response_window(
        activity, query_end_frame=contract.query_end_frame, quiet_frames=quiet_frames)
    cap_active = bool(text.cap_active or audio.cap_active)
    truncated = bool(primary.status == "unevaluable_truncated" or cap_active)
    no_response = primary.status == "unevaluable_no_response"
    response_complete = primary.status == "complete" and not cap_active
    evaluation_status = (
        "evaluable" if response_complete else
        "unevaluable_truncated" if truncated else
        "unevaluable_no_response" if no_response else
        "unevaluable_unknown"
    )
    full_levels = frame_rms_dbfs(pcm, frame_samples=FRAME_SAMPLES)
    greeting = None
    suppression = None
    if startup_mode == REQUIRED_EXPERIMENTAL_STARTUP_MODES[0]:
        startup_frames = int(result.startup_frame_count)
        terminal = result.handshake_terminal_frame
        lexical = any(
            any(character.isalnum() for character in piece)
            for piece in sequence.text_pieces[:startup_frames]
        )
        audio_active = bool(np.any(full_levels[:startup_frames] >= threshold_dbfs))
        quiet_start = startup_frames - FROZEN_GREETING_QUIET_FRAMES
        quiet_ok = bool(
            quiet_start >= 0
            and not any(piece.strip() for piece in sequence.text_pieces[quiet_start:startup_frames])
            and np.all(full_levels[quiet_start:startup_frames] < threshold_dbfs)
        )
        greeting = {
            "measured": bool(
                isinstance(terminal, int)
                and terminal >= 0
                and terminal + FROZEN_GREETING_QUIET_FRAMES < startup_frames
                and isinstance(result.handshake_terminal_piece, str)
                and bool(result.handshake_terminal_piece.strip())
                and result.handshake_completion_signal
                == "terminal_punctuation_plus_text_audio_quiet"
                and result.handshake_replay_identical is True
                and result.continuous_mimi_input_verified is True
                and lexical and audio_active and quiet_ok
            ),
            "startup_frame_count": startup_frames,
            "terminal_frame": terminal,
            "terminal_piece": result.handshake_terminal_piece,
            "completion_signal": result.handshake_completion_signal,
            "lexical_activity_observed": lexical,
            "audio_activity_observed": audio_active,
            "final_quiet_run_verified": quiet_ok,
            "replay_identical": result.handshake_replay_identical,
            "continuous_mimi_input_verified": result.continuous_mimi_input_verified,
        }
    elif startup_mode == REQUIRED_EXPERIMENTAL_STARTUP_MODES[1]:
        suppression = {
            "text_quiet_through_user_end": not any(
                piece.strip() for piece in pieces[:contract.user_end_frame]),
            "audio_quiet_through_user_end": bool(
                np.all(audio_levels[:contract.user_end_frame] < threshold_dbfs)),
        }
        suppression["verified"] = all(suppression.values())
    exact_coverage = bool(
        conversation_pcm.size == contract.target_end_frame_count * FRAME_SAMPLES
        and tokens[..., start:end].shape[-1] == contract.target_end_frame_count
    )
    return ({
        "exact_output_coverage": exact_coverage,
        "full_frame_count": expected_full_frames,
        "conversation_frame_count": contract.target_end_frame_count,
        "conversation_pcm_sample_count": int(conversation_pcm.size),
        "text_tail": asdict(text),
        "audio_tail": audio.to_dict(),
        "primary_response": primary.to_dict(),
        "combined_cap_active": cap_active,
        "combined_truncated": truncated,
        "combined_no_response": no_response,
        "response_complete": response_complete,
        "evaluation_status": evaluation_status,
        "greeting": greeting,
        "suppression": suppression,
        "text_token_ids": ids,
        "text_pieces": pieces,
    }, conversation_pcm)


def _full_duplex_record_hash(row: Mapping[str, Any]) -> str:
    return sha256_value({name: value for name, value in row.items() if name != "record_sha256"})


def _read_full_duplex_cell(path: Path, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    row = read_json(path)
    if row.get("status") != "completed" or row.get("cell_identity") != expected_identity:
        raise ContractError(f"completed conversation cell identity mismatch: {path.name}")
    if row.get("record_sha256") != _full_duplex_record_hash(row):
        raise ContractError(f"completed conversation cell record hash mismatch: {path.name}")
    return row


def _verify_full_duplex_record_artifacts(row: Mapping[str, Any], output_root: Path) -> None:
    root = output_root.resolve()
    arms = row.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        raise ContractError("completed conversation cell must contain exactly two arms")
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise ContractError("completed conversation arm must be an object")
        for prefix in ("full_audio", "primary_audio"):
            relative = require_relative_uri(str(arm.get(f"{prefix}_uri", "")))
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ContractError("blind audio artifact escapes the output root") from error
            if not path.is_file() or sha256_file(path) != validate_sha256(
                str(arm.get(f"{prefix}_sha256", "")), f"{prefix} artifact"):
                raise ContractError(f"completed blind audio artifact failed hash verification: {relative}")


def _portable_full_duplex_error(error: Exception, roots: Sequence[Path]) -> str:
    message = str(error)
    for index, root in enumerate(roots):
        message = message.replace(str(root.resolve()), f"<ROOT_{index}>")
    return message[:2000]


def _record_full_duplex_failure(
    failure_root: Path, *, cell_id: str, cell_identity: Mapping[str, Any],
    error: Exception, roots: Sequence[Path], synthetic: bool,
) -> None:
    failure_root.mkdir(parents=True, exist_ok=True)
    attempt = len(list(failure_root.glob(f"{cell_id}.*.json"))) + 1
    body = {
        "schema_version": "1.0.0",
        "status": "failed",
        "cell_id": cell_id,
        "cell_identity": dict(cell_identity),
        "attempt": attempt,
        "failure_type": type(error).__name__,
        "failure_message": _portable_full_duplex_error(error, roots),
        "synthetic": synthetic,
    }
    failure_id = sha256_value(body)
    write_json(failure_root / f"{cell_id}.{failure_id}.json", {**body, "failure_id": failure_id})


def _read_full_duplex_failures(
    failure_root: Path, expected_identities: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(failure_root.glob("*.json")):
        row = read_json(path)
        cell_id = str(row.get("cell_id", ""))
        failure_id = str(row.get("failure_id", ""))
        body = {name: value for name, value in row.items() if name != "failure_id"}
        if (
            row.get("status") != "failed"
            or cell_id not in expected_identities
            or row.get("cell_identity") != expected_identities[cell_id]
            or failure_id != sha256_value(body)
            or path.name != f"{cell_id}.{failure_id}.json"
        ):
            raise ContractError(f"failed conversation attempt has invalid identity: {path.name}")
        rows.append(row)
    return rows


def _full_duplex_review_rows(
    records: Sequence[Mapping[str, Any]], *, run_identity_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validations: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: str(row["cell_id"])):
        public_arms: list[dict[str, Any]] = []
        for arm in sorted(record["arms"], key=lambda row: str(row["blind_label"])):
            review_id = sha256_value({
                "run_identity_sha256": run_identity_sha256,
                "cell_id": record["cell_id"],
                "blind_label": arm["blind_label"],
            })
            public_arm = {
                "review_id": review_id,
                "blind_label": arm["blind_label"],
                "full_audio_uri": arm["full_audio_uri"],
                "full_audio_sha256": arm["full_audio_sha256"],
                "primary_audio_uri": arm["primary_audio_uri"],
                "primary_audio_sha256": arm["primary_audio_sha256"],
                "technical_status": arm["technical_status"],
                "exact_output_coverage": arm["diagnostics"]["exact_output_coverage"],
                "response_complete": arm["diagnostics"]["response_complete"],
                "combined_cap_active": arm["diagnostics"]["combined_cap_active"],
                "combined_truncated": arm["diagnostics"]["combined_truncated"],
                "combined_no_response": arm["diagnostics"]["combined_no_response"],
            }
            public_arms.append(public_arm)
            for slot in (1, 2):
                immutable = {
                    "schema_version": "1.0.0",
                    "run_identity_sha256": run_identity_sha256,
                    "review_id": review_id,
                    "submission_id": sha256_value({"review_id": review_id, "reviewer_slot": slot}),
                    "pair_id": record["pair_id"],
                    "blind_label": arm["blind_label"],
                    "startup_mode": record["startup_mode"],
                    "full_audio_uri": arm["full_audio_uri"],
                    "full_audio_sha256": arm["full_audio_sha256"],
                    "primary_audio_uri": arm["primary_audio_uri"],
                    "primary_audio_sha256": arm["primary_audio_sha256"],
                    "reviewer_slot": slot,
                }
                templates.append({
                    **immutable,
                    "reviewer_id": None,
                    **{name: None for name in _FULL_DUPLEX_REVIEW_FIELDS},
                })
        validations.append({
            "schema_version": "1.0.0",
            "cell_id": record["cell_id"],
            "pair_id": record["pair_id"],
            "startup_mode": record["startup_mode"],
            "seed": record["seed"],
            "trial_pseudonym": hashlib.sha256(
                f"{run_identity_sha256}:{record['trial_id']}".encode()).hexdigest()[:24],
            "analysis_scope": record["analysis_scope"],
            "status": "awaiting_double_blind_human_review",
            "arms": public_arms,
            "synthetic": record["synthetic"],
        })
    return validations, templates


def _write_or_verify_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    expected = [dict(row) for row in rows]
    if path.exists() and read_jsonl(path) != expected:
        raise ContractError(f"resume-derived artifact differs from immutable cells: {path.name}")
    if not path.exists():
        write_jsonl(path, expected)


def _resolve_full_duplex_reviews(
    templates: Sequence[Mapping[str, Any]], reviews_path: Path,
    adjudications_path: Path | None, *, output_root: Path,
) -> tuple[dict[str, dict[str, bool]], dict[str, Any]]:
    reviews = read_jsonl(reviews_path)
    template_by_submission = {str(row["submission_id"]): row for row in templates}
    review_by_submission = {str(row.get("submission_id", "")): row for row in reviews}
    if (
        len(review_by_submission) != len(reviews)
        or set(review_by_submission) != set(template_by_submission)
    ):
        missing = len(set(template_by_submission) - set(review_by_submission))
        extra = len(set(review_by_submission) - set(template_by_submission))
        write_json(output_root / "conversation_review_status.json", {
            "schema_version": "1.0.0", "passed": False,
            "status": "incomplete_or_duplicate_reviews", "missing": missing, "extra": extra,
        })
        raise ContractError(
            f"double-blind review coverage is incomplete (missing={missing}, extra={extra})")
    immutable_names = tuple(
        name for name in templates[0]
        if name not in {"reviewer_id", *_FULL_DUPLEX_REVIEW_FIELDS}
    ) if templates else ()
    by_review_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for submission_id, template in template_by_submission.items():
        row = review_by_submission[submission_id]
        if any(row.get(name) != template.get(name) for name in immutable_names):
            raise ContractError(f"review submission altered blinded identity: {submission_id}")
        reviewer = row.get("reviewer_id")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ContractError(f"review submission has no reviewer identity: {submission_id}")
        for name in _FULL_DUPLEX_REVIEW_FIELDS:
            if not isinstance(row.get(name), bool):
                raise ContractError(f"review judgment {name} must be boolean: {submission_id}")
        by_review_id[str(row["review_id"])].append(row)
    conflicts: list[dict[str, Any]] = []
    resolved: dict[str, dict[str, bool]] = {}
    raw_agreements = 0
    raw_comparisons = 0
    reviewer_ids: dict[str, tuple[str, str]] = {}
    for review_id, rows in by_review_id.items():
        if len(rows) != 2 or {row["reviewer_slot"] for row in rows} != {1, 2}:
            raise ContractError(f"review {review_id} does not have exact reviewer slots 1 and 2")
        rows.sort(key=lambda row: int(row["reviewer_slot"]))
        ids = (str(rows[0]["reviewer_id"]).strip(), str(rows[1]["reviewer_id"]).strip())
        if ids[0] == ids[1]:
            raise ContractError(f"review {review_id} was not assessed by independent reviewers")
        reviewer_ids[review_id] = ids
        disagreements = []
        consensus: dict[str, bool] = {}
        for name in _FULL_DUPLEX_REVIEW_FIELDS:
            left, right = bool(rows[0][name]), bool(rows[1][name])
            raw_comparisons += 1
            raw_agreements += int(left == right)
            if left == right:
                consensus[name] = left
            else:
                disagreements.append(name)
        if disagreements:
            exemplar = rows[0]
            conflicts.append({
                "schema_version": "1.0.0",
                "run_identity_sha256": exemplar["run_identity_sha256"],
                "review_id": review_id,
                "adjudication_id": sha256_value({"review_id": review_id, "kind": "adjudication"}),
                "pair_id": exemplar["pair_id"],
                "blind_label": exemplar["blind_label"],
                "startup_mode": exemplar["startup_mode"],
                "conflicting_fields": disagreements,
                "adjudicator_id": None,
                **{name: None for name in _FULL_DUPLEX_REVIEW_FIELDS},
            })
        else:
            resolved[review_id] = consensus
    _write_or_verify_jsonl(output_root / "adjudication_template.jsonl", conflicts)
    adjudications = read_jsonl(adjudications_path) if adjudications_path is not None else []
    adjudication_by_review = {str(row.get("review_id", "")): row for row in adjudications}
    expected_conflicts = {str(row["review_id"]): row for row in conflicts}
    if len(adjudication_by_review) != len(adjudications) or set(adjudication_by_review) != set(expected_conflicts):
        if conflicts:
            raise ContractError(
                "reviewer disagreements require one independent adjudication per conflicted clip")
        if adjudications:
            raise ContractError("adjudication file contains entries without reviewer disagreements")
    for review_id, template in expected_conflicts.items():
        row = adjudication_by_review[review_id]
        immutable = (
            "schema_version", "run_identity_sha256", "review_id", "adjudication_id",
            "pair_id", "blind_label", "startup_mode", "conflicting_fields",
        )
        if any(row.get(name) != template.get(name) for name in immutable):
            raise ContractError(f"adjudication altered blinded identity: {review_id}")
        adjudicator = row.get("adjudicator_id")
        if (
            not isinstance(adjudicator, str) or not adjudicator.strip()
            or adjudicator.strip() in reviewer_ids[review_id]
        ):
            raise ContractError(f"adjudicator for {review_id} is missing or not independent")
        judgments: dict[str, bool] = {}
        for name in _FULL_DUPLEX_REVIEW_FIELDS:
            if not isinstance(row.get(name), bool):
                raise ContractError(f"adjudicated judgment {name} must be boolean: {review_id}")
            judgments[name] = bool(row[name])
        resolved[review_id] = judgments
    summary = {
        "schema_version": "1.0.0",
        "passed": len(resolved) == len(by_review_id),
        "reviewed_clip_count": len(by_review_id),
        "independent_submission_count": len(reviews),
        "conflicted_clip_count": len(conflicts),
        "adjudicated_clip_count": len(adjudications),
        "raw_field_agreement": raw_agreements / raw_comparisons if raw_comparisons else 0.0,
        "reviews_sha256": sha256_file(reviews_path),
        "adjudications_sha256": (
            sha256_file(adjudications_path) if adjudications_path is not None else None),
    }
    write_json(output_root / "conversation_review_status.json", summary)
    return resolved, summary


def run_full_duplex(argv: Sequence[str]) -> int:
    parser = _parser("Run the bounded, blind, frozen full-duplex behavioral bridge.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoded-manifest", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--input-artifact-root", type=Path, required=True)
    parser.add_argument("--primary-intervention", required=True,
                        choices=("within_repair_erasure", "identity_noop"))
    parser.add_argument("--donor-arms", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--trial-ids")
    parser.add_argument("--limit-trials", type=int, default=1)
    parser.add_argument("--max-paired-cells", type=int, default=16)
    parser.add_argument("--identity-branch-anchor", default="query_end")
    parser.add_argument("--include-natural-diagnostic", action="store_true")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.limit_trials < 1 or args.limit_trials > 8:
        raise ContractError("--limit-trials must keep the bounded conversation canary in [1, 8]")
    if args.max_paired_cells < 1:
        raise ContractError("--max-paired-cells must be positive")
    if args.donor_arms not in {"conditional_on_feedback_divergence", "none"}:
        raise ContractError("full-duplex donor arms are conditional diagnostics, never the primary arm")
    if args.primary_intervention == "within_repair_erasure" and args.donor_arms == "none":
        # Donor interventions are intentionally absent, but retain the frozen
        # CLI label that documents their conditional-only inferential status.
        raise ContractError("within-repair bridge requires donor-arms=conditional_on_feedback_divergence")
    config = read_json(args.config)
    _frozen_model_config(config)
    policy, quiet_frames = _full_duplex_audio_policy(config)
    config_sha = sha256_file(args.config)
    manifest_rows = read_jsonl(args.manifest)
    encoded_rows, encoded_by_id = _encoded_rows(args.encoded_manifest)
    if not manifest_rows:
        raise ContractError("full-duplex canary manifest is empty")
    manifest_ids = [str(row.get("trial_id", "")) for row in manifest_rows]
    if any(not trial_id for trial_id in manifest_ids) or len(set(manifest_ids)) != len(manifest_ids):
        raise ContractError("full-duplex manifest has missing or duplicate trial IDs")
    if set(encoded_by_id) != set(manifest_ids):
        raise ContractError("encoded full-duplex manifest does not exactly cover the canary manifest")
    manifest_sha = sha256_file(args.manifest)
    encoded_sha = sha256_file(args.encoded_manifest)
    anchors_sha = sha256_file(args.anchors)
    source_manifest_sha, canary_sidecar_sha = _full_duplex_source_identity(
        args.manifest, manifest_rows, manifest_sha)
    anchor_rows_value, anchors_by_key = _anchor_lookup(args.anchors)
    if not anchor_rows_value:
        raise ContractError("full-duplex anchor map is empty")
    code_commit = _git_commit()
    model = _frozen_model_config(config)
    selection, selection_sha, selection_file_sha = _full_duplex_selection(
        args.selection,
        primary_intervention=args.primary_intervention,
        config_sha256=config_sha,
        identity_anchor=args.identity_branch_anchor,
    )
    anchor_name = str(selection["anchor"])
    seed_values = _ints(args.seeds)
    if (
        not seed_values or len(set(seed_values)) != len(seed_values)
        or any(seed < 0 for seed in seed_values)
    ):
        raise ContractError("full-duplex seeds must be unique non-negative integers")
    requested_ids = set(_csv(args.trial_ids)) if args.trial_ids else None
    repairs = [
        row for row in manifest_rows
        if not str(row.get("condition", "")).startswith("clean")
        and (requested_ids is None or str(row["trial_id"]) in requested_ids)
    ]
    repairs.sort(key=lambda row: str(row["trial_id"]))
    if requested_ids is not None and requested_ids != {str(row["trial_id"]) for row in repairs}:
        raise ContractError("--trial-ids contains a missing or non-repair canary trial")
    if requested_ids is not None and len(repairs) > args.limit_trials:
        raise ContractError("--trial-ids exceeds the explicit --limit-trials bound")
    repairs = repairs[:args.limit_trials]
    if not repairs:
        raise ContractError("full-duplex canary selected no repair trials")
    modes = list(REQUIRED_EXPERIMENTAL_STARTUP_MODES)
    if args.include_natural_diagnostic:
        modes.append(STARTUP_MODE_NATURAL)

    # Validate every source WAV and encoded row before a checkpoint can be
    # constructed.  The exact canary subset remains tied to its full source
    # manifest through the optional selection sidecar.
    source_by_id = {str(row["trial_id"]): row for row in manifest_rows}
    wav_by_id: dict[str, Path] = {}
    for source in manifest_rows:
        trial_id = str(source["trial_id"])
        frames = _encoding_frame_contract(source)
        wav, source_audio_sha = _validate_source_wav(
            args.input_artifact_root, source, frames["user_frame_count"])
        wav_by_id[trial_id] = wav
        encoded_row = encoded_by_id[trial_id]
        expected_static = {
            "source_manifest_sha256": manifest_sha,
            "source_row_sha256": sha256_value(source),
            "source_audio_sha256": source_audio_sha,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "code_commit": code_commit,
            "synthetic": bool(args.synthetic),
        }
        if any(encoded_row.get(name) != value for name, value in expected_static.items()):
            raise ContractError(f"{trial_id}: encoded provenance differs from this exact canary run")

    jobs: list[dict[str, Any]] = []
    selected_trial_hashes: dict[str, str] = {}
    silence_code_hashes: set[str] = set()
    for trial in repairs:
        trial_id = str(trial["trial_id"])
        try:
            contract = ConversationContract.from_manifest_row(trial)
        except ConversationContractError as error:
            raise ContractError(f"{trial_id}: {error}") from error
        if contract.response_capture_frames != RESPONSE_CAPTURE_FRAMES:
            raise ContractError(f"{trial_id}: response window is not exactly 500 frames")
        if contract.user_start_frame != FROZEN_PREPARED_LEADIN_FRAMES:
            raise ContractError(
                f"{trial_id}: common-handshake mode requires the frozen 480 ms prepared lead-in")
        encoded_row = encoded_by_id[trial_id]
        arrays = {
            name: _load_encoded_array(
                args.encoded_manifest, encoded_row, name, require_current_contract=True)
            for name in ("conversation_codes", "assistant_silence_codes")
        }
        if arrays["conversation_codes"].shape[-1] != contract.target_end_frame_count:
            raise ContractError(f"{trial_id}: encoded conversation target differs from contract")
        silence_code_hashes.add(_array_sha256(arrays["assistant_silence_codes"]))
        anchor = anchors_by_key.get((trial_id, anchor_name))
        if anchor is None:
            raise ContractError(f"{trial_id}: frozen full-duplex anchor {anchor_name!r} is absent")
        frame = int(anchor["frame"])
        if not 0 <= frame < contract.target_end_frame_count:
            raise ContractError(f"{trial_id}: frozen full-duplex anchor is outside the target")
        interventions = _full_duplex_interventions(
            selection, frame=frame, primary_intervention=args.primary_intervention)
        selected_trial_hashes[trial_id] = sha256_value(trial)
        for mode in modes:
            for seed in seed_values:
                jobs.append({
                    "trial": trial,
                    "encoded_row": encoded_row,
                    "contract": contract,
                    "arrays": arrays,
                    "anchor": anchor,
                    "frame": frame,
                    "interventions": interventions,
                    "startup_mode": mode,
                    "seed": seed,
                })
    if len(jobs) > args.max_paired_cells:
        raise ContractError(
            f"bounded conversation grid has {len(jobs)} paired cells, exceeding "
            f"--max-paired-cells={args.max_paired_cells}; raise it explicitly only after cost review")

    run_identity_body = {
        "schema_version": "1.0.0",
        "operation": "bounded_full_duplex_conversation_canary",
        "harness_version": HARNESS_VERSION,
        "code_commit": code_commit,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_dtype": model.get("dtype"),
        "config_sha256": config_sha,
        "source_manifest_sha256": source_manifest_sha,
        "canary_manifest_sha256": manifest_sha,
        "canary_selection_sidecar_sha256": canary_sidecar_sha,
        "encoded_manifest_sha256": encoded_sha,
        "anchor_map_sha256": anchors_sha,
        "selection_sha256": selection_sha,
        "selection_file_sha256": selection_file_sha,
        "primary_intervention": args.primary_intervention,
        "donor_arms_policy": args.donor_arms,
        "startup_modes": modes,
        "seeds": seed_values,
        "max_paired_cells": args.max_paired_cells,
        "selected_trial_sha256": selected_trial_hashes,
        "audio_activity_policy_sha256": policy["policy_sha256"],
        "synthetic": bool(args.synthetic),
    }
    run_identity_sha = sha256_value(run_identity_body)
    run_identity = {**run_identity_body, "run_identity_sha256": run_identity_sha}
    identity_path = args.output_root / "run_identity.json"
    if (
        args.output_root.exists()
        and not identity_path.exists()
        and any(args.output_root.iterdir())
    ):
        raise ContractError("full-duplex output root is non-empty but has no run identity")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if identity_path.exists():
        if not args.resume:
            raise ContractError("full-duplex output already exists; use --resume")
        if read_json(identity_path) != run_identity:
            raise ContractError("full-duplex resume identity differs from the existing run")
    else:
        write_json(identity_path, run_identity)
    reserved_output_frames = sum(
        2 * (
            int(job["contract"].target_end_frame_count)
            + (FROZEN_GREETING_MAX_FRAMES
               if job["startup_mode"] == REQUIRED_EXPERIMENTAL_STARTUP_MODES[0] else 0)
        )
        for job in jobs
    )
    planned_workload = {
        "schema_version": "1.0.0",
        "run_identity_sha256": run_identity_sha,
        "paired_cell_count": len(jobs),
        "arm_generation_count": 2 * len(jobs),
        "reserved_output_frame_count": reserved_output_frames,
        "reserved_output_audio_hours": reserved_output_frames * FRAME_MS / 1000 / 3600,
        # Reserve both full and primary WAVs at the full-stream size; the real
        # primary clips are shorter, but compression/early stop is never used
        # to justify a smaller pre-paid volume.
        "reserved_pcm16_wav_payload_bytes": (
            reserved_output_frames * FRAME_SAMPLES * 2 * 2),
        "common_handshake_reserved_frames_per_arm": FROZEN_GREETING_MAX_FRAMES,
        "max_paired_cells": args.max_paired_cells,
    }
    workload_path = args.output_root / "planned_workload.json"
    if workload_path.exists() and read_json(workload_path) != planned_workload:
        raise ContractError("full-duplex planned workload changed during resume")
    if not workload_path.exists():
        write_json(workload_path, planned_workload)
    private_root = args.output_root / "private_cells"
    failure_root = args.output_root / "private_failures"
    private_root.mkdir(parents=True, exist_ok=True)
    unexpected_private_map = (
        any(private_root.glob("*.json"))
        and not (args.output_root / "private_blind_map.json").is_file()
    )
    if unexpected_private_map:
        raise ContractError("completed blind cells exist without their private HMAC map")
    blind_store = BlindAssignmentStore(
        args.output_root, run_identity_sha256=run_identity_sha)

    for job in jobs:
        trial = job["trial"]
        trial_id = str(trial["trial_id"])
        cell_identity = {
            "run_identity_sha256": run_identity_sha,
            "trial_id": trial_id,
            "source_row_sha256": sha256_value(trial),
            "encoded_row_sha256": sha256_value(job["encoded_row"]),
            "startup_mode": job["startup_mode"],
            "seed": job["seed"],
            "anchor": anchor_name,
            "anchor_frame": job["frame"],
            "interventions_sha256": sha256_value(job["interventions"]),
        }
        job["cell_identity"] = cell_identity
        job["cell_id"] = sha256_value(cell_identity)
        job["cell_path"] = private_root / f"{job['cell_id']}.json"
    expected_cell_ids = {str(job["cell_id"]) for job in jobs}
    expected_cell_identities = {
        str(job["cell_id"]): job["cell_identity"] for job in jobs}
    observed_cell_ids = {path.stem for path in private_root.glob("*.json")}
    if not observed_cell_ids <= expected_cell_ids:
        raise ContractError("full-duplex output contains cells outside the current immutable grid")

    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for job in jobs:
        if job["cell_path"].exists():
            record = _read_full_duplex_cell(job["cell_path"], job["cell_identity"])
            _verify_full_duplex_record_artifacts(record, args.output_root)
            completed[job["cell_id"]] = record
        else:
            pending.append(job)
    if completed and not args.resume:
        raise ContractError("completed full-duplex cells require --resume")

    calibration_path = args.output_root / "forced_silence_calibration.json"
    expected_calibration_identity = sha256_value({
        "run_identity_sha256": run_identity_sha,
        "assistant_silence_codes_sha256": sorted(silence_code_hashes),
        "policy_sha256": policy["policy_sha256"],
    })
    calibration = None
    if calibration_path.exists():
        calibration = read_json(calibration_path)
        calibration_body = {
            name: value for name, value in calibration.items() if name != "record_sha256"
        }
        if (
            calibration.get("calibration_identity_sha256") != expected_calibration_identity
            or calibration.get("record_sha256") != sha256_value(calibration_body)
        ):
            raise ContractError("forced-silence calibration failed resume identity verification")

    backend = None
    if calibration is None or pending:
        if not args.synthetic:
            backend = MoshiBackend(
                model_repo=MODEL_REPO,
                model_revision=MODEL_REVISION,
                dtype=str(model.get("dtype", "bfloat16")),
                use_sampling=True,
            )
    if calibration is None:
        decoded_levels: list[float] = []
        measured_hashes: list[str] = []
        seen_silence: set[str] = set()
        for job in jobs:
            silence_codes = job["arrays"]["assistant_silence_codes"]
            codes_sha = _array_sha256(silence_codes)
            if codes_sha in seen_silence:
                continue
            seen_silence.add(codes_sha)
            if args.synthetic:
                pcm = np.zeros(silence_codes.shape[-1] * FRAME_SAMPLES, dtype=np.float32)
            else:
                assert backend is not None
                if hasattr(backend, "decode_assistant_silence"):
                    pcm = np.asarray(
                        backend.decode_assistant_silence(silence_codes), dtype=np.float32)
                else:
                    torch = backend.torch
                    audio_codes = torch.as_tensor(
                        silence_codes, device=backend.device, dtype=torch.long)
                    text = torch.full(
                        (1, 1, audio_codes.shape[-1]),
                        int(backend.lm_gen.lm_model.text_padding_token_id),
                        device=backend.device,
                        dtype=torch.long,
                    )
                    pcm = backend._decode_tokens(
                        torch.cat([text, audio_codes], dim=1),
                        expected_frames=int(audio_codes.shape[-1]),
                    )
            levels = frame_rms_dbfs(pcm, frame_samples=FRAME_SAMPLES)
            decoded_levels.append(float(np.max(levels)))
            measured_hashes.append(hashlib.sha256(np.ascontiguousarray(pcm).tobytes()).hexdigest())
        forced_silence_max = max(decoded_levels)
        body = {
            "schema_version": "1.0.0",
            "analysis_status": (
                "synthetic_local_validation" if args.synthetic else "real_checkpoint_measurement"),
            "calibration_identity_sha256": expected_calibration_identity,
            "run_identity_sha256": run_identity_sha,
            "audio_activity_policy_sha256": policy["policy_sha256"],
            "assistant_silence_codes_sha256": sorted(silence_code_hashes),
            "decoded_pcm_sha256": sorted(measured_hashes),
            "forced_silence_decode_max_dbfs": forced_silence_max,
            "threshold_dbfs": float(policy["threshold_dbfs"]),
            "passed": forced_silence_max < float(policy["threshold_dbfs"]),
            "synthetic": bool(args.synthetic),
        }
        calibration = {**body, "record_sha256": sha256_value(body)}
        write_json(calibration_path, calibration)
    if calibration.get("passed") is not True:
        raise ContractError("forced-silence decode is not below the frozen audio threshold")

    for job in pending:
        trial = job["trial"]
        contract = job["contract"]
        trial_id = str(trial["trial_id"])
        assignment = blind_store.assign(
            cell_key=job["cell_identity"], arms=("baseline", "patched"))
        try:
            if args.synthetic:
                result = _synthetic_full_duplex_pair(
                    contract,
                    startup_mode=str(job["startup_mode"]),
                    seed=int(job["seed"]),
                    branch_frame=int(job["frame"]),
                )
            else:
                assert backend is not None
                raw_pcm = backend._read_pcm(wav_by_id[trial_id])
                result = backend.generate_paired_conversation(
                    job["arrays"]["conversation_codes"],
                    assistant_silence_codes=job["arrays"]["assistant_silence_codes"],
                    conversation_pcm=(
                        raw_pcm
                        if job["startup_mode"] == REQUIRED_EXPERIMENTAL_STARTUP_MODES[0]
                        else None
                    ),
                    seed=int(job["seed"]),
                    branch_frame=int(job["frame"]),
                    intervention=(job["interventions"] or None),
                    startup_mode=str(job["startup_mode"]),
                    target_frame_count=contract.target_end_frame_count,
                    user_start_frame=contract.user_start_frame,
                    query_end_frame=contract.query_end_frame,
                    user_end_frame=contract.user_end_frame,
                    handshake_max_frames=FROZEN_GREETING_MAX_FRAMES,
                    handshake_quiet_frames=FROZEN_GREETING_QUIET_FRAMES,
                    prepared_leadin_frames=FROZEN_PREPARED_LEADIN_FRAMES,
                    handshake_silence_threshold_dbfs=float(policy["threshold_dbfs"]),
                )
            absolute_branch = int(result.startup_frame_count) + int(job["frame"])
            if (
                result.pre_intervention_identical is not True
                or int(result.shared_prefix_frames) != absolute_branch
                or not np.array_equal(
                    _numpy(result.baseline.tokens)[..., :absolute_branch],
                    _numpy(result.patched.tokens)[..., :absolute_branch],
                )
                or not np.array_equal(
                    _numpy(result.baseline.feedback_tokens)[..., :absolute_branch],
                    _numpy(result.patched.feedback_tokens)[..., :absolute_branch],
                )
            ):
                raise ContractError("paired arms do not share the exact pre-intervention state/RNG prefix")
            if args.primary_intervention == "identity_noop" and (
                not np.array_equal(_numpy(result.baseline.tokens), _numpy(result.patched.tokens))
                or not np.array_equal(
                    _numpy(result.baseline.feedback_tokens),
                    _numpy(result.patched.feedback_tokens),
                )
            ):
                raise ContractError("identity/no-op conversation branches diverged")
            arm_rows: list[dict[str, Any]] = []
            for arm_name, sequence in (
                ("baseline", result.baseline), ("patched", result.patched)):
                diagnostics, conversation_pcm = _full_duplex_sequence_diagnostics(
                    sequence,
                    contract,
                    startup_mode=str(job["startup_mode"]),
                    threshold_dbfs=float(policy["threshold_dbfs"]),
                    threshold_source=str(policy["policy_sha256"]),
                    quiet_frames=quiet_frames,
                    result=result,
                )
                blind_stem = assignment.arm_to_audio_stem[arm_name]
                full_path = args.output_root / "audio" / f"{blind_stem}.full.wav"
                primary_path = args.output_root / "audio" / f"{blind_stem}.primary.wav"
                primary = diagnostics["primary_response"]
                clip_start = int(primary["query_end_frame"])
                clip_end = int(primary["response_end_frame_exclusive"] or primary["capture_end_frame"])
                primary_pcm = conversation_pcm[
                    clip_start * FRAME_SAMPLES : clip_end * FRAME_SAMPLES]
                _atomic_pcm16_wav(full_path, np.asarray(sequence.pcm, dtype=np.float32))
                _atomic_pcm16_wav(primary_path, primary_pcm)
                arm_rows.append({
                    "arm": arm_name,
                    "blind_label": assignment.arm_to_label[arm_name],
                    "full_audio_uri": full_path.relative_to(args.output_root).as_posix(),
                    "full_audio_sha256": sha256_file(full_path),
                    "primary_audio_uri": primary_path.relative_to(args.output_root).as_posix(),
                    "primary_audio_sha256": sha256_file(primary_path),
                    "primary_clip_start_frame": clip_start,
                    "primary_clip_end_frame_exclusive": clip_end,
                    "technical_status": diagnostics["evaluation_status"],
                    "diagnostics": diagnostics,
                })
            body = {
                "schema_version": "1.0.0",
                "status": "completed",
                "cell_id": job["cell_id"],
                "cell_identity": job["cell_identity"],
                "pair_id": assignment.pair_id,
                "trial_id": trial_id,
                "scenario_id": trial.get("scenario_id"),
                "condition": trial.get("condition"),
                "startup_mode": job["startup_mode"],
                "seed": job["seed"],
                "analysis_scope": (
                    "diagnostic_only_known_greeting_confound"
                    if job["startup_mode"] == STARTUP_MODE_NATURAL
                    else "required_conversation_canary"),
                "primary_intervention": args.primary_intervention,
                "selection_sha256": selection_sha,
                "anchor": anchor_name,
                "anchor_frame": job["frame"],
                "pre_intervention_identical": result.pre_intervention_identical,
                "shared_prefix_frames": result.shared_prefix_frames,
                "shared_prefix_sha256": result.shared_prefix_sha256,
                "shared_feedback_sha256": result.shared_feedback_sha256,
                "first_feedback_divergence_frame": result.first_feedback_divergence_frame,
                "first_output_divergence_frame": result.first_output_divergence_frame,
                "lm_step_count": result.lm_step_count,
                "handshake_probe_lm_step_count": result.handshake_probe_lm_step_count,
                "arms": arm_rows,
                "provenance": {
                    "code_commit": code_commit,
                    "harness_version": HARNESS_VERSION,
                    "model_repo": MODEL_REPO,
                    "model_revision": MODEL_REVISION,
                    "config_sha256": config_sha,
                    "source_manifest_sha256": source_manifest_sha,
                    "canary_manifest_sha256": manifest_sha,
                    "encoded_manifest_sha256": encoded_sha,
                    "anchor_map_sha256": anchors_sha,
                    "selection_file_sha256": selection_file_sha,
                    "source_audio_sha256": trial["audio_sha256"],
                    "encoded_artifact_identity_sha256": job["encoded_row"].get(
                        "artifact_identity_sha256"),
                },
                "synthetic": bool(args.synthetic),
            }
            record = {**body, "record_sha256": sha256_value(body)}
            write_json(job["cell_path"], record)
            completed[job["cell_id"]] = record
        except Exception as error:
            _record_full_duplex_failure(
                failure_root,
                cell_id=str(job["cell_id"]),
                cell_identity=job["cell_identity"],
                error=error,
                roots=(args.output_root, args.input_artifact_root),
                synthetic=bool(args.synthetic),
            )
            write_jsonl(
                args.output_root / "failures.jsonl",
                _read_full_duplex_failures(failure_root, expected_cell_identities),
            )
            if backend is not None and hasattr(backend, "torch"):
                torch = backend.torch
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise

    records = [
        _read_full_duplex_cell(job["cell_path"], job["cell_identity"]) for job in jobs
    ]
    for record in records:
        _verify_full_duplex_record_artifacts(record, args.output_root)
    failures = _read_full_duplex_failures(failure_root, expected_cell_identities)
    write_jsonl(args.output_root / "failures.jsonl", failures)
    validations, templates = _full_duplex_review_rows(
        records, run_identity_sha256=run_identity_sha)
    _write_or_verify_jsonl(args.output_root / "validation.jsonl", validations)
    _write_or_verify_jsonl(args.output_root / "blind_review_template.jsonl", templates)
    write_json(args.output_root / "resume_summary.json", {
        "schema_version": "1.0.0",
        "run_identity_sha256": run_identity_sha,
        "expected_cells": len(jobs),
        "completed_cells": len(records),
        "unresolved_cells": 0,
        "preserved_failed_attempts": len(failures),
        "duplicate_cells": 0,
        "backend_constructed": backend is not None,
    })
    if args.reviews is None:
        print(
            f"wrote {len(records)} paired conversation cells and {len(templates)} review slots; "
            "conversation_canary.json remains absent until double review/adjudication")
        return 0
    resolved, review_summary = _resolve_full_duplex_reviews(
        templates, args.reviews, args.adjudications, output_root=args.output_root)

    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_mode[str(record["startup_mode"])].append(record)
    per_mode: dict[str, dict[str, int]] = {}
    arm_annotation_totals: dict[str, Counter[str]] = {
        "baseline": Counter(), "patched": Counter()}
    for mode in REQUIRED_EXPERIMENTAL_STARTUP_MODES:
        mode_records = by_mode.get(mode, [])
        measurement = {
            "trial_count": len(mode_records),
            "truncated_count": 0,
            "cap_active_count": 0,
            "exact_output_coverage_count": 0,
            "response_complete_count": 0,
            "text_tail_checked_count": 0,
            "audio_tail_checked_count": 0,
            "human_flow_review_pass_count": 0,
        }
        for record in mode_records:
            arms = record["arms"]
            measurement["truncated_count"] += int(any(
                arm["diagnostics"]["combined_truncated"] for arm in arms))
            measurement["cap_active_count"] += int(any(
                arm["diagnostics"]["combined_cap_active"] for arm in arms))
            measurement["exact_output_coverage_count"] += int(all(
                arm["diagnostics"]["exact_output_coverage"] for arm in arms))
            measurement["response_complete_count"] += int(all(
                arm["diagnostics"]["response_complete"] for arm in arms))
            measurement["text_tail_checked_count"] += int(all(
                isinstance(arm["diagnostics"].get("text_tail"), Mapping) for arm in arms))
            measurement["audio_tail_checked_count"] += int(all(
                isinstance(arm["diagnostics"].get("audio_tail"), Mapping) for arm in arms))
            validation = next(
                row for row in validations if row["cell_id"] == record["cell_id"])
            review_id_by_label = {
                str(arm["blind_label"]): str(arm["review_id"])
                for arm in validation["arms"]
            }
            cell_judgments = [
                resolved[review_id_by_label[str(arm["blind_label"])]] for arm in arms
            ]
            measurement["human_flow_review_pass_count"] += int(all(
                row["natural_flow"] and row["primary_response_scorable"]
                for row in cell_judgments))
            for arm, judgments in zip(arms, cell_judgments, strict=True):
                totals = arm_annotation_totals[str(arm["arm"])]
                totals["clips"] += 1
                for name in _FULL_DUPLEX_REVIEW_FIELDS:
                    totals[name] += int(judgments[name])
        per_mode[mode] = measurement
    aggregate_names = (
        "trial_count", "truncated_count", "cap_active_count",
        "exact_output_coverage_count", "response_complete_count",
        "text_tail_checked_count", "audio_tail_checked_count",
        "human_flow_review_pass_count",
    )
    aggregate = {
        ("required_mode_trial_count" if name == "trial_count" else name):
        sum(per_mode[mode][name] for mode in REQUIRED_EXPERIMENTAL_STARTUP_MODES)
        for name in aggregate_names
    }
    common_rows = by_mode.get(REQUIRED_EXPERIMENTAL_STARTUP_MODES[0], [])
    suppressed_rows = by_mode.get(REQUIRED_EXPERIMENTAL_STARTUP_MODES[1], [])
    common_greetings = all(
        arm["diagnostics"].get("greeting", {}).get("measured") is True
        for record in common_rows for arm in record["arms"]
    ) and bool(common_rows)
    suppression_verified = all(
        arm["diagnostics"].get("suppression", {}).get("verified") is True
        for record in suppressed_rows for arm in record["arms"]
    ) and bool(suppressed_rows)
    minimum_trials = int(config.get("gates", {}).get(
        "conversation_canary_min_trials_per_mode", 0))
    checks = {
        "initial_greeting_measured": common_greetings,
        "turn_taking_reviewed": len(resolved) == len(templates) // 2,
        "human_flow_review_passed": all(
            per_mode[mode]["human_flow_review_pass_count"] == per_mode[mode]["trial_count"]
            for mode in REQUIRED_EXPERIMENTAL_STARTUP_MODES),
        "response_capture_complete": all(
            per_mode[mode]["response_complete_count"] == per_mode[mode]["trial_count"]
            for mode in REQUIRED_EXPERIMENTAL_STARTUP_MODES),
        "output_coverage_complete": all(
            per_mode[mode]["exact_output_coverage_count"] == per_mode[mode]["trial_count"]
            for mode in REQUIRED_EXPERIMENTAL_STARTUP_MODES),
        "text_tail_checked": all(
            per_mode[mode]["text_tail_checked_count"] == per_mode[mode]["trial_count"]
            for mode in REQUIRED_EXPERIMENTAL_STARTUP_MODES),
        "audio_tail_checked": all(
            per_mode[mode]["audio_tail_checked_count"] == per_mode[mode]["trial_count"]
            for mode in REQUIRED_EXPERIMENTAL_STARTUP_MODES),
        "no_tail_truncation": all(
            per_mode[mode]["truncated_count"] == 0
            and per_mode[mode]["cap_active_count"] == 0
            for mode in REQUIRED_EXPERIMENTAL_STARTUP_MODES),
        "greeting_suppression_verified": suppression_verified,
        "minimum_trials_per_mode": minimum_trials >= 4 and all(
            per_mode[mode]["trial_count"] >= minimum_trials
            for mode in REQUIRED_EXPERIMENTAL_STARTUP_MODES),
        "paired_prefix_and_rng_identity": all(
            record.get("pre_intervention_identical") is True
            and isinstance(record.get("shared_prefix_sha256"), str)
            and len(record["shared_prefix_sha256"]) == 64
            and isinstance(record.get("shared_feedback_sha256"), str)
            and len(record["shared_feedback_sha256"]) == 64
            for record in records),
        "forced_silence_calibrated": calibration.get("passed") is True,
    }
    validation_passed = all(checks.values()) and review_summary["passed"]
    report = {
        "schema_version": "1.0.0",
        "analysis_status": (
            "synthetic_local_validation" if args.synthetic
            else "bounded_real_checkpoint_conversation_canary"),
        # Synthetic execution may validate orchestration, but it can never be
        # accepted as paid-run readiness evidence.
        "passed": validation_passed and not args.synthetic,
        "synthetic_validation_passed": validation_passed if args.synthetic else None,
        "checks": checks,
        "code_commit": code_commit,
        "harness_version": HARNESS_VERSION,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "config_sha256": config_sha,
        "source_manifest_sha256": source_manifest_sha,
        "canary_manifest_sha256": manifest_sha,
        "encoded_manifest_sha256": encoded_sha,
        "anchor_map_sha256": anchors_sha,
        "selection_sha256": selection_sha,
        "selection_file_sha256": selection_file_sha,
        "run_identity_sha256": run_identity_sha,
        "primary_intervention": args.primary_intervention,
        "canary_purpose": (
            "prepaid_conversation_flow_identity_noop"
            if args.primary_intervention == "identity_noop"
            else "post_selection_within_repair_erasure_bridge"),
        "per_mode": per_mode,
        "measurements": aggregate,
        "tail_detection": {
            "text_quiet_frames": quiet_frames,
            "tail_guard_frames": TAIL_GUARD_FRAMES,
            "audio_activity_policy_version": policy["version"],
            "audio_activity_detector": policy["detector"],
            "audio_activity_frame_samples": policy["frame_samples"],
            "audio_activity_threshold_dbfs": float(policy["threshold_dbfs"]),
            "audio_activity_calibration": policy["calibration"],
            "audio_activity_policy_sha256": policy["policy_sha256"],
            "forced_silence_decode_max_dbfs": calibration[
                "forced_silence_decode_max_dbfs"],
            "forced_silence_calibration_sha256": sha256_file(calibration_path),
        },
        "review": review_summary,
        "behavioral_annotations_by_arm": {
            arm: dict(values) for arm, values in arm_annotation_totals.items()},
        "natural_model_start": {
            "status": "diagnostic_only_known_greeting_confound",
            "included": STARTUP_MODE_NATURAL in by_mode,
            "trial_count": len(by_mode.get(STARTUP_MODE_NATURAL, [])),
        },
        "preserved_failed_attempt_count": len(failures),
        "synthetic": bool(args.synthetic),
    }
    status_path = args.output_root / "conversation_canary_status.json"
    write_json(status_path, report)
    if args.synthetic:
        report_path = args.output_root / "synthetic_conversation_canary.json"
        if report_path.exists() and read_json(report_path) != report:
            raise ContractError("existing synthetic canary report differs from finalized evidence")
        if not report_path.exists():
            write_json(report_path, report)
        if not validation_passed:
            print("synthetic NO_GO: local conversation orchestration checks did not pass")
            return 3
        print(f"wrote non-authorizing synthetic conversation validation -> {report_path}")
        return 0
    if not report["passed"]:
        print("NO_GO: conversation canary did not pass every frozen technical/review gate")
        return 3
    report_path = args.output_root / "conversation_canary.json"
    if report_path.exists() and read_json(report_path) != report:
        raise ContractError("existing conversation canary report differs from finalized evidence")
    if not report_path.exists():
        write_json(report_path, report)
    print(f"finalized readiness-compatible conversation canary -> {report_path}")
    return 0


def _metrics(rows: Sequence[Mapping[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed" and math.isfinite(float(row.get("delta_M", math.nan)))]
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for row in completed:
        by_scenario[str(row.get("scenario_id", row.get("recipient_trial_id")))].append(float(row["delta_M"]))
    cluster_values = [float(np.mean(values)) for values in by_scenario.values()]
    if len(cluster_values) < 2:
        cluster_values = [float(row["delta_M"]) for row in completed]
    if len(cluster_values) < 2:
        return {"analysis_status": "insufficient_data", "n_cells": len(completed), "passed": False}
    estimate, low, high = bootstrap_mean_ci(cluster_values, replicates, seed)
    # Sign-flip approximation is deterministic and two-sided.
    rng = np.random.default_rng(seed + 1)
    array = np.asarray(cluster_values)
    null = np.mean(rng.choice([-1.0, 1.0], size=(max(2000, replicates), len(array))) * array, axis=1)
    p = float((np.sum(np.abs(null) >= abs(estimate)) + 1) / (len(null) + 1))
    return {"analysis_status": "synthetic_local_validation", "n_cells": len(completed),
            "n_scenario_clusters": len(cluster_values), "estimate": estimate, "ci95": [low, high],
            "raw_p_two_sided": p, "holm_p": holm_adjust([p])[0], "passed": low > 0}


def freeze_analysis_plan(argv: Sequence[str]) -> int:
    parser = _parser(
        "Freeze a self-hashed empirical analysis plan from pristine planned-cell grids."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path, action="append", required=True,
        help="Frozen discovery selection; repeat for independently frozen sites.",
    )
    parser.add_argument(
        "--planned-cells", type=Path, action="append", required=True,
        help="Pristine planned_cells.jsonl; repeat to preregister multiple arms/scans.",
    )
    parser.add_argument("--output-spec", type=Path, required=True)
    parser.add_argument("--output-expected-cells", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.synthetic:
        raise ContractError(
            "frozen analysis plans are empirical preregistrations; synthetic runs use the fixture analyzer"
        )
    spec, expected = freeze_analysis_artifacts(
        run_root=args.run_root,
        config_path=args.config,
        template_path=args.template,
        selection_paths=args.selection,
        planned_cell_paths=args.planned_cells,
        output_spec=args.output_spec,
        output_expected_cells=args.output_expected_cells,
    )
    print(json.dumps({
        "analysis_spec": str(args.output_spec),
        "analysis_spec_sha256": spec["analysis_spec_sha256"],
        "expected_cells": str(args.output_expected_cells),
        "expected_cells_sha256": expected["expected_cells_sha256"],
        "expected_cell_count": expected["cell_count"],
        "source_plan_count": len(expected["source_plans"]),
    }, sort_keys=True))
    return 0


def analyze(argv: Sequence[str]) -> int:
    parser = _parser("Analyze all completed cells with scenario-cluster uncertainty.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--analysis-spec", type=Path)
    parser.add_argument(
        "--expected-cells", type=Path,
        help="Optional explicit path; must equal the URI frozen in --analysis-spec.",
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int,
        help="Synthetic fixture override; empirical value is frozen in the analysis spec.",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int,
        help="Synthetic fixture override; empirical value is frozen in the analysis spec.",
    )
    args = parser.parse_args(argv)
    discovered_result_files = sorted(args.run_root.rglob("*patch_results.jsonl"))
    if not discovered_result_files:
        raise ContractError("analysis found no patch result files")
    discovered_rows = [row for path in discovered_result_files for row in read_jsonl(path)]
    if not discovered_rows:
        raise ContractError("analysis found no patch result rows")
    synthetic_values = {row.get("synthetic") for row in discovered_rows}
    if not synthetic_values <= {True, False} or len(synthetic_values) != 1:
        raise ContractError("analysis cannot mix synthetic and empirical patch results")
    all_synthetic = synthetic_values == {True}
    reports = args.run_root / "reports"
    if not all_synthetic:
        if args.analysis_spec is None or args.config is None:
            raise ContractError(
                "empirical analysis requires --config and --analysis-spec"
            )
        analysis_spec, expected, result_files = load_frozen_analysis_inputs(
            run_root=args.run_root,
            analysis_spec_path=args.analysis_spec,
            expected_cells_path=args.expected_cells,
        )
        if analysis_spec.get("config_sha256") != sha256_file(args.config):
            raise ContractError("frozen analysis spec targets a different config")
        frozen_replicates = int(analysis_spec["bootstrap_replicates"])
        frozen_seed = int(analysis_spec["bootstrap_seed"])
        if args.bootstrap_replicates is not None and args.bootstrap_replicates != frozen_replicates:
            raise ContractError("--bootstrap-replicates differs from the frozen analysis spec")
        if args.bootstrap_seed is not None and args.bootstrap_seed != frozen_seed:
            raise ContractError("--bootstrap-seed differs from the frozen analysis spec")
        rows = [row for path in result_files for row in read_jsonl(path)]
        if any(row.get("synthetic") is not False for row in rows):
            raise ContractError("frozen empirical analysis received a synthetic result row")
        metrics = analyze_frozen_contrasts(
            rows,
            analysis_spec["hypotheses"],
            expected_cells=expected["cells"],
            bootstrap_replicates=frozen_replicates,
            seed=frozen_seed,
            alpha=float(analysis_spec.get("alpha", 0.05)),
        )
        metrics["analysis_spec_sha256"] = sha256_file(args.analysis_spec)
        expected_path = args.run_root / str(analysis_spec["expected_cells_uri"])
        try:
            analysis_spec_uri = args.analysis_spec.resolve().relative_to(
                args.run_root.resolve()).as_posix()
        except ValueError as error:
            raise ContractError("--analysis-spec must be inside --run-root") from error
        metrics["analysis_spec_uri"] = require_relative_uri(analysis_spec_uri)
        metrics["expected_cells_uri"] = require_relative_uri(
            str(analysis_spec["expected_cells_uri"]))
        metrics["analysis_spec_value_sha256"] = analysis_spec["analysis_spec_sha256"]
        metrics["expected_cells_sha256"] = sha256_file(expected_path)
        metrics["expected_cells_value_sha256"] = expected["expected_cells_sha256"]
        metrics["source_plans_sha256"] = expected["source_plans_sha256"]
        metrics["selection_bindings"] = analysis_spec["selection_bindings"]
    else:
        if args.analysis_spec is not None or args.expected_cells is not None or args.config is not None:
            raise ContractError(
                "synthetic fixture analysis cannot consume an empirical frozen analysis plan"
            )
        # The local analytic fixture intentionally remains small and has no
        # inferential meaning.  Missing-cell/SESOI behavior is exercised by the
        # frozen-protocol unit suite; empirical paths can never use this branch.
        rows = discovered_rows
        result_files = discovered_result_files
        metrics = _metrics(
            rows,
            args.bootstrap_replicates if args.bootstrap_replicates is not None else 10000,
            args.bootstrap_seed if args.bootstrap_seed is not None else 20260826,
        )
        metrics["analysis_status"] = "synthetic_local_validation"
        metrics["passed"] = bool(metrics.get("passed"))
    metrics.update({
        "schema_version": "1.1.0",
        "harness_version": HARNESS_VERSION,
        "provenance": {
            path.relative_to(args.run_root).as_posix(): sha256_file(path)
            for path in result_files
        },
        "limitations": (
            ["Synthetic/local validation is not evidence about Boston, Seattle, or Moshiko."]
            if all_synthetic
            else ["Interpretation remains conditional on every frozen readiness and review gate."]
        ),
    })
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "mechanistic_discovery_summary.json", metrics)
    write_csv(reports / "tables/all_scenario_effects.csv", rows)
    if isinstance(metrics.get("registry"), list):
        registry = [
            {
                "hypothesis": row.get("hypothesis_id"),
                "family": row.get("family"),
                "direction": row.get("direction"),
                "statistic": row.get("estimate"),
                "n": row.get("n_scenario_clusters"),
                "raw_p": row.get("raw_p_two_sided"),
                "adjusted_p": row.get("holm_p"),
                "ci_type": "scenario-cluster bootstrap",
                "sesoi": row.get("sesoi"),
                "passed": row.get("passed"),
            }
            for row in metrics["registry"]
        ]
    else:
        registry = [{"hypothesis": "synthetic target-minus-stale fixture effect", "family": "fixture",
                     "direction": "positive", "statistic": metrics.get("estimate"),
                     "n": metrics.get("n_scenario_clusters"),
                     "raw_p": metrics.get("raw_p_two_sided"), "adjusted_p": metrics.get("holm_p"),
                     "ci_type": "scenario-cluster bootstrap", "sesoi": "not_inferential",
                     "passed": metrics.get("passed")}]
    write_csv(reports / "tables/multiplicity_registry.csv", registry)
    print(f"analyzed {len(rows)} cells -> {reports}")
    return 0


def _svg(title: str, value: float) -> str:
    magnitude = min(280, max(0, int(abs(value) * 100)))
    x = 360 if value >= 0 else 360 - magnitude
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="180" role="img" '
            f'aria-label="{title}"><rect width="720" height="180" fill="white"/>'
            f'<text x="30" y="40" font-family="sans-serif" font-size="20">{title}</text>'
            f'<line x1="360" y1="60" x2="360" y2="145" stroke="#555"/>'
            f'<rect x="{x}" y="80" width="{magnitude}" height="35" fill="#3264a8"/>'
            f'<text x="30" y="108" font-family="monospace" font-size="16">estimate {value:.4f}</text></svg>\n')


def render_report(argv: Sequence[str]) -> int:
    parser = _parser("Render a self-contained Markdown/SVG mechanistic report.")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = args.run_root / "reports"
    summary_path = reports / "mechanistic_discovery_summary.json"
    if not summary_path.exists():
        raise ContractError("analyze_mechanistic_results.py must run before rendering")
    summary = read_json(summary_path)
    status = summary.get("analysis_status", "unknown")
    text = f"# Mechanistic stale-binding results\n\n## Status\n\n`{status}`\n\n"
    if status == "synthetic_local_validation":
        text += "This report is a **synthetic/local harness validation**, not empirical evidence about Moshiko.\n\n"
    else:
        text += "This report contains model-run outputs; causal wording is allowed only after every frozen gate passes.\n\n"
    text += f"- Completed cells: {summary.get('n_cells', 0)}\n- Scenario clusters: {summary.get('n_scenario_clusters', 0)}\n"
    effect_label = "Mean synthetic fixture effect" if status == "synthetic_local_validation" else "Frozen contrast estimate"
    text += f"- {effect_label}: {summary.get('estimate', 'see multiplicity registry')}\n"
    text += f"- 95% cluster bootstrap CI: {summary.get('ci95', 'see multiplicity registry')}\n"
    text += f"- Summary SHA-256: `{sha256_file(summary_path)}`\n\n"
    text += "## Required next gates\n\n- Run the exact pinned checkpoint on RunPod.\n- Pass open-loop and baseline capability gates.\n- Obtain independent alignment and double-listen review before formal confirmation.\n"
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    _path = reports / "MECHANISTIC_RESULTS.md"
    _path.write_text(text, encoding="utf-8")
    titles = ["Baseline margin", "Probe layer-time", "Residual patch", "Frozen confirmation",
              "Temporal propagation", "Controls and no-ops"]
    for index, title in enumerate(titles, 1):
        (reports / "figures" / f"{index:02d}_{title.lower().replace(' ', '_')}.svg").write_text(
            _svg(title, float(summary.get("estimate", 0.0) or 0.0)), encoding="utf-8")
    print(f"rendered report -> {_path}")
    return 0


def verify_run(argv: Sequence[str]) -> int:
    parser = _parser("Verify provenance, rows, reports, and artifact hashes fail-closed.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--allow-local-synthetic", action="store_true")
    args = parser.parse_args(argv)
    required = ["reports/MECHANISTIC_RESULTS.md", "reports/mechanistic_discovery_summary.json",
                "reports/tables/all_scenario_effects.csv", "reports/tables/multiplicity_registry.csv"]
    missing = [path for path in required if not (args.run_root / path).is_file()]
    if missing:
        raise ContractError(f"run is missing required report artifacts: {missing}")
    report_text = (args.run_root / required[0]).read_text(encoding="utf-8")
    summary = read_json(args.run_root / required[1])
    if not isinstance(summary, Mapping):
        raise ContractError("mechanistic analysis summary must be an object")
    patch_cell_count, patch_rows_are_synthetic = verify_patch_artifacts(args.run_root)
    verify_analysis_provenance(args.run_root, summary)
    if summary.get("analysis_status") == "synthetic_local_validation":
        if summary.get("n_cells") != patch_cell_count:
            raise ContractError("verified patch-cell count differs from analysis summary")
    else:
        scoped_count = sum(
            len(read_jsonl(args.run_root / require_relative_uri(str(uri))))
            for uri in summary.get("provenance", {})
        )
        if summary.get("n_cells") != scoped_count or scoped_count > patch_cell_count:
            raise ContractError(
                "verified preregistered patch-cell count differs from empirical analysis summary"
            )
    if summary.get("analysis_status") == "synthetic_local_validation" and not args.allow_local_synthetic:
        raise ContractError("synthetic run requires --allow-local-synthetic and cannot satisfy empirical gates")
    if summary.get("analysis_status") == "synthetic_local_validation" and "not empirical evidence" not in report_text:
        raise ContractError("synthetic report lacks the mandatory evidence disclaimer")
    if (summary.get("analysis_status") == "synthetic_local_validation") != patch_rows_are_synthetic:
        raise ContractError("analysis status disagrees with patch-result synthetic provenance")
    if summary.get("analysis_status") != "synthetic_local_validation":
        empirical_required = [
            "preflight/model_contract/run_identity.json",
            "preflight/model_contract/environment.json",
            "preflight/model_contract/model_contract.json",
            "preflight/model_contract/readouts.bound.json",
            "encoded_user_manifest.jsonl", "anchor_map.jsonl",
            "frame_trace.jsonl", "gpu_canary/open_loop_validation.json",
            "baseline_readout.jsonl",
            "mechanistic_frozen_selection.json",
        ]
        empirical_missing = [path for path in empirical_required if not (args.run_root / path).is_file()]
        if empirical_missing:
            raise ContractError(f"empirical run is missing gate evidence: {empirical_missing}")
        open_loop = read_json(args.run_root / "gpu_canary/open_loop_validation.json")
        if not open_loop.get("passed"):
            raise ContractError("open-loop gate did not pass")
    artifact_count, created = verify_or_create_artifact_manifest(args.run_root)
    action = "created and verified" if created else "verified existing"
    print(
        f"{action} immutable hash manifest for {artifact_count} artifacts "
        f"and {patch_cell_count} patch cells under {args.run_root}"
    )
    return 0


def package_results(argv: Sequence[str]) -> int:
    parser = _parser("Create separately verified public and private result archives.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact_manifest_path = args.run_root / "artifact_sha256.json"
    if not artifact_manifest_path.is_file():
        raise ContractError("verify_mechanistic_run.py must establish artifact_sha256.json first")
    verify_artifact_manifest(args.run_root, read_json(artifact_manifest_path))
    hashes = package_tree(args.run_root, args.public_output, args.private_output)
    verify_archive(args.public_output, public=True)
    verify_archive(args.private_output, public=False)
    checksum_manifest = package_checksum_manifest(args.public_output, args.private_output)
    if (
        hashes.get("public_sha256") != checksum_manifest["archives"]["public"]["sha256"]
        or hashes.get("private_sha256") != checksum_manifest["archives"]["private"]["sha256"]
    ):
        raise ContractError("archive SHA-256 changed during post-package verification")
    checksum_path = args.public_output.with_suffix(args.public_output.suffix + ".sha256.json")
    write_json(checksum_path, checksum_manifest)
    verify_package_checksums(
        read_json(checksum_path),
        public_path=args.public_output,
        private_path=args.private_output,
    )
    print(json.dumps(checksum_manifest, sort_keys=True))
    return 0


COMMANDS = {
    "build_mech_manifest.py": build_mech_manifest,
    "build_anchor_map.py": build_anchor_map,
    "build_multivalue_controls.py": build_multivalue_controls,
    "simulate_multivalue_power.py": simulate_multivalue_power,
    "validate_multivalue_controls.py": validate_multivalue_controls,
    "encode_user_audio.py": encode_user_audio,
    "validate_mechanistic_contract.py": validate_mechanistic_contract,
    "validate_open_loop.py": validate_open_loop,
    "capture_activations.py": capture_activations,
    "score_readouts.py": score_readouts,
    "fit_probes.py": fit_probes,
    "scan_residual_patches.py": lambda argv: _scan(argv, "residual"),
    "scan_component_patches.py": lambda argv: _scan(argv, "component"),
    "scan_kv_patches.py": lambda argv: _scan(argv, "kv"),
    "run_path_patches.py": lambda argv: _scan(argv, "path"),
    "freeze_mechanistic_selection.py": freeze_mechanistic_selection,
    "run_confirmatory_patches.py": run_confirmatory,
    "run_full_duplex_validation.py": run_full_duplex,
    "freeze_analysis_plan.py": freeze_analysis_plan,
    "analyze_mechanistic_results.py": analyze,
    "render_mechanistic_report.py": render_report,
    "verify_mechanistic_run.py": verify_run,
    "package_mechanistic_results.py": package_results,
}


def main_for(program: str, argv: Sequence[str] | None = None) -> int:
    try:
        command = COMMANDS[Path(program).name]
        return command(list(sys.argv[1:] if argv is None else argv))
    except (ContractError, ReadinessError) as error:
        print(f"CONTRACT ERROR: {error}", file=sys.stderr)
        return 2
