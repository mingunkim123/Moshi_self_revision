"""Strict validation for v2 evaluation inputs and completed response evidence."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any, Mapping
import wave

try:  # Support direct script execution and package-style imports.
    from .common import CONDITIONS, sha256_file, sha256_value
    from .timing import validate_timing
except ImportError:  # pragma: no cover
    from common import CONDITIONS, sha256_file, sha256_value
    from timing import validate_timing


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
INPUT_FIELDS = {
    "prepared_stimulus_id",
    "preparation_hash",
    "uri",
    "sha256",
    "duration_ms",
    "sample_rate",
    "channels",
    "sample_width_bytes",
    "timeline",
    "mimi_frame_samples",
}
CAPTURE_FIELDS = {
    "condition",
    "timebase",
    "stream_origin_ms",
    "prepared_timing",
    "prepared_timing_sha256",
    "primary_window_start_ms",
    "utterance_end_ms",
    "response_capture_ms",
    "requested_target_end_ms",
    "target_end_sample_count",
    "target_end_frame_count",
    "actual_target_end_ms",
}
MATRIX_FIELDS = {
    "production_matrix",
    "accepted_audio_count",
    "generation_seeds",
    "eval_trial_count",
    "accepted_audio_set_sha256",
    "prepared_stimulus_set_sha256",
    "prepared_manifest_content_sha256",
}
EXECUTION_FIELDS = {
    "runner_version",
    "runner_source_sha256",
    "input_sample_rate",
    "mimi_frame_samples",
    "prefix_silence_ms",
    "response_capture_ms",
    "reset_model_stream_between_trials",
    "reset_rng_for_each_trial_seed",
    "required_model_type",
    "required_max_lm_delay",
}
COMPLETED_RESPONSE_FIELDS = {
    "status",
    "transcript",
    "transcript_sha256",
    "audio_path",
    "audio_sha256",
    "audio_duration_ms",
    "audio_sample_rate",
    "audio_channels",
    "audio_sample_width_bytes",
    "elapsed_seconds",
    "generation_seed",
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
    "stream_reset",
    "rng_reset",
    "backend",
    "runner_source_sha256",
    "effective_generation_config_sha256",
    "stream_events_sha256",
    "evidence_sha256",
}
BACKEND_FIELDS = {
    "name",
    "version",
    "model_repo",
    "resolved_revision",
    "snapshot_revision",
    "code_commit",
    "model_type",
    "mimi_sample_rate",
    "frame_size",
    "max_lm_delay",
    "effective_generation_config",
    "effective_generation_config_sha256",
}
STREAM_EVENT_FIELDS = {"frame_index", "time_ms", "token_id", "piece"}


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    return number > 0 if positive else number >= 0


def validate_relative_content_uri(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty artifact-root-relative path")
    if "\\" in value or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        raise ValueError(f"{label} must be a portable relative content path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must not be absolute or contain traversal segments")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"{label} must already be normalized as a POSIX path")
    return normalized


def resolve_content_uri(value: Any, root: Path, *, label: str) -> Path:
    uri = validate_relative_content_uri(value, label=label)
    resolved_root = root.expanduser().resolve()
    resolved = (resolved_root / Path(*PurePosixPath(uri).parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:  # Defensive after lexical traversal validation.
        raise ValueError(f"{label} escapes its declared artifact root") from error
    return resolved


def validate_input_stimulus(value: Any, *, label: str = "input_stimulus") -> None:
    if not isinstance(value, Mapping) or set(value) != INPUT_FIELDS:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} must contain exactly {sorted(INPUT_FIELDS)}; got {actual}")
    if not isinstance(value["prepared_stimulus_id"], str) or not value["prepared_stimulus_id"]:
        raise ValueError(f"{label}.prepared_stimulus_id must be a non-empty string")
    validate_relative_content_uri(value["uri"], label=f"{label}.uri")
    for field in ("preparation_hash", "sha256"):
        if not isinstance(value[field], str) or not HASH_RE.fullmatch(value[field]):
            raise ValueError(f"{label}.{field} must be a lowercase SHA-256")
    if not _finite_number(value["duration_ms"], positive=True):
        raise ValueError(f"{label}.duration_ms must be finite and positive")
    for field in ("sample_rate", "channels", "sample_width_bytes", "mimi_frame_samples"):
        field_value = value[field]
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value <= 0
        ):
            raise ValueError(f"{label}.{field} must be a positive integer")
    if value["channels"] != 1 or value["sample_width_bytes"] != 2:
        raise ValueError(f"{label} must describe mono PCM16 audio")
    if value["timeline"] != "prepared_stream_relative":
        raise ValueError(f"{label}.timeline must be prepared_stream_relative")


def validate_capture_contract(
    value: Any,
    input_stimulus: Mapping[str, Any],
    *,
    label: str = "capture_contract",
) -> None:
    if not isinstance(value, Mapping) or set(value) != CAPTURE_FIELDS:
        raise ValueError(f"{label} must contain exactly {sorted(CAPTURE_FIELDS)}")
    condition = value["condition"]
    if condition not in CONDITIONS:
        raise ValueError(f"{label}.condition is invalid")
    if value["timebase"] != "prepared_stream_relative" or value["stream_origin_ms"] != 0:
        raise ValueError(f"{label} must use prepared_stream_relative origin 0")
    timing = value["prepared_timing"]
    if not isinstance(timing, Mapping):
        raise ValueError(f"{label}.prepared_timing must be an object")
    if value["prepared_timing_sha256"] != sha256_value(timing):
        raise ValueError(f"{label}.prepared_timing_sha256 mismatch")
    timing_errors = validate_timing(
        str(condition), dict(timing), float(input_stimulus["duration_ms"])
    )
    if timing_errors:
        raise ValueError(f"{label}.prepared_timing: " + "; ".join(timing_errors))
    primary_start = timing.get("closing_prompt_offset_ms")
    utterance_end = timing.get("utterance_end_ms")
    if value["primary_window_start_ms"] != primary_start:
        raise ValueError(f"{label}.primary_window_start_ms must equal closing_prompt_offset_ms")
    if value["utterance_end_ms"] != utterance_end:
        raise ValueError(f"{label}.utterance_end_ms must equal prepared timing")
    if not _finite_number(value["response_capture_ms"], positive=True):
        raise ValueError(f"{label}.response_capture_ms must be finite and positive")
    requested = float(utterance_end) + float(value["response_capture_ms"])
    if abs(float(value["requested_target_end_ms"]) - requested) > 1e-6:
        raise ValueError(f"{label}.requested_target_end_ms is inconsistent")
    for field in ("target_end_sample_count", "target_end_frame_count"):
        field_value = value[field]
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"{label}.{field} must be a positive integer")
    frame_samples = int(input_stimulus["mimi_frame_samples"])
    if value["target_end_sample_count"] % frame_samples:
        raise ValueError(f"{label}.target_end_sample_count must be frame aligned")
    if value["target_end_frame_count"] * frame_samples != value["target_end_sample_count"]:
        raise ValueError(f"{label} target frame/sample counts disagree")
    prepared_samples = round(
        float(input_stimulus["duration_ms"])
        * int(input_stimulus["sample_rate"])
        / 1000.0
    )
    if value["target_end_sample_count"] < prepared_samples:
        raise ValueError(f"{label} target ends before the prepared stimulus file")
    minimum_samples = requested * float(input_stimulus["sample_rate"]) / 1000.0
    if value["target_end_sample_count"] + 1e-9 < minimum_samples:
        raise ValueError(f"{label} target ends before the requested capture window")
    actual_end = (
        float(value["target_end_sample_count"])
        * 1000.0
        / float(input_stimulus["sample_rate"])
    )
    if abs(float(value["actual_target_end_ms"]) - actual_end) > 1e-6:
        raise ValueError(f"{label}.actual_target_end_ms is inconsistent")
    if actual_end - requested >= frame_samples * 1000.0 / float(input_stimulus["sample_rate"]) + 1e-6:
        raise ValueError(f"{label} target was not rounded to the next Mimi frame")


def validate_matrix_contract(value: Any, *, label: str = "matrix_contract") -> None:
    if not isinstance(value, Mapping) or set(value) != MATRIX_FIELDS:
        raise ValueError(f"{label} must contain exactly {sorted(MATRIX_FIELDS)}")
    if not isinstance(value["production_matrix"], bool):
        raise ValueError(f"{label}.production_matrix must be a boolean")
    for field in ("accepted_audio_count", "eval_trial_count"):
        field_value = value[field]
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"{label}.{field} must be a positive integer")
    seeds = value["generation_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError(f"{label}.generation_seeds must be ordered unique non-negative integers")
    if value["eval_trial_count"] != value["accepted_audio_count"] * len(seeds):
        raise ValueError(f"{label} trial count is not accepted_audio_count x seeds")
    if value["production_matrix"] is True and (
        value["accepted_audio_count"] != 600
        or len(seeds) != 5
        or value["eval_trial_count"] != 3000
    ):
        raise ValueError(f"{label} production matrix must be exactly 600 x 5 = 3000")
    for field in (
        "accepted_audio_set_sha256",
        "prepared_stimulus_set_sha256",
        "prepared_manifest_content_sha256",
    ):
        if not isinstance(value[field], str) or not HASH_RE.fullmatch(value[field]):
            raise ValueError(f"{label}.{field} must be a lowercase SHA-256")


def validate_execution_contract(value: Any, *, label: str = "execution_contract") -> None:
    if not isinstance(value, Mapping) or set(value) != EXECUTION_FIELDS:
        raise ValueError(f"{label} must contain exactly {sorted(EXECUTION_FIELDS)}")
    if not isinstance(value["runner_version"], str) or not value["runner_version"]:
        raise ValueError(f"{label}.runner_version must be non-empty")
    if not isinstance(value["runner_source_sha256"], str) or not HASH_RE.fullmatch(
        value["runner_source_sha256"]
    ):
        raise ValueError(f"{label}.runner_source_sha256 must be a lowercase SHA-256")
    for field in ("input_sample_rate", "mimi_frame_samples", "required_max_lm_delay"):
        field_value = value[field]
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"{label}.{field} must be a positive integer")
    for field in ("prefix_silence_ms", "response_capture_ms"):
        if not _finite_number(value[field], positive=True):
            raise ValueError(f"{label}.{field} must be finite and positive")
    if value["reset_model_stream_between_trials"] is not True:
        raise ValueError(f"{label} must require stream reset")
    if value["reset_rng_for_each_trial_seed"] is not True:
        raise ValueError(f"{label} must require RNG reset")
    if value["required_model_type"] != "moshi" or value["required_max_lm_delay"] != 1:
        raise ValueError(f"{label} only supports standard Moshi with max LM delay 1")
    if value["input_sample_rate"] != 24000 or value["mimi_frame_samples"] != 1920:
        raise ValueError(f"{label} standard Moshi requires 24 kHz audio and 1,920-sample frames")


def verify_input_stimulus_file(
    value: Mapping[str, Any],
    *,
    artifact_root: Path,
    hash_cache: dict[Path, str] | None = None,
) -> Path:
    validate_input_stimulus(value)
    path = resolve_content_uri(value["uri"], artifact_root, label="input_stimulus.uri")
    if not path.is_file():
        raise FileNotFoundError(f"prepared stimulus is missing: {path}")
    resolved = path.resolve()
    cache = hash_cache if hash_cache is not None else {}
    actual_hash = cache.get(resolved)
    if actual_hash is None:
        actual_hash = sha256_file(resolved)
        cache[resolved] = actual_hash
    if actual_hash != value["sha256"]:
        raise ValueError(f"prepared stimulus hash mismatch: {path}")
    try:
        with wave.open(str(resolved), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
    except (EOFError, wave.Error) as error:
        raise ValueError(f"prepared stimulus is not a readable WAV: {path}") from error
    if channels != value["channels"]:
        raise ValueError(f"prepared stimulus channel count changed: {path}")
    if sample_width != value["sample_width_bytes"]:
        raise ValueError(f"prepared stimulus sample width changed: {path}")
    if sample_rate != value["sample_rate"]:
        raise ValueError(f"prepared stimulus sample rate changed: {path}")
    frame_samples = int(value["mimi_frame_samples"])
    if frame_count % frame_samples:
        raise ValueError(f"prepared stimulus is not Mimi-frame aligned: {path}")
    observed_duration = frame_count * 1000.0 / sample_rate
    if abs(observed_duration - float(value["duration_ms"])) > 0.05:
        raise ValueError(f"prepared stimulus duration changed: {path}")
    return resolved


def validate_stream_events(events: Any, *, label: str = "stream_events") -> None:
    if not isinstance(events, list) or not events:
        raise ValueError(f"{label} must be a non-empty list")
    previous_time = -1.0
    for index, event in enumerate(events):
        event_label = f"{label}[{index}]"
        if not isinstance(event, Mapping) or set(event) != STREAM_EVENT_FIELDS:
            raise ValueError(
                f"{event_label} must contain exactly {sorted(STREAM_EVENT_FIELDS)}"
            )
        frame_index = event["frame_index"]
        token_id = event["token_id"]
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise ValueError(f"{event_label}.frame_index must be an integer")
        if frame_index != index:
            raise ValueError(f"{event_label}.frame_index must be contiguous from zero")
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{event_label}.token_id must be a non-negative integer")
        if not _finite_number(event["time_ms"]):
            raise ValueError(f"{event_label}.time_ms must be finite and non-negative")
        time_ms = float(event["time_ms"])
        if time_ms < previous_time:
            raise ValueError(f"{label}.time_ms must be monotonic")
        previous_time = time_ms
        if not isinstance(event["piece"], str):
            raise ValueError(f"{event_label}.piece must be a string")


def response_evidence_hash(
    response_without_hash: Mapping[str, Any], stream_events: list[dict[str, Any]]
) -> str:
    payload = {
        "response": {
            key: value
            for key, value in response_without_hash.items()
            if key != "evidence_sha256"
        },
        "stream_events": stream_events,
    }
    return sha256_value(payload)


def validate_trial_response(
    row: Mapping[str, Any],
    *,
    verify_audio: bool = False,
    response_root: Path | None = None,
) -> None:
    trial_id = row.get("eval_trial_id", "<unknown>")
    response = row.get("response")
    if not isinstance(response, Mapping):
        raise ValueError(f"{trial_id}: response must be an object")
    status = response.get("status")
    if status == "pending":
        if set(response) != {"status"}:
            raise ValueError(f"{trial_id}: pending response may contain only status")
        if "stream_events" in row:
            raise ValueError(f"{trial_id}: pending response must not contain stream_events")
        return
    if status != "completed":
        raise ValueError(f"{trial_id}: response.status must be pending or completed")
    if set(response) != COMPLETED_RESPONSE_FIELDS:
        missing = sorted(COMPLETED_RESPONSE_FIELDS - set(response))
        extra = sorted(set(response) - COMPLETED_RESPONSE_FIELDS)
        raise ValueError(
            f"{trial_id}: completed response fields mismatch; missing={missing}, extra={extra}"
        )
    transcript = response["transcript"]
    if not isinstance(transcript, str):
        raise ValueError(f"{trial_id}: response.transcript must be a string")
    transcript_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    if response["transcript_sha256"] != transcript_hash:
        raise ValueError(f"{trial_id}: response transcript hash mismatch")
    for field in ("audio_sha256", "stream_events_sha256", "evidence_sha256"):
        if not isinstance(response[field], str) or not HASH_RE.fullmatch(response[field]):
            raise ValueError(f"{trial_id}: response.{field} must be a lowercase SHA-256")
    validate_relative_content_uri(
        response["audio_path"], label=f"{trial_id}: response.audio_path"
    )
    if not _finite_number(response["audio_duration_ms"], positive=True):
        raise ValueError(f"{trial_id}: response.audio_duration_ms must be positive")
    if not _finite_number(response["elapsed_seconds"]):
        raise ValueError(f"{trial_id}: response.elapsed_seconds must be non-negative")
    for field in ("audio_sample_rate", "audio_channels", "audio_sample_width_bytes"):
        value = response[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{trial_id}: response.{field} must be a positive integer")
    if response["audio_channels"] != 1 or response["audio_sample_width_bytes"] != 2:
        raise ValueError(f"{trial_id}: response audio must be mono PCM16")
    seed = response["generation_seed"]
    if seed != row.get("generation_seed") or isinstance(seed, bool):
        raise ValueError(f"{trial_id}: response generation seed mismatch")
    if response["stream_reset"] is not True or response["rng_reset"] is not True:
        raise ValueError(f"{trial_id}: stream and RNG resets must be evidenced as true")
    capture = row.get("capture_contract")
    execution = row.get("execution_contract")
    if not isinstance(capture, Mapping) or not isinstance(execution, Mapping):
        raise ValueError(f"{trial_id}: completed response requires frozen run contracts")
    if response["audio_sample_rate"] != execution["input_sample_rate"]:
        raise ValueError(f"{trial_id}: response audio sample rate mismatch")
    if response["timebase"] != "prepared_stream_relative" or response["stream_origin_ms"] != 0:
        raise ValueError(f"{trial_id}: response must use prepared-stream origin zero")
    exact_timing = {
        "primary_window_start_ms": capture["primary_window_start_ms"],
        "requested_target_end_ms": capture["requested_target_end_ms"],
        "actual_target_end_ms": capture["actual_target_end_ms"],
    }
    for field, expected in exact_timing.items():
        if response[field] != expected:
            raise ValueError(f"{trial_id}: response.{field} disagrees with capture contract")
    for field in (
        "fed_sample_count",
        "fed_frame_count",
        "output_sample_count",
        "output_frame_count",
        "appended_zero_sample_count",
    ):
        value = response[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{trial_id}: response.{field} must be a non-negative integer")
    frame_samples = int(execution["mimi_frame_samples"])
    target_samples = int(capture["target_end_sample_count"])
    target_frames = int(capture["target_end_frame_count"])
    if response["fed_sample_count"] != target_samples:
        raise ValueError(f"{trial_id}: fed samples do not cover the capture contract")
    if response["fed_frame_count"] != target_frames:
        raise ValueError(f"{trial_id}: fed frames do not cover the capture contract")
    if response["fed_sample_count"] != response["fed_frame_count"] * frame_samples:
        raise ValueError(f"{trial_id}: fed sample/frame counts disagree")
    if response["output_sample_count"] != response["fed_sample_count"]:
        raise ValueError(f"{trial_id}: response audio does not cover every fed sample")
    if response["output_frame_count"] != response["fed_frame_count"]:
        raise ValueError(f"{trial_id}: response audio does not cover every fed frame")
    if response["output_sample_count"] != response["output_frame_count"] * frame_samples:
        raise ValueError(f"{trial_id}: output sample/frame counts disagree")
    prepared_samples = round(
        float(row["input_stimulus"]["duration_ms"])
        * int(row["input_stimulus"]["sample_rate"])
        / 1000.0
    )
    if response["appended_zero_sample_count"] != target_samples - prepared_samples:
        raise ValueError(f"{trial_id}: appended zero count is inconsistent")
    expected_audio_duration = (
        response["output_sample_count"]
        * 1000.0
        / response["audio_sample_rate"]
    )
    if abs(float(response["audio_duration_ms"]) - expected_audio_duration) > 0.05:
        raise ValueError(f"{trial_id}: response audio duration/count mismatch")
    if response["coverage_complete"] is not True or response["eos_reached"] is not False:
        raise ValueError(f"{trial_id}: early EOS or incomplete capture coverage is forbidden")
    if response["runner_source_sha256"] != execution["runner_source_sha256"]:
        raise ValueError(f"{trial_id}: runner source hash mismatch")
    backend = response["backend"]
    if not isinstance(backend, Mapping) or set(backend) != BACKEND_FIELDS:
        raise ValueError(f"{trial_id}: response.backend fields are invalid")
    string_backend_fields = {
        "name",
        "version",
        "model_repo",
        "resolved_revision",
        "snapshot_revision",
        "code_commit",
        "model_type",
        "effective_generation_config_sha256",
    }
    for field in string_backend_fields:
        if not isinstance(backend[field], str) or not backend[field]:
            raise ValueError(f"{trial_id}: response.backend.{field} must be non-empty")
    if backend["model_repo"] != row.get("model_repo"):
        raise ValueError(f"{trial_id}: response backend model_repo mismatch")
    if backend["resolved_revision"] != row.get("resolved_revision"):
        raise ValueError(f"{trial_id}: response backend resolved_revision mismatch")
    if backend["snapshot_revision"] != row.get("resolved_revision"):
        raise ValueError(f"{trial_id}: response backend snapshot revision mismatch")
    if backend["code_commit"] != row.get("code_commit"):
        raise ValueError(f"{trial_id}: response backend code commit mismatch")
    if backend["model_type"] != "moshi" or backend["max_lm_delay"] != 1:
        raise ValueError(f"{trial_id}: only standard Moshi with max LM delay 1 is valid")
    for field in ("mimi_sample_rate", "frame_size", "max_lm_delay"):
        value = backend[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{trial_id}: response.backend.{field} must be positive")
    if backend["mimi_sample_rate"] != execution["input_sample_rate"]:
        raise ValueError(f"{trial_id}: backend Mimi sample rate mismatch")
    if backend["frame_size"] != execution["mimi_frame_samples"]:
        raise ValueError(f"{trial_id}: backend frame size mismatch")
    effective = backend["effective_generation_config"]
    if not isinstance(effective, Mapping) or not effective:
        raise ValueError(f"{trial_id}: effective generation config must be non-empty")
    effective_hash = sha256_value(effective)
    if backend["effective_generation_config_sha256"] != effective_hash:
        raise ValueError(f"{trial_id}: backend effective generation hash mismatch")
    if response["effective_generation_config_sha256"] != effective_hash:
        raise ValueError(f"{trial_id}: response effective generation hash mismatch")
    events = row.get("stream_events")
    validate_stream_events(events, label=f"{trial_id}.stream_events")
    assert isinstance(events, list)
    if len(events) != response["fed_frame_count"]:
        raise ValueError(f"{trial_id}: token timeline does not cover every fed frame")
    rendered = "".join(str(event["piece"]) for event in events).strip()
    if transcript != rendered:
        raise ValueError(f"{trial_id}: response transcript disagrees with stream events")
    if response["stream_events_sha256"] != sha256_value(events):
        raise ValueError(f"{trial_id}: stream-events hash mismatch")
    expected_evidence_hash = response_evidence_hash(response, events)
    if response["evidence_sha256"] != expected_evidence_hash:
        raise ValueError(f"{trial_id}: response evidence hash mismatch")
    if verify_audio:
        if response_root is None:
            raise ValueError("response_root is required to verify response audio")
        path = resolve_content_uri(
            response["audio_path"],
            response_root,
            label=f"{trial_id}: response.audio_path",
        )
        if not path.is_file():
            raise FileNotFoundError(f"{trial_id}: response audio is missing: {path}")
        if sha256_file(path) != response["audio_sha256"]:
            raise ValueError(f"{trial_id}: response audio hash mismatch")
        try:
            with wave.open(str(path), "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                sample_rate = handle.getframerate()
                frame_count = handle.getnframes()
        except (EOFError, wave.Error) as error:
            raise ValueError(f"{trial_id}: response audio is not a readable WAV") from error
        if (
            channels != response["audio_channels"]
            or sample_width != response["audio_sample_width_bytes"]
            or sample_rate != response["audio_sample_rate"]
        ):
            raise ValueError(f"{trial_id}: response audio metadata mismatch")
        duration_ms = frame_count * 1000.0 / sample_rate
        if abs(duration_ms - float(response["audio_duration_ms"])) > 0.05:
            raise ValueError(f"{trial_id}: response audio duration mismatch")
