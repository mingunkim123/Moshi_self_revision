"""Fail-closed conversation timing, response-boundary, and cost contracts.

The mechanistic runner deals in 80 ms model frames.  This module deliberately
does not load Torch, Mimi, or a checkpoint so manifests can be audited before a
GPU is rented.  Frame fields use half-open boundaries: a frame at
``user_end_frame`` is the first frame after the user has finished speaking.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


STARTUP_MODE_NATURAL = "natural_model_start"
STARTUP_MODE_GREETING_SUPPRESSED = "greeting_suppressed"
STARTUP_MODE_COMMON_HANDSHAKE = "common_handshake_then_request"
REQUIRED_EXPERIMENTAL_STARTUP_MODES = (
    STARTUP_MODE_COMMON_HANDSHAKE,
    STARTUP_MODE_GREETING_SUPPRESSED,
)
NATURAL_START_STATUS = "diagnostic_only_known_greeting_confound"
DATASET_V2_CONTRACT_SOURCE = "dataset_v2_frozen_capture_contract"
REVIEWED_MULTIVALUE_CONTRACT_SOURCE = "reviewed_multivalue_frozen_capture_contract"
CONTRACT_SOURCES = frozenset(
    {DATASET_V2_CONTRACT_SOURCE, REVIEWED_MULTIVALUE_CONTRACT_SOURCE}
)
STARTUP_MODES = frozenset(
    {
        STARTUP_MODE_NATURAL,
        STARTUP_MODE_GREETING_SUPPRESSED,
        STARTUP_MODE_COMMON_HANDSHAKE,
    }
)

MOSHIKO_SAMPLE_RATE = 24_000
MOSHIKO_FRAME_SAMPLES = 1_920
RESPONSE_CAPTURE_FRAMES = 500
TAIL_GUARD_FRAMES = 25


class ConversationContractError(ValueError):
    """Raised when timing evidence is missing, ambiguous, or inconsistent."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConversationContractError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConversationContractError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversationContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive " if positive else ""
        raise ConversationContractError(f"{label} must be a finite {qualifier}number")
    return result


def _consistent(label: str, values: Sequence[Any]) -> Any:
    present = [value for value in values if value is not None]
    if not present:
        raise ConversationContractError(f"missing {label}")
    first = present[0]
    if any(value != first for value in present[1:]):
        raise ConversationContractError(f"inconsistent {label}: {present}")
    return first


def _optional_consistent(label: str, values: Sequence[Any]) -> Any | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return _consistent(label, present)


def _sha256_value(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exclusive_frame_from_ms(
    milliseconds: Any, sample_rate: int, frame_samples: int, label: str
) -> int:
    value = _number(milliseconds, label)
    if value < 0:
        raise ConversationContractError(f"{label} must be non-negative")
    exact_samples = value * sample_rate / 1000.0
    rounded_samples = round(exact_samples)
    if abs(exact_samples - rounded_samples) > 1e-6:
        raise ConversationContractError(f"{label} does not resolve to an integer sample count")
    return int(math.ceil(rounded_samples / frame_samples))


def _exact_frame_count_from_duration(
    duration_ms: Any, sample_rate: int, frame_samples: int, label: str
) -> int:
    milliseconds = _number(duration_ms, label, positive=True)
    exact_samples = milliseconds * sample_rate / 1000.0
    rounded_samples = round(exact_samples)
    if abs(exact_samples - rounded_samples) > 1e-6:
        raise ConversationContractError(f"{label} does not resolve to an integer sample count")
    if rounded_samples % frame_samples:
        raise ConversationContractError(f"{label} is not aligned to a complete model frame")
    return rounded_samples // frame_samples


def _exact_frame_count_from_ms(
    milliseconds: Any,
    sample_rate: int,
    frame_samples: int,
    label: str,
    *,
    allow_zero: bool = False,
) -> int:
    value = _number(milliseconds, label)
    if value < 0 or (not allow_zero and value == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConversationContractError(f"{label} must be {qualifier}")
    exact_frames = value * sample_rate / (1000.0 * frame_samples)
    rounded = round(exact_frames)
    if abs(exact_frames - rounded) > 1e-9:
        raise ConversationContractError(f"{label} is not aligned to a model-frame boundary")
    return rounded


@dataclass(frozen=True)
class ConversationContract:
    """Immutable, frame-exact contract for one conversational generation.

    ``user_frame_count`` is the complete prepared user WAV (including any file
    tail). ``user_end_frame`` is the semantic end of the user utterance.
    ``target_end_frame_count`` covers the user utterance plus the complete fixed
    response window, and therefore can be longer than ``user_frame_count``.
    """

    trial_id: str
    startup_mode: str
    startup_status: str
    required_startup_modes: tuple[str, ...]
    sample_rate: int
    frame_samples: int
    user_start_frame: int
    query_end_frame: int
    user_end_frame: int
    user_frame_count: int
    response_capture_frames: int
    target_end_frame_count: int
    tail_guard_frames: int
    appended_zero_frame_count: int

    @property
    def frame_ms(self) -> float:
        return self.frame_samples * 1000.0 / self.sample_rate

    @property
    def response_capture_ms(self) -> float:
        return self.response_capture_frames * self.frame_ms

    @property
    def target_duration_ms(self) -> float:
        return self.target_end_frame_count * self.frame_ms

    @classmethod
    def from_manifest_row(cls, row: Mapping[str, Any]) -> "ConversationContract":
        """Parse redundant manifest evidence and reject every disagreement.

        A canonical row carries a ``conversation_contract`` object.  Relevant
        v2 ``input_stimulus``, ``execution_contract``, and ``capture_contract``
        fields are also checked when present so a stale or partially rebound
        manifest cannot silently change the generation horizon.
        """

        if not isinstance(row, Mapping):
            raise ConversationContractError("manifest row must be an object")
        contract = _mapping(row.get("conversation_contract"), "conversation_contract")
        stimulus = _mapping(row.get("input_stimulus"), "input_stimulus")
        execution = _mapping(row.get("execution_contract"), "execution_contract")
        capture = _mapping(row.get("capture_contract"), "capture_contract")

        trial_id = _consistent(
            "trial_id",
            [row.get("trial_id"), row.get("eval_trial_id"), contract.get("trial_id")],
        )
        if not isinstance(trial_id, str) or not trial_id:
            raise ConversationContractError("trial_id must be a non-empty string")

        startup_mode = _consistent(
            "startup_mode", [contract.get("startup_mode"), row.get("startup_mode")]
        )
        if startup_mode not in STARTUP_MODES:
            raise ConversationContractError(
                f"startup_mode must be one of {sorted(STARTUP_MODES)}"
            )
        startup_status = _consistent(
            "startup_status", [contract.get("startup_status"), row.get("startup_status")]
        )
        if startup_mode == STARTUP_MODE_NATURAL and startup_status != NATURAL_START_STATUS:
            raise ConversationContractError(
                "natural_model_start must be marked as the known greeting-confounded diagnostic"
            )
        required_modes_value = contract.get("required_startup_modes")
        if not isinstance(required_modes_value, list) or any(
            not isinstance(mode, str) for mode in required_modes_value
        ):
            raise ConversationContractError("required_startup_modes must be a string array")
        required_startup_modes = tuple(required_modes_value)
        if required_startup_modes != REQUIRED_EXPERIMENTAL_STARTUP_MODES:
            raise ConversationContractError(
                "required_startup_modes must freeze common handshake and greeting-suppressed runs"
            )

        source = contract.get("source")
        if source not in CONTRACT_SOURCES:
            raise ConversationContractError(
                f"conversation_contract.source must be one of {sorted(CONTRACT_SOURCES)}"
            )
        if not capture or not execution or not stimulus:
            raise ConversationContractError(
                "conversation rows must preserve input, capture, and execution contracts"
            )
        expected_capture_hash = contract.get("source_capture_contract_sha256")
        expected_execution_hash = contract.get("source_execution_contract_sha256")
        if expected_capture_hash != _sha256_value(capture):
            raise ConversationContractError("source capture contract hash mismatch")
        if expected_execution_hash != _sha256_value(execution):
            raise ConversationContractError("source execution contract hash mismatch")
        prepared_timing = _mapping(capture.get("prepared_timing"), "capture_contract.prepared_timing")
        if capture.get("prepared_timing_sha256") != _sha256_value(prepared_timing):
            raise ConversationContractError("source prepared timing hash mismatch")
        if contract.get("file_replay_startup") != "prime_once_then_consume_first_mimi_frame":
            raise ConversationContractError("unknown file replay startup contract")
        if contract.get("assistant_output_origin_frame") != 0:
            raise ConversationContractError("assistant output origin must be frame zero")
        if stimulus.get("timeline") != "prepared_stream_relative":
            raise ConversationContractError("source input stimulus has an unsupported timebase")
        if capture.get("timebase") != "prepared_stream_relative" or capture.get("stream_origin_ms") != 0:
            raise ConversationContractError("source capture contract has an unsupported timebase")
        if (
            execution.get("required_model_type") != "moshi"
            or execution.get("required_max_lm_delay") != 1
            or execution.get("reset_model_stream_between_trials") is not True
            or execution.get("reset_rng_for_each_trial_seed") is not True
        ):
            raise ConversationContractError("source execution contract is not the pinned Moshiko contract")
        if row.get("condition") is not None and capture.get("condition") != row.get("condition"):
            raise ConversationContractError("source capture condition disagrees with the trial")
        if row.get("prepared_stimulus_id") is not None and (
            stimulus.get("prepared_stimulus_id") != row.get("prepared_stimulus_id")
        ):
            raise ConversationContractError("source prepared stimulus binding disagrees with the trial")
        audio_sha = _consistent(
            "source audio sha256", [row.get("audio_sha256"), stimulus.get("sha256")]
        )
        if not isinstance(audio_sha, str) or len(audio_sha) != 64:
            raise ConversationContractError("source audio sha256 is malformed")

        sample_rate = _integer(
            _consistent(
                "sample_rate",
                [
                    contract.get("sample_rate"),
                    row.get("sample_rate"),
                    stimulus.get("sample_rate"),
                    execution.get("input_sample_rate"),
                ],
            ),
            "sample_rate",
            minimum=1,
        )
        frame_samples = _integer(
            _consistent(
                "frame_samples",
                [
                    contract.get("frame_samples"),
                    row.get("frame_samples"),
                    row.get("mimi_frame_samples"),
                    stimulus.get("mimi_frame_samples"),
                    execution.get("mimi_frame_samples"),
                ],
            ),
            "frame_samples",
            minimum=1,
        )
        if sample_rate != MOSHIKO_SAMPLE_RATE or frame_samples != MOSHIKO_FRAME_SAMPLES:
            raise ConversationContractError(
                "conversation contract is not the pinned mono 24 kHz / 1,920-sample Mimi timebase"
            )
        if stimulus.get("channels") != 1 or stimulus.get("sample_width_bytes") != 2:
            raise ConversationContractError("source input stimulus is not mono PCM16")

        inferred_user_frames = None
        if stimulus.get("duration_ms") is not None:
            inferred_user_frames = _exact_frame_count_from_duration(
                stimulus["duration_ms"], sample_rate, frame_samples, "input_stimulus.duration_ms"
            )
        user_frame_count = _integer(
            _consistent(
                "user_frame_count",
                [
                    contract.get("user_frame_count"),
                    row.get("user_frame_count"),
                    row.get("frame_count"),
                    inferred_user_frames,
                ],
            ),
            "user_frame_count",
            minimum=1,
        )

        inferred_start = None
        prefix_ms = _optional_consistent(
            "prefix_silence_ms",
            [
                contract.get("prefix_silence_ms"),
                contract.get("input_prefix_silence_ms"),
                execution.get("prefix_silence_ms"),
            ],
        )
        if prefix_ms is not None:
            inferred_start = _exact_frame_count_from_ms(
                prefix_ms,
                sample_rate,
                frame_samples,
                "prefix_silence_ms",
                allow_zero=True,
            )
        user_start_frame = _integer(
            _consistent(
                "user_start_frame",
                [contract.get("user_start_frame"), row.get("user_start_frame"), inferred_start],
            ),
            "user_start_frame",
        )

        query_end_ms = _consistent(
            "query_end_ms",
            [
                contract.get("query_end_ms"),
                capture.get("utterance_end_ms"),
                capture.get("primary_window_start_ms"),
                prepared_timing.get("utterance_end_ms"),
            ],
        )
        inferred_query_end = _exclusive_frame_from_ms(
            query_end_ms, sample_rate, frame_samples, "query_end_ms"
        )
        query_end_frame = _integer(
            _consistent(
                "query_end_frame",
                [
                    contract.get("query_end_frame"),
                    row.get("query_end_frame"),
                    inferred_query_end,
                ],
            ),
            "query_end_frame",
            minimum=1,
        )
        user_end_frame = _integer(
            _consistent(
                "user_end_frame",
                [
                    contract.get("user_end_frame"),
                    row.get("user_end_frame"),
                    inferred_query_end,
                ],
            ),
            "user_end_frame",
            minimum=1,
        )

        inferred_response_frames = None
        response_ms = _optional_consistent(
            "response_capture_ms",
            [
                contract.get("response_capture_ms"),
                capture.get("response_capture_ms"),
                execution.get("response_capture_ms"),
            ],
        )
        if response_ms is not None:
            inferred_response_frames = _exact_frame_count_from_ms(
                response_ms, sample_rate, frame_samples, "response_capture_ms"
            )
        response_capture_frames = _integer(
            _consistent(
                "response_capture_frames",
                [
                    contract.get("response_capture_frames"),
                    row.get("response_capture_frames"),
                    inferred_response_frames,
                ],
            ),
            "response_capture_frames",
            minimum=1,
        )
        if response_capture_frames != RESPONSE_CAPTURE_FRAMES:
            raise ConversationContractError(
                f"response_capture_frames must be {RESPONSE_CAPTURE_FRAMES} (40 seconds)"
            )
        target_end_frame_count = _integer(
            _consistent(
                "target_end_frame_count",
                [
                    contract.get("target_end_frame_count"),
                    row.get("target_end_frame_count"),
                    capture.get("target_end_frame_count"),
                ],
            ),
            "target_end_frame_count",
            minimum=1,
        )
        tail_guard_frames = _integer(
            _consistent(
                "tail_guard_frames",
                [contract.get("tail_guard_frames"), row.get("tail_guard_frames")],
            ),
            "tail_guard_frames",
            minimum=1,
        )
        if tail_guard_frames != TAIL_GUARD_FRAMES:
            raise ConversationContractError(
                f"tail_guard_frames must be {TAIL_GUARD_FRAMES} (2 seconds)"
            )

        appended_zero_frame_count = _integer(
            _consistent(
                "appended_zero_frame_count",
                [
                    contract.get("appended_zero_frame_count"),
                    row.get("appended_zero_frame_count"),
                    target_end_frame_count - user_frame_count,
                ],
            ),
            "appended_zero_frame_count",
        )

        if user_start_frame >= query_end_frame:
            raise ConversationContractError("user_start_frame must precede query_end_frame")
        if query_end_frame != user_end_frame:
            raise ConversationContractError(
                "query_end_frame and user_end_frame must be the same half-open semantic boundary"
            )
        if user_end_frame > user_frame_count:
            raise ConversationContractError("user_end_frame exceeds the prepared user frames")
        if user_frame_count > target_end_frame_count:
            raise ConversationContractError("prepared user frames exceed the capture target")
        if target_end_frame_count != user_end_frame + response_capture_frames:
            raise ConversationContractError(
                "target_end_frame_count must equal user_end_frame + response_capture_frames"
            )
        if tail_guard_frames > response_capture_frames:
            raise ConversationContractError("tail_guard_frames exceeds the response window")
        if appended_zero_frame_count != target_end_frame_count - user_frame_count:
            raise ConversationContractError("appended-zero frame count is inconsistent")

        requested_target_end_ms = _number(
            capture.get("requested_target_end_ms"),
            "capture_contract.requested_target_end_ms",
            positive=True,
        )
        expected_requested_ms = _number(query_end_ms, "query_end_ms") + (
            response_capture_frames * frame_samples * 1000.0 / sample_rate
        )
        if abs(requested_target_end_ms - expected_requested_ms) > 1e-6:
            raise ConversationContractError("source requested target end is inconsistent")
        actual_target_end_ms = _number(
            capture.get("actual_target_end_ms"),
            "capture_contract.actual_target_end_ms",
            positive=True,
        )
        expected_actual_ms = target_end_frame_count * frame_samples * 1000.0 / sample_rate
        if abs(actual_target_end_ms - expected_actual_ms) > 1e-6:
            raise ConversationContractError("source actual target end is inconsistent")

        sample_count = _optional_consistent(
            "user_sample_count",
            [contract.get("user_sample_count"), row.get("sample_count"), row.get("user_sample_count")],
        )
        if sample_count is not None and _integer(sample_count, "user_sample_count", minimum=1) != (
            user_frame_count * frame_samples
        ):
            raise ConversationContractError("user sample/frame counts disagree")
        target_samples = _optional_consistent(
            "target_end_sample_count",
            [contract.get("target_end_sample_count"), capture.get("target_end_sample_count")],
        )
        if target_samples is not None and _integer(
            target_samples, "target_end_sample_count", minimum=1
        ) != target_end_frame_count * frame_samples:
            raise ConversationContractError("target sample/frame counts disagree")

        for label, observed in (
            ("fed_frame_count", _mapping(row.get("response"), "response").get("fed_frame_count")),
            ("output_frame_count", _mapping(row.get("response"), "response").get("output_frame_count")),
        ):
            if observed is not None and _integer(observed, label) != target_end_frame_count:
                raise ConversationContractError(f"{label} does not cover the capture target")

        return cls(
            trial_id=trial_id,
            startup_mode=startup_mode,
            startup_status=str(startup_status),
            required_startup_modes=required_startup_modes,
            sample_rate=sample_rate,
            frame_samples=frame_samples,
            user_start_frame=user_start_frame,
            query_end_frame=query_end_frame,
            user_end_frame=user_end_frame,
            user_frame_count=user_frame_count,
            response_capture_frames=response_capture_frames,
            target_end_frame_count=target_end_frame_count,
            tail_guard_frames=tail_guard_frames,
            appended_zero_frame_count=appended_zero_frame_count,
        )


def _lexical(piece: str) -> bool:
    return any(character.isalnum() for character in piece)


@dataclass(frozen=True)
class ConversationDiagnostics:
    """Exact text-token boundary diagnostics for one captured output stream."""

    trial_id: str
    frame_ms: float
    total_frames: int
    first_lexical_frame: int | None
    first_post_query_lexical_frame: int | None
    first_post_user_lexical_frame: int | None
    last_activity_frame: int | None
    greeting_before_user: bool
    overlap: bool
    overlap_activity_frames: int
    first_post_user_latency_frames: int | None
    trailing_quiet_frames: int
    cap_active: bool
    truncated: bool
    no_response: bool

    @property
    def first_post_user_latency_ms(self) -> float | None:
        if self.first_post_user_latency_frames is None:
            return None
        return self.first_post_user_latency_frames * self.frame_ms


def diagnose_response_boundaries(
    contract: ConversationContract,
    token_ids: Sequence[int],
    token_pieces: Sequence[str],
    *,
    blank_token_ids: frozenset[int] = frozenset({0, 3}),
) -> ConversationDiagnostics:
    """Classify startup, overlap, response, and fixed-cap boundaries.

    ``no_response`` means that no lexical text token occurred at or after the
    query boundary; it does not claim that the decoded audio is silent.
    ``truncated`` is conservative: any non-whitespace text activity in the
    frozen tail guard marks the capture as cap-active and unevaluable.
    """

    if len(token_ids) != len(token_pieces):
        raise ConversationContractError("token IDs and pieces have different lengths")
    if len(token_ids) != contract.target_end_frame_count:
        raise ConversationContractError(
            "token timeline does not exactly cover target_end_frame_count"
        )
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids):
        raise ConversationContractError("token IDs must be integers")
    if any(not isinstance(piece, str) for piece in token_pieces):
        raise ConversationContractError("token pieces must be strings")
    if any(token_id in blank_token_ids and piece.strip() for token_id, piece in zip(token_ids, token_pieces)):
        raise ConversationContractError("blank token ID has a non-blank rendered piece")
    if any(token_id not in blank_token_ids and piece == "" for token_id, piece in zip(token_ids, token_pieces)):
        raise ConversationContractError("non-blank token ID has an empty rendered piece")

    activity_frames = tuple(index for index, piece in enumerate(token_pieces) if piece.strip())
    lexical_frames = tuple(index for index, piece in enumerate(token_pieces) if _lexical(piece))
    first_lexical = lexical_frames[0] if lexical_frames else None
    first_post_query = next(
        (frame for frame in lexical_frames if frame >= contract.query_end_frame), None
    )
    first_post_user = next(
        (frame for frame in lexical_frames if frame >= contract.user_end_frame), None
    )
    overlap_frames = tuple(
        frame
        for frame in activity_frames
        if contract.user_start_frame <= frame < contract.user_end_frame
    )
    last_activity = activity_frames[-1] if activity_frames else None
    trailing_quiet = (
        contract.target_end_frame_count
        if last_activity is None
        else contract.target_end_frame_count - last_activity - 1
    )
    tail_start = contract.target_end_frame_count - contract.tail_guard_frames
    cap_active = any(frame >= tail_start for frame in activity_frames)

    return ConversationDiagnostics(
        trial_id=contract.trial_id,
        frame_ms=contract.frame_ms,
        total_frames=len(token_ids),
        first_lexical_frame=first_lexical,
        first_post_query_lexical_frame=first_post_query,
        first_post_user_lexical_frame=first_post_user,
        last_activity_frame=last_activity,
        greeting_before_user=first_lexical is not None and first_lexical < contract.user_start_frame,
        overlap=bool(overlap_frames),
        overlap_activity_frames=len(overlap_frames),
        first_post_user_latency_frames=(
            None if first_post_user is None else first_post_user - contract.user_end_frame
        ),
        trailing_quiet_frames=trailing_quiet,
        cap_active=cap_active,
        truncated=cap_active,
        no_response=first_post_query is None,
    )


@dataclass(frozen=True)
class GenerationCostEstimate:
    trial_count: int
    seed_count: int
    arm_count: int
    generation_count: int
    output_frame_count: int
    model_step_count: int
    output_audio_hours: float
    output_pcm16_bytes: int
    estimated_gpu_hours: float | None
    estimated_cost_usd: float | None


def estimate_generation_count(trial_count: int, seed_count: int, arm_count: int) -> int:
    """Return the number of trial/seed/arm outputs after validating each axis."""

    return (
        _integer(trial_count, "trial_count", minimum=1)
        * _integer(seed_count, "seed_count", minimum=1)
        * _integer(arm_count, "arm_count", minimum=1)
    )


def estimate_generation_work(
    contracts: Sequence[ConversationContract],
    *,
    seed_count: int,
    arm_count: int,
    branch_frames: Mapping[str, int] | None = None,
    real_time_factor: float | None = None,
    gpu_hourly_usd: float | None = None,
) -> GenerationCostEstimate:
    """Estimate full replay or shared-prefix branching work.

    ``real_time_factor`` is observed GPU seconds per generated audio second.
    Without ``branch_frames`` every arm pays one LM priming step.  With it, the
    prefix and priming step are paid once per trial/seed and arms branch before
    the named frame.
    """

    if not contracts:
        raise ConversationContractError("at least one conversation contract is required")
    seeds = _integer(seed_count, "seed_count", minimum=1)
    arms = _integer(arm_count, "arm_count", minimum=1)
    if len({contract.trial_id for contract in contracts}) != len(contracts):
        raise ConversationContractError("conversation contracts contain duplicate trial IDs")

    output_frames = sum(contract.target_end_frame_count for contract in contracts) * seeds * arms
    model_frame_seconds = 0.0
    if branch_frames is None:
        model_steps = sum(
            arms * (contract.target_end_frame_count + 1) for contract in contracts
        ) * seeds
        model_frame_seconds = sum(
            arms
            * (contract.target_end_frame_count + 1)
            * contract.frame_samples
            / contract.sample_rate
            for contract in contracts
        ) * seeds
    else:
        unknown = set(branch_frames) - {contract.trial_id for contract in contracts}
        missing = {contract.trial_id for contract in contracts} - set(branch_frames)
        if unknown or missing:
            raise ConversationContractError(
                f"branch frame IDs differ from contracts; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        model_steps = 0
        for contract in contracts:
            branch = _integer(branch_frames[contract.trial_id], "branch_frame")
            if branch >= contract.target_end_frame_count:
                raise ConversationContractError("branch_frame must precede the capture end")
            # One delay-priming call plus the shared prefix and each arm's suffix.
            trial_steps = branch + 1 + arms * (contract.target_end_frame_count - branch)
            model_steps += trial_steps
            model_frame_seconds += trial_steps * contract.frame_samples / contract.sample_rate
        model_steps *= seeds
        model_frame_seconds *= seeds

    audio_seconds = sum(
        contract.target_end_frame_count * contract.frame_samples / contract.sample_rate
        for contract in contracts
    ) * seeds * arms
    pcm_bytes = sum(
        contract.target_end_frame_count * contract.frame_samples * 2
        for contract in contracts
    ) * seeds * arms

    gpu_hours = None
    if real_time_factor is not None:
        factor = _number(real_time_factor, "real_time_factor", positive=True)
        # Model-step time includes the extra delay-prime calls.
        gpu_hours = model_frame_seconds * factor / 3600.0
    cost = None
    if gpu_hourly_usd is not None:
        if gpu_hours is None:
            raise ConversationContractError(
                "gpu_hourly_usd requires an observed real_time_factor"
            )
        rate = _number(gpu_hourly_usd, "gpu_hourly_usd", positive=True)
        cost = gpu_hours * rate

    return GenerationCostEstimate(
        trial_count=len(contracts),
        seed_count=seeds,
        arm_count=arms,
        generation_count=estimate_generation_count(len(contracts), seeds, arms),
        output_frame_count=output_frames,
        model_step_count=model_steps,
        output_audio_hours=audio_seconds / 3600.0,
        output_pcm16_bytes=pcm_bytes,
        estimated_gpu_hours=gpu_hours,
        estimated_cost_usd=cost,
    )
