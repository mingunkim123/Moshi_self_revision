from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from experiments.self_repair.mechanistic.conversation import (
    DATASET_V2_CONTRACT_SOURCE,
    NATURAL_START_STATUS,
    REQUIRED_EXPERIMENTAL_STARTUP_MODES,
    RESPONSE_CAPTURE_FRAMES,
    STARTUP_MODE_COMMON_HANDSHAKE,
    STARTUP_MODE_GREETING_SUPPRESSED,
    STARTUP_MODE_NATURAL,
    STARTUP_MODES,
    ConversationContract,
    ConversationContractError,
    diagnose_response_boundaries,
    estimate_generation_count,
    estimate_generation_work,
)
from experiments.self_repair.mechanistic.core import sha256_value


def _manifest_row() -> dict[str, object]:
    prepared_timing = {"utterance_end_ms": 640}
    capture_contract = {
        "condition": "repair",
        "timebase": "prepared_stream_relative",
        "stream_origin_ms": 0,
        "prepared_timing": prepared_timing,
        "prepared_timing_sha256": sha256_value(prepared_timing),
        "utterance_end_ms": 640,
        "primary_window_start_ms": 640,
        "response_capture_ms": 40_000,
        "requested_target_end_ms": 40_640,
        "target_end_frame_count": 508,
        "target_end_sample_count": 508 * 1_920,
        "actual_target_end_ms": 40_640,
    }
    execution_contract = {
        "input_sample_rate": 24_000,
        "mimi_frame_samples": 1_920,
        "prefix_silence_ms": 160,
        "response_capture_ms": 40_000,
        "required_model_type": "moshi",
        "required_max_lm_delay": 1,
        "reset_model_stream_between_trials": True,
        "reset_rng_for_each_trial_seed": True,
    }
    row = {
        "trial_id": "trial-1",
        "prepared_stimulus_id": "prepared-1",
        "condition": "repair",
        "audio_sha256": "a" * 64,
        "sample_rate": 24_000,
        "frame_count": 12,
        "sample_count": 12 * 1_920,
        "conversation_contract": {
            "trial_id": "trial-1",
            "source": DATASET_V2_CONTRACT_SOURCE,
            "startup_mode": STARTUP_MODE_NATURAL,
            "startup_status": NATURAL_START_STATUS,
            "required_startup_modes": list(REQUIRED_EXPERIMENTAL_STARTUP_MODES),
            "file_replay_startup": "prime_once_then_consume_first_mimi_frame",
            "assistant_output_origin_frame": 0,
            "sample_rate": 24_000,
            "frame_samples": 1_920,
            "prefix_silence_ms": 160,
            "user_start_frame": 2,
            "query_end_frame": 8,
            "query_end_ms": 640,
            "user_end_frame": 8,
            "user_frame_count": 12,
            "user_sample_count": 12 * 1_920,
            "response_capture_frames": RESPONSE_CAPTURE_FRAMES,
            "response_capture_ms": 40_000,
            "target_end_frame_count": 508,
            "target_end_sample_count": 508 * 1_920,
            "tail_guard_frames": 25,
            "appended_zero_frame_count": 496,
            "source_capture_contract_sha256": sha256_value(capture_contract),
            "source_execution_contract_sha256": sha256_value(execution_contract),
        },
        "input_stimulus": {
            "prepared_stimulus_id": "prepared-1",
            "sha256": "a" * 64,
            "sample_rate": 24_000,
            "channels": 1,
            "sample_width_bytes": 2,
            "mimi_frame_samples": 1_920,
            "duration_ms": 960,
            "timeline": "prepared_stream_relative",
        },
        "execution_contract": execution_contract,
        "capture_contract": capture_contract,
    }
    return row


def _contract() -> ConversationContract:
    return ConversationContract.from_manifest_row(_manifest_row())


def test_contract_is_immutable_and_cross_checks_redundant_manifest_evidence() -> None:
    contract = _contract()
    assert contract.frame_ms == 80
    assert contract.response_capture_ms == 40_000
    assert contract.target_duration_ms == 40_640
    assert contract.query_end_frame == contract.user_end_frame == 8
    assert contract.appended_zero_frame_count == 496
    assert STARTUP_MODES == {
        STARTUP_MODE_NATURAL,
        STARTUP_MODE_GREETING_SUPPRESSED,
        STARTUP_MODE_COMMON_HANDSHAKE,
    }
    with pytest.raises(FrozenInstanceError):
        contract.user_end_frame = 11  # type: ignore[misc]


def test_zero_length_startup_prefix_is_valid_when_explicitly_frame_aligned() -> None:
    row = _manifest_row()
    row["conversation_contract"]["user_start_frame"] = 0
    row["conversation_contract"]["prefix_silence_ms"] = 0
    row["execution_contract"]["prefix_silence_ms"] = 0
    row["conversation_contract"]["source_execution_contract_sha256"] = sha256_value(
        row["execution_contract"]
    )
    contract = ConversationContract.from_manifest_row(row)
    assert contract.user_start_frame == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["conversation_contract"].update(target_end_frame_count=507), "inconsistent target_end_frame_count"),
        (lambda row: row.update(sample_count=1), "inconsistent user_sample_count"),
        (lambda row: row["input_stimulus"].update(duration_ms=961), "complete model frame"),
        (lambda row: row["conversation_contract"].update(startup_mode="implicit"), "startup_mode"),
        (lambda row: row["conversation_contract"].update(query_end_frame=9), "inconsistent query_end_frame"),
        (lambda row: row["conversation_contract"].update(tail_guard_frames=26), "must be 25"),
        (lambda row: row["capture_contract"].update(response_capture_ms=39_920), "hash mismatch"),
    ],
)
def test_contract_fails_closed_on_misalignment_or_inconsistent_counts(mutation, message: str) -> None:
    row = _manifest_row()
    mutation(row)
    with pytest.raises(ConversationContractError, match=message):
        ConversationContract.from_manifest_row(row)


def test_response_boundaries_use_half_open_user_and_cap_boundaries() -> None:
    contract = _contract()
    token_ids = [3] * contract.target_end_frame_count
    pieces = [""] * contract.target_end_frame_count
    token_ids[1], pieces[1] = 10, " Hi"
    token_ids[2], pieces[2] = 11, ","  # exactly at user start: overlap, not greeting
    token_ids[7], pieces[7] = 12, " okay"
    token_ids[8], pieces[8] = 13, " Seattle"  # exactly at half-open user end: post-user
    token_ids[10], pieces[10] = 14, " plan"

    result = diagnose_response_boundaries(contract, token_ids, pieces)
    assert result.first_lexical_frame == 1
    assert result.greeting_before_user is True
    assert result.overlap is True
    assert result.overlap_activity_frames == 2
    assert result.first_post_query_lexical_frame == 8
    assert result.first_post_user_lexical_frame == 8
    assert result.first_post_user_latency_frames == 0
    assert result.first_post_user_latency_ms == 0
    assert result.trailing_quiet_frames == 497
    assert result.cap_active is False
    assert result.truncated is False
    assert result.no_response is False


def test_cap_activity_is_conservatively_truncated() -> None:
    contract = _contract()
    token_ids = [3] * contract.target_end_frame_count
    pieces = [""] * contract.target_end_frame_count
    token_ids[-1], pieces[-1] = 10, " continuing"
    result = diagnose_response_boundaries(contract, token_ids, pieces)
    assert result.cap_active is True
    assert result.truncated is True
    assert result.trailing_quiet_frames == 0


def test_greeting_only_is_not_a_post_query_response() -> None:
    contract = _contract()
    token_ids = [3] * contract.target_end_frame_count
    pieces = [""] * contract.target_end_frame_count
    token_ids[1], pieces[1] = 10, " Hello"
    result = diagnose_response_boundaries(contract, token_ids, pieces)
    assert result.greeting_before_user is True
    assert result.no_response is True
    assert result.first_post_user_lexical_frame is None
    assert result.trailing_quiet_frames == 506


@pytest.mark.parametrize(
    ("ids", "pieces", "message"),
    [
        ([3] * 507, [""] * 507, "exactly cover"),
        ([3] * 508, [""] * 507, "different lengths"),
        ([3] * 507 + [True], [""] * 508, "must be integers"),
        ([3] * 508, [""] * 507 + [1], "must be strings"),
        ([3] * 508, [""] * 507 + [" speech"], "blank token ID"),
        ([3] * 507 + [10], [""] * 508, "non-blank token ID"),
    ],
)
def test_response_diagnostics_fail_closed_on_bad_timelines(ids, pieces, message: str) -> None:
    with pytest.raises(ConversationContractError, match=message):
        diagnose_response_boundaries(_contract(), ids, pieces)


def test_cost_helpers_report_naive_and_shared_prefix_work() -> None:
    first = _contract()
    row = _manifest_row()
    row["trial_id"] = "trial-2"
    row["conversation_contract"]["trial_id"] = "trial-2"
    second = ConversationContract.from_manifest_row(row)

    assert estimate_generation_count(2, 5, 2) == 20
    naive = estimate_generation_work(
        [first, second], seed_count=5, arm_count=2, real_time_factor=0.4, gpu_hourly_usd=2.0
    )
    shared = estimate_generation_work(
        [first, second], seed_count=5, arm_count=2,
        branch_frames={"trial-1": 8, "trial-2": 8}, real_time_factor=0.4,
        gpu_hourly_usd=2.0,
    )
    assert naive.generation_count == 20
    assert naive.output_frame_count == 10_160
    assert naive.model_step_count == 10_180
    assert shared.model_step_count == 10_090
    assert shared.model_step_count < naive.model_step_count
    assert naive.output_audio_hours == pytest.approx(812.8 / 3600)
    assert naive.output_pcm16_bytes == 10_160 * 1_920 * 2
    assert naive.estimated_gpu_hours == pytest.approx(10_180 * 0.08 * 0.4 / 3600)
    assert naive.estimated_cost_usd == pytest.approx(naive.estimated_gpu_hours * 2)


def test_cost_helpers_fail_closed_on_incomplete_branch_map_or_missing_rtf() -> None:
    contract = _contract()
    with pytest.raises(ConversationContractError, match="branch frame IDs"):
        estimate_generation_work(
            [contract], seed_count=1, arm_count=2, branch_frames={"other": 3}
        )
    with pytest.raises(ConversationContractError, match="requires an observed"):
        estimate_generation_work(
            [contract], seed_count=1, arm_count=2, gpu_hourly_usd=1.0
        )
