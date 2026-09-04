from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.self_repair.mechanistic.readiness import (
    CONVERSATION_CHECKS,
    FROZEN_AUDIO_ACTIVITY_POLICY_SHA256,
    GPU_CANARY_CHECKS,
    MODEL_CHECKS,
    OPEN_LOOP_CHECKS,
    ReadinessError,
    assess_readiness,
    estimate_workload,
)


def _config() -> dict:
    return {
        "model": {"heads": 4, "hidden_size": 16},
        "audio": {"sample_rate": 24_000, "mimi_frame_samples": 1_920, "frame_ms": 80},
        "conversation": {
            "required_modes": ["common_handshake_then_request", "greeting_suppressed"],
            "startup": {"natural_max_ms": 160},
            "response": {
                "trailing_text_quiet_ms": 160,
                "tail_guard_ms": 160,
                "audio_activity": {
                    "version": "1.0.0",
                    "detector": "frame_rms_dbfs",
                    "frame_samples": 1_920,
                    "threshold_dbfs": -45.0,
                    "calibration": "forced_silence_decode_max_must_remain_below_threshold",
                    "policy_sha256": FROZEN_AUDIO_ACTIVITY_POLICY_SHA256,
                },
            },
        },
        "gates": {
            "conversation_canary_min_trials_per_mode": 2,
            "conversation_canary_truncated_max": 0,
            "conversation_canary_coverage_min": 1.0,
        },
    }


def _manifest() -> list[dict]:
    rows = []
    for index, (condition, frames, fold) in enumerate(
        (("clean_current", 10, 1), ("repair", 12, 1), ("repair", 20, 4))
    ):
        rows.append({
            "trial_id": f"t{index}",
            "scenario_id": f"s{index // 2}",
            "condition": condition,
            "speaker_id": "spk",
            "role": "discovery" if fold == 1 else "internal_validation",
            "analysis_fold": fold,
            "sample_rate": 24_000,
            "frame_count": frames,
            "sample_count": frames * 1_920,
            "conversation_contract": {
                "user_frame_count": frames,
                "user_end_frame": frames,
                "response_capture_frames": 2,
                "target_end_frame_count": frames + 2,
                "target_end_sample_count": (frames + 2) * 1_920,
                "appended_zero_frame_count": 2,
            },
        })
    return rows


def _spec() -> dict:
    return {
        "trial_selector": {"roles": ["discovery"]},
        "recipient_selector": {"exclude_clean": True},
        "scans": [{
            "name": "residual_and_head",
            "layers": [0, 3],
            "anchors": ["new_end", "query_end"],
            "donor_arms": ["clean", "self"],
            "components": ["resid_post", {"name": "head_z", "heads": [1, 2]}],
            "full_replays_per_cell": 3,
            "readout_steps_per_cell": 4,
            "expected_cell_count": 24,
        }],
        "generation": {
            "trial_selector": {"exclude_clean": True},
            "seeds": [17, 29],
            "branches": ["baseline", "patched"],
            "startup_modes": ["common_handshake_then_request", "greeting_suppressed"],
            "response_capture_ms": 160,
            "expected_generation_count": 8,
        },
        "storage": {
            "user_codebooks": 8,
            "code_dtype_bytes": 8,
            "audio_sample_width_bytes": 2,
            "wav_header_bytes": 44,
            "result_bytes_per_cell": 100,
            "fixed_reserved_bytes": 1_000,
            "captures": [{
                "selector": {"conditions": ["repair"]},
                "layers": [0, 3],
                "anchors": ["new_end"],
                "sites": ["resid_post"],
                "dtype_bytes": 4,
            }],
        },
    }


def _passing_evidence() -> dict:
    def passed(names: tuple[str, ...]) -> dict:
        return {"passed": True, "checks": {name: True for name in names}}

    return {
        "model_contract": passed(MODEL_CHECKS),
        "open_loop": passed(OPEN_LOOP_CHECKS),
        "conversation_canary": {
            **passed(CONVERSATION_CHECKS),
            "per_mode": {
                mode: {
                    "trial_count": 2,
                    "truncated_count": 0,
                    "cap_active_count": 0,
                    "exact_output_coverage_count": 2,
                    "response_complete_count": 2,
                    "text_tail_checked_count": 2,
                    "audio_tail_checked_count": 2,
                    "human_flow_review_pass_count": 2,
                }
                for mode in ("common_handshake_then_request", "greeting_suppressed")
            },
            "measurements": {
                "required_mode_trial_count": 4,
                "truncated_count": 0,
                "cap_active_count": 0,
                "exact_output_coverage_count": 4,
                "text_tail_checked_count": 4,
                "audio_tail_checked_count": 4,
                "human_flow_review_pass_count": 4,
            },
            "tail_detection": {
                "text_quiet_frames": 2,
                "tail_guard_frames": 2,
                "audio_activity_policy_version": "1.0.0",
                "audio_activity_detector": "frame_rms_dbfs",
                "audio_activity_frame_samples": 1_920,
                "audio_activity_threshold_dbfs": -45.0,
                "audio_activity_calibration": "forced_silence_decode_max_must_remain_below_threshold",
                "audio_activity_policy_sha256": FROZEN_AUDIO_ACTIVITY_POLICY_SHA256,
                "forced_silence_decode_max_dbfs": -70.0,
            },
        },
        "gpu_canary": {
            **passed(GPU_CANARY_CHECKS),
            "measurements": {
                "completed_cells": 2,
                "failed_cells": 0,
                "duplicate_cells": 0,
                "model_frame_count": 100,
                "elapsed_seconds": 4.0,
                "mean_cell_seconds": 2.0,
                "seconds_per_model_frame": 0.04,
                "peak_vram_bytes": 1_000_000,
                "device_total_vram_bytes": 2_000_000,
                "activation_bytes": 128,
            },
        },
    }


def test_exact_workload_arithmetic() -> None:
    estimate = estimate_workload(_manifest(), _config(), _spec())
    assert estimate.manifest_trial_count == 3
    assert estimate.manifest_frame_count == 42
    assert estimate.selected_trial_count == 2
    assert estimate.selected_frame_count == 22
    assert estimate.recipient_trial_count == 1
    assert estimate.cell_count == 24  # 1 recipient * 2 layers * 2 anchors * 2 donors * (1 + 2 heads)
    assert estimate.cells_per_recipient == 24
    assert estimate.replay_pass_count == 72
    assert estimate.replay_frame_count == 14 * 24 * 3
    assert estimate.readout_frame_count == 24 * 4
    assert estimate.generation_trial_count == 1
    assert estimate.generation_count == 8
    assert estimate.generation_frame_count == ((14 + 2) + 14) * 4
    assert estimate.generated_audio_frame_count == 120
    assert estimate.generated_audio_hours == pytest.approx(120 * 0.08 / 3600)
    assert estimate.encoded_tensor_bytes == (10 + 2 * 12 + 12 + 2 * 14) * 8 * 8
    assert estimate.activation_tensor_bytes == 1 * 2 * 1 * 1 * 16 * 4
    assert estimate.cell_record_reserved_bytes == 24 * 100
    assert estimate.generated_wav_reserved_bytes == 120 * 1_920 * 2 + 8 * 44
    assert estimate.total_model_frames == 14 * 24 * 3 + 24 * 4 + 120
    expected_storage = (74 * 8 * 8) + 128 + 2_400 + (120 * 1_920 * 2 + 352) + 1_000
    assert estimate.total_storage_reserved_bytes == expected_storage


def test_head_grid_is_detected_as_explosive() -> None:
    spec = _spec()
    spec["scans"][0]["expected_cell_count"] = 24
    report = assess_readiness(
        _manifest(), _config(), spec,
        evidence=_passing_evidence(),
        limits={
            "max_cells": 20,
            "max_cells_per_recipient": 20,
            "max_model_frames": 10_000,
            "max_generation_runs": 10,
            "max_generated_audio_hours": 1,
            "max_storage_bytes": 10_000_000,
        },
    )
    assert report["decision"] == "NO_GO"
    codes = {blocker["code"] for blocker in report["blockers"]}
    assert "explosive_cell_count" in codes
    assert "explosive_cells_per_recipient" in codes
    assert report["stages"][-1]["stage"] == "paid_scan"
    assert report["stages"][-1]["decision"] == "NO_GO"


@pytest.mark.parametrize("missing", ["model_contract", "open_loop", "conversation_canary", "gpu_canary"])
def test_missing_required_evidence_is_fail_closed(missing: str) -> None:
    evidence = _passing_evidence()
    del evidence[missing]
    report = assess_readiness(_manifest(), _config(), _spec(), evidence=evidence)
    assert report["decision"] == "NO_GO"
    stage = next(row for row in report["stages"] if row["stage"] == missing)
    assert stage["decision"] == "NO_GO"
    assert any(blocker["code"] == f"missing_{missing}_evidence" for blocker in report["blockers"])


def test_false_conversation_tail_check_blocks_paid_scan() -> None:
    evidence = _passing_evidence()
    evidence["conversation_canary"]["checks"]["no_tail_truncation"] = False
    report = assess_readiness(_manifest(), _config(), _spec(), evidence=evidence)
    assert report["decision"] == "NO_GO"
    assert any("no_tail_truncation" in blocker["code"] for blocker in report["blockers"])


def test_every_gate_and_budget_must_pass_for_go() -> None:
    report = assess_readiness(_manifest(), _config(), _spec(), evidence=_passing_evidence())
    assert report["decision"] == "GO"
    assert report["blockers"] == []
    assert all(stage["decision"] == "GO" for stage in report["stages"])
    assert report["runtime_projection"]["estimated_gpu_hours_by_cell"] == pytest.approx(24 * 2 / 3600)


def test_malformed_evidence_and_limits_fail_closed() -> None:
    report = assess_readiness(_manifest(), _config(), _spec(), evidence="not-an-object")  # type: ignore[arg-type]
    assert report["decision"] == "NO_GO"
    assert any(blocker["code"] == "invalid_evidence_bundle" for blocker in report["blockers"])

    report = assess_readiness(
        _manifest(), _config(), _spec(), evidence=_passing_evidence(), limits={"max_cells": 1.5}
    )
    assert report["decision"] == "NO_GO"
    assert report["blockers"][0]["code"] == "invalid_workload_contract"


def test_manifest_timebase_mismatch_and_grid_drift_fail_closed() -> None:
    broken_manifest = _manifest()
    broken_manifest[0]["sample_count"] += 1
    with pytest.raises(ReadinessError, match=r"frame_count \* frame_samples"):
        estimate_workload(broken_manifest, _config(), _spec())

    broken_spec = deepcopy(_spec())
    broken_spec["scans"][0]["expected_cell_count"] = 23
    report = assess_readiness(_manifest(), _config(), broken_spec, evidence=_passing_evidence())
    assert report["decision"] == "NO_GO"
    assert report["estimate"] is None
    assert report["blockers"][0]["code"] == "invalid_workload_contract"


def test_explicit_replay_frame_count_supports_suffix_plans() -> None:
    spec = _spec()
    scan = spec["scans"][0]
    scan["replay_frames_per_cell"] = 7
    estimate = estimate_workload(_manifest(), _config(), spec)
    assert estimate.replay_pass_count == 72
    assert estimate.replay_frame_count == 24 * 7


def test_conversation_numeric_counts_and_gpu_measurements_are_fail_closed() -> None:
    evidence = _passing_evidence()
    evidence["conversation_canary"]["per_mode"]["common_handshake_then_request"]["cap_active_count"] = 1
    report = assess_readiness(_manifest(), _config(), _spec(), evidence=evidence)
    assert report["decision"] == "NO_GO"
    assert any(blocker["code"] == "cap_active_response_observed" for blocker in report["blockers"])

    evidence = _passing_evidence()
    evidence["gpu_canary"]["measurements"]["mean_cell_seconds"] = 1.0
    report = assess_readiness(_manifest(), _config(), _spec(), evidence=evidence)
    assert report["decision"] == "NO_GO"
    assert any(blocker["code"] == "gpu_canary_cell_timing_mismatch" for blocker in report["blockers"])


def test_common_handshake_startup_cap_is_reserved() -> None:
    estimate = estimate_workload(_manifest(), _config(), _spec())
    # One generated repair trial, 4 seed/branch combinations: the handshake mode
    # reserves 2 extra frames each while the suppressed mode does not.
    assert estimate.generation_frame_count == (16 + 14) * 4


def test_stale_audio_activity_policy_hash_blocks_readiness() -> None:
    config = _config()
    config["conversation"]["response"]["audio_activity"]["policy_sha256"] = "a" * 64
    report = assess_readiness(_manifest(), config, _spec(), evidence=_passing_evidence())
    assert report["decision"] == "NO_GO"
    assert any(
        blocker["code"] == "invalid_conversation_gate_config"
        and "canonical contents" in blocker["message"]
        for blocker in report["blockers"]
    )


def test_exact_estimate_binds_kind_specific_execution_arms() -> None:
    residual = _spec()
    residual["scans"][0]["components"] = ["resid_post"]
    residual["scans"][0]["expected_cell_count"] = 8
    residual["execution"] = {
        "kind": "residual",
        "role": "discovery",
        "layers": [0, 3],
        "anchors": ["new_end", "query_end"],
        "donors": ["clean", "self"],
        "controls": ["ignored-default-control"],
        "components": ["resid_post"],
        "limit_scenarios": None,
        "selection_sha256": None,
    }
    assert estimate_workload(_manifest(), _config(), residual).cell_count == 8

    mismatched = deepcopy(residual)
    mismatched["execution"]["donors"] = ["clean"]
    with pytest.raises(ReadinessError, match="kind-active execution.donors"):
        estimate_workload(_manifest(), _config(), mismatched)

    component = deepcopy(residual)
    component["execution"]["kind"] = "component"
    component["scans"][0]["components"] = ["attn_out"]
    component["execution"]["components"] = ["attn_out"]
    component["execution"]["controls"] = ["clean", "self"]
    component["execution"]["donors"] = ["ignored-default-donor"]
    assert estimate_workload(_manifest(), _config(), component).cell_count == 8
