from __future__ import annotations

import numpy as np
import pytest

from experiments.self_repair.mechanistic.audio_activity import (
    AudioActivityError,
    diagnose_audio_tail,
    frame_rms_dbfs,
)


FRAME = 1_920


def _pcm(frames: int = 10) -> np.ndarray:
    return np.zeros(frames * FRAME, dtype=np.float32)


def test_frame_rms_and_quiet_tail_are_frame_exact() -> None:
    pcm = _pcm()
    pcm[2 * FRAME:3 * FRAME] = 0.1
    levels = frame_rms_dbfs(pcm, frame_samples=FRAME)
    assert len(levels) == 10
    assert levels[2] == pytest.approx(-20.0)
    result = diagnose_audio_tail(
        pcm,
        sample_rate=24_000,
        frame_samples=FRAME,
        expected_frame_count=10,
        tail_guard_frames=2,
        threshold_dbfs=-45.0,
        threshold_source="canary-silence-calibration:abc123",
    )
    assert result.first_active_frame == result.last_active_frame == 2
    assert result.trailing_quiet_frames == 7
    assert result.cap_active is False
    assert len(result.pcm_sha256) == 64


def test_activity_in_tail_guard_is_cap_active() -> None:
    pcm = _pcm()
    pcm[-FRAME:] = 0.01
    result = diagnose_audio_tail(
        pcm,
        sample_rate=24_000,
        frame_samples=FRAME,
        expected_frame_count=10,
        tail_guard_frames=2,
        threshold_dbfs=-45.0,
        threshold_source="frozen-calibration",
    )
    assert result.last_active_frame == 9
    assert result.trailing_quiet_frames == 0
    assert result.cap_active is True


def test_audio_tail_rejects_partial_coverage_and_self_fitted_threshold() -> None:
    with pytest.raises(AudioActivityError, match="expected exactly"):
        diagnose_audio_tail(
            _pcm()[:-1], sample_rate=24_000, frame_samples=FRAME,
            expected_frame_count=10, tail_guard_frames=2,
            threshold_dbfs=-45.0, threshold_source="frozen-calibration",
        )
    with pytest.raises(AudioActivityError, match="provenance"):
        diagnose_audio_tail(
            _pcm(), sample_rate=24_000, frame_samples=FRAME,
            expected_frame_count=10, tail_guard_frames=2,
            threshold_dbfs=-45.0, threshold_source="",
        )


@pytest.mark.parametrize("bad", [np.zeros((1, FRAME)), np.zeros(FRAME - 1), np.array([np.nan])])
def test_frame_rms_rejects_invalid_pcm(bad: np.ndarray) -> None:
    with pytest.raises(AudioActivityError):
        frame_rms_dbfs(bad, frame_samples=FRAME)
