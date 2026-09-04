"""Conservative, dependency-free audio-tail diagnostics for Moshiko output.

This is not a speech recognizer.  It answers the narrower safety question used
by the paid-run gate: did decoded audio activity reach the frozen capture cap?
Any uncertain/noisy tail is treated as active, so it cannot be silently called
a complete response.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

import numpy as np


class AudioActivityError(ValueError):
    """Raised when PCM coverage or the frozen detector contract is invalid."""


@dataclass(frozen=True)
class AudioTailDiagnostics:
    sample_rate: int
    frame_samples: int
    frame_count: int
    threshold_dbfs: float
    threshold_source: str
    active_frame_count: int
    first_active_frame: int | None
    last_active_frame: int | None
    trailing_quiet_frames: int
    tail_guard_frames: int
    cap_active: bool
    clipped_sample_count: int
    pcm_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frame_rms_dbfs(pcm: np.ndarray, *, frame_samples: int) -> np.ndarray:
    """Return one finite RMS dBFS value per complete mono frame."""
    if isinstance(frame_samples, bool) or not isinstance(frame_samples, int) or frame_samples <= 0:
        raise AudioActivityError("frame_samples must be a positive integer")
    array = np.asarray(pcm)
    if array.ndim != 1:
        raise AudioActivityError("decoded PCM must be a one-dimensional mono timeline")
    if array.size == 0 or array.size % frame_samples:
        raise AudioActivityError("decoded PCM must contain a positive whole number of frames")
    if not np.issubdtype(array.dtype, np.number):
        raise AudioActivityError("decoded PCM must be numeric")
    floating = np.asarray(array, dtype=np.float64)
    if not np.isfinite(floating).all():
        raise AudioActivityError("decoded PCM contains NaN or infinity")
    framed = floating.reshape(-1, frame_samples)
    rms = np.sqrt(np.mean(np.square(framed), axis=1))
    floor = np.finfo(np.float64).tiny
    return 20.0 * np.log10(np.maximum(rms, floor))


def diagnose_audio_tail(
    pcm: np.ndarray,
    *,
    sample_rate: int,
    frame_samples: int,
    expected_frame_count: int,
    tail_guard_frames: int,
    threshold_dbfs: float,
    threshold_source: str,
) -> AudioTailDiagnostics:
    """Validate exact coverage and conservatively flag activity at the cap.

    ``threshold_source`` is mandatory provenance (for example a hash-bound
    silence calibration record).  The function deliberately will not infer a
    convenient threshold from the evaluated response itself.
    """
    for value, label in (
        (sample_rate, "sample_rate"),
        (frame_samples, "frame_samples"),
        (expected_frame_count, "expected_frame_count"),
        (tail_guard_frames, "tail_guard_frames"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AudioActivityError(f"{label} must be a positive integer")
    if tail_guard_frames > expected_frame_count:
        raise AudioActivityError("tail_guard_frames exceeds the output timeline")
    if isinstance(threshold_dbfs, bool) or not isinstance(threshold_dbfs, (int, float)):
        raise AudioActivityError("threshold_dbfs must be finite")
    threshold = float(threshold_dbfs)
    if not math.isfinite(threshold) or threshold >= 0:
        raise AudioActivityError("threshold_dbfs must be finite and below 0 dBFS")
    if not isinstance(threshold_source, str) or not threshold_source.strip():
        raise AudioActivityError("threshold_source provenance is required")

    array = np.asarray(pcm)
    expected_samples = expected_frame_count * frame_samples
    if array.ndim != 1 or array.size != expected_samples:
        raise AudioActivityError(
            f"decoded PCM coverage is {array.size} samples; expected exactly {expected_samples}")
    levels = frame_rms_dbfs(array, frame_samples=frame_samples)
    active = np.flatnonzero(levels >= threshold)
    first = int(active[0]) if active.size else None
    last = int(active[-1]) if active.size else None
    trailing = expected_frame_count if last is None else expected_frame_count - last - 1
    tail_start = expected_frame_count - tail_guard_frames
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return AudioTailDiagnostics(
        sample_rate=sample_rate,
        frame_samples=frame_samples,
        frame_count=expected_frame_count,
        threshold_dbfs=threshold,
        threshold_source=threshold_source,
        active_frame_count=int(active.size),
        first_active_frame=first,
        last_active_frame=last,
        trailing_quiet_frames=trailing,
        tail_guard_frames=tail_guard_frames,
        cap_active=bool(active.size and int(active[-1]) >= tail_start),
        clipped_sample_count=int(np.count_nonzero(np.abs(np.asarray(array, dtype=np.float64)) >= 1.0)),
        pcm_sha256=digest,
    )
