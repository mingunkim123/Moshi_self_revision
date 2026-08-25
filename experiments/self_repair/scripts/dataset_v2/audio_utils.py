from __future__ import annotations

import math
from pathlib import Path
import wave

import numpy as np


def dbfs(amplitude: float) -> float:
    return float("-inf") if amplitude <= 0 else 20.0 * math.log10(amplitude)


def read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1:
        raise ValueError(f"{path}: expected mono WAV, found {channels} channels")
    if sample_width != 2:
        raise ValueError(f"{path}: expected signed PCM16 WAV, found {sample_width * 8}-bit")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sample_rate


def write_pcm16_mono(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    if audio.ndim != 1:
        raise ValueError("audio must be a mono vector")
    clipped = np.clip(audio, -1.0, 32767.0 / 32768.0)
    pcm = np.rint(clipped * 32768.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=True)
    target_size = round(audio.size * target_rate / source_rate)
    if target_size < 1:
        return np.zeros(0, dtype=np.float32)
    source_positions = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def active_rms(audio: np.ndarray, threshold_dbfs: float = -50.0) -> float:
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    active = audio[np.abs(audio) >= threshold]
    return 0.0 if active.size == 0 else float(np.sqrt(np.mean(np.square(active, dtype=np.float64))))


def normalize_audio(
    audio: np.ndarray,
    target_rms_dbfs: float,
    peak_limit_dbfs: float,
    active_threshold_dbfs: float = -50.0,
) -> tuple[np.ndarray, dict[str, float]]:
    before = active_rms(audio, active_threshold_dbfs)
    if before == 0:
        raise ValueError("no active audio above the configured threshold")
    desired = 10.0 ** (target_rms_dbfs / 20.0)
    peak_limit = 10.0 ** (peak_limit_dbfs / 20.0)
    peak = float(np.max(np.abs(audio)))
    gain = desired / before
    if peak:
        gain = min(gain, peak_limit / peak)
    normalized = np.asarray(audio * gain, dtype=np.float32)
    after = active_rms(normalized, active_threshold_dbfs)
    return normalized, {
        "active_rms_dbfs_before": dbfs(before),
        "active_rms_dbfs_after": dbfs(after),
        "peak_dbfs_after": dbfs(float(np.max(np.abs(normalized)))),
        "gain_db": dbfs(gain),
    }


def duration_ms(audio: np.ndarray, sample_rate: int) -> float:
    return audio.size * 1000.0 / sample_rate


def append_silence_and_frame_pad(
    audio: np.ndarray,
    sample_rate: int,
    prefix_ms: float,
    frame_samples: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    if prefix_ms < 0 or frame_samples < 1:
        raise ValueError("invalid preparation settings")
    prefix_samples = round(prefix_ms * sample_rate / 1000.0)
    prepared = np.concatenate((np.zeros(prefix_samples, dtype=np.float32), audio))
    pad_samples = (-prepared.size) % frame_samples
    if pad_samples:
        prepared = np.pad(prepared, (0, pad_samples))
    return prepared, {
        "prefix_samples": prefix_samples,
        "frame_pad_samples": pad_samples,
        "prefix_ms_actual": prefix_samples * 1000.0 / sample_rate,
    }
