#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    EXPERIMENT_ROOT,
    parse_bool,
    parse_optional_float,
    read_csv,
    read_json,
    resolve_experiment_path,
    sha256_file,
    validate_id,
    write_csv,
)


OUTPUT_FIELDS = [
    "trial_id",
    "speaker_id",
    "condition_id",
    "language",
    "track",
    "utterance",
    "target",
    "stale",
    "is_repair",
    "is_clean",
    "is_long_gap",
    "clean_match_id",
    "raw_audio_path",
    "prepared_audio_path",
    "prepared_audio_sha256",
    "repair_marker_onset_ms",
    "repair_onset_ms",
    "repair_end_ms",
    "user_end_ms",
    "duration_ms",
    "active_rms_dbfs_before",
    "active_rms_dbfs_after",
    "peak_dbfs_after",
    "gain_db",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and frame-pad recorded stimuli for Moshi."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "config/experiment.json",
    )
    parser.add_argument(
        "--conditions",
        type=Path,
        default=EXPERIMENT_ROOT / "data/conditions.csv",
    )
    parser.add_argument(
        "--recordings",
        type=Path,
        default=EXPERIMENT_ROOT / "data/recordings.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "data/manifest.prepared.csv",
    )
    parser.add_argument(
        "--prepared-root",
        default="data/prepared",
        help="Output audio root, relative to the experiment directory unless absolute.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dbfs(amplitude: float) -> float:
    if amplitude <= 0:
        return float("-inf")
    return 20.0 * math.log10(amplitude)


def active_rms(audio: np.ndarray, threshold_dbfs: float) -> float:
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    active = audio[np.abs(audio) >= threshold]
    if active.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(active, dtype=np.float64))))


def normalize_audio(
    audio: np.ndarray,
    target_rms_dbfs: float,
    active_threshold_dbfs: float,
    peak_limit_dbfs: float,
) -> tuple[np.ndarray, dict[str, float]]:
    before = active_rms(audio, active_threshold_dbfs)
    if before == 0.0:
        raise ValueError("No active speech found above the configured threshold")
    desired = 10.0 ** (target_rms_dbfs / 20.0)
    gain = desired / before
    peak = float(np.max(np.abs(audio)))
    peak_limit = 10.0 ** (peak_limit_dbfs / 20.0)
    if peak > 0:
        gain = min(gain, peak_limit / peak)
    output = np.asarray(audio * gain, dtype=np.float32)
    # The absolute activity threshold can admit a few extra samples after gain.
    # Re-estimate twice so the reported active RMS stays close to the target.
    for _ in range(2):
        current = active_rms(output, active_threshold_dbfs)
        current_peak = float(np.max(np.abs(output)))
        if current == 0.0 or current_peak == 0.0:
            break
        correction = min(desired / current, peak_limit / current_peak)
        output = np.asarray(output * correction, dtype=np.float32)
        gain *= correction
    after = active_rms(output, active_threshold_dbfs)
    return output, {
        "active_rms_dbfs_before": dbfs(before),
        "active_rms_dbfs_after": dbfs(after),
        "peak_dbfs_after": dbfs(float(np.max(np.abs(output)))),
        "gain_db": dbfs(gain),
    }


def shifted_timestamp(
    value: float | None,
    prefix_ms: float,
    insert_at_ms: float | None,
    insert_ms: float,
) -> str:
    if value is None:
        return ""
    shifted = value + prefix_ms
    if insert_at_ms is not None and value >= insert_at_ms:
        shifted += insert_ms
    return f"{shifted:.3f}"


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    try:
        import sphn
    except ImportError as error:
        raise RuntimeError(
            "sphn is not installed. Run runpod/setup.sh or `pip install -e ./moshi`."
        ) from error
    pcm, _ = sphn.read(str(path), sample_rate=sample_rate)
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.ndim == 1:
        return pcm
    if pcm.ndim != 2 or pcm.shape[0] < 1:
        raise ValueError(f"Unexpected audio shape {pcm.shape} for {path}")
    return pcm[0]


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import sphn

    path.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(path), audio, sample_rate=sample_rate)


def condition_map(path: Path) -> dict[str, dict[str, str]]:
    output = {}
    for row in read_csv(path):
        condition_id = validate_id(row["condition_id"], "condition_id")
        if condition_id in output:
            raise ValueError(f"Duplicate condition_id: {condition_id}")
        output[condition_id] = row
    return output


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")

    config = read_json(args.config)
    audio_config: dict[str, Any] = config["audio"]
    sample_rate = int(audio_config["sample_rate"])
    frame_size = int(audio_config["frame_size"])
    prefix_ms = float(audio_config["prefix_silence_ms"])
    suffix_ms = float(audio_config["suffix_silence_ms"])
    prefix_samples = round(prefix_ms * sample_rate / 1000.0)
    suffix_samples = round(suffix_ms * sample_rate / 1000.0)
    conditions = condition_map(args.conditions)
    prepared_root = resolve_experiment_path(args.prepared_root)

    output_rows: list[dict[str, Any]] = []
    seen_trials: set[str] = set()
    for recording in read_csv(args.recordings):
        if not parse_bool(recording.get("accepted"), default=True):
            continue
        trial_id = validate_id(recording["trial_id"], "trial_id")
        speaker_id = validate_id(recording["speaker_id"], "speaker_id")
        condition_id = validate_id(recording["condition_id"], "condition_id")
        if trial_id in seen_trials:
            raise ValueError(f"Duplicate trial_id: {trial_id}")
        seen_trials.add(trial_id)
        if condition_id not in conditions:
            raise ValueError(f"Unknown condition_id {condition_id} in {args.recordings}")
        condition = conditions[condition_id]

        raw_path = resolve_experiment_path(recording["raw_audio_path"])
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing recording for {trial_id}: {raw_path}")
        raw_audio = load_mono(raw_path, sample_rate)
        raw_duration_ms = len(raw_audio) * 1000.0 / sample_rate

        insert_at_ms = parse_optional_float(recording.get("insert_silence_at_ms"))
        insert_ms = parse_optional_float(recording.get("insert_silence_ms")) or 0.0
        if insert_ms < 0:
            raise ValueError(f"insert_silence_ms cannot be negative for {trial_id}")
        if insert_at_ms is not None:
            if insert_at_ms < 0 or insert_at_ms > raw_duration_ms:
                raise ValueError(f"insert_silence_at_ms is outside {trial_id}")
            insert_sample = round(insert_at_ms * sample_rate / 1000.0)
            inserted = np.zeros(round(insert_ms * sample_rate / 1000.0), dtype=np.float32)
            raw_audio = np.concatenate(
                (raw_audio[:insert_sample], inserted, raw_audio[insert_sample:])
            )

        normalized, levels = normalize_audio(
            raw_audio,
            float(audio_config["target_active_rms_dbfs"]),
            float(audio_config["active_threshold_dbfs"]),
            float(audio_config["peak_limit_dbfs"]),
        )
        prepared = np.concatenate(
            (
                np.zeros(prefix_samples, dtype=np.float32),
                normalized,
                np.zeros(suffix_samples, dtype=np.float32),
            )
        )
        remainder = len(prepared) % frame_size
        if remainder:
            prepared = np.pad(prepared, (0, frame_size - remainder))
        if len(prepared) % frame_size:
            raise AssertionError("Prepared audio is not frame aligned")

        prepared_path = prepared_root / speaker_id / f"{condition_id}.wav"
        if prepared_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {prepared_path}; pass --overwrite"
            )
        write_wav(prepared_path, prepared, sample_rate)
        try:
            display_raw = raw_path.relative_to(EXPERIMENT_ROOT)
        except ValueError:
            display_raw = raw_path
        try:
            display_prepared = prepared_path.relative_to(EXPERIMENT_ROOT)
        except ValueError:
            display_prepared = prepared_path

        user_end = parse_optional_float(recording.get("user_end_ms"))
        if user_end is None:
            user_end = raw_duration_ms
        output_rows.append(
            {
                "trial_id": trial_id,
                "speaker_id": speaker_id,
                "condition_id": condition_id,
                "language": condition["language"],
                "track": condition["track"],
                "utterance": condition["utterance"],
                "target": condition["target"],
                "stale": condition["stale"],
                "is_repair": condition["is_repair"],
                "is_clean": condition["is_clean"],
                "is_long_gap": condition["is_long_gap"],
                "clean_match_id": condition["clean_match_id"],
                "raw_audio_path": str(display_raw),
                "prepared_audio_path": str(display_prepared),
                "prepared_audio_sha256": sha256_file(prepared_path),
                "repair_marker_onset_ms": shifted_timestamp(
                    parse_optional_float(recording.get("repair_marker_onset_ms")),
                    prefix_ms,
                    insert_at_ms,
                    insert_ms,
                ),
                "repair_onset_ms": shifted_timestamp(
                    parse_optional_float(recording.get("repair_onset_ms")),
                    prefix_ms,
                    insert_at_ms,
                    insert_ms,
                ),
                "repair_end_ms": shifted_timestamp(
                    parse_optional_float(recording.get("repair_end_ms")),
                    prefix_ms,
                    insert_at_ms,
                    insert_ms,
                ),
                "user_end_ms": shifted_timestamp(
                    user_end, prefix_ms, insert_at_ms, insert_ms
                ),
                "duration_ms": f"{len(prepared) * 1000.0 / sample_rate:.3f}",
                **{key: f"{value:.4f}" for key, value in levels.items()},
                "notes": recording.get("notes", ""),
            }
        )

    if not output_rows:
        raise ValueError("No accepted recordings were found")
    write_csv(args.output, output_rows, OUTPUT_FIELDS)
    print(f"Prepared {len(output_rows)} stimuli")
    print(f"Manifest: {args.output}")
    print(f"Audio root: {prepared_root}")


if __name__ == "__main__":
    main()
