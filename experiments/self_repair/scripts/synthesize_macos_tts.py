#!/usr/bin/env python3
from __future__ import annotations

import argparse
from array import array
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

from common import EXPERIMENT_ROOT, read_csv, write_csv, write_json
from generate_recording_manifest import FIELDS as RECORDING_FIELDS


SPEAKERS = {
    "S01": {"voice": "Yuna", "rate": 180},
    "S02": {"voice": "Eddy (한국어(한국))", "rate": 175},
}

# Each tuple is (kind, value, role). Silence values are milliseconds.
PLANS: dict[str, list[tuple[str, str | int, str]]] = {
    "K1": [("speech", "서울 날씨 알려줘", "clean")],
    "K2": [("speech", "부산 날씨 알려줘", "clean")],
    "K3": [
        ("speech", "부산 날씨", "reparandum"),
        ("silence", 200, "gap"),
        ("speech", "아니", "marker"),
        ("silence", 80, "gap"),
        ("speech", "서울 날씨 알려줘", "repair"),
    ],
    "K4": [
        ("speech", "서울 날씨", "reparandum"),
        ("silence", 200, "gap"),
        ("speech", "아니", "marker"),
        ("silence", 80, "gap"),
        ("speech", "부산 날씨 알려줘", "repair"),
    ],
    "K5": [
        ("speech", "부산 날씨", "reparandum"),
        ("silence", 800, "gap"),
        ("speech", "아니", "marker"),
        ("silence", 80, "gap"),
        ("speech", "서울 날씨 알려줘", "repair"),
    ],
    "K6": [
        ("speech", "서울 날씨", "reparandum"),
        ("silence", 800, "gap"),
        ("speech", "아니", "marker"),
        ("silence", 80, "gap"),
        ("speech", "부산 날씨 알려줘", "repair"),
    ],
    "K7": [("speech", "부산과 서울 날씨 둘 다 알려줘", "both")],
    "K8": [
        ("speech", "아니", "discourse_marker"),
        ("silence", 80, "gap"),
        ("speech", "부산과 서울 둘 다 날씨 알려줘", "both"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize the K1-K8 Korean smoke set with macOS voices."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=EXPERIMENT_ROOT / "data/raw",
    )
    parser.add_argument(
        "--recordings",
        type=Path,
        default=EXPERIMENT_ROOT / "data/recordings.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=EXPERIMENT_ROOT / "data/tts_metadata.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_pcm(path: Path) -> tuple[wave._wave_params, bytes]:
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        frames = handle.readframes(params.nframes)
    if params.nchannels != 1 or params.sampwidth != 2 or params.framerate != 24000:
        raise ValueError(f"Unexpected TTS WAV format for {path}: {params}")
    return params, frames


def trim_silence(frames: bytes, threshold: int = 180, pad_ms: int = 20) -> bytes:
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    active = [index for index, value in enumerate(samples) if abs(value) >= threshold]
    if not active:
        raise ValueError("TTS segment contains no detectable speech")
    pad = round(24000 * pad_ms / 1000)
    start = max(0, active[0] - pad)
    end = min(len(samples), active[-1] + pad + 1)
    trimmed = array("h", samples[start:end])
    if sys.byteorder != "little":
        trimmed.byteswap()
    return trimmed.tobytes()


def synthesize_segment(
    text: str,
    voice: str,
    rate: int,
    temporary_dir: Path,
    cache: dict[tuple[str, int, str], bytes],
) -> bytes:
    key = (voice, rate, text)
    if key in cache:
        return cache[key]
    path = temporary_dir / f"segment_{len(cache):03d}.wav"
    subprocess.run(
        [
            "say",
            "-v",
            voice,
            "-r",
            str(rate),
            "-o",
            str(path),
            "--file-format=WAVE",
            "--data-format=LEI16@24000",
            "--channels=1",
            text,
        ],
        check=True,
    )
    _, frames = read_pcm(path)
    cache[key] = trim_silence(frames)
    return cache[key]


def synthesize_plan(
    plan: list[tuple[str, str | int, str]],
    voice: str,
    rate: int,
    temporary_dir: Path,
    cache: dict[tuple[str, int, str], bytes],
) -> tuple[bytes, dict[str, float | None]]:
    chunks: list[bytes] = []
    sample_offset = 0
    timings: dict[str, float | None] = {
        "repair_marker_onset_ms": None,
        "repair_onset_ms": None,
        "repair_end_ms": None,
        "user_end_ms": None,
    }
    for kind, value, role in plan:
        if kind == "silence":
            sample_count = round(24000 * int(value) / 1000)
            chunk = b"\x00\x00" * sample_count
        else:
            if role == "marker":
                timings["repair_marker_onset_ms"] = sample_offset * 1000.0 / 24000
            if role == "repair":
                timings["repair_onset_ms"] = sample_offset * 1000.0 / 24000
            chunk = synthesize_segment(
                str(value), voice, rate, temporary_dir, cache
            )
            sample_count = len(chunk) // 2
        chunks.append(chunk)
        sample_offset += sample_count
        if role == "repair":
            timings["repair_end_ms"] = sample_offset * 1000.0 / 24000
    timings["user_end_ms"] = sample_offset * 1000.0 / 24000
    return b"".join(chunks), timings


def write_pcm(path: Path, frames: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(frames)


def update_recordings(
    path: Path, generated: dict[tuple[str, str], dict[str, object]]
) -> None:
    existing = read_csv(path) if path.exists() else []
    by_key = {(row["speaker_id"], row["condition_id"]): row for row in existing}
    for (speaker_id, condition_id), metadata in generated.items():
        row = by_key.get(
            (speaker_id, condition_id),
            {
                "trial_id": f"{speaker_id}__{condition_id}",
                "speaker_id": speaker_id,
                "condition_id": condition_id,
                "insert_silence_at_ms": "",
                "insert_silence_ms": "",
                "accepted": "1",
                "notes": "",
            },
        )
        row["raw_audio_path"] = str(metadata["relative_path"])
        for field in (
            "repair_marker_onset_ms",
            "repair_onset_ms",
            "repair_end_ms",
            "user_end_ms",
        ):
            value = metadata[field]
            row[field] = "" if value is None else f"{float(value):.3f}"
        row["accepted"] = "1"
        row["notes"] = (
            f"macOS say TTS; voice={metadata['voice']}; rate={metadata['rate']}"
        )
        by_key[(speaker_id, condition_id)] = row
    rows = sorted(by_key.values(), key=lambda row: (row["speaker_id"], row["condition_id"]))
    write_csv(path, rows, RECORDING_FIELDS)


def main() -> None:
    args = parse_args()
    if shutil.which("say") is None:
        raise RuntimeError("macOS `say` command was not found")
    generated: dict[tuple[str, str], dict[str, object]] = {}
    cache: dict[tuple[str, int, str], bytes] = {}
    with tempfile.TemporaryDirectory(prefix="moshi_tts_") as temporary:
        temporary_dir = Path(temporary)
        for speaker_id, speaker in SPEAKERS.items():
            voice = str(speaker["voice"])
            rate = int(speaker["rate"])
            for condition_id, plan in PLANS.items():
                output_path = args.raw_root / speaker_id / f"{condition_id}.wav"
                if output_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"Refusing to overwrite {output_path}; pass --overwrite"
                    )
                frames, timings = synthesize_plan(
                    plan, voice, rate, temporary_dir, cache
                )
                write_pcm(output_path, frames)
                try:
                    relative_path = output_path.relative_to(EXPERIMENT_ROOT)
                except ValueError:
                    relative_path = output_path
                generated[(speaker_id, condition_id)] = {
                    "speaker_id": speaker_id,
                    "condition_id": condition_id,
                    "voice": voice,
                    "rate": rate,
                    "relative_path": str(relative_path),
                    "duration_ms": len(frames) / 2 * 1000.0 / 24000,
                    **timings,
                }
                print(
                    f"{speaker_id}/{condition_id}: "
                    f"{generated[(speaker_id, condition_id)]['duration_ms']:.0f} ms"
                )
    update_recordings(args.recordings, generated)
    write_json(
        args.metadata,
        {
            "engine": "macOS say",
            "sample_rate": 24000,
            "sample_width_bytes": 2,
            "speakers": SPEAKERS,
            "items": list(generated.values()),
        },
    )
    print(f"Generated {len(generated)} WAV files under {args.raw_root}")
    print(f"Updated recording manifest: {args.recordings}")
    print(f"TTS metadata: {args.metadata}")


if __name__ == "__main__":
    main()
