#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

from common import EXPERIMENT_ROOT, read_csv, write_csv, write_json
from generate_recording_manifest import FIELDS as RECORDING_FIELDS

try:
    import edge_tts
except ImportError as error:
    raise SystemExit(
        "edge-tts is required. Install experiments/self_repair/requirements-tts.txt"
    ) from error


SPEAKERS = {
    "EN01": {
        "voice": "en-US-AvaMultilingualNeural",
        "rate": "-3%",
        "pitch": "+0Hz",
    },
    "EN02": {
        "voice": "en-US-AndrewMultilingualNeural",
        "rate": "-2%",
        "pitch": "+0Hz",
    },
}


@dataclass(frozen=True)
class Source:
    text: str
    marker_word: str | None = None
    marker_occurrence: int = 1
    repair_word: str | None = None
    repair_occurrence: int = 1
    repair_end_word: str | None = None
    repair_end_occurrence: int = 1


SOURCES = {
    "E1": Source("Could you tell me what the weather's like in Seoul?"),
    "E2": Source("Could you tell me what the weather's like in Busan?"),
    # No punctuation is placed before the repair marker. The neural model creates
    # one continuous prosodic contour; the controlled pause is inserted later.
    "E3": Source(
        "Could you tell me what the weather's like in Busan uh, sorry, I meant Seoul?",
        marker_word="uh",
        repair_word="I",
        repair_end_word="Seoul",
    ),
    "E4": Source(
        "Could you tell me what the weather's like in Seoul uh, sorry, I meant Busan?",
        marker_word="uh",
        repair_word="I",
        repair_end_word="Busan",
    ),
    "E7": Source(
        "Could you tell me what the weather's like in both Busan and Seoul?"
    ),
    "E8": Source(
        "Could you tell me what the weather's like in Busan actually, could you check both Busan and Seoul?",
        marker_word="actually",
        repair_word="could",
        repair_occurrence=2,
        repair_end_word="Seoul",
    ),
    "E9": Source(
        "Actually, could you tell me what the weather's like in both Busan and Seoul?"
    ),
}

# E5/E6 reuse the exact E3/E4 neural waveform. Only inserted silence differs.
CONDITIONS = {
    "E1": ("E1", 0),
    "E2": ("E2", 0),
    "E3": ("E3", 200),
    "E4": ("E4", 200),
    "E5": ("E3", 800),
    "E6": ("E4", 800),
    "E7": ("E7", 0),
    "E8": ("E8", 200),
    "E9": ("E9", 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize E1-E9 as continuous neural English speech."
    )
    parser.add_argument(
        "--raw-root", type=Path, default=EXPERIMENT_ROOT / "data/raw_en"
    )
    parser.add_argument(
        "--recordings",
        type=Path,
        default=EXPERIMENT_ROOT / "data/recordings.en.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=EXPERIMENT_ROOT / "data/tts_metadata.en.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    if ffmpeg := shutil.which("ffmpeg"):
        command = [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(mp3_path),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-sample_fmt",
            "s16",
            str(wav_path),
        ]
    elif afconvert := shutil.which("afconvert"):
        command = [
            afconvert,
            "-f",
            "WAVE",
            "-d",
            "LEI16@24000",
            "-c",
            "1",
            str(mp3_path),
            str(wav_path),
        ]
    else:
        raise RuntimeError("ffmpeg or macOS afconvert is required")
    subprocess.run(command, check=True)


def synthesize_source(
    source: Source,
    voice: str,
    rate: str,
    pitch: str,
    temporary_dir: Path,
    stem: str,
) -> tuple[bytes, list[dict[str, object]]]:
    mp3_path = temporary_dir / f"{stem}.mp3"
    wav_path = temporary_dir / f"{stem}.wav"
    boundaries: list[dict[str, object]] = []
    communicate = edge_tts.Communicate(
        source.text,
        voice,
        rate=rate,
        pitch=pitch,
        boundary="WordBoundary",
    )
    with mp3_path.open("wb") as audio:
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                audio.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                boundaries.append(
                    {
                        "text": chunk["text"],
                        "offset_ms": float(chunk["offset"]) / 10_000,
                        "duration_ms": float(chunk["duration"]) / 10_000,
                    }
                )
    if not boundaries:
        raise RuntimeError(f"No word boundaries returned for: {source.text}")
    convert_mp3_to_wav(mp3_path, wav_path)
    with wave.open(str(wav_path), "rb") as handle:
        if (
            handle.getnchannels(),
            handle.getsampwidth(),
            handle.getframerate(),
        ) != (1, 2, 24000):
            raise ValueError(f"Unexpected decoded WAV format: {handle.getparams()}")
        frames = handle.readframes(handle.getnframes())
    return frames, boundaries


def find_boundary(
    boundaries: list[dict[str, object]], word: str, occurrence: int
) -> dict[str, object]:
    matches = [
        boundary
        for boundary in boundaries
        if str(boundary["text"]).casefold() == word.casefold()
    ]
    if occurrence < 1 or occurrence > len(matches):
        raise ValueError(f"Could not find occurrence {occurrence} of {word!r}")
    return matches[occurrence - 1]


def add_pause(
    frames: bytes, at_ms: float | None, pause_ms: int
) -> tuple[bytes, int | None]:
    if not pause_ms:
        return frames, None
    if at_ms is None:
        raise ValueError("A pause requires a marker boundary")
    sample_index = round(at_ms * 24000 / 1000)
    byte_index = max(0, min(len(frames), sample_index * 2))
    silence = b"\x00\x00" * round(pause_ms * 24000 / 1000)
    return frames[:byte_index] + silence + frames[byte_index:], byte_index // 2


def write_wav(path: Path, frames: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(frames)


def shifted_ms(boundary: dict[str, object] | None, pause_ms: int) -> float | None:
    if boundary is None:
        return None
    return float(boundary["offset_ms"]) + pause_ms


def update_recordings(
    path: Path, generated: dict[tuple[str, str], dict[str, object]]
) -> None:
    existing = read_csv(path) if path.exists() else []
    by_key = {(row["speaker_id"], row["condition_id"]): row for row in existing}
    for key, metadata in generated.items():
        speaker_id, condition_id = key
        row = by_key.get(
            key,
            {
                "trial_id": f"{speaker_id}__{condition_id}",
                "speaker_id": speaker_id,
                "condition_id": condition_id,
            },
        )
        row.update(
            {
                "raw_audio_path": metadata["relative_path"],
                "repair_marker_onset_ms": metadata["repair_marker_onset_ms"],
                "repair_onset_ms": metadata["repair_onset_ms"],
                "repair_end_ms": metadata["repair_end_ms"],
                "user_end_ms": metadata["user_end_ms"],
                # The neural WAV already contains the controlled pause. These
                # columns are only for pauses that prepare_stimuli.py must add.
                "insert_silence_at_ms": "",
                "insert_silence_ms": "",
                "accepted": "1",
                "notes": (
                    f"neural TTS; voice={metadata['voice']}; "
                    f"rate={metadata['rate']}; continuous utterance"
                ),
            }
        )
        by_key[key] = row
    rows = sorted(by_key.values(), key=lambda row: (row["speaker_id"], row["condition_id"]))
    write_csv(path, rows, RECORDING_FIELDS)


def main() -> None:
    args = parse_args()
    for speaker_id in SPEAKERS:
        for condition_id in CONDITIONS:
            path = args.raw_root / speaker_id / f"{condition_id}.wav"
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")

    generated: dict[tuple[str, str], dict[str, object]] = {}
    metadata_items: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="moshi_neural_tts_") as temporary:
        temporary_dir = Path(temporary)
        for speaker_id, settings in SPEAKERS.items():
            source_cache: dict[str, tuple[bytes, list[dict[str, object]]]] = {}
            for source_id, source in SOURCES.items():
                source_cache[source_id] = synthesize_source(
                    source,
                    str(settings["voice"]),
                    str(settings["rate"]),
                    str(settings["pitch"]),
                    temporary_dir,
                    f"{speaker_id}_{source_id}",
                )
            for condition_id, (source_id, pause_ms) in CONDITIONS.items():
                source = SOURCES[source_id]
                base_frames, boundaries = source_cache[source_id]
                marker = (
                    find_boundary(boundaries, source.marker_word, source.marker_occurrence)
                    if source.marker_word
                    else None
                )
                repair = (
                    find_boundary(boundaries, source.repair_word, source.repair_occurrence)
                    if source.repair_word
                    else None
                )
                repair_end = (
                    find_boundary(
                        boundaries,
                        source.repair_end_word,
                        source.repair_end_occurrence,
                    )
                    if source.repair_end_word
                    else None
                )
                marker_base_ms = float(marker["offset_ms"]) if marker else None
                frames, inserted_at_sample = add_pause(
                    base_frames, marker_base_ms, pause_ms
                )
                output_path = args.raw_root / speaker_id / f"{condition_id}.wav"
                write_wav(output_path, frames)
                try:
                    relative_path = output_path.relative_to(EXPERIMENT_ROOT)
                except ValueError:
                    relative_path = output_path
                item = {
                    "speaker_id": speaker_id,
                    "condition_id": condition_id,
                    "source_id": source_id,
                    "text": source.text,
                    **settings,
                    "relative_path": str(relative_path),
                    "duration_ms": len(frames) / 2 * 1000 / 24000,
                    "repair_marker_onset_ms": shifted_ms(marker, pause_ms),
                    "repair_onset_ms": shifted_ms(repair, pause_ms),
                    "repair_end_ms": (
                        shifted_ms(repair_end, pause_ms)
                        + float(repair_end["duration_ms"])
                        if repair_end
                        else None
                    ),
                    "user_end_ms": len(frames) / 2 * 1000 / 24000,
                    "insert_silence_at_ms": (
                        inserted_at_sample * 1000 / 24000
                        if inserted_at_sample is not None
                        else None
                    ),
                    "insert_silence_ms": pause_ms or None,
                    "word_boundaries": [
                        {
                            **boundary,
                            "offset_ms": float(boundary["offset_ms"])
                            + (pause_ms if marker and float(boundary["offset_ms"]) >= marker_base_ms else 0),
                        }
                        for boundary in boundaries
                    ],
                }
                generated[(speaker_id, condition_id)] = item
                metadata_items.append(item)
                print(f"{speaker_id}/{condition_id}: {item['duration_ms']:.0f} ms")

    update_recordings(args.recordings, generated)
    write_json(
        args.metadata,
        {
            "engine": "Microsoft Edge online neural TTS via edge-tts",
            "edge_tts_version": importlib.metadata.version("edge-tts"),
            "language": "en-US",
            "sample_rate": 24000,
            "sample_width_bytes": 2,
            "speakers": SPEAKERS,
            "items": metadata_items,
        },
    )
    print(f"Generated {len(generated)} neural WAV files under {args.raw_root}")
    print(f"Recording manifest: {args.recordings}")
    print(f"TTS metadata: {args.metadata}")


if __name__ == "__main__":
    main()
