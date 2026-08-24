#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

from common import EXPERIMENT_ROOT, write_json
from synthesize_macos_tts import synthesize_plan, update_recordings, write_pcm


SPEAKERS = {
    "EN01": {"voice": "Samantha", "rate": 180},
    "EN02": {"voice": "Eddy (영어(미국))", "rate": 175},
}

PLANS: dict[str, list[tuple[str, str | int, str]]] = {
    "E1": [
        ("speech", "Could you tell me what the weather's like in Seoul", "clean")
    ],
    "E2": [
        ("speech", "Could you tell me what the weather's like in Busan", "clean")
    ],
    "E3": [
        ("speech", "Could you tell me what the weather's like in Busan", "reparandum"),
        ("silence", 200, "gap"),
        ("speech", "uh, sorry", "marker"),
        ("silence", 60, "gap"),
        ("speech", "I meant Seoul", "repair"),
    ],
    "E4": [
        ("speech", "Could you tell me what the weather's like in Seoul", "reparandum"),
        ("silence", 200, "gap"),
        ("speech", "uh, sorry", "marker"),
        ("silence", 60, "gap"),
        ("speech", "I meant Busan", "repair"),
    ],
    "E5": [
        ("speech", "Could you tell me what the weather's like in Busan", "reparandum"),
        ("silence", 800, "gap"),
        ("speech", "uh, sorry", "marker"),
        ("silence", 60, "gap"),
        ("speech", "I meant Seoul", "repair"),
    ],
    "E6": [
        ("speech", "Could you tell me what the weather's like in Seoul", "reparandum"),
        ("silence", 800, "gap"),
        ("speech", "uh, sorry", "marker"),
        ("silence", 60, "gap"),
        ("speech", "I meant Busan", "repair"),
    ],
    "E7": [
        (
            "speech",
            "Could you tell me what the weather's like in both Busan and Seoul",
            "both",
        )
    ],
    "E8": [
        ("speech", "Could you tell me what the weather's like in Busan", "reparandum"),
        ("silence", 200, "gap"),
        ("speech", "actually", "marker"),
        ("silence", 60, "gap"),
        ("speech", "could you check both Busan and Seoul", "repair"),
    ],
    "E9": [
        ("speech", "Actually", "discourse_marker"),
        ("silence", 60, "gap"),
        (
            "speech",
            "could you tell me what the weather's like in both Busan and Seoul",
            "both",
        ),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize the E1-E9 English Moshi experiment with macOS voices."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=EXPERIMENT_ROOT / "data/raw_en",
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


def main() -> None:
    args = parse_args()
    if shutil.which("say") is None:
        raise RuntimeError("macOS `say` command was not found")
    generated: dict[tuple[str, str], dict[str, object]] = {}
    cache: dict[tuple[str, int, str], bytes] = {}
    with tempfile.TemporaryDirectory(prefix="moshi_en_tts_") as temporary:
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
            "language": "en_US",
            "sample_rate": 24000,
            "sample_width_bytes": 2,
            "speakers": SPEAKERS,
            "items": list(generated.values()),
        },
    )
    print(f"Generated {len(generated)} WAV files under {args.raw_root}")
    print(f"English recording manifest: {args.recordings}")
    print(f"English TTS metadata: {args.metadata}")


if __name__ == "__main__":
    main()
