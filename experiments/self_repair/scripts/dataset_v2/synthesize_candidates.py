#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping

import numpy as np

from common import DATASET_ROOT, DEFAULT_SCRIPTS, read_config, read_jsonl, sha256_file, sha256_value, write_json, write_jsonl
from ids import candidate_id


DEFAULT_TARGETS = DATASET_ROOT / "assignments/rendition_targets.jsonl"
DEFAULT_OUTPUT = DATASET_ROOT / "manifests/raw_candidates.jsonl"
DEFAULT_AUDIO_ROOT = DATASET_ROOT / "artifacts/raw_candidates"
SSML_TEMPLATE_VERSION = "2.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize immutable raw rendition candidates.")
    parser.add_argument(
        "--provider",
        required=True,
        choices=("edge_private_smoke", "azure_speech_s0", "kokoro_local_v1_0"),
    )
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target-id", action="append", dest="target_ids")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output manifest and checkpoint after every candidate.",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=DATASET_ROOT / "artifacts/model_cache",
    )
    return parser.parse_args()


def render_ssml(script: dict[str, Any], voice: str) -> str:
    parts = []
    for segment in script["segments"]:
        mark = f"seg_{segment['segment_index']:02d}_{segment['role']}"
        if segment.get("unit_id"):
            mark += f"_{segment['unit_id']}"
        parts.append(f'<bookmark mark="{html.escape(mark, quote=True)}"/>{html.escape(str(segment["text"]))}')
    body = "; ".join(parts)
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{html.escape(voice, quote=True)}"><prosody rate="0%" pitch="0Hz">'
        f"{body}</prosody></voice></speak>"
    )


def _convert_mp3(mp3_path: Path, wav_path: Path) -> None:
    converter = shutil.which("ffmpeg")
    if converter:
        command = [converter, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3_path), "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(wav_path)]
    elif converter := shutil.which("afconvert"):
        command = [converter, "-f", "WAVE", "-d", "LEI16@24000", "-c", "1", str(mp3_path), str(wav_path)]
    else:
        raise RuntimeError("ffmpeg or macOS afconvert is required for Edge smoke")
    subprocess.run(command, check=True)


def _edge_synthesize(text: str, voice: str, wav_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import edge_tts
    except ImportError as error:
        raise RuntimeError("install requirements-v2.txt") from error
    boundaries: list[dict[str, Any]] = []
    mp3_path = wav_path.with_suffix(".mp3")
    communicate = edge_tts.Communicate(text, voice, rate="+0%", pitch="+0Hz", boundary="WordBoundary")
    with mp3_path.open("wb") as audio:
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                audio.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                boundaries.append(
                    {
                        "type": "word",
                        "text": chunk["text"],
                        "offset_ms": float(chunk["offset"]) / 10_000.0,
                        "duration_ms": float(chunk["duration"]) / 10_000.0,
                    }
                )
    if not boundaries:
        raise RuntimeError("Edge returned no word boundaries")
    _convert_mp3(mp3_path, wav_path)
    provider_artifact = {"uri": str(mp3_path.resolve()), "sha256": sha256_file(mp3_path)}
    return boundaries, provider_artifact


def _azure_synthesize(ssml: str, wav_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as error:
        raise RuntimeError("install requirements-azure-tts.txt") from error
    key = os.environ.get("AZURE_SPEECH_KEY") or os.environ.get("SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")
    if not key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY (or SPEECH_KEY) and AZURE_SPEECH_REGION are required")
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    output_config = speechsdk.audio.AudioOutputConfig(filename=str(wav_path))
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=output_config)
    boundaries: list[dict[str, Any]] = []

    def word(event: Any) -> None:
        duration = event.duration
        duration_ms = (
            duration.total_seconds() * 1000.0
            if hasattr(duration, "total_seconds")
            else float(duration) / 10_000.0
        )
        boundaries.append(
            {
                "type": "word",
                "text": event.text,
                "offset_ms": event.audio_offset / 10_000.0,
                "duration_ms": duration_ms,
                "text_offset": event.text_offset,
                "word_length": event.word_length,
                "boundary_type": str(event.boundary_type),
            }
        )

    def bookmark(event: Any) -> None:
        boundaries.append(
            {"type": "bookmark", "text": event.text, "offset_ms": event.audio_offset / 10_000.0, "duration_ms": 0.0}
        )

    synthesizer.synthesis_word_boundary.connect(word)
    synthesizer.bookmark_reached.connect(bookmark)
    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
        raise RuntimeError(f"Azure synthesis failed: {details.reason}: {details.error_details}")
    if not boundaries:
        raise RuntimeError("Azure returned no timing events")
    return sorted(boundaries, key=lambda item: (item["offset_ms"], item["type"])), {"result_id": result.result_id}


class KokoroLocalEngine:
    """Pinned local Kokoro engine with hash-verified model and voice artifacts."""

    def __init__(self, calibration: Mapping[str, Any], cache_dir: Path):
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from kokoro import KModel, KPipeline
        except ImportError as error:
            raise RuntimeError("install requirements-kokoro-tts.txt") from error

        repo = str(calibration["model_repo"])
        revision = str(calibration["model_revision"])
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Kokoro model_revision must be a full 40-hex commit")

        def pinned_file(filename: str, expected_sha256: str) -> Path:
            path = Path(
                hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    revision=revision,
                    cache_dir=cache_dir,
                )
            )
            observed = sha256_file(path)
            if observed != expected_sha256:
                raise ValueError(
                    f"Kokoro artifact hash mismatch for {filename}: {observed}"
                )
            return path

        self.repo = repo
        self.revision = revision
        self.sample_rate = int(calibration["sample_rate"])
        if self.sample_rate != 24000:
            raise ValueError("Kokoro calibration must use native 24 kHz output")
        self.speed = float(calibration["speed"])
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError("Kokoro speed must be in [0.5, 2.0]")
        config_path = pinned_file(
            str(calibration["config_file"]), str(calibration["config_sha256"])
        )
        model_path = pinned_file(
            str(calibration["model_file"]), str(calibration["model_sha256"])
        )
        self.model_sha256 = str(calibration["model_sha256"])
        self.config_sha256 = str(calibration["config_sha256"])
        self.voice_paths: dict[str, Path] = {}
        self.voice_hashes: dict[str, str] = {}
        for speaker in calibration["speakers"]:
            voice = str(speaker["voice"])
            voice_hash = str(speaker["voice_sha256"])
            if voice in self.voice_paths:
                raise ValueError(f"duplicate Kokoro voice: {voice}")
            self.voice_paths[voice] = pinned_file(f"voices/{voice}.pt", voice_hash)
            self.voice_hashes[voice] = voice_hash

        # CPU is deliberate for the M1/8 GiB local production host.  It avoids
        # device-specific numerical changes and keeps the frozen source track
        # reproducible across local and Linux CPU reruns.
        self.device = "cpu"
        self.model = KModel(
            repo_id=repo, config=str(config_path), model=str(model_path)
        ).to(self.device).eval()
        self.pipeline = KPipeline(
            lang_code="a", repo_id=repo, model=self.model, device=self.device
        )
        self.torch_version = torch.__version__

    def synthesize(
        self, text: str, voice: str, wav_path: Path
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            import soundfile as sf
        except ImportError as error:
            raise RuntimeError("install requirements-kokoro-tts.txt") from error
        voice_path = self.voice_paths.get(voice)
        if voice_path is None:
            raise ValueError(f"voice is not in the frozen Kokoro inventory: {voice}")
        chunks: list[np.ndarray] = []
        boundaries: list[dict[str, Any]] = []
        accumulated_ms = 0.0
        for result in self.pipeline(
            text, voice=str(voice_path), speed=self.speed, split_pattern=None
        ):
            if result.audio is None:
                continue
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError("Kokoro returned empty or non-finite audio")
            for token in result.tokens or []:
                if token.start_ts is None or token.end_ts is None:
                    continue
                onset_ms = accumulated_ms + float(token.start_ts) * 1000.0
                offset_ms = accumulated_ms + float(token.end_ts) * 1000.0
                if offset_ms <= onset_ms:
                    continue
                boundaries.append(
                    {
                        "type": "word",
                        "text": str(token.text),
                        "offset_ms": onset_ms,
                        "duration_ms": offset_ms - onset_ms,
                        "confidence": None,
                        "timing_source": "kokoro_predicted_duration_seed",
                    }
                )
            chunks.append(audio)
            accumulated_ms += audio.size * 1000.0 / self.sample_rate
        if not chunks:
            raise RuntimeError("Kokoro returned no audio chunks")
        waveform = np.concatenate(chunks)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, waveform, self.sample_rate, subtype="PCM_16")
        if not boundaries:
            raise RuntimeError("Kokoro returned no predicted token timings")
        return boundaries, {
            "model_repo": self.repo,
            "model_revision": self.revision,
            "model_sha256": self.model_sha256,
            "config_sha256": self.config_sha256,
            "voice_sha256": self.voice_hashes[voice],
            "torch_version": self.torch_version,
            "device": self.device,
        }


def synthesize(
    targets: list[dict[str, Any]],
    scripts: list[dict[str, Any]],
    provider: str,
    attempts: int,
    audio_root: Path,
    config: Mapping[str, Any] | None = None,
    kokoro_engine: KokoroLocalEngine | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if attempts < 1 or attempts > 5:
        raise ValueError("attempts must be in 1..5")
    if provider == "kokoro_local_v1_0" and attempts != 1:
        raise ValueError(
            "Kokoro is deterministic at frozen settings; use one candidate and "
            "regenerate the complete matched bundle under an approved retry policy"
        )
    if provider == "kokoro_local_v1_0" and kokoro_engine is None:
        raise ValueError("Kokoro provider requires a hash-verified engine")
    script_map = {str(row["script_id"]): row for row in scripts}
    if len(script_map) != len(scripts):
        raise ValueError("duplicate script IDs")
    rows: list[dict[str, Any]] = []
    audio_root.mkdir(parents=True, exist_ok=True)
    for target in targets:
        target_id = str(target["rendition_target_id"])
        script = script_map.get(str(target["script_id"]))
        if script is None:
            raise ValueError(f"{target_id}: unknown script")
        voice = str(target["voice"])
        ssml = render_ssml(script, voice)
        if provider == "kokoro_local_v1_0":
            assert kokoro_engine is not None
            fixed_parameters = {
                "provider": provider,
                "voice": voice,
                "speed": kokoro_engine.speed,
                "style": None,
                "model_revision": kokoro_engine.revision,
                "pause_policy": "model_natural_semicolon_boundaries",
                "candidate_policy": "deterministic_single_candidate_then_bundle_level_retry",
            }
        else:
            fixed_parameters = {
                "provider": provider,
                "voice": voice,
                "rate": "+0%",
                "pitch": "+0Hz",
                "style": None,
                "ssml_template_version": SSML_TEMPLATE_VERSION,
                "pause_policy": "provider_natural_semicolon_boundaries",
            }
        for attempt in range(1, attempts + 1):
            item_id = candidate_id(target_id, attempt)
            wav_path = audio_root / f"{item_id}.wav"
            boundary_path = audio_root / f"{item_id}.boundaries.json"
            if wav_path.exists() or boundary_path.exists():
                raise FileExistsError(f"immutable candidate exists: {item_id}")
            if provider == "edge_private_smoke":
                boundaries, provider_artifact = _edge_synthesize(script["transcript"], voice, wav_path)
                engine_version = importlib.metadata.version("edge-tts")
                request_payload = {"text": script["transcript"], **fixed_parameters}
            elif provider == "azure_speech_s0":
                boundaries, provider_artifact = _azure_synthesize(ssml, wav_path)
                engine_version = importlib.metadata.version("azure-cognitiveservices-speech")
                request_payload = {"ssml": ssml, **fixed_parameters}
            else:
                assert kokoro_engine is not None
                boundaries, provider_artifact = kokoro_engine.synthesize(
                    script["transcript"], voice, wav_path
                )
                engine_version = importlib.metadata.version("kokoro")
                request_payload = {
                    "text": script["transcript"],
                    "model_sha256": kokoro_engine.model_sha256,
                    "voice_sha256": kokoro_engine.voice_hashes[voice],
                    **fixed_parameters,
                }
            write_json(boundary_path, {"schema_version": "2.0.0", "events": boundaries})
            import wave
            with wave.open(str(wav_path), "rb") as handle:
                channels, width, rate, frames = (
                    handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()
                )
            if (channels, width, rate) != (1, 2, 24000):
                raise ValueError(f"{item_id}: provider WAV is not mono PCM16 24kHz")
            row = {
                **target,
                "candidate_id": item_id,
                "selected_candidate_id": None,
                "accepted_audio_id": None,
                "lifecycle_status": "raw_candidate",
                "raw_candidate": {
                    "uri": str(wav_path.resolve()),
                    "sha256": sha256_file(wav_path),
                    "duration_ms": frames * 1000.0 / rate,
                    "sample_rate": rate,
                    "channels": channels,
                    "sample_width_bytes": width,
                    "timeline": "content_relative",
                },
                "canonical_candidate": None,
                "accepted_utterance": None,
                "prepared_stimulus": None,
                "timing": None,
                "alignment": {
                    "status": "provider_events_only_not_independent_alignment",
                    "provider_event_uri": str(boundary_path.resolve()),
                    "provider_event_sha256": sha256_file(boundary_path),
                },
                "synthesis": {
                    **fixed_parameters,
                    "engine_version": engine_version,
                    "candidate_index": attempt,
                    "request_hash": sha256_value(request_payload),
                    "provider_artifact": provider_artifact,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                "selection": None,
                "qc": {"status": "pending"},
            }
            rows.append(row)
            if checkpoint is not None:
                checkpoint(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.resume and args.attempts != 1:
        raise ValueError("checkpoint resume currently requires --attempts 1")
    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"immutable output manifest exists; pass --resume to verify and continue: {args.output}"
        )
    targets = read_jsonl(args.targets)
    if args.target_ids:
        requested = set(args.target_ids)
        targets = [row for row in targets if row["rendition_target_id"] in requested]
        missing = sorted(requested - {row["rendition_target_id"] for row in targets})
        if missing:
            raise ValueError(f"unknown target IDs: {missing}")
    if args.limit is not None:
        targets = targets[: args.limit]
    completed = read_jsonl(args.output) if args.resume and args.output.is_file() else []
    completed_ids = [str(row.get("candidate_id")) for row in completed]
    if len(set(completed_ids)) != len(completed_ids):
        raise ValueError("resume manifest contains duplicate candidate IDs")
    expected_target_ids = {str(row["rendition_target_id"]) for row in targets}
    for row in completed:
        if row.get("rendition_target_id") not in expected_target_ids:
            raise ValueError("resume manifest contains a candidate outside the requested target set")
        raw = row.get("raw_candidate")
        if not isinstance(raw, dict):
            raise ValueError("resume manifest candidate is missing raw_candidate")
        raw_path = Path(str(raw.get("uri", "")))
        if not raw_path.is_file() or raw.get("sha256") != sha256_file(raw_path):
            raise ValueError("resume manifest raw candidate is missing or has a hash mismatch")
    completed_target_ids = {str(row["rendition_target_id"]) for row in completed}
    targets = [
        row for row in targets if str(row["rendition_target_id"]) not in completed_target_ids
    ]
    config = read_config()
    engine = None
    if args.provider == "kokoro_local_v1_0":
        calibration = config.get("open_source_calibration")
        if not isinstance(calibration, dict):
            raise ValueError("dataset config is missing open_source_calibration")
        expected_track = str(calibration["source_track_id"])
        if any(str(row.get("source_track_id")) != expected_track for row in targets):
            raise ValueError("Kokoro targets do not match the frozen calibration source track")
        engine = KokoroLocalEngine(calibration, args.model_cache)
    checkpoint_rows = list(completed)

    def checkpoint(row: dict[str, Any]) -> None:
        checkpoint_rows.append(row)
        write_jsonl(
            args.output,
            sorted(checkpoint_rows, key=lambda item: str(item["candidate_id"])),
        )

    rows = synthesize(
        targets,
        read_jsonl(args.scripts),
        args.provider,
        args.attempts,
        args.audio_root,
        config=config,
        kokoro_engine=engine,
        checkpoint=checkpoint,
    )
    if not rows and not args.output.is_file():
        write_jsonl(args.output, checkpoint_rows)
    print(
        f"Synthesized {len(rows)} new raw candidates; "
        f"manifest total={len(checkpoint_rows)} -> {args.output}"
    )


if __name__ == "__main__":
    main()
