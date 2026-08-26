#!/usr/bin/env python3
"""Recover a private Kokoro raw manifest from immutable audio sidecars.

The first local 10-voice run demonstrated that an 8 GiB host can be terminated
after writing audio but before the aggregate JSONL is flushed.  This command is
limited to the non-release voice-calibration track.  It reconstructs only
deterministic request metadata, verifies every model/voice/file hash, and marks
the recovery explicitly; it must never be used to invent missing production
audio provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import wave
from typing import Any

from common import DATASET_ROOT, DEFAULT_SCRIPTS, read_config, read_jsonl, sha256_file, sha256_value, write_jsonl
from ids import candidate_id


DEFAULT_TARGETS = DATASET_ROOT / "calibration/kokoro_voice_targets.jsonl"
DEFAULT_AUDIO_ROOT = DATASET_ROOT / "artifacts/kokoro_voice_raw"
DEFAULT_OUTPUT = DATASET_ROOT / "release_evidence/kokoro_voice_raw_candidates.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def recover_rows(
    targets: list[dict[str, Any]],
    scripts: list[dict[str, Any]],
    config: dict[str, Any],
    audio_root: Path,
) -> list[dict[str, Any]]:
    calibration = config.get("open_source_calibration")
    if not isinstance(calibration, dict) or calibration.get("release_eligible") is not False:
        raise ValueError("recovery is restricted to non-release open-source calibration")
    if calibration.get("provider") != "kokoro_local_v1_0":
        raise ValueError("unexpected open-source calibration provider")
    scripts_by_id = {str(row["script_id"]): row for row in scripts}
    speakers = {str(row["speaker_id"]): row for row in calibration["speakers"]}
    if len(targets) != 10 or len(speakers) != 10:
        raise ValueError("recovery requires the exact 10-voice audition matrix")
    recovered: list[dict[str, Any]] = []
    for target in targets:
        target_id = str(target["rendition_target_id"])
        item_id = candidate_id(target_id, 1)
        wav_path = audio_root / f"{item_id}.wav"
        boundary_path = audio_root / f"{item_id}.boundaries.json"
        if not wav_path.is_file() or not boundary_path.is_file():
            raise FileNotFoundError(f"{item_id}: incomplete immutable audio/sidecar pair")
        payload = json.loads(boundary_path.read_text(encoding="utf-8"))
        events = payload.get("events") if isinstance(payload, dict) else None
        if not isinstance(events, list) or not events:
            raise ValueError(f"{item_id}: boundary sidecar has no events")
        with wave.open(str(wav_path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
        if (channels, width, rate) != (1, 2, 24000) or frames <= 0:
            raise ValueError(f"{item_id}: expected nonempty mono PCM16 24 kHz WAV")
        script = scripts_by_id.get(str(target["script_id"]))
        speaker = speakers.get(str(target["speaker_id"]))
        if script is None or speaker is None:
            raise ValueError(f"{item_id}: target is not in the frozen script/voice inventory")
        if target.get("voice") != speaker.get("voice"):
            raise ValueError(f"{item_id}: target voice does not match the frozen inventory")
        fixed_parameters = {
            "provider": "kokoro_local_v1_0",
            "voice": target["voice"],
            "speed": float(calibration["speed"]),
            "style": None,
            "model_revision": calibration["model_revision"],
            "pause_policy": "model_natural_semicolon_boundaries",
            "candidate_policy": calibration["candidate_policy"],
        }
        request_payload = {
            "text": script["transcript"],
            "model_sha256": calibration["model_sha256"],
            "voice_sha256": speaker["voice_sha256"],
            **fixed_parameters,
        }
        generated_at = datetime.fromtimestamp(
            wav_path.stat().st_mtime, timezone.utc
        ).isoformat()
        recovered.append(
            {
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
                    "engine_version": importlib.metadata.version("kokoro"),
                    "candidate_index": 1,
                    "request_hash": sha256_value(request_payload),
                    "provider_artifact": {
                        "model_repo": calibration["model_repo"],
                        "model_revision": calibration["model_revision"],
                        "model_sha256": calibration["model_sha256"],
                        "config_sha256": calibration["config_sha256"],
                        "voice_sha256": speaker["voice_sha256"],
                        "device": "cpu",
                    },
                    "generated_at": generated_at,
                    "manifest_recovered_after_interrupted_aggregate_write": True,
                    "recovery_basis": "immutable_wav_and_boundary_hashes_plus_frozen_request",
                },
                "selection": None,
                "qc": {"status": "pending"},
            }
        )
    ids = [row["candidate_id"] for row in recovered]
    if len(set(ids)) != 10:
        raise ValueError("recovered candidate IDs are not unique")
    return sorted(recovered, key=lambda row: row["candidate_id"])


def main() -> None:
    args = parse_args()
    rows = recover_rows(
        read_jsonl(args.targets),
        read_jsonl(args.scripts),
        read_config(),
        args.audio_root,
    )
    write_jsonl(args.output, rows)
    print(f"Recovered {len(rows)} private Kokoro calibration rows -> {args.output}")


if __name__ == "__main__":
    main()
