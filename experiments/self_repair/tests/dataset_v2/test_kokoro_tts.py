from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import wave


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts/dataset_v2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_config, read_jsonl  # noqa: E402
from select_kokoro_voice_calibration import select_targets  # noqa: E402
from synthesize_candidates import synthesize  # noqa: E402


DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset_v2"


class FakeKokoroEngine:
    speed = 1.0
    revision = "1" * 40
    model_sha256 = "2" * 64
    voice_hashes = {"af_fixture": "3" * 64}

    def synthesize(self, text: str, voice: str, wav_path: Path):
        self.asserted = (text, voice)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(b"\0\0" * 2400)
        return (
            [
                {
                    "type": "word",
                    "text": "Hello",
                    "offset_ms": 10.0,
                    "duration_ms": 40.0,
                    "timing_source": "kokoro_predicted_duration_seed",
                }
            ],
            {
                "model_revision": self.revision,
                "model_sha256": self.model_sha256,
                "voice_sha256": self.voice_hashes[voice],
                "device": "cpu",
            },
        )


class KokoroTtsTests(unittest.TestCase):
    def test_voice_calibration_is_exact_shared_script_matrix(self) -> None:
        targets, report = select_targets(
            read_jsonl(DATASET_ROOT / "generated/scripts.jsonl"),
            read_config(DATASET_ROOT / "config/dataset.yaml"),
        )
        self.assertEqual(len(targets), 10)
        self.assertEqual(len({row["script_id"] for row in targets}), 1)
        self.assertEqual(len({row["voice"] for row in targets}), 10)
        self.assertEqual(report["candidate_count"], 10)
        self.assertRegex(report["model_revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(report["model_sha256"], r"^[0-9a-f]{64}$")

    def test_kokoro_candidate_records_pinned_request_and_checkpoints(self) -> None:
        target = {
            "schema_version": "2.0.0",
            "rendition_target_id": "fixture__kokoro__speaker",
            "script_id": "fixture",
            "source_track_id": "kokoro",
            "speaker_id": "speaker",
            "voice": "af_fixture",
        }
        script = {
            "script_id": "fixture",
            "transcript": "Hello world",
            "segments": [
                {"segment_index": 0, "role": "closing_prompt", "text": "Hello world"}
            ],
        }
        engine = FakeKokoroEngine()
        checkpoints = []
        with tempfile.TemporaryDirectory() as temporary:
            rows = synthesize(
                [target],
                [script],
                "kokoro_local_v1_0",
                1,
                Path(temporary),
                kokoro_engine=engine,
                checkpoint=checkpoints.append,
            )
        self.assertEqual(rows, checkpoints)
        self.assertEqual(len(rows), 1)
        synthesis = rows[0]["synthesis"]
        self.assertEqual(synthesis["candidate_policy"], "deterministic_single_candidate_then_bundle_level_retry")
        self.assertEqual(synthesis["model_revision"], engine.revision)
        self.assertEqual(
            synthesis["provider_artifact"]["voice_sha256"], "3" * 64
        )

    def test_kokoro_rejects_duplicate_deterministic_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "deterministic"):
                synthesize(
                    [],
                    [],
                    "kokoro_local_v1_0",
                    2,
                    Path(temporary),
                    kokoro_engine=FakeKokoroEngine(),
                )


if __name__ == "__main__":
    unittest.main()
