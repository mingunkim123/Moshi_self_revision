from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "dataset_v2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audio_utils import read_pcm16_mono, write_pcm16_mono  # noqa: E402
from common import read_jsonl, sha256_file, write_jsonl  # noqa: E402
from materialize_provisional_eval_audio import run_materialization  # noqa: E402


class ProvisionalMaterializationTests(unittest.TestCase):
    def _source_row(self, root: Path) -> dict[str, object]:
        wav = root / "canonical.wav"
        t = np.arange(16800, dtype=np.float32) / 24000.0
        write_pcm16_mono(wav, 0.08 * np.sin(2.0 * np.pi * 220.0 * t), 24000)
        return {
            "schema_version": "2.0.0",
            "candidate_id": "travel_001__a_to_b__clean_final__tts_test_v1__tts01__cand01",
            "rendition_target_id": "travel_001__a_to_b__clean_final__tts_test_v1__tts01",
            "script_id": "travel_001__a_to_b__clean_final",
            "text_bundle_id": "travel_001__a_to_b",
            "matched_audio_bundle_id": "travel_001__a_to_b__tts_test_v1__tts01",
            "scenario_id": "travel_001",
            "direction_id": "a_to_b",
            "source_track_id": "tts_test_v1",
            "speaker_id": "tts01",
            "condition": "clean_final",
            "lifecycle_status": "canonical_candidate",
            "canonical_candidate": {
                "uri": str(wav),
                "sha256": sha256_file(wav),
                "duration_ms": 700.0,
                "sample_rate": 24000,
                "channels": 1,
                "sample_width_bytes": 2,
                "timeline": "content_relative",
            },
            "qc": {"automatic_status": "passed", "errors": [], "outcome_blind": True},
            "alignment": {
                "independent_forced_alignment": True,
                "confidence": {"aggregate": 0.0, "calibrated": False, "threshold_passed": False},
                "manual_review": {
                    "required": True,
                    "status": "pending",
                    "reviewer_id": None,
                    "reviewed_at": None,
                    "audit_log": [],
                },
            },
            "timing": {
                "old_value_onset_ms": None,
                "old_value_offset_ms": None,
                "repair_cue_onset_ms": None,
                "repair_cue_offset_ms": None,
                "repeated_old_onset_ms": None,
                "repeated_old_offset_ms": None,
                "new_value_onset_ms": 100.0,
                "new_value_offset_ms": 200.0,
                "closing_prompt_onset_ms": 300.0,
                "closing_prompt_offset_ms": 400.0,
                "utterance_end_ms": 500.0,
                "actual_latency_ms": None,
                "post_final_value_duration_ms": 300.0,
                "post_repair_duration_ms": 300.0,
                "post_cue_duration_ms": None,
            },
        }

    def test_materializes_without_claiming_review_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "qc.jsonl"
            write_jsonl(source, [self._source_row(root)])
            accepted_path = root / "accepted.jsonl"
            prepared_path = root / "prepared.jsonl"
            waiver_path = root / "waiver.json"
            report_path = root / "report.json"
            accepted, prepared = run_materialization(
                input_path=source,
                accepted_output=accepted_path,
                prepared_output=prepared_path,
                accepted_root=root / "accepted",
                prepared_root=root / "prepared",
                waiver_path=waiver_path,
                report_path=report_path,
                expected_count=1,
                acknowledge_missing_review_record=True,
            )
            self.assertEqual(len(accepted), 1)
            self.assertEqual(len(prepared), 1)
            self.assertFalse(accepted[0]["release_eligible"])
            self.assertFalse(prepared[0]["release_eligible"])
            self.assertEqual(prepared[0]["lifecycle_status"], "prepared")
            self.assertFalse(prepared[0]["provisional_engineering"]["all_items_passed_claimed"])
            audio, rate = read_pcm16_mono(Path(prepared[0]["prepared_stimulus"]["uri"]))
            self.assertEqual(rate, 24000)
            self.assertEqual(audio.size % 1920, 0)
            self.assertTrue(np.all(audio[:11520] == 0.0))
            self.assertEqual(len(read_jsonl(accepted_path)), 1)
            self.assertEqual(len(read_jsonl(prepared_path)), 1)

    def test_requires_explicit_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "qc.jsonl"
            write_jsonl(source, [self._source_row(root)])
            with self.assertRaisesRegex(ValueError, "acknowledge-missing-review-record"):
                run_materialization(
                    input_path=source,
                    accepted_output=root / "accepted.jsonl",
                    prepared_output=root / "prepared.jsonl",
                    accepted_root=root / "accepted",
                    prepared_root=root / "prepared",
                    waiver_path=root / "waiver.json",
                    report_path=root / "report.json",
                    expected_count=1,
                    acknowledge_missing_review_record=False,
                )


if __name__ == "__main__":
    unittest.main()
