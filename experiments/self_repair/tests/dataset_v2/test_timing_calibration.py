from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "dataset_v2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_timing_calibration import analyze_calibration  # noqa: E402
from common import CONDITIONS, read_config  # noqa: E402


DELAYED = (
    "delayed_neutral",
    "delayed_one_dependency",
    "delayed_three_dependencies",
)


def _pre_units(condition: str) -> list[str]:
    return {
        "clean_final": [],
        "immediate_repair": [],
        "delayed_neutral": ["N1", "N2", "N3"],
        "delayed_one_dependency": ["N1", "D1", "N2"],
        "delayed_three_dependencies": ["D1", "D2", "D3"],
    }[condition]


def _scripts() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        rows.append(
            {
                "schema_version": "2.0.0",
                "script_id": f"travel_001__a_to_b__{condition}",
                "text_bundle_id": "travel_001__a_to_b",
                "scenario_id": "travel_001",
                "direction_id": "a_to_b",
                "condition": condition,
                "pre_repair_units": _pre_units(condition),
                "one_dependency_pre_position": (
                    2 if condition == "delayed_one_dependency" else None
                ),
            }
        )
    return rows


def _timing(condition: str, latency: float, post: float) -> dict[str, float | None]:
    old_offset = 1000.0
    cue_onset = old_offset + latency
    new_offset = cue_onset + 350.0
    end = new_offset + post
    return {
        "old_value_offset_ms": None if condition == "clean_final" else old_offset,
        "repair_cue_onset_ms": None if condition == "clean_final" else cue_onset,
        "new_value_offset_ms": 1000.0 if condition == "clean_final" else new_offset,
        "utterance_end_ms": 1000.0 + post if condition == "clean_final" else end,
        "actual_latency_ms": None if condition == "clean_final" else latency,
        "post_final_value_duration_ms": post,
    }


def _candidates() -> list[dict[str, object]]:
    voices = {
        "edge_fast": "en-US-GuyNeural",
        "edge_slow": "en-US-RogerNeural",
    }
    rows: list[dict[str, object]] = []
    delayed_index = {condition: index for index, condition in enumerate(DELAYED)}
    for speaker_index, (speaker_id, voice) in enumerate(voices.items()):
        for condition in CONDITIONS:
            script_id = f"travel_001__a_to_b__{condition}"
            target_id = f"{script_id}__edge_private_calibration_r1__{speaker_id}"
            for attempt in (1, 2):
                if condition in DELAYED:
                    latency = (
                        3000.0
                        + speaker_index * 10.0
                        + delayed_index[condition] * 20.0
                        + (attempt - 1) * 100.0
                    )
                    post = (
                        5000.0
                        + speaker_index * 10.0
                        + delayed_index[condition] * 20.0
                        + (attempt - 1) * 100.0
                    )
                else:
                    latency = 200.0
                    post = 6500.0 + speaker_index * 100.0 + attempt * 10.0
                timing = _timing(condition, latency, post)
                duration = float(timing["utterance_end_ms"]) + 200.0
                rows.append(
                    {
                        "schema_version": "2.0.0",
                        "candidate_id": f"{target_id}__cand{attempt:02d}",
                        "rendition_target_id": target_id,
                        "script_id": script_id,
                        "text_bundle_id": "travel_001__a_to_b",
                        "matched_audio_bundle_id": (
                            f"travel_001__a_to_b__edge_private_calibration_r1__{speaker_id}"
                        ),
                        "source_track_id": "edge_private_calibration_r1",
                        "speaker_id": speaker_id,
                        "voice": voice,
                        "condition": condition,
                        "inferential_role": "engineering_calibration_only",
                        "lifecycle_status": "canonical_candidate",
                        "selected_candidate_id": None,
                        "accepted_audio_id": None,
                        "accepted_utterance": None,
                        "prepared_stimulus": None,
                        "canonical_candidate": {"duration_ms": duration},
                        "timing": timing,
                        "synthesis": {
                            "provider": "edge_private_smoke",
                            "voice": voice,
                        },
                        "qc": {
                            "automatic_status": "passed",
                            "errors": [],
                            "metrics": {"duration_ms": duration},
                        },
                    }
                )
    return rows


class TimingCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = read_config()
        self.scripts = _scripts()
        self.candidates = _candidates()

    def test_report_is_deterministic_private_and_empirically_derived(self) -> None:
        report = analyze_calibration(self.scripts, self.candidates, self.config)
        reordered = analyze_calibration(
            list(reversed(self.scripts)), list(reversed(self.candidates)), self.config
        )
        self.assertEqual(report, reordered)
        self.assertTrue(report["provisional"])
        self.assertTrue(report["private"])
        self.assertFalse(report["release_eligible"])
        self.assertFalse(report["accepted_release_audio_selected"])
        self.assertFalse(report["provenance"]["uses_config_illustrative_target_latency"])

        latency = report["empirical_common_overlap_recommendations"][
            "delayed_actual_latency"
        ]
        recommendation = latency["recommendation"]
        self.assertEqual(recommendation["mode"], "single_global_provisional_target")
        self.assertEqual(recommendation["target_ms"], 3075.0)
        self.assertEqual(recommendation["tolerance_ms"], 15.0)
        self.assertFalse(recommendation["production_timing_freeze_eligible"])

        shifted = copy.deepcopy(self.candidates)
        for row in shifted:
            if row["condition"] in DELAYED:
                timing = row["timing"]
                timing["actual_latency_ms"] += 500.0
                timing["repair_cue_onset_ms"] += 500.0
                timing["new_value_offset_ms"] += 500.0
                timing["utterance_end_ms"] += 500.0
                row["canonical_candidate"]["duration_ms"] += 500.0
                row["qc"]["metrics"]["duration_ms"] += 500.0
        shifted_report = analyze_calibration(self.scripts, shifted, self.config)
        shifted_target = shifted_report["empirical_common_overlap_recommendations"][
            "delayed_actual_latency"
        ]["recommendation"]["target_ms"]
        self.assertEqual(shifted_target, 3575.0)

    def test_voice_condition_duration_and_design_balance_summaries(self) -> None:
        report = analyze_calibration(self.scripts, self.candidates, self.config)
        self.assertEqual(len(report["per_voice_condition"]), 10)
        delayed_fast = next(
            row
            for row in report["per_voice_condition"]
            if row["speaker_id"] == "edge_fast"
            and row["condition"] == "delayed_neutral"
        )
        self.assertEqual(delayed_fast["metrics"]["actual_latency_ms"]["count"], 2)
        self.assertEqual(
            delayed_fast["metrics"]["post_final_value_duration_ms"]["median"],
            5050.0,
        )
        self.assertEqual(
            delayed_fast["metrics"]["utterance_duration_ms"]["count"], 2
        )

        balance = report["design_balance"]
        delayed_neutral = next(
            row for row in balance["by_condition"] if row["condition"] == "delayed_neutral"
        )
        delayed_one = next(
            row
            for row in balance["by_condition"]
            if row["condition"] == "delayed_one_dependency"
        )
        delayed_three = next(
            row
            for row in balance["by_condition"]
            if row["condition"] == "delayed_three_dependencies"
        )
        self.assertEqual(
            delayed_neutral["pre_repair_unit_counts_by_binding"],
            {"dependent": 0, "neutral": 6},
        )
        self.assertEqual(
            delayed_one["pre_repair_unit_counts_by_binding"],
            {"dependent": 2, "neutral": 4},
        )
        self.assertEqual(
            delayed_three["pre_repair_unit_counts_by_binding"],
            {"dependent": 6, "neutral": 0},
        )
        self.assertEqual(
            balance["delayed_one_dependency_balance"]["dependent_pre_position_counts"],
            {"1": 0, "2": 2, "3": 0},
        )

    def test_qc_and_timing_failures_are_excluded_and_reported(self) -> None:
        candidates = copy.deepcopy(self.candidates)
        failed_qc_id = candidates[0]["candidate_id"]
        candidates[0]["qc"] = {
            "automatic_status": "failed",
            "errors": ["digital clipping detected"],
            "metrics": candidates[0]["qc"]["metrics"],
        }
        missing_timing_id = candidates[1]["candidate_id"]
        candidates[1]["timing"] = None
        report = analyze_calibration(self.scripts, candidates, self.config)
        failures = {
            item["reason"]: item for item in report["failures"]["by_reason"]
        }
        self.assertEqual(
            failures["automatic_qc_failed"]["candidate_ids"], [failed_qc_id]
        )
        self.assertEqual(failures["timing_missing"]["candidate_ids"], [missing_timing_id])
        self.assertEqual(report["failures"]["excluded_candidate_count"], 2)
        self.assertEqual(report["coverage"]["eligible_candidate_count"], 18)

    def test_rejects_any_attempt_to_treat_private_edge_rows_as_release_audio(self) -> None:
        candidates = copy.deepcopy(self.candidates)
        candidates[0]["lifecycle_status"] = "accepted"
        candidates[0]["accepted_audio_id"] = "not-allowed"
        with self.assertRaisesRegex(ValueError, "lifecycle_status=canonical_candidate"):
            analyze_calibration(self.scripts, candidates, self.config)

        unsafe_config = copy.deepcopy(self.config)
        unsafe_config["engineering_calibration"]["release_eligible"] = True
        with self.assertRaisesRegex(ValueError, "release_eligible=false"):
            analyze_calibration(self.scripts, self.candidates, unsafe_config)


if __name__ == "__main__":
    unittest.main()
