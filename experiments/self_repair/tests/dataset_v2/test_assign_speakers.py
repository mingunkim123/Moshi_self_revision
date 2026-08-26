from __future__ import annotations

from collections import Counter, defaultdict
import copy
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "dataset_v2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from assign_speakers import (  # noqa: E402
    build_manifests,
    make_report,
    validate_manifests,
    write_manifests,
)
from common import CONDITIONS, DEFAULT_CONFIG, read_config, read_jsonl, sha256_value  # noqa: E402


SOURCE_TRACK_ID = "tts_kokoro_v1_0_r1"


def synthetic_scripts() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_number in range(1, 31):
        scenario_id = f"travel_{scenario_number:03d}"
        for direction_id in ("a_to_b", "b_to_a"):
            text_bundle_id = f"{scenario_id}__{direction_id}"
            for condition in CONDITIONS:
                rows.append(
                    {
                        "schema_version": "2.0.0",
                        "scenario_id": scenario_id,
                        "direction_id": direction_id,
                        "text_bundle_id": text_bundle_id,
                        "condition": condition,
                        "script_id": f"{text_bundle_id}__{condition}",
                    }
                )
    return rows


class SpeakerAssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = read_config(DEFAULT_CONFIG)
        self.scripts = synthetic_scripts()
        self.manifests = build_manifests(
            self.scripts, self.config, SOURCE_TRACK_ID
        )

    def test_exact_counts_and_balanced_speakers(self) -> None:
        assignments = self.manifests["speaker_bundles"]
        targets = self.manifests["rendition_targets"]
        self.assertEqual(len(assignments), 120)
        self.assertEqual(len(targets), 600)
        self.assertEqual(len({row["matched_audio_bundle_id"] for row in assignments}), 120)
        self.assertEqual(len({row["rendition_target_id"] for row in targets}), 600)

        assignment_counts = Counter(row["speaker_id"] for row in assignments)
        target_counts = Counter(row["speaker_id"] for row in targets)
        self.assertEqual(set(assignment_counts.values()), {12})
        self.assertEqual(set(target_counts.values()), {60})

        direction_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in assignments:
            direction_counts[str(row["speaker_id"])][str(row["direction_id"])] += 1
        self.assertTrue(
            all(counts == Counter({"a_to_b": 6, "b_to_a": 6}) for counts in direction_counts.values())
        )

    def test_each_bundle_has_two_distinct_speakers_and_five_conditions(self) -> None:
        assignments_by_bundle: dict[str, list[dict[str, object]]] = defaultdict(list)
        targets_by_assignment: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in self.manifests["speaker_bundles"]:
            assignments_by_bundle[str(row["text_bundle_id"])].append(row)
        for row in self.manifests["rendition_targets"]:
            targets_by_assignment[str(row["matched_audio_bundle_id"])].append(row)

        self.assertEqual(len(assignments_by_bundle), 60)
        for rows in assignments_by_bundle.values():
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["speaker_id"] for row in rows}), 2)
        for rows in targets_by_assignment.values():
            self.assertEqual(len(rows), 5)
            self.assertEqual({row["condition"] for row in rows}, set(CONDITIONS))

    def test_canonical_ids_include_source_track(self) -> None:
        assignment = self.manifests["speaker_bundles"][0]
        expected_matched = (
            f"{assignment['text_bundle_id']}__{SOURCE_TRACK_ID}__{assignment['speaker_id']}"
        )
        self.assertEqual(assignment["matched_audio_bundle_id"], expected_matched)

        target = self.manifests["rendition_targets"][0]
        expected_target = f"{target['script_id']}__{SOURCE_TRACK_ID}__{target['speaker_id']}"
        self.assertEqual(target["rendition_target_id"], expected_target)

    def test_folds_have_six_scenarios_and_keep_directions_together(self) -> None:
        folds = self.manifests["analysis_folds"]
        self.assertEqual(Counter(row["analysis_fold"] for row in folds), Counter({1: 6, 2: 6, 3: 6, 4: 6, 5: 6}))
        fold_by_scenario = {row["scenario_id"]: row["analysis_fold"] for row in folds}

        observed: dict[str, set[int]] = defaultdict(set)
        directions: dict[str, set[str]] = defaultdict(set)
        for row in self.manifests["speaker_bundles"]:
            scenario_id = str(row["scenario_id"])
            observed[scenario_id].add(int(row["analysis_fold"]))
            directions[scenario_id].add(str(row["direction_id"]))
        for scenario_id in fold_by_scenario:
            self.assertEqual(observed[scenario_id], {fold_by_scenario[scenario_id]})
            self.assertEqual(directions[scenario_id], {"a_to_b", "b_to_a"})

    def test_recording_order_is_complete_and_never_repeats_bundle_adjacent(self) -> None:
        recording_by_speaker: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in self.manifests["recording_order"]:
            recording_by_speaker[str(row["speaker_id"])].append(row)
        self.assertEqual(len(recording_by_speaker), 10)

        target_ids = {
            row["rendition_target_id"] for row in self.manifests["rendition_targets"]
        }
        recording_ids = {
            row["rendition_target_id"] for row in self.manifests["recording_order"]
        }
        self.assertEqual(recording_ids, target_ids)
        for rows in recording_by_speaker.values():
            ordered = sorted(rows, key=lambda row: int(row["recording_position"]))
            self.assertEqual(
                [row["recording_position"] for row in ordered], list(range(1, 61))
            )
            self.assertTrue(
                all(
                    first["text_bundle_id"] != second["text_bundle_id"]
                    for first, second in zip(ordered, ordered[1:])
                )
            )

    def test_deterministic_across_input_order_and_seed_sensitive(self) -> None:
        reversed_manifests = build_manifests(
            list(reversed(self.scripts)), self.config, SOURCE_TRACK_ID
        )
        self.assertEqual(sha256_value(self.manifests), sha256_value(reversed_manifests))

        changed_config = copy.deepcopy(self.config)
        changed_config["generation_seed"] += 1
        changed = build_manifests(self.scripts, changed_config, SOURCE_TRACK_ID)
        self.assertNotEqual(sha256_value(self.manifests), sha256_value(changed))

    def test_validator_and_report_pass(self) -> None:
        self.assertEqual(
            validate_manifests(self.manifests, self.config, SOURCE_TRACK_ID), []
        )
        report = make_report(
            self.manifests, self.config, SOURCE_TRACK_ID
        )
        self.assertEqual(report["validation"]["status"], "passed")
        self.assertEqual(report["counts"]["matched_audio_bundles"], 120)
        self.assertEqual(report["counts"]["rendition_targets"], 600)

    def test_write_jsonl_manifests_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = write_manifests(
                root / "assignments",
                root / "reports" / "assignment.json",
                self.manifests,
                self.config,
                SOURCE_TRACK_ID,
            )
            self.assertEqual(len(read_jsonl(outputs["speaker_bundles"])), 120)
            self.assertEqual(len(read_jsonl(outputs["rendition_targets"])), 600)
            self.assertTrue(outputs["report"].is_file())

    def test_rejects_duplicate_or_incomplete_script_matrix(self) -> None:
        duplicate = [*self.scripts[:-1], self.scripts[0]]
        with self.assertRaisesRegex(ValueError, "duplicate script IDs"):
            build_manifests(duplicate, self.config, SOURCE_TRACK_ID)
        with self.assertRaisesRegex(ValueError, "expected 300 scripts"):
            build_manifests(self.scripts[:-1], self.config, SOURCE_TRACK_ID)

    def test_validator_rejects_target_bundle_permutation(self) -> None:
        tampered = copy.deepcopy(self.manifests)
        first, second = tampered["rendition_targets"][:2]
        first["matched_audio_bundle_id"], second["matched_audio_bundle_id"] = (
            second["matched_audio_bundle_id"],
            first["matched_audio_bundle_id"],
        )
        errors = validate_manifests(tampered, self.config, SOURCE_TRACK_ID)
        self.assertTrue(any("matched audio bundle" in error for error in errors))

    def test_validator_rejects_recording_metadata_permutation(self) -> None:
        tampered = copy.deepcopy(self.manifests)
        first = tampered["recording_order"][0]
        second = next(
            row
            for row in tampered["recording_order"][1:]
            if row["condition"] != first["condition"]
        )
        first["condition"], second["condition"] = second["condition"], first["condition"]
        errors = validate_manifests(tampered, self.config, SOURCE_TRACK_ID)
        self.assertTrue(any("recording row" in error and "condition mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
