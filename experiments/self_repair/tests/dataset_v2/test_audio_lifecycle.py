from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "dataset_v2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from align_from_boundaries import align_rows  # noqa: E402
from alignment_evidence import (  # noqa: E402
    manual_review_evidence_binding,
    set_manual_review_evidence_binding,
    validate_downstream_alignment_evidence,
)
from audio_utils import duration_ms, read_pcm16_mono, write_pcm16_mono  # noqa: E402
from canonicalize_audio import canonicalize_rows  # noqa: E402
from common import DATASET_ROOT, normalized_text, read_config, sha256_file  # noqa: E402
from import_independent_alignment import (  # noqa: E402
    alignment_input_binding_sha256,
    frozen_transcript_sha256,
    import_alignments,
)
from prepare_v2_stimuli import prepare_rows  # noqa: E402
from select_candidates import (  # noqa: E402
    DEFAULT_INPUT as DEFAULT_SELECTION_INPUT,
    assert_outcome_blind,
    materialize_accepted_rows,
    parse_args as parse_selection_args,
    select_candidate_rows,
    validate_selection_policy,
)
from validate_audio import (  # noqa: E402
    DEFAULT_QC_INPUT,
    parse_args as parse_audio_validation_args,
    validate_audio_lifecycle,
)


def _script() -> dict[str, object]:
    raw_segments = [
        ("initial_old_root", None, "Set the destination to Boston", "root", "destination"),
        ("semantic_unit", "D1", "Book a window seat", "bound", "seat_preference"),
        ("repair_cue", None, "Sorry, I mean Seattle, not Boston", None, None),
        ("semantic_unit", "N1", "Use the morning train", "neutral", "departure_period"),
        ("closing_prompt", None, "Could you help me plan all of that?", None, None),
    ]
    segments = [
        {
            "segment_index": index,
            "role": role,
            "unit_id": unit_id,
            "text": text,
            "binding": binding,
            "relation": relation,
            "boundary_after": "terminal" if role == "closing_prompt" else "nonterminal_semicolon",
        }
        for index, (role, unit_id, text, binding, relation) in enumerate(raw_segments)
    ]
    transcript = "; ".join(str(segment["text"]) for segment in segments)
    return {
        "schema_version": "2.0.0",
        "script_id": "travel_001__a_to_b__delayed_one_dependency",
        "text_bundle_id": "travel_001__a_to_b",
        "scenario_id": "travel_001",
        "direction_id": "a_to_b",
        "condition": "delayed_one_dependency",
        "old_value": "Boston",
        "new_value": "Seattle",
        "segments": segments,
        "pre_repair_units": ["D1"],
        "post_repair_units": ["N1"],
        "transcript": transcript,
        "normalized_transcript": normalized_text(transcript),
    }


def _boundaries(script: dict[str, object], confidence: float) -> list[dict[str, object]]:
    # Deliberately vary case/punctuation; the aligner must map lexical tokens, not
    # compare raw provider strings or global string occurrences.
    words = str(script["normalized_transcript"]).split()
    result: list[dict[str, object]] = []
    for index, word in enumerate(words):
        surface = word.upper() if index % 4 == 0 else word
        if index % 5 == 0:
            surface += ","
        result.append(
            {
                "text": surface,
                "offset_ms": 50.0 + index * 90.0,
                "duration_ms": 70.0,
                "confidence": confidence,
            }
        )
    return result


def _policy(timing: dict[str, object]) -> dict[str, object]:
    latency = float(timing["actual_latency_ms"])
    post = float(timing["post_final_value_duration_ms"])
    targets = {
        "clean_final": {"latency_ms": None, "post_duration_ms": post},
        "immediate_repair": {"latency_ms": latency, "post_duration_ms": post},
        "delayed_neutral": {"latency_ms": latency, "post_duration_ms": post},
        "delayed_one_dependency": {"latency_ms": latency, "post_duration_ms": post},
        "delayed_three_dependencies": {"latency_ms": latency, "post_duration_ms": post},
    }
    return {
        "schema_version": "2.0.0",
        "policy_version": "fixture-v1",
        "status": "frozen",
        "frozen_at": "2026-08-26T12:00:00+09:00",
        "weights": {
            "latency_error": 1.0,
            "post_duration_error": 1.0,
            "alignment_confidence_penalty": 10.0,
            "clipping_penalty": 100.0,
            "noise_penalty": 2.0,
        },
        "scales_ms": {"latency": 100.0, "post_duration": 100.0},
        "targets_by_condition": targets,
        "tie_break": [
            "selection_score",
            "alignment_confidence_desc",
            "canonical_sha256",
            "candidate_id",
        ],
        "require_alignment_review_complete": True,
        "alignment_gate": {
            "minimum_aggregate_confidence": 0.90,
            "require_calibrated_confidence": False,
            "allow_audited_manual_review": True,
        },
        "tail_after_utterance_ms": 200.0,
    }


def _speaker_policy(
    timing: dict[str, object], speaker_ids: tuple[str, ...] = ("tts01", "tts02")
) -> dict[str, object]:
    policy = _policy(timing)
    targets = policy.pop("targets_by_condition")
    policy["policy_version"] = "fixture-speaker-v1"
    policy["timing_target_scope"] = "speaker_specific"
    policy["speaker_ids"] = sorted(speaker_ids)
    policy["targets_by_speaker"] = {
        speaker_id: copy.deepcopy(targets) for speaker_id in sorted(speaker_ids)
    }
    return policy


def _shift_repair_and_following_events(row: dict[str, object], shift_ms: float) -> None:
    timing = row["timing"]
    assert isinstance(timing, dict)
    for key in (
        "repair_cue_onset_ms",
        "new_value_onset_ms",
        "new_value_offset_ms",
        "repeated_old_onset_ms",
        "repeated_old_offset_ms",
        "repair_cue_offset_ms",
        "closing_prompt_onset_ms",
        "closing_prompt_offset_ms",
        "utterance_end_ms",
    ):
        timing[key] = float(timing[key]) + shift_ms
    timing["actual_latency_ms"] = float(timing["actual_latency_ms"]) + shift_ms


def _refresh_completed_review_binding(
    row: dict[str, object], script: dict[str, object]
) -> None:
    binding = set_manual_review_evidence_binding(row, script)
    review = row["alignment"]["manual_review"]
    for entry in review["audit_log"]:
        entry["evidence_binding_sha256"] = binding["evidence_binding_sha256"]


class AudioLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = read_config()
        self.script = _script()
        self.target_id = (
            "travel_001__a_to_b__delayed_one_dependency__tts_controlled_r1__tts01"
        )
        self.raw_rows = self._make_raw_rows()
        canonical = canonicalize_rows(
            self.raw_rows, self.config, self.root / "canonical"
        )
        confidence = {"cand01": 0.90, "cand02": 0.99, "cand03": 0.90}
        for row in canonical:
            suffix = str(row["candidate_id"])[-6:]
            row["synthesis"]["provider_word_boundaries"] = _boundaries(
                self.script, confidence[suffix]
            )
        self.canonical_rows = align_rows(canonical, [self.script])
        for row in self.canonical_rows:
            review = row["alignment"]["manual_review"]
            binding = review["evidence_binding_sha256"]
            review.update(
                {
                    "status": "passed",
                    "reviewer_id": "reviewer_blind_01",
                    "reviewed_at": "2026-08-26T12:30:00+09:00",
                    "audit_log": [
                        {
                            "action": "verified_without_boundary_change",
                            "reviewer_id": "reviewer_blind_01",
                            "evidence_binding_sha256": binding,
                        }
                    ],
                }
            )
            row["qc"] = {
                "automatic_status": "passed",
                "clipping": False,
                "noise_penalty": 0.0 if str(row["candidate_id"]).endswith("cand02") else 0.1,
            }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_raw_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        sample_rate = 16000
        sample_count = round(2.90 * sample_rate)
        time = np.arange(sample_count, dtype=np.float32) / sample_rate
        waveforms = {
            1: (0.07 * np.sin(2 * np.pi * 210 * time)).astype(np.float32),
            2: (0.05 * np.sin(2 * np.pi * 260 * time)).astype(np.float32),
        }
        for index in (1, 2, 3):
            audio = waveforms[1 if index == 3 else index]
            path = self.root / "raw" / f"candidate_{index}.wav"
            write_pcm16_mono(path, audio, sample_rate)
            candidate_id = f"{self.target_id}__cand{index:02d}"
            rows.append(
                {
                    "schema_version": "2.0.0",
                    "rendition_target_id": self.target_id,
                    "candidate_id": candidate_id,
                    "script_id": self.script["script_id"],
                    "text_bundle_id": self.script["text_bundle_id"],
                    "matched_audio_bundle_id": "travel_001__a_to_b__tts_controlled_r1__tts01",
                    "source_track_id": "tts_controlled_r1",
                    "speaker_id": "tts01",
                    "condition": self.script["condition"],
                    "analysis_fold": 1,
                    "inferential_role": "confirmatory_evaluation",
                    "lifecycle_status": "raw_candidate",
                    "raw_candidate": {
                        "uri": str(path.resolve()),
                        "sha256": sha256_file(path),
                        "duration_ms": duration_ms(audio, sample_rate),
                        "sample_rate": sample_rate,
                        "channels": 1,
                        "sample_width_bytes": 2,
                        "timeline": "content_relative",
                    },
                    "synthesis": {
                        "provider": "synthetic-test-provider",
                        "provider_version": "1",
                        "model": "fixture-voice-model",
                        "voice": "fixture01",
                        "request_id": f"request-{index}",
                    },
                }
            )
        return rows

    def test_alignment_maps_occurrences_cue_and_units_with_seed_provenance(self) -> None:
        row = self.canonical_rows[0]
        alignment = row["alignment"]
        self.assertEqual(alignment["method"], "provider_word_boundaries_seed")
        self.assertFalse(alignment["independent_forced_alignment"])
        self.assertTrue(alignment["manual_review"]["required"])
        events = alignment["event_spans"]
        self.assertEqual(events["old_value"]["segment_index"], 0)
        self.assertEqual(events["repair_cue"]["segment_index"], 2)
        self.assertEqual(events["new_value"]["text"].casefold(), "seattle")
        self.assertEqual(events["repeated_old"]["text"].casefold(), "boston")
        self.assertGreater(
            events["repeated_old"]["onset_ms"], events["new_value"]["offset_ms"]
        )
        units = {span["unit_id"]: span for span in alignment["unit_spans"]}
        self.assertEqual(units["D1"]["repair_position"], "pre")
        self.assertGreater(units["D1"]["stale_dependency_age_ms"], 0)
        self.assertEqual(units["N1"]["repair_position"], "post")
        self.assertIsNone(units["N1"]["stale_dependency_age_ms"])

    def test_selection_cli_defaults_to_qc_manifest(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["select_candidates.py", "--policy", "frozen-policy.json"],
        ):
            args = parse_selection_args()
        self.assertEqual(DEFAULT_SELECTION_INPUT, DATASET_ROOT / "manifests/qc_candidates.jsonl")
        self.assertEqual(args.input, DEFAULT_SELECTION_INPUT)
        with patch.object(
            sys,
            "argv",
            ["validate_audio.py", "--policy", "frozen-policy.json"],
        ):
            validation_args = parse_audio_validation_args()
        self.assertEqual(DEFAULT_QC_INPUT, DATASET_ROOT / "manifests/qc_candidates.jsonl")
        self.assertEqual(validation_args.canonical, DEFAULT_QC_INPUT)

    def test_independent_import_verifies_local_audio_transcript_and_run_binding(self) -> None:
        candidate = copy.deepcopy(self.canonical_rows[0])
        candidate_id = str(candidate["candidate_id"])
        script_id = str(candidate["script_id"])
        audio_hash = sha256_file(Path(candidate["canonical_candidate"]["uri"]))
        transcript_hash = frozen_transcript_sha256(self.script)
        run = {
            "alignment_run_id": "mfa-run-20260826-001",
            "tool": "montreal_forced_aligner",
            "tool_version": "3.3.0",
            "model_id": "english_us_arpa",
        }
        binding_hash = alignment_input_binding_sha256(
            candidate_id=candidate_id,
            script_id=script_id,
            canonical_audio_sha256=audio_hash,
            transcript_sha256=transcript_hash,
            **run,
        )
        external = {
            "candidate_id": candidate_id,
            "script_id": script_id,
            "words": _boundaries(self.script, 0.96),
            "aggregate_confidence": 0.96,
            "minimum_word_confidence": 0.91,
            "confidence_kind": "fixture_probability",
            "confidence_calibrated": True,
            "audio_sha256": audio_hash,
            "transcript_sha256": transcript_hash,
            "input_binding_sha256": binding_hash,
            **run,
        }
        imported = import_alignments(
            [candidate],
            [self.script],
            [external],
            minimum_confidence=0.90,
            **run,
        )
        self.assertEqual(len(imported), 1)
        alignment = imported[0]["alignment"]
        self.assertTrue(alignment["independent_forced_alignment"])
        self.assertTrue(alignment["confidence"]["threshold_passed"])
        provenance = alignment["external_provenance"]
        self.assertTrue(provenance["verified_against_local_inputs"])
        self.assertEqual(provenance["canonical_audio_sha256"], audio_hash)
        self.assertEqual(provenance["transcript_sha256"], transcript_hash)
        # The independent boundary rows must replace, not silently lose to, any
        # synthesis-provider boundaries retained on the canonical row.
        self.assertEqual(
            alignment["transcript_mapping"][0]["provider_confidence"], 0.96
        )
        policy = _policy(imported[0]["timing"])
        self.assertEqual(
            validate_downstream_alignment_evidence(
                imported[0], self.script, policy["alignment_gate"]
            ),
            [],
        )
        selected, _ = select_candidate_rows(imported, policy, [self.script])
        self.assertEqual(selected[0]["candidate_id"], candidate_id)

        flag_only = copy.deepcopy(candidate)
        flag_only["alignment"]["method"] = "montreal_forced_aligner"
        flag_only["alignment"]["independent_forced_alignment"] = True
        flag_only["alignment"]["confidence"] = {
            "aggregate": 0.99,
            "threshold": 0.90,
            "threshold_passed": True,
            "calibrated": True,
        }
        flag_only["alignment"].pop("external_provenance", None)
        with self.assertRaisesRegex(ValueError, "external_provenance is missing"):
            select_candidate_rows([flag_only], policy, [self.script])

        manipulated_flags = copy.deepcopy(imported)
        manipulated_flags[0]["alignment"]["confidence"]["aggregate"] = 0.999
        manipulated_flags[0]["alignment"]["confidence"]["threshold_passed"] = True
        with self.assertRaisesRegex(ValueError, "alignment payload hash mismatch"):
            select_candidate_rows(manipulated_flags, policy, [self.script])

        mismatched_threshold_policy = copy.deepcopy(policy)
        mismatched_threshold_policy["alignment_gate"][
            "minimum_aggregate_confidence"
        ] = 0.91
        with self.assertRaisesRegex(ValueError, "importer threshold does not equal"):
            select_candidate_rows(imported, mismatched_threshold_policy, [self.script])

        uncalibrated_external = copy.deepcopy(external)
        uncalibrated_external["confidence_calibrated"] = False
        uncalibrated = import_alignments(
            [candidate],
            [self.script],
            [uncalibrated_external],
            minimum_confidence=0.90,
            **run,
        )
        calibrated_policy = copy.deepcopy(policy)
        calibrated_policy["alignment_gate"]["require_calibrated_confidence"] = True
        with self.assertRaisesRegex(ValueError, "confidence is not calibrated"):
            select_candidate_rows(uncalibrated, calibrated_policy, [self.script])

        tampering = (
            ("audio_sha256", "0" * 64, "audio_sha256"),
            ("transcript_sha256", "1" * 64, "transcript_sha256"),
            ("alignment_run_id", "another-run", "alignment_run_id"),
            ("tool_version", "9.9.9", "tool_version"),
            ("input_binding_sha256", "2" * 64, "input_binding_sha256"),
        )
        for field, value, pattern in tampering:
            tampered = copy.deepcopy(external)
            tampered[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, pattern):
                import_alignments(
                    [candidate],
                    [self.script],
                    [tampered],
                    minimum_confidence=0.90,
                    **run,
                )

        bad_candidate = copy.deepcopy(candidate)
        bad_candidate["canonical_candidate"]["sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "canonical WAV manifest hash mismatch"):
            import_alignments(
                [bad_candidate],
                [self.script],
                [external],
                minimum_confidence=0.90,
                **run,
            )

    def test_selection_deduplicates_hashes_and_is_outcome_blind(self) -> None:
        policy = _policy(self.canonical_rows[0]["timing"])
        self.assertEqual(len(validate_selection_policy(policy)), 64)
        selected, decisions = select_candidate_rows(
            self.canonical_rows, policy, [self.script]
        )
        self.assertEqual(len(selected), 1)
        self.assertTrue(str(selected[0]["candidate_id"]).endswith("cand02"))
        decision_by_id = {str(row["candidate_id"]): row for row in decisions}
        duplicate = next(
            row for candidate_id, row in decision_by_id.items() if candidate_id.endswith("cand03")
        )
        self.assertEqual(
            duplicate["reason"],
            "duplicate_canonical_hash_of:" + f"{self.target_id}__cand01",
        )
        contaminated = copy.deepcopy(self.canonical_rows[0])
        contaminated["model_response"] = "Boston"
        with self.assertRaisesRegex(ValueError, "model-outcome fields"):
            assert_outcome_blind(contaminated)

        pre_qc = copy.deepcopy(self.canonical_rows)
        for row in pre_qc:
            row["qc"] = {"status": "passed"}
        with self.assertRaisesRegex(ValueError, "automatic_audio_qc_not_passed"):
            select_candidate_rows(pre_qc, policy, [self.script])

    def test_speaker_specific_policy_changes_target_without_changing_global_behavior(self) -> None:
        rows = copy.deepcopy(self.canonical_rows)
        _shift_repair_and_following_events(rows[0], 200.0)
        _refresh_completed_review_binding(rows[0], self.script)

        global_policy = _policy(rows[1]["timing"])
        global_selected, _ = select_candidate_rows(rows, global_policy, [self.script])
        self.assertTrue(str(global_selected[0]["candidate_id"]).endswith("cand02"))
        self.assertEqual(global_selected[0]["selection"]["timing_target_scope"], "global")
        self.assertIsNone(
            global_selected[0]["selection"]["timing_target_speaker_id"]
        )

        speaker_policy = _speaker_policy(rows[1]["timing"])
        speaker_targets = speaker_policy["targets_by_speaker"]
        self.assertIsInstance(speaker_targets, dict)
        speaker_targets["tts01"]["delayed_one_dependency"]["latency_ms"] = float(
            rows[0]["timing"]["actual_latency_ms"]
        )
        speaker_selected, decisions = select_candidate_rows(
            rows, speaker_policy, [self.script]
        )
        self.assertTrue(str(speaker_selected[0]["candidate_id"]).endswith("cand01"))
        selection = speaker_selected[0]["selection"]
        self.assertEqual(selection["timing_target_scope"], "speaker_specific")
        self.assertEqual(selection["timing_target_speaker_id"], "tts01")
        self.assertTrue(
            all(
                decision["timing_target_speaker_id"] == "tts01"
                for decision in decisions
            )
        )

    def test_speaker_policy_validation_is_exact_and_hash_is_deterministic(self) -> None:
        policy = _speaker_policy(self.canonical_rows[0]["timing"])
        policy_hash = validate_selection_policy(policy)
        reordered = copy.deepcopy(policy)
        reordered["targets_by_speaker"] = dict(
            reversed(list(reordered["targets_by_speaker"].items()))
        )
        self.assertEqual(validate_selection_policy(reordered), policy_hash)

        declared = copy.deepcopy(policy)
        declared["policy_hash"] = policy_hash
        self.assertEqual(validate_selection_policy(declared), policy_hash)
        declared["targets_by_speaker"]["tts01"]["immediate_repair"][
            "latency_ms"
        ] += 1
        with self.assertRaisesRegex(ValueError, "policy_hash does not match"):
            validate_selection_policy(declared)

        invalid_policies: list[tuple[str, dict[str, object], str]] = []
        missing = copy.deepcopy(policy)
        missing["targets_by_speaker"].pop("tts02")
        invalid_policies.append(("missing speaker", missing, "missing=\\['tts02'\\]"))

        extra = copy.deepcopy(policy)
        extra["targets_by_speaker"]["tts03"] = copy.deepcopy(
            extra["targets_by_speaker"]["tts01"]
        )
        invalid_policies.append(("extra speaker", extra, "extra=\\['tts03'\\]"))

        missing_condition = copy.deepcopy(policy)
        missing_condition["targets_by_speaker"]["tts01"].pop("clean_final")
        invalid_policies.append(
            ("missing condition", missing_condition, "must contain exactly")
        )

        extra_target_field = copy.deepcopy(policy)
        extra_target_field["targets_by_speaker"]["tts01"]["clean_final"][
            "tolerance_ms"
        ] = 200.0
        invalid_policies.append(
            ("extra target field", extra_target_field, "target must contain exactly")
        )

        non_finite = copy.deepcopy(policy)
        non_finite["targets_by_speaker"]["tts01"]["clean_final"][
            "post_duration_ms"
        ] = float("nan")
        invalid_policies.append(("non-finite target", non_finite, "must be finite"))

        zero_alignment_threshold = copy.deepcopy(policy)
        zero_alignment_threshold["alignment_gate"][
            "minimum_aggregate_confidence"
        ] = 0.0
        invalid_policies.append(
            (
                "zero alignment threshold",
                zero_alignment_threshold,
                "must be finite and in \\(0,1\\]",
            )
        )

        unsorted = copy.deepcopy(policy)
        unsorted["speaker_ids"] = list(reversed(unsorted["speaker_ids"]))
        invalid_policies.append(("unsorted IDs", unsorted, "lexicographic order"))

        contaminated = copy.deepcopy(policy)
        contaminated["audit"] = {"model_output": "Seattle"}
        invalid_policies.append(("outcome contamination", contaminated, "outcome-blind"))

        for label, invalid, pattern in invalid_policies:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, pattern):
                validate_selection_policy(invalid)

    def test_speaker_policy_rejects_undeclared_or_mixed_candidate_speakers(self) -> None:
        policy = _speaker_policy(self.canonical_rows[0]["timing"])
        undeclared = copy.deepcopy(self.canonical_rows)
        for row in undeclared:
            row["speaker_id"] = "tts03"
        with self.assertRaisesRegex(ValueError, "is not declared"):
            select_candidate_rows(undeclared, policy, [self.script])

        mixed = copy.deepcopy(self.canonical_rows)
        mixed[1]["speaker_id"] = "tts02"
        with self.assertRaisesRegex(ValueError, "candidates mix speaker IDs"):
            select_candidate_rows(mixed, policy, [self.script])

    def test_full_lifecycle_preserves_lineage_and_exact_audio_coordinates(self) -> None:
        policy = _policy(self.canonical_rows[0]["timing"])
        selected, _ = select_candidate_rows(self.canonical_rows, policy, [self.script])
        accepted = materialize_accepted_rows(
            selected, self.config, policy, self.root / "accepted", [self.script]
        )
        prepared = prepare_rows(accepted, self.config, self.root / "prepared")
        self.assertEqual(
            validate_audio_lifecycle(
                self.raw_rows,
                self.canonical_rows,
                accepted,
                prepared,
                self.config,
                selection_policy=policy,
                scripts=[self.script],
                enforce_config_counts=False,
            ),
            [],
        )

        accepted_audio, rate = read_pcm16_mono(Path(accepted[0]["accepted_utterance"]["uri"]))
        end_ms = float(accepted[0]["timing"]["utterance_end_ms"])
        self.assertEqual(accepted_audio.size, round((end_ms + 200.0) * rate / 1000.0))
        self.assertEqual(
            accepted[0]["selection"]["tail_policy"]["leading_coordinate_shift_samples"], 0
        )
        self.assertFalse(
            accepted[0]["selection"]["tail_policy"]["frame_padding_applied"]
        )

        prepared_audio, _ = read_pcm16_mono(Path(prepared[0]["prepared_stimulus"]["uri"]))
        prefix_samples = round(
            float(self.config["audio"]["prefix_silence_ms"]) * rate / 1000.0
        )
        self.assertTrue(np.all(prepared_audio[:prefix_samples] == 0))
        self.assertTrue(
            np.array_equal(
                prepared_audio[prefix_samples : prefix_samples + accepted_audio.size],
                accepted_audio,
            )
        )
        self.assertEqual(
            prepared_audio.size % int(self.config["audio"]["mimi_frame_samples"]), 0
        )

    def test_manual_review_binding_rejects_stale_timing_or_alignment(self) -> None:
        policy = _policy(self.canonical_rows[0]["timing"])
        row = copy.deepcopy(self.canonical_rows[0])
        binding = manual_review_evidence_binding(row, self.script)
        self.assertEqual(
            row["alignment"]["manual_review"]["evidence_binding_sha256"],
            binding["evidence_binding_sha256"],
        )
        self.assertEqual(
            binding["transcript_sha256"], frozen_transcript_sha256(self.script)
        )
        self.assertEqual(
            validate_downstream_alignment_evidence(
                row, self.script, policy["alignment_gate"]
            ),
            [],
        )
        missing_script_errors = validate_downstream_alignment_evidence(
            row, None, policy["alignment_gate"]  # type: ignore[arg-type]
        )
        self.assertTrue(
            any(
                "frozen script identity mismatch" in error
                for error in missing_script_errors
            )
        )

        stale_timing = copy.deepcopy(row)
        stale_timing["timing"]["closing_prompt_onset_ms"] += 40.0
        timing_errors = validate_downstream_alignment_evidence(
            stale_timing, self.script, policy["alignment_gate"]
        )
        self.assertTrue(any("binding is stale" in error for error in timing_errors))
        with self.assertRaisesRegex(ValueError, "binding is stale"):
            select_candidate_rows([stale_timing], policy, [self.script])

        stale_alignment = copy.deepcopy(row)
        stale_alignment["alignment"]["source_kind"] = "mutated_after_review"
        alignment_errors = validate_downstream_alignment_evidence(
            stale_alignment, self.script, policy["alignment_gate"]
        )
        self.assertTrue(any("binding is stale" in error for error in alignment_errors))

        changed_script = copy.deepcopy(self.script)
        changed_script["segments"][-1]["text"] = "Could you help me plan every detail?"
        changed_script["transcript"] = "; ".join(
            segment["text"] for segment in changed_script["segments"]
        )
        changed_script["normalized_transcript"] = normalized_text(
            changed_script["transcript"]
        )
        self.assertEqual(changed_script["script_id"], self.script["script_id"])
        script_errors = validate_downstream_alignment_evidence(
            row, changed_script, policy["alignment_gate"]
        )
        self.assertTrue(any("binding is stale" in error for error in script_errors))
        with self.assertRaisesRegex(ValueError, "binding is stale"):
            select_candidate_rows([row], policy, [changed_script])

    def test_validator_rejects_accepted_timing_or_alignment_mutation(self) -> None:
        policy = _policy(self.canonical_rows[0]["timing"])
        selected, _ = select_candidate_rows(self.canonical_rows, policy, [self.script])
        accepted = materialize_accepted_rows(
            selected,
            self.config,
            policy,
            self.root / "accepted_exact_lineage",
            [self.script],
        )
        prepared = prepare_rows(
            accepted, self.config, self.root / "prepared_exact_lineage"
        )

        changed_timing = copy.deepcopy(accepted)
        changed_timing[0]["timing"]["closing_prompt_onset_ms"] += 40.0
        timing_errors = validate_audio_lifecycle(
            self.raw_rows,
            self.canonical_rows,
            changed_timing,
            prepared,
            self.config,
            selection_policy=policy,
            scripts=[self.script],
            enforce_config_counts=False,
        )
        self.assertTrue(
            any("accepted timing does not exactly match selected QC row" in error for error in timing_errors)
        )

        changed_alignment = copy.deepcopy(accepted)
        changed_alignment[0]["alignment"]["source_kind"] = "mutated_after_selection"
        alignment_errors = validate_audio_lifecycle(
            self.raw_rows,
            self.canonical_rows,
            changed_alignment,
            prepared,
            self.config,
            selection_policy=policy,
            scripts=[self.script],
            enforce_config_counts=False,
        )
        self.assertTrue(
            any("accepted alignment does not exactly match selected QC row" in error for error in alignment_errors)
        )

    def test_validator_detects_prepared_prefix_corruption(self) -> None:
        policy = _policy(self.canonical_rows[0]["timing"])
        selected, _ = select_candidate_rows(self.canonical_rows, policy, [self.script])
        accepted = materialize_accepted_rows(
            selected, self.config, policy, self.root / "accepted", [self.script]
        )
        prepared = prepare_rows(accepted, self.config, self.root / "prepared")
        prepared_path = Path(prepared[0]["prepared_stimulus"]["uri"])
        audio, rate = read_pcm16_mono(prepared_path)
        audio[0] = 0.25
        write_pcm16_mono(prepared_path, audio, rate)
        prepared[0]["prepared_stimulus"]["sha256"] = sha256_file(prepared_path)
        errors = validate_audio_lifecycle(
            self.raw_rows,
            self.canonical_rows,
            accepted,
            prepared,
            self.config,
            selection_policy=policy,
            scripts=[self.script],
            enforce_config_counts=False,
        )
        self.assertTrue(any("exact prefix + accepted + frame pad" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
