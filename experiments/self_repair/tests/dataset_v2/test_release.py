from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts/dataset_v2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_release import (  # noqa: E402
    CONFIG_FILES,
    CONDITIONS,
    DOC_FILES,
    FULL_AUDIO_RELEASE,
    FULL_EVIDENCE_FILES,
    FULL_PUBLIC_FILES,
    REQUIRED_APPROVAL_GATES,
    SCHEMA_FILES,
    TEXT_DEVELOPMENT,
    ReleaseError,
    build_release,
    canonical_json,
    collect_source_hashes,
    sha256_value,
    sha256_file,
)
from verify_release import verify_release  # noqa: E402
from alignment_evidence import (  # noqa: E402
    alignment_input_binding_sha256,
    frozen_transcript_sha256,
    independent_alignment_payload_sha256,
)
from build_eval_adapter import build_eval_trials, generation_parameters  # noqa: E402
from ids import prepared_stimulus_id  # noqa: E402
from run_eval_v2 import BackendOutput, _completed_row  # noqa: E402
from timing import derived_timing, shift_events  # noqa: E402


COMMIT = "a" * 40
RESOLVED_REVISION = "b" * 40


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def fixture_config() -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
        "dataset_id": "release_fixture_v2",
        "language": "en",
        "domain": "travel",
        "counts": {
            "scenarios": 1,
            "text_bundles": 2,
            "scripts": 10,
            "matched_audio_bundles_per_track": 4,
            "rendition_targets_per_track": 20,
        },
        "conditions": list(CONDITIONS),
        "timing": {"status": "frozen"},
        "source_tracks": {
            "tts_release": {
                "status": "release_approved",
                "speaker_count": 2,
                "speakers": [
                    {"speaker_id": "voice01", "voice": "voice-one"},
                    {"speaker_id": "voice02", "voice": "voice-two"},
                ],
            }
        },
        "evaluation": {
            "generation_seeds": [17, 29],
            "annotations_per_trial": 2,
        },
    }


def fixture_eval_config() -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
        "model_repo": "kyutai/moshi-fixture",
        "device": "cuda",
        "dtype": "bfloat16",
        "cfg_coef": 1.0,
        "generation_seeds": [17, 29],
        "generation": {
            "use_sampling": True,
            "temp": 0.8,
            "temp_text": 0.7,
            "top_k": 250,
            "top_k_text": 25,
        },
        "streaming": {
            "input_sample_rate": 24000,
            "mimi_frame_samples": 1920,
            "prefix_silence_ms": 480,
            "response_capture_ms": 320.0,
            "reset_model_stream_between_trials": True,
            "reset_rng_for_each_trial_seed": True,
        },
    }


def fixture_timing(condition: str, *, prepared: bool = False) -> dict[str, float | None]:
    if condition == "clean_final":
        events: dict[str, float | None] = {
            "old_value_onset_ms": None,
            "old_value_offset_ms": None,
            "repair_cue_onset_ms": None,
            "new_value_onset_ms": 100.0,
            "new_value_offset_ms": 200.0,
            "repeated_old_onset_ms": None,
            "repeated_old_offset_ms": None,
            "repair_cue_offset_ms": None,
            "closing_prompt_onset_ms": 900.0,
            "closing_prompt_offset_ms": 1300.0,
            "utterance_end_ms": 1400.0,
        }
    else:
        events = {
            "old_value_onset_ms": 100.0,
            "old_value_offset_ms": 200.0,
            "repair_cue_onset_ms": 400.0,
            "new_value_onset_ms": 500.0,
            "new_value_offset_ms": 600.0,
            "repeated_old_onset_ms": 650.0,
            "repeated_old_offset_ms": 700.0,
            "repair_cue_offset_ms": 750.0,
            "closing_prompt_onset_ms": 900.0,
            "closing_prompt_offset_ms": 1300.0,
            "utterance_end_ms": 1400.0,
        }
    timing = derived_timing(condition, events)
    return shift_events(timing, 480.0) if prepared else timing


def blueprint() -> dict[str, object]:
    dependent = [
        {
            "unit_id": f"D{index}",
            "text": f"dependent statement {index}",
            "relation": f"dependent.{index}",
            "binding": "root_dependent",
            "state_patch": {f"dependent.{index}": "{root_value}"},
            "balance_pair_id": f"P{index}",
            "speech_act": "statement",
            "boundary_type": "nonterminal",
        }
        for index in range(1, 4)
    ]
    neutral = [
        {
            "unit_id": f"N{index}",
            "text": f"neutral statement {index}",
            "relation": f"neutral.{index}",
            "binding": "root_invariant",
            "state_patch": {f"neutral.{index}": f"value-{index}"},
            "balance_pair_id": f"P{index}",
            "speech_act": "statement",
            "boundary_type": "nonterminal",
        }
        for index in range(1, 4)
    ]
    return {
        "schema_version": "2.0.0",
        "scenario_id": "travel_001",
        "context_label": "fixture",
        "language": "en",
        "domain": "travel",
        "root_slot": "destination",
        "value_a": "Boston",
        "value_b": "Seattle",
        "root_template": "I am going to {value}",
        "dependent_units": dependent,
        "neutral_units": neutral,
        "repair_template": "Sorry, I mean {new}, not {old}",
        "closing_prompt": "Could you help me plan all of that?",
        "gold_state_template": {"destination": "{root_value}"},
        "one_dependency_unit": "D1",
        "one_dependency_pre_position": 1,
        "rotation_id": "R1",
        "reviews": [
            {
                "reviewer_id": "reviewer_a",
                "decision": "approved",
                "reviewed_at": "2026-08-26T00:00:00Z",
            },
            {
                "reviewer_id": "reviewer_b",
                "decision": "approved",
                "reviewed_at": "2026-08-26T00:01:00Z",
            },
        ],
        "review_status": "approved",
        "source": {"authoring_method": "fixture", "license": "CC0-1.0"},
    }


def create_text_fixture(root: Path) -> dict[str, object]:
    root.mkdir(parents=True)
    (root / "VERSION").write_text("2.0.0\n", encoding="utf-8")
    for relative in DOC_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(f"# {relative}\n\nRelease fixture documentation.\n", encoding="utf-8")
    config = fixture_config()
    write_json(root / CONFIG_FILES[0], config)
    write_json(root / CONFIG_FILES[1], fixture_eval_config())
    write_json(root / CONFIG_FILES[2], {"schema_version": "2.0.0", "values": {}})
    write_json(
        root / CONFIG_FILES[3],
        {
            "schema_version": "2.0.0",
            "status": "pending_user_decision",
            "provider": "azure_speech_s0",
        },
    )
    for relative in SCHEMA_FILES:
        write_json(root / relative, {"$schema": "https://json-schema.org/draft/2020-12/schema"})

    bp = blueprint()
    write_jsonl(root / "blueprints/scenarios.jsonl", [bp])
    bp_hash = sha256_value(bp)
    config_hash = sha256_value(config)
    scripts: list[dict[str, object]] = []
    for direction in ("a_to_b", "b_to_a"):
        bundle_id = f"travel_001__{direction}"
        for condition in CONDITIONS:
            transcript = f"fixture transcript {direction} {condition}"
            scripts.append(
                {
                    "schema_version": "2.0.0",
                    "scenario_id": "travel_001",
                    "direction_id": direction,
                    "text_bundle_id": bundle_id,
                    "script_id": f"{bundle_id}__{condition}",
                    "condition": condition,
                    "blueprint_hash": bp_hash,
                    "config_hash": config_hash,
                    "transcript": transcript,
                    "segments": [
                        {"segment_index": 0, "text": transcript},
                    ],
                }
            )
    write_jsonl(root / "generated/scripts.jsonl", scripts)
    answer_keys = [
        {
            "schema_version": "2.0.0",
            "answer_key_id": f"travel_001__{direction}",
            "scenario_id": "travel_001",
            "direction_id": direction,
        }
        for direction in ("a_to_b", "b_to_a")
    ]
    write_jsonl(root / "answer_keys/answer_keys.jsonl", answer_keys)

    folds = [
        {
            "schema_version": "2.0.0",
            "scenario_id": "travel_001",
            "analysis_fold": 1,
            "inferential_role": "confirmatory_evaluation",
        }
    ]
    bundles: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    recording: list[dict[str, object]] = []
    position_by_speaker = {"voice01": 0, "voice02": 0}
    for direction in ("a_to_b", "b_to_a"):
        text_bundle_id = f"travel_001__{direction}"
        for speaker in ("voice01", "voice02"):
            matched_id = f"{text_bundle_id}__tts_release__{speaker}"
            script_ids = [f"{text_bundle_id}__{condition}" for condition in CONDITIONS]
            target_ids = [f"{script_id}__tts_release__{speaker}" for script_id in script_ids]
            voice = "voice-one" if speaker == "voice01" else "voice-two"
            bundles.append(
                {
                    "schema_version": "2.0.0",
                    "matched_audio_bundle_id": matched_id,
                    "text_bundle_id": text_bundle_id,
                    "scenario_id": "travel_001",
                    "direction_id": direction,
                    "source_track_id": "tts_release",
                    "speaker_id": speaker,
                    "voice": voice,
                    "script_ids": script_ids,
                    "rendition_target_ids": target_ids,
                    "analysis_fold": 1,
                    "inferential_role": "confirmatory_evaluation",
                }
            )
            for script_id, condition in zip(script_ids, CONDITIONS):
                target_id = f"{script_id}__tts_release__{speaker}"
                target = {
                    "schema_version": "2.0.0",
                    "rendition_target_id": target_id,
                    "script_id": script_id,
                    "text_bundle_id": text_bundle_id,
                    "matched_audio_bundle_id": matched_id,
                    "scenario_id": "travel_001",
                    "direction_id": direction,
                    "condition": condition,
                    "source_track_id": "tts_release",
                    "speaker_id": speaker,
                    "voice": voice,
                    "analysis_fold": 1,
                    "inferential_role": "confirmatory_evaluation",
                }
                targets.append(target)
                position_by_speaker[speaker] += 1
                recording.append(
                    {
                        **target,
                        "recording_order_id": (
                            f"tts_release__{speaker}__position_"
                            f"{position_by_speaker[speaker]:03d}"
                        ),
                        "recording_position": position_by_speaker[speaker],
                    }
                )
    write_jsonl(root / "assignments/analysis_folds.jsonl", folds)
    write_jsonl(root / "assignments/speaker_bundles.jsonl", bundles)
    write_jsonl(root / "assignments/rendition_targets.jsonl", targets)
    write_jsonl(root / "assignments/recording_order.jsonl", recording)
    return {"config": config, "scripts": scripts, "targets": targets}


def add_full_inputs(root: Path, fixture: dict[str, object]) -> None:
    (root / "LICENSE").write_text("CC0 1.0 Universal\n", encoding="utf-8")
    selection_policy = {
        "schema_version": "2.0.0",
        "status": "frozen",
        "policy_version": "fixture-selection-v1",
        "outcome_blind": True,
        "alignment_gate": {
            "minimum_aggregate_confidence": 0.9,
            "require_calibrated_confidence": True,
            "allow_audited_manual_review": True,
        },
    }
    selection_policy["policy_hash"] = sha256_value(selection_policy)
    write_json(root / FULL_EVIDENCE_FILES["selection_policy"], selection_policy)
    timing_policy = {
        "schema_version": "2.0.0",
        "status": "frozen",
        "policy_version": "fixture-timing-v1",
        "dataset_config_canonical_sha256": sha256_value(fixture["config"]),
        "selection_policy_hash": selection_policy["policy_hash"],
    }
    timing_policy["policy_hash"] = sha256_value(timing_policy)
    write_json(root / FULL_EVIDENCE_FILES["timing_policy"], timing_policy)
    accepted: list[dict[str, object]] = []
    prepared: list[dict[str, object]] = []
    for target in fixture["targets"]:  # type: ignore[index]
        target = dict(target)  # type: ignore[arg-type]
        target_id = str(target["rendition_target_id"])
        accepted_id = f"{target_id}__accepted"
        candidate_id = f"{target_id}__cand01"
        condition = str(target["condition"])
        artifact_hash = hashlib.sha256(accepted_id.encode()).hexdigest()
        script = next(
            row for row in fixture["scripts"] if row["script_id"] == target["script_id"]  # type: ignore[index]
        )
        transcript_hash = frozen_transcript_sha256(script)  # type: ignore[arg-type]
        alignment_run_id = "fixture-alignment-run"
        alignment_tool = "fixture-aligner"
        alignment_version = "1"
        alignment_model = "fixture-model"
        input_binding = alignment_input_binding_sha256(
            candidate_id=candidate_id,
            script_id=str(target["script_id"]),
            canonical_audio_sha256=artifact_hash,
            transcript_sha256=transcript_hash,
            alignment_run_id=alignment_run_id,
            tool=alignment_tool,
            tool_version=alignment_version,
            model_id=alignment_model,
        )
        alignment = {
            "method": alignment_tool,
            "tool": alignment_tool,
            "tool_version": alignment_version,
            "model_id": alignment_model,
            "confidence": {
                "aggregate": 0.99,
                "threshold": 0.9,
                "calibrated": True,
                "threshold_passed": True,
            },
            "independent_forced_alignment": True,
        }
        alignment_payload_hash = independent_alignment_payload_sha256(alignment)
        alignment["external_provenance"] = {
            "candidate_id": candidate_id,
            "script_id": str(target["script_id"]),
            "canonical_audio_sha256": artifact_hash,
            "transcript_sha256": transcript_hash,
            "alignment_run_id": alignment_run_id,
            "tool": alignment_tool,
            "tool_version": alignment_version,
            "model_id": alignment_model,
            "input_binding_sha256": input_binding,
            "transcript_hash_encoding": "exact_utf8_sha256",
            "binding_version": "2.0.0",
            "verified_against_local_inputs": True,
            "external_row_content_sha256": hashlib.sha256(candidate_id.encode()).hexdigest(),
            "alignment_payload_sha256": alignment_payload_hash,
        }
        alignment["manual_review"] = {"required": False, "status": "not_required"}
        accepted.append(
            {
                **target,
                "accepted_audio_id": accepted_id,
                "selected_candidate_id": candidate_id,
                "lifecycle_status": "accepted",
                "canonical_candidate": {
                    "uri": f"private/canonical/{candidate_id}.wav",
                    "sha256": artifact_hash,
                    "duration_ms": 1600.0,
                    "sample_rate": 24000,
                    "channels": 1,
                },
                "accepted_utterance": {
                    "uri": f"audio/accepted/{accepted_id}.wav",
                    "sha256": artifact_hash,
                    "duration_ms": 1600.0,
                    "sample_rate": 24000,
                    "channels": 1,
                    "sample_width_bytes": 2,
                    "timeline": "content_relative",
                    "source_canonical_sha256": artifact_hash,
                },
                "timing": fixture_timing(condition),
                "alignment": alignment,
                "synthesis": {
                    "provider": "fixture-provider",
                    "model": "fixture-model",
                    "voice": target["speaker_id"],
                },
                "selection": {
                    "policy_version": selection_policy["policy_version"],
                    "policy_hash": selection_policy["policy_hash"],
                    "status": "materialized_accepted",
                    "selected_candidate_id": candidate_id,
                    "selected_canonical_sha256": artifact_hash,
                    "alignment_gate_hash": sha256_value(selection_policy["alignment_gate"]),
                    "outcome_blind": True,
                },
                "qc": {
                    "automatic_status": "passed",
                    "manual_status": "passed",
                    "outcome_blind": True,
                    "errors": [],
                },
                "license": {
                    "identifier": "CC0-1.0",
                    "scope": "public redistribution",
                    "redistribution_allowed": True,
                },
            }
        )
        preparation_basis = {
            "sample_rate": 24000,
            "prefix_silence_ms": 480,
            "mimi_frame_samples": 1920,
            "normalization_stage": "accepted_canonical",
        }
        preparation_hash = sha256_value(preparation_basis)
        prepared_id = prepared_stimulus_id(accepted_id, preparation_hash)
        prepared.append(
            {
                **target,
                "prepared_stimulus_id": prepared_id,
                "accepted_audio_id": accepted_id,
                "lifecycle_status": "prepared",
                "preparation_hash": preparation_hash,
                "prepared_stimulus": {
                    "uri": f"audio/prepared/{prepared_id}.wav",
                    "sha256": hashlib.sha256(prepared_id.encode()).hexdigest(),
                    "duration_ms": 2080.0,
                    "sample_rate": 24000,
                    "channels": 1,
                    "sample_width_bytes": 2,
                    "timeline": "prepared_stream_relative",
                },
                "preparation": {
                    **preparation_basis,
                    "prefix_ms_actual": 480.0,
                    "frame_pad_samples": 0,
                },
                "prepared_timing": fixture_timing(condition, prepared=True),
            }
        )
    write_jsonl(root / "manifests/accepted_audio.jsonl", accepted)
    write_jsonl(root / "manifests/prepared_stimuli.jsonl", prepared)

    eval_config = fixture_eval_config()
    pending_trials = build_eval_trials(
        prepared,
        model_repo="kyutai/moshi-fixture",
        resolved_revision=RESOLVED_REVISION,
        generation_config=eval_config,
        code_commit=COMMIT,
        generation_seeds=(17, 29),
        expected_audio_count=len(prepared),
        artifact_root=root,
        allow_nonproduction_matrix=True,
    )
    response_root = root / "evaluation/response_artifacts"
    effective_generation = {
        "lm_gen": generation_parameters(eval_config),
        "cfg_coef": eval_config["cfg_coef"],
        "device": "fixture",
        "dtype": "float32",
    }
    backend_metadata = {
        "name": "fixture",
        "version": "1",
        "model_repo": "kyutai/moshi-fixture",
        "resolved_revision": RESOLVED_REVISION,
        "snapshot_revision": RESOLVED_REVISION,
        "code_commit": COMMIT,
        "model_type": "moshi",
        "mimi_sample_rate": 24000,
        "frame_size": 1920,
        "max_lm_delay": 1,
        "effective_generation_config": effective_generation,
        "effective_generation_config_sha256": sha256_value(effective_generation),
    }
    trials: list[dict[str, object]] = []
    for trial in pending_trials:
        frame_count = int(trial["capture_contract"]["target_end_frame_count"])
        sample_count = int(trial["capture_contract"]["target_end_sample_count"])
        pieces = [""] * frame_count
        pieces[-1] = "private model response omitted from release"
        output = BackendOutput(
            audio=np.zeros(sample_count, dtype=np.float32),
            sample_rate=24000,
            token_ids=[0] * frame_count,
            token_pieces=pieces,
            frame_step_ms=80.0,
            eos_reached=False,
        )
        audio_name = hashlib.sha256(str(trial["eval_trial_id"]).encode()).hexdigest()
        trials.append(
            _completed_row(
                trial,
                output,
                elapsed_seconds=1.0,
                audio_path=response_root / "audio" / f"{audio_name}.wav",
                response_root=response_root,
                appended_zero_sample_count=max(
                    0,
                    sample_count
                    - round(float(trial["input_stimulus"]["duration_ms"]) * 24),
                ),
                backend_metadata=backend_metadata,
            )
        )
    write_jsonl(root / "evaluation/eval_trials.jsonl", trials)

    annotations: list[dict[str, object]] = []
    for trial in trials:
        trial_id = str(trial["eval_trial_id"])
        for annotator in ("annotator_a", "annotator_b"):
            annotations.append(
                {
                    "schema_version": "2.0.0",
                    "blind_id": "blind_" + hashlib.sha256(trial_id.encode()).hexdigest()[:16],
                    "eval_trial_id": trial_id,
                    "annotator_id": annotator,
                    "overall_label": "target_only",
                    "relation_labels": {"D1": "new_bound", "D2": "new_bound", "D3": "new_bound"},
                    "final_target_correct": True,
                    "stale_state_error": False,
                    "assistant_started_before_repair": False,
                    "notes": "",
                    "adjudicator": False,
                }
            )
    write_jsonl(root / "annotations/annotations.jsonl", annotations)

    config_hash = sha256_value(fixture["config"])
    accepted_hash = sha256_file(root / "manifests/accepted_audio.jsonl")
    eval_hash = sha256_file(root / "evaluation/eval_trials.jsonl")
    annotation_hash = sha256_file(root / "annotations/annotations.jsonl")
    common = {
        "schema_version": "2.0.0",
        "dataset_config_canonical_sha256": config_hash,
        "selection_policy_hash": selection_policy["policy_hash"],
        "timing_policy_hash": timing_policy["policy_hash"],
    }
    write_json(
        root / FULL_EVIDENCE_FILES["alignment_report"],
        {
            **common,
            "status": "passed",
            "accepted_manifest_sha256": accepted_hash,
            "accepted_audio_count": len(accepted),
            "eligible_alignment_count": len(accepted),
        },
    )
    write_json(
        root / FULL_EVIDENCE_FILES["audio_qc_report"],
        {
            **common,
            "status": "passed",
            "accepted_manifest_sha256": accepted_hash,
            "accepted_audio_count": len(accepted),
            "automatic_pass_count": len(accepted),
            "unresolved_count": 0,
        },
    )
    write_json(
        root / FULL_EVIDENCE_FILES["double_listen_report"],
        {
            **common,
            "status": "passed",
            "accepted_manifest_sha256": accepted_hash,
            "accepted_audio_count": len(accepted),
            "reviewed_count": 4,
            "unresolved_count": 0,
        },
    )
    analysis_common = {
        **common,
        "status": "completed",
        "accepted_manifest_sha256": accepted_hash,
        "eval_manifest_sha256": eval_hash,
        "annotation_manifest_sha256": annotation_hash,
        "eval_trial_count": len(trials),
        "eval_run_id": trials[0]["eval_run_id"],
    }
    write_json(root / FULL_EVIDENCE_FILES["analysis_result"], analysis_common)
    write_json(root / FULL_EVIDENCE_FILES["baseline_report"], analysis_common)


def write_approval(root: Path, path: Path) -> None:
    write_json(
        path,
        {
            "schema_version": "2.0.0",
            "dataset_version": "2.0.0",
            "release_kind": FULL_AUDIO_RELEASE,
            "status": "approved",
            "public_release": True,
            "approved_git_commit": COMMIT,
            "approved_source_hashes": collect_source_hashes(root, full=True),
            "gates": {gate: "passed" for gate in REQUIRED_APPROVAL_GATES},
            "license": {
                "identifier": "CC0-1.0",
                "scope": "public metadata and audio redistribution",
                "redistribution_allowed": True,
            },
            "approver_id": "kept-private-and-not-packaged",
        },
    )


class ReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.dataset = self.base / "dataset"
        self.fixture = create_text_fixture(self.dataset)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_text_snapshot_is_deterministic_and_explicitly_not_a_release(self) -> None:
        first = self.base / "snapshot-one"
        second = self.base / "snapshot-two"
        manifest = build_release(
            self.dataset, first, kind=TEXT_DEVELOPMENT, git_commit=COMMIT
        )
        build_release(self.dataset, second, kind=TEXT_DEVELOPMENT, git_commit=COMMIT)
        self.assertFalse(manifest["release_eligible"])
        self.assertEqual(manifest["status"], "development_snapshot_not_public_release")
        self.assertEqual(
            (first / "RELEASE_MANIFEST.json").read_bytes(),
            (second / "RELEASE_MANIFEST.json").read_bytes(),
        )
        self.assertEqual(
            (first / "CHECKSUMS.sha256").read_bytes(),
            (second / "CHECKSUMS.sha256").read_bytes(),
        )
        report = verify_release(first)
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["release_eligible"])
        self.assertFalse((first / "manifests").exists())
        self.assertFalse((first / "evaluation").exists())
        self.assertFalse((first / "annotations").exists())

    def test_verifier_detects_tamper_and_unexpected_file(self) -> None:
        tampered = self.base / "tampered"
        build_release(self.dataset, tampered, kind=TEXT_DEVELOPMENT, git_commit=COMMIT)
        with (tampered / "README.md").open("a", encoding="utf-8") as handle:
            handle.write("tamper\n")
        with self.assertRaisesRegex(ReleaseError, "checksum mismatch"):
            verify_release(tampered)

        unexpected = self.base / "unexpected"
        build_release(self.dataset, unexpected, kind=TEXT_DEVELOPMENT, git_commit=COMMIT)
        (unexpected / "surprise.txt").write_text("unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(ReleaseError, "file-set mismatch"):
            verify_release(unexpected)

    def test_full_release_requires_bound_approval_and_strips_private_payloads(self) -> None:
        add_full_inputs(self.dataset, self.fixture)
        with self.assertRaisesRegex(ReleaseError, "release-approval"):
            build_release(
                self.dataset,
                self.base / "not-approved",
                kind=FULL_AUDIO_RELEASE,
                git_commit=COMMIT,
            )

        approval = self.base / "approval.json"
        write_approval(self.dataset, approval)
        first = self.base / "full-one"
        second = self.base / "full-two"
        build_release(
            self.dataset,
            first,
            kind=FULL_AUDIO_RELEASE,
            git_commit=COMMIT,
            approval_path=approval,
        )
        build_release(
            self.dataset,
            second,
            kind=FULL_AUDIO_RELEASE,
            git_commit=COMMIT,
            approval_path=approval,
        )
        report = verify_release(first)
        self.assertTrue(report["release_eligible"])
        self.assertEqual(
            (first / "CHECKSUMS.sha256").read_bytes(),
            (second / "CHECKSUMS.sha256").read_bytes(),
        )
        public_eval = read_lines(first / FULL_PUBLIC_FILES["eval_trials"])
        self.assertTrue(public_eval)
        self.assertTrue(all("response" not in row for row in public_eval))
        public_annotations = read_lines(first / FULL_PUBLIC_FILES["annotations"])
        self.assertTrue(
            all("annotator_id" not in row and "blind_id" not in row for row in public_annotations)
        )
        file_names = {path.name.casefold() for path in first.rglob("*") if path.is_file()}
        self.assertNotIn("raw_candidates.jsonl", file_names)
        self.assertNotIn("private_blind_map.jsonl", file_names)
        self.assertTrue(
            all((first / relative).is_file() for relative in FULL_EVIDENCE_FILES.values())
        )

        with (self.dataset / "README.md").open("a", encoding="utf-8") as handle:
            handle.write("approval-stale\n")
        with self.assertRaisesRegex(ReleaseError, "source hashes"):
            build_release(
                self.dataset,
                self.base / "stale-approval",
                kind=FULL_AUDIO_RELEASE,
                git_commit=COMMIT,
                approval_path=approval,
            )

    def test_full_release_rejects_absolute_audio_uri(self) -> None:
        add_full_inputs(self.dataset, self.fixture)
        accepted_path = self.dataset / "manifests/accepted_audio.jsonl"
        accepted = read_lines(accepted_path)
        accepted[0]["accepted_utterance"]["uri"] = "/tmp/private.wav"
        write_jsonl(accepted_path, accepted)
        approval = self.base / "approval.json"
        write_approval(self.dataset, approval)
        with self.assertRaisesRegex(ReleaseError, "relative POSIX path|artifact root"):
            build_release(
                self.dataset,
                self.base / "absolute-uri",
                kind=FULL_AUDIO_RELEASE,
                git_commit=COMMIT,
                approval_path=approval,
            )

    def test_full_release_rejects_assignment_bundle_permutation(self) -> None:
        add_full_inputs(self.dataset, self.fixture)
        path = self.dataset / "assignments/rendition_targets.jsonl"
        rows = read_lines(path)
        rows[0]["matched_audio_bundle_id"], rows[5]["matched_audio_bundle_id"] = (
            rows[5]["matched_audio_bundle_id"],
            rows[0]["matched_audio_bundle_id"],
        )
        write_jsonl(path, rows)
        approval = self.base / "approval.json"
        write_approval(self.dataset, approval)
        with self.assertRaisesRegex(ReleaseError, "matched-audio bundle|lineage"):
            build_release(
                self.dataset,
                self.base / "assignment-permutation",
                kind=FULL_AUDIO_RELEASE,
                git_commit=COMMIT,
                approval_path=approval,
            )

    def test_full_release_rejects_accepted_and_prepared_metadata_permutation(self) -> None:
        add_full_inputs(self.dataset, self.fixture)
        accepted_path = self.dataset / "manifests/accepted_audio.jsonl"
        rows = read_lines(accepted_path)
        rows[0]["condition"], rows[1]["condition"] = rows[1]["condition"], rows[0]["condition"]
        write_jsonl(accepted_path, rows)
        approval = self.base / "approval.json"
        write_approval(self.dataset, approval)
        with self.assertRaisesRegex(ReleaseError, "condition does not match target"):
            build_release(
                self.dataset,
                self.base / "accepted-permutation",
                kind=FULL_AUDIO_RELEASE,
                git_commit=COMMIT,
                approval_path=approval,
            )

    def test_full_release_rejects_unbound_gate_evidence(self) -> None:
        add_full_inputs(self.dataset, self.fixture)
        path = self.dataset / FULL_EVIDENCE_FILES["audio_qc_report"]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["accepted_manifest_sha256"] = "0" * 64
        write_json(path, report)
        approval = self.base / "approval.json"
        write_approval(self.dataset, approval)
        with self.assertRaisesRegex(ReleaseError, "audio QC report.*approved input"):
            build_release(
                self.dataset,
                self.base / "unbound-evidence",
                kind=FULL_AUDIO_RELEASE,
                git_commit=COMMIT,
                approval_path=approval,
            )

    def test_full_release_rejects_private_payload_in_packaged_gate_evidence(self) -> None:
        add_full_inputs(self.dataset, self.fixture)
        path = self.dataset / FULL_EVIDENCE_FILES["analysis_result"]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["model_outputs"] = [{"transcript": "must stay private"}]
        write_json(path, report)
        approval = self.base / "approval-private-evidence.json"
        write_approval(self.dataset, approval)
        with self.assertRaisesRegex(ReleaseError, "forbidden private response/reviewer fields"):
            build_release(
                self.dataset,
                self.base / "private-gate-evidence",
                kind=FULL_AUDIO_RELEASE,
                git_commit=COMMIT,
                approval_path=approval,
            )

    def test_full_release_rejects_zero_alignment_threshold_and_forged_provenance(self) -> None:
        add_full_inputs(self.dataset, self.fixture)
        policy_path = self.dataset / FULL_EVIDENCE_FILES["selection_policy"]
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy["alignment_gate"]["minimum_aggregate_confidence"] = 0
        policy["policy_hash"] = sha256_value(
            {key: value for key, value in policy.items() if key != "policy_hash"}
        )
        write_json(policy_path, policy)
        approval = self.base / "approval-zero-threshold.json"
        write_approval(self.dataset, approval)
        with self.assertRaisesRegex(ReleaseError, "invalid alignment_gate"):
            build_release(
                self.dataset,
                self.base / "zero-alignment-threshold",
                kind=FULL_AUDIO_RELEASE,
                git_commit=COMMIT,
                approval_path=approval,
            )

        second_dataset = self.base / "dataset-forged-alignment"
        second_fixture = create_text_fixture(second_dataset)
        add_full_inputs(second_dataset, second_fixture)
        accepted_path = second_dataset / "manifests/accepted_audio.jsonl"
        accepted = read_lines(accepted_path)
        accepted[0]["alignment"]["external_provenance"]["input_binding_sha256"] = "0" * 64
        write_jsonl(accepted_path, accepted)
        second_approval = self.base / "approval-forged-alignment.json"
        write_approval(second_dataset, second_approval)
        with self.assertRaisesRegex(ReleaseError, "input binding mismatch"):
            build_release(
                second_dataset,
                self.base / "forged-alignment",
                kind=FULL_AUDIO_RELEASE,
                git_commit=COMMIT,
                approval_path=second_approval,
            )

    def test_full_release_rejects_missing_response_evidence_and_contradictory_label(self) -> None:
        add_full_inputs(self.dataset, self.fixture)
        eval_path = self.dataset / "evaluation/eval_trials.jsonl"
        trials = read_lines(eval_path)
        trials[0].pop("stream_events")
        write_jsonl(eval_path, trials)
        approval = self.base / "approval-response.json"
        write_approval(self.dataset, approval)
        with self.assertRaisesRegex(ReleaseError, "stream_events must be a non-empty list"):
            build_release(
                self.dataset,
                self.base / "missing-response-evidence",
                kind=FULL_AUDIO_RELEASE,
                git_commit=COMMIT,
                approval_path=approval,
            )

        # Recreate a clean fixture in a separate directory for the annotation invariant.
        second_dataset = self.base / "dataset-annotation"
        second_fixture = create_text_fixture(second_dataset)
        add_full_inputs(second_dataset, second_fixture)
        annotations_path = second_dataset / "annotations/annotations.jsonl"
        annotations = read_lines(annotations_path)
        annotations[0]["stale_state_error"] = True
        write_jsonl(annotations_path, annotations)
        second_approval = self.base / "approval-annotation.json"
        write_approval(second_dataset, second_approval)
        with self.assertRaisesRegex(ReleaseError, "stale_state_error contradicts"):
            build_release(
                second_dataset,
                self.base / "contradictory-annotation",
                kind=FULL_AUDIO_RELEASE,
                git_commit=COMMIT,
                approval_path=second_approval,
            )


def read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
