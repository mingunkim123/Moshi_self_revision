from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts/dataset_v2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_eval_adapter import (  # noqa: E402
    build_eval_trials,
    generation_config_hash,
    generation_parameters,
    guard_output_reuse,
    make_eval_run_id,
    validate_eval_trials,
    write_eval_manifest,
)
from audio_utils import duration_ms, write_pcm16_mono  # noqa: E402
from common import (  # noqa: E402
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
    write_jsonl,
)
from ids import prepared_stimulus_id  # noqa: E402
from make_annotation_sheet_v2 import (  # noqa: E402
    DEFAULT_ACCEPTED_MANIFEST,
    FORBIDDEN_PUBLIC_FIELDS,
    annotation_from_sheet_row,
    build_annotation_package,
    resolve_annotations,
    validate_annotation,
)
from score_results_v2 import frozen_contrast_inference, score_primary  # noqa: E402
from response_validation import (  # noqa: E402
    validate_execution_contract,
    validate_trial_response,
)
from run_eval_v2 import (  # noqa: E402
    BackendOutput,
    _completed_row,
    _snapshot_revision,
    _verify_clean_git_identity,
    main as run_eval_main,
    run_evaluation,
)
from validate_schemas import validate_rows  # noqa: E402


CONDITIONS = (
    "clean_final",
    "immediate_repair",
    "delayed_neutral",
    "delayed_one_dependency",
    "delayed_three_dependencies",
)
SEEDS = (17, 29, 42, 101, 2026)
RESOLVED_REVISION = "1" * 40
CODE_COMMIT = "2" * 40


def eval_config(
    seeds: tuple[int, ...] = SEEDS,
    *,
    sample_rate: int = 24000,
    frame_samples: int = 1920,
    response_capture_ms: float = 320.0,
    temp: float = 0.8,
) -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
        "model_repo": "kyutai/moshi",
        "device": "cuda",
        "dtype": "bfloat16",
        "cfg_coef": 1.0,
        "generation_seeds": list(seeds),
        "generation": {
            "use_sampling": True,
            "temp": temp,
            "temp_text": 0.7,
            "top_k": 250,
            "top_k_text": 25,
        },
        "streaming": {
            "input_sample_rate": sample_rate,
            "mimi_frame_samples": frame_samples,
            "prefix_silence_ms": 480,
            "response_capture_ms": response_capture_ms,
            "reset_model_stream_between_trials": True,
            "reset_rng_for_each_trial_seed": True,
        },
    }


def prepared_timing(condition: str) -> dict[str, float | None]:
    if condition == "clean_final":
        return {
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
            "actual_latency_ms": None,
            "post_final_value_duration_ms": 1200.0,
            "post_repair_duration_ms": 1200.0,
            "post_cue_duration_ms": None,
        }
    return {
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
        "actual_latency_ms": 200.0,
        "post_final_value_duration_ms": 800.0,
        "post_repair_duration_ms": 800.0,
        "post_cue_duration_ms": 650.0,
    }


def accepted_fixture(count: int = 600) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_index in range(1, 31):
        scenario_id = f"travel_{scenario_index:03d}"
        for direction in ("a_to_b", "b_to_a"):
            bundle_id = f"{scenario_id}__{direction}"
            for speaker_number in (1, 2):
                speaker_id = f"tts{speaker_number:02d}"
                matched_id = f"{bundle_id}__tts_controlled_r1__{speaker_id}"
                for condition in CONDITIONS:
                    script_id = f"{bundle_id}__{condition}"
                    rendition_id = f"{script_id}__tts_controlled_r1__{speaker_id}"
                    accepted_id = f"{rendition_id}__accepted"
                    preparation = {
                        "sample_rate": 24000,
                        "prefix_silence_ms": 480,
                        "mimi_frame_samples": 1920,
                        "normalization_stage": "accepted_canonical",
                    }
                    preparation_hash = sha256_value(preparation)
                    rows.append(
                        {
                            "schema_version": "2.0.0",
                            "accepted_audio_id": accepted_id,
                            "rendition_target_id": rendition_id,
                            "script_id": script_id,
                            "text_bundle_id": bundle_id,
                            "matched_audio_bundle_id": matched_id,
                            "scenario_id": scenario_id,
                            "direction_id": direction,
                            "condition": condition,
                            "source_track_id": "tts_controlled_r1",
                            "speaker_id": speaker_id,
                            "lifecycle_status": "prepared",
                            "prepared_stimulus_id": prepared_stimulus_id(
                                accepted_id, preparation_hash
                            ),
                            "preparation_hash": preparation_hash,
                            "prepared_stimulus": {
                                "uri": f"artifacts/prepared/{rendition_id}.wav",
                                "sha256": "b" * 64,
                                "duration_ms": 1600.0,
                                "sample_rate": 24000,
                                "channels": 1,
                                "sample_width_bytes": 2,
                                "timeline": "prepared_stream_relative",
                            },
                            "preparation": preparation,
                            "prepared_timing": prepared_timing(condition),
                        }
                    )
    if count == 600:
        return rows
    return rows[:count]


def build_trials(accepted: list[dict[str, object]]) -> list[dict[str, object]]:
    config = eval_config()
    return build_eval_trials(
        accepted,
        model_repo="kyutai/moshi",
        resolved_revision=RESOLVED_REVISION,
        generation_config=config,
        code_commit=CODE_COMMIT,
        generation_seeds=SEEDS,
        expected_audio_count=len(accepted),
        allow_nonproduction_matrix=len(accepted) != 600,
    )


class FakeEvalBackend:
    def __init__(
        self, identity: dict[str, str], generation_config: dict[str, object]
    ) -> None:
        effective_generation = {
            "lm_gen": generation_parameters(generation_config),
            "cfg_coef": generation_config["cfg_coef"],
            "device": "fake-ci",
            "dtype": "float32",
        }
        self._metadata = {
            "name": "fake-ci-backend",
            "version": "1.0",
            "model_repo": identity["model_repo"],
            "resolved_revision": identity["resolved_revision"],
            "snapshot_revision": identity["resolved_revision"],
            "code_commit": identity["code_commit"],
            "model_type": "moshi",
            "mimi_sample_rate": 24000,
            "frame_size": 1920,
            "max_lm_delay": 1,
            "effective_generation_config": effective_generation,
            "effective_generation_config_sha256": sha256_value(
                effective_generation
            ),
        }
        self.reset_seeds: list[int] = []
        self.inference_inputs: list[np.ndarray] = []
        self._active_seed: int | None = None

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def reset_trial(self, seed: int) -> None:
        self._active_seed = seed
        self.reset_seeds.append(seed)

    def infer(
        self, input_audio: np.ndarray, input_stimulus: dict[str, object]
    ) -> BackendOutput:
        if self._active_seed is None:
            raise AssertionError("reset_trial must be called before inference")
        self.inference_inputs.append(np.asarray(input_audio).copy())
        sample_rate = int(input_stimulus["sample_rate"])
        frame_samples = int(input_stimulus["mimi_frame_samples"])
        frame_count = input_audio.size // frame_samples
        samples = np.arange(input_audio.size, dtype=np.float32) / sample_rate
        frequency = 180 + self._active_seed % 100
        audio = (0.03 * np.sin(2 * np.pi * frequency * samples)).astype(np.float32)
        token_ids = [0] * frame_count
        token_pieces = [""] * frame_count
        token_ids[-1] = 400 + self._active_seed
        token_pieces[-1] = " Seattle"
        return BackendOutput(
            audio=audio,
            sample_rate=sample_rate,
            token_ids=token_ids,
            token_pieces=token_pieces,
            frame_step_ms=80.0,
            eos_reached=False,
        )


def runnable_eval_fixture(
    root: Path, seeds: tuple[int, ...] = (17, 29)
) -> tuple[
    Path,
    Path,
    Path,
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
]:
    prepared = accepted_fixture(1)
    input_path = root / "prepared.wav"
    sample_rate = 24000
    frame_samples = 1920
    samples = np.arange(frame_samples * 20, dtype=np.float32) / sample_rate
    input_audio = (0.04 * np.sin(2 * np.pi * 220 * samples)).astype(np.float32)
    write_pcm16_mono(input_path, input_audio, sample_rate)
    prepared[0]["prepared_stimulus"] = {
        "uri": input_path.name,
        "sha256": sha256_file(input_path),
        "duration_ms": duration_ms(input_audio, sample_rate),
        "sample_rate": sample_rate,
        "channels": 1,
        "sample_width_bytes": 2,
        "timeline": "prepared_stream_relative",
    }
    generation_config = eval_config(seeds)
    trials = build_eval_trials(
        prepared,
        model_repo="kyutai/moshi",
        resolved_revision=RESOLVED_REVISION,
        generation_config=generation_config,
        code_commit=CODE_COMMIT,
        generation_seeds=seeds,
        expected_audio_count=1,
        artifact_root=root,
        allow_nonproduction_matrix=True,
    )
    manifest_path = root / "pending.jsonl"
    config_path = root / "generation.json"
    output_path = root / "completed.jsonl"
    write_jsonl(manifest_path, trials)
    write_json(config_path, generation_config)
    return manifest_path, config_path, output_path, trials, generation_config, prepared


def answer_key_fixture(
    accepted: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_bundle: dict[str, dict[str, object]] = {}
    for audio in accepted:
        bundle_id = str(audio["text_bundle_id"])
        direction = str(audio["direction_id"])
        target, stale = (
            ("Seattle", "Boston")
            if direction == "a_to_b"
            else ("Boston", "Seattle")
        )
        by_bundle[bundle_id] = {
            "schema_version": "2.0.0",
            "answer_key_id": bundle_id,
            "scenario_id": audio["scenario_id"],
            "context_label": "museum_food_hotel",
            "direction_id": direction,
            "target_value": target,
            "stale_value": stale,
            "target_evidence": [target],
            "stale_evidence": [stale],
            "dependent_relations": [
                {
                    "unit_id": relation,
                    "relation": relation_name,
                    "planning_constraint": constraint,
                    "new_bound_state": {relation_name: target},
                    "old_bound_state": {relation_name: stale},
                }
                for relation, relation_name, constraint in (
                    ("D1", "activity.location", "Recommend a museum activity"),
                    ("D2", "food.location", "Recommend a seafood-free restaurant"),
                    ("D3", "accommodation.location", "Recommend a hotel under budget"),
                )
            ],
            "root_invariant_constraints": [
                {
                    "unit_id": "N1",
                    "relation": "user.preference",
                    "state": {"user.preference": "museums"},
                }
            ],
            "final_window_rule": "Use the primary response window.",
            "partial_response_rule": "Do not infer unaddressed relations.",
            "generic_response_rule": "Generic advice is unresolved.",
            "safety_note": "Do not guarantee allergy safety.",
        }
    return [by_bundle[key] for key in sorted(by_bundle)]


def annotation(
    trial_id: str,
    blind_id: str,
    annotator_id: str,
    *,
    correct: bool,
    adjudicator: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "2.0.0",
        "blind_id": blind_id,
        "eval_trial_id": trial_id,
        "annotator_id": annotator_id,
        "overall_label": "target_only" if correct else "no_evidence",
        "relation_labels": {
            "D1": "new_bound" if correct else "not_addressed",
            "D2": "new_bound" if correct else "not_addressed",
            "D3": "new_bound" if correct else "not_addressed",
        },
        "final_target_correct": correct,
        "stale_state_error": False,
        "assistant_started_before_repair": None,
        "notes": "",
        "adjudicator": adjudicator,
    }
    return row


class EvalManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.accepted = accepted_fixture()
        cls.trials = build_trials(cls.accepted)

    def test_production_matrix_has_3000_unique_trials(self) -> None:
        self.assertEqual(len(self.accepted), 600)
        self.assertEqual(len(self.trials), 3000)
        self.assertEqual(len({row["eval_trial_id"] for row in self.trials}), 3000)
        self.assertEqual(len({row["eval_run_id"] for row in self.trials}), 1)
        run_id = str(self.trials[0]["eval_run_id"])
        expected_hash = generation_config_hash(eval_config())
        self.assertIn(RESOLVED_REVISION, run_id)
        self.assertIn(expected_hash, run_id)
        self.assertIn(CODE_COMMIT, run_id)
        self.assertEqual(
            self.trials[0]["execution_contract"]["runner_version"], "2.2.2"
        )
        self.assertEqual(
            self.trials[0]["matrix_contract"]["generation_seeds"], list(SEEDS)
        )
        self.assertTrue(self.trials[0]["matrix_contract"]["production_matrix"])
        self.assertTrue(
            all(row["response"] == {"status": "pending"} for row in self.trials)
        )

    def test_snapshot_revision_accepts_only_same_cache_blob_symlink(self) -> None:
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_cache = root / "models--kyutai--moshiko-pytorch-bf16"
            blob = model_cache / "blobs" / "weight"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"weights")
            snapshot = model_cache / "snapshots" / revision
            snapshot.mkdir(parents=True)
            linked_weight = snapshot / "model.safetensors"
            linked_weight.symlink_to(blob)
            self.assertEqual(_snapshot_revision(linked_weight), revision)

            direct_weight = snapshot / "tokenizer.model"
            direct_weight.write_bytes(b"tokenizer")
            self.assertEqual(_snapshot_revision(direct_weight), revision)

            outside = root / "outside.safetensors"
            outside.write_bytes(b"untrusted")
            escaped_weight = snapshot / "escaped.safetensors"
            escaped_weight.symlink_to(outside)
            self.assertIsNone(_snapshot_revision(escaped_weight))

    def test_existing_output_refuses_different_identity(self) -> None:
        accepted = self.accepted[:1]
        first = build_trials(accepted)
        changed = build_eval_trials(
            accepted,
            model_repo="kyutai/moshi",
            resolved_revision=RESOLVED_REVISION,
            generation_config=eval_config(temp=0.9),
            code_commit=CODE_COMMIT,
            generation_seeds=SEEDS,
            expected_audio_count=1,
            allow_nonproduction_matrix=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "trials.jsonl"
            write_eval_manifest(output, first)
            guard_output_reuse(output, first)  # exact reuse is idempotent
            with self.assertRaisesRegex(RuntimeError, "different run identity"):
                guard_output_reuse(output, changed)

    def test_incomplete_production_matrices_are_rejected(self) -> None:
        without_one_audio = [
            row
            for row in self.trials
            if row["accepted_audio_id"] != self.accepted[-1]["accepted_audio_id"]
        ]
        self.assertEqual(len(without_one_audio), 2995)
        with self.assertRaisesRegex(ValueError, "accepted-audio count mismatch"):
            validate_eval_trials(without_one_audio)

        without_one_seed = [
            row for row in self.trials if row["generation_seed"] != SEEDS[-1]
        ]
        self.assertEqual(len(without_one_seed), 2400)
        with self.assertRaisesRegex(ValueError, "eval-trial count mismatch"):
            validate_eval_trials(without_one_seed)

    def test_seed_config_unknown_generation_and_portable_uri_fail_closed(self) -> None:
        one = accepted_fixture(1)
        with self.assertRaisesRegex(ValueError, "generation_seeds must exactly equal"):
            build_eval_trials(
                one,
                model_repo="kyutai/moshi",
                resolved_revision=RESOLVED_REVISION,
                generation_config=eval_config(),
                code_commit=CODE_COMMIT,
                generation_seeds=SEEDS[:-1],
                expected_audio_count=1,
                allow_nonproduction_matrix=True,
            )

        changed = eval_config()
        changed["generation"]["temperature"] = 0.8
        with self.assertRaisesRegex(ValueError, "unknown Moshi generation parameters"):
            generation_parameters(changed)

        escaped = accepted_fixture(1)
        escaped[0]["prepared_stimulus"]["uri"] = "../outside.wav"
        with self.assertRaisesRegex(ValueError, "traversal segments"):
            build_trials(escaped)

    def test_nonstandard_mimi_stream_contract_is_rejected(self) -> None:
        contract = deepcopy(self.trials[0]["execution_contract"])
        contract["input_sample_rate"] = 16000
        contract["mimi_frame_samples"] = 1280
        with self.assertRaisesRegex(ValueError, "requires 24 kHz.*1,920"):
            validate_execution_contract(contract)


class EvalRunnerTests(unittest.TestCase):
    def test_tracked_dirty_tree_is_rejected_by_runtime_identity_gate(self) -> None:
        with mock.patch(
            "run_eval_v2.subprocess.check_output", return_value=CODE_COMMIT + "\n"
        ), mock.patch("run_eval_v2.subprocess.run") as run:
            run.return_value.returncode = 1
            with self.assertRaisesRegex(RuntimeError, "tracked source differs"):
                _verify_clean_git_identity(CODE_COMMIT)

    def test_incomplete_coverage_and_early_eos_are_rejected(self) -> None:
        trial = build_trials(accepted_fixture(1))[0]
        frames = int(trial["capture_contract"]["target_end_frame_count"])
        samples = int(trial["capture_contract"]["target_end_sample_count"])
        backend = FakeEvalBackend(
            {
                "model_repo": str(trial["model_repo"]),
                "resolved_revision": str(trial["resolved_revision"]),
                "code_commit": str(trial["code_commit"]),
            },
            eval_config(),
        )
        metadata = backend.metadata
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete = BackendOutput(
                audio=np.zeros(samples, dtype=np.float32),
                sample_rate=24000,
                token_ids=[0] * (frames - 1),
                token_pieces=[""] * (frames - 1),
                frame_step_ms=80.0,
                eos_reached=False,
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete model coverage"):
                _completed_row(
                    trial,
                    incomplete,
                    elapsed_seconds=0.1,
                    audio_path=root / "incomplete.wav",
                    response_root=root,
                    appended_zero_sample_count=0,
                    backend_metadata=metadata,
                )

            early_eos = BackendOutput(
                audio=np.zeros(samples, dtype=np.float32),
                sample_rate=24000,
                token_ids=[0] * frames,
                token_pieces=[""] * frames,
                frame_step_ms=80.0,
                eos_reached=True,
            )
            with self.assertRaisesRegex(RuntimeError, "early model EOS"):
                _completed_row(
                    trial,
                    early_eos,
                    elapsed_seconds=0.1,
                    audio_path=root / "eos.wav",
                    response_root=root,
                    appended_zero_sample_count=0,
                    backend_metadata=metadata,
                )

    def test_both_atomic_crash_points_resume_safely(self) -> None:
        for phase in ("after_audio_before_record", "after_record_before_manifest"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, _, output, trials, config, _ = runnable_eval_fixture(
                    root, seeds=(17,)
                )
                del manifest
                response_root = root / "responses"
                identity = {
                    "model_repo": str(trials[0]["model_repo"]),
                    "resolved_revision": str(trials[0]["resolved_revision"]),
                    "generation_config_hash": str(
                        trials[0]["generation_config_hash"]
                    ),
                    "code_commit": str(trials[0]["code_commit"]),
                    "eval_run_id": str(trials[0]["eval_run_id"]),
                }
                failed = False

                def inject(observed_phase: str, trial_id: str) -> None:
                    nonlocal failed
                    del trial_id
                    if not failed and observed_phase == phase:
                        failed = True
                        raise RuntimeError(f"injected {phase}")

                with self.assertRaisesRegex(RuntimeError, f"injected {phase}"):
                    run_evaluation(
                        trials,
                        generation_config=config,
                        output_path=output,
                        artifact_root=root,
                        response_root=response_root,
                        backend=FakeEvalBackend(identity, config),
                        failure_injector=inject,
                    )

                resumed_backend = FakeEvalBackend(identity, config)
                report = run_evaluation(
                    trials,
                    generation_config=config,
                    output_path=output,
                    artifact_root=root,
                    response_root=response_root,
                    backend=resumed_backend,
                )
                self.assertEqual(report["status"], "completed")
                self.assertEqual(
                    report["executed_count"],
                    1 if phase == "after_audio_before_record" else 0,
                )
                completed = read_jsonl(output)
                self.assertEqual(completed[0]["response"]["status"], "completed")
                validate_trial_response(
                    completed[0], verify_audio=True, response_root=response_root
                )

    def test_cli_dry_run_verifies_inputs_without_loading_backend_or_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, config, output, _, _, _ = runnable_eval_fixture(root)

            def forbidden_factory(
                identity: dict[str, str], generation: dict[str, object]
            ) -> FakeEvalBackend:
                raise AssertionError("dry-run must not load a model backend")

            report = run_eval_main(
                [
                    "--input",
                    str(manifest),
                    "--output",
                    str(output),
                    "--generation-config",
                    str(config),
                    "--artifact-root",
                    str(root),
                    "--response-root",
                    str(root / "responses"),
                    "--dry-run",
                ],
                backend_factory=forbidden_factory,
            )
            self.assertEqual(report["status"], "dry_run_validated")
            self.assertEqual(report["trial_count"], 2)
            self.assertEqual(report["selected_count"], 2)
            self.assertEqual(report["unique_prepared_files_verified"], 1)
            self.assertFalse(output.exists())
            self.assertFalse((root / "responses").exists())

    def test_cli_only_seed_executes_and_resumes_one_frozen_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, config, output, _, _, _ = runnable_eval_fixture(root)
            response_root = root / "responses"
            backends: list[FakeEvalBackend] = []

            def factory(
                identity: dict[str, str], generation: dict[str, object]
            ) -> FakeEvalBackend:
                backend = FakeEvalBackend(identity, generation)
                backends.append(backend)
                return backend

            args = [
                "--input",
                str(manifest),
                "--output",
                str(output),
                "--generation-config",
                str(config),
                "--artifact-root",
                str(root),
                "--response-root",
                str(response_root),
                "--only-seed",
                "29",
            ]
            first = run_eval_main(args, backend_factory=factory)
            self.assertEqual(first["status"], "selection_completed")
            self.assertEqual(first["executed_count"], 1)
            self.assertEqual(first["remaining_count"], 1)
            self.assertEqual(first["selection_remaining_count"], 0)
            self.assertEqual(first["only_generation_seeds"], [29])
            self.assertEqual(backends[0].reset_seeds, [29])
            rows = read_jsonl(output)
            self.assertEqual(
                [row["response"]["status"] for row in rows],
                ["pending", "completed"],
            )

            created_before = len(backends)
            resumed = run_eval_main(args, backend_factory=factory)
            self.assertEqual(resumed["status"], "selection_completed")
            self.assertEqual(resumed["executed_count"], 0)
            self.assertEqual(len(backends), created_before)

            with self.assertRaisesRegex(ValueError, "not in the frozen matrix"):
                run_eval_main(
                    [*args[:-2], "--only-seed", "999"],
                    backend_factory=factory,
                )

    def test_cli_executes_atomic_records_and_resumes_without_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, config, output, _, _, _ = runnable_eval_fixture(root)
            response_root = root / "responses"
            backends: list[FakeEvalBackend] = []

            def factory(
                identity: dict[str, str], generation: dict[str, object]
            ) -> FakeEvalBackend:
                backend = FakeEvalBackend(identity, generation)
                backends.append(backend)
                return backend

            common_args = [
                "--input",
                str(manifest),
                "--output",
                str(output),
                "--generation-config",
                str(config),
                "--artifact-root",
                str(root),
                "--response-root",
                str(response_root),
                "--checkpoint-every",
                "1",
            ]
            first = run_eval_main(
                [*common_args, "--limit", "1"], backend_factory=factory
            )
            self.assertEqual(first["status"], "partially_completed")
            self.assertEqual(first["executed_count"], 1)
            self.assertEqual(backends[0].reset_seeds, [17])
            partial = read_jsonl(output)
            self.assertEqual(
                [row["response"]["status"] for row in partial],
                ["completed", "pending"],
            )
            first_evidence = deepcopy(partial[0])

            second = run_eval_main(common_args, backend_factory=factory)
            self.assertEqual(second["status"], "completed")
            self.assertEqual(second["executed_count"], 1)
            self.assertEqual(backends[1].reset_seeds, [29])
            completed = read_jsonl(output)
            self.assertEqual(completed[0], first_evidence)
            self.assertTrue(all(row["response"]["status"] == "completed" for row in completed))
            self.assertEqual(len(list(response_root.glob("records/*/*.json"))), 2)
            self.assertEqual(len(list(response_root.glob("audio/*/*.wav"))), 2)
            for row in completed:
                validate_trial_response(
                    row, verify_audio=True, response_root=response_root
                )
                self.assertEqual(row["response"]["transcript"], "Seattle")
                self.assertTrue(row["response"]["stream_reset"])
                self.assertTrue(row["response"]["rng_reset"])
                self.assertEqual(
                    row["response"]["timebase"], "prepared_stream_relative"
                )
                self.assertEqual(row["response"]["stream_origin_ms"], 0)
                self.assertEqual(
                    row["response"]["fed_sample_count"],
                    row["capture_contract"]["target_end_sample_count"],
                )
                self.assertEqual(
                    row["response"]["output_sample_count"],
                    row["response"]["fed_sample_count"],
                )
                self.assertFalse(Path(row["response"]["audio_path"]).is_absolute())

            schema = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "dataset_v2/schemas/eval_trial.schema.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(validate_rows(completed, schema), [])

            created_before = len(backends)
            third = run_eval_main(common_args, backend_factory=factory)
            self.assertEqual(third["executed_count"], 0)
            self.assertEqual(len(backends), created_before)
            self.assertEqual(read_jsonl(output), completed)

    def test_cli_rejects_identity_mismatch_before_loading_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, config, output, _, _, prepared = runnable_eval_fixture(root)
            response_root = root / "responses"
            created = 0

            def factory(
                identity: dict[str, str], generation: dict[str, object]
            ) -> FakeEvalBackend:
                nonlocal created
                created += 1
                return FakeEvalBackend(identity, generation)

            args = [
                "--input",
                str(manifest),
                "--output",
                str(output),
                "--generation-config",
                str(config),
                "--artifact-root",
                str(root),
                "--response-root",
                str(response_root),
                "--limit",
                "1",
            ]
            run_eval_main(args, backend_factory=factory)
            self.assertEqual(created, 1)

            changed_config = eval_config((17, 29), temp=0.9)
            changed_trials = build_eval_trials(
                prepared,
                model_repo="kyutai/moshi",
                resolved_revision=RESOLVED_REVISION,
                generation_config=changed_config,
                code_commit=CODE_COMMIT,
                generation_seeds=(17, 29),
                expected_audio_count=1,
                artifact_root=root,
                allow_nonproduction_matrix=True,
            )
            changed_manifest = root / "changed-pending.jsonl"
            changed_config_path = root / "changed-generation.json"
            write_jsonl(changed_manifest, changed_trials)
            write_json(changed_config_path, changed_config)
            with self.assertRaisesRegex(RuntimeError, "different run identity"):
                run_eval_main(
                    [
                        "--input",
                        str(changed_manifest),
                        "--output",
                        str(output),
                        "--generation-config",
                        str(changed_config_path),
                        "--artifact-root",
                        str(root),
                        "--response-root",
                        str(response_root),
                    ],
                    backend_factory=factory,
                )
            self.assertEqual(created, 1)

    def test_prepared_hash_tamper_is_rejected_before_backend_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, config, output, trials, _, _ = runnable_eval_fixture(root)
            input_path = root / str(trials[0]["input_stimulus"]["uri"])
            input_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "prepared stimulus hash mismatch"):
                run_eval_main(
                    [
                        "--input",
                        str(manifest),
                        "--output",
                        str(output),
                        "--generation-config",
                        str(config),
                        "--artifact-root",
                        str(root),
                        "--response-root",
                        str(root / "responses"),
                    ],
                    backend_factory=lambda identity, generation: (_ for _ in ()).throw(
                        AssertionError("backend must not load after input tamper")
                    ),
                )


class AnnotationSheetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.accepted = accepted_fixture(8)
        self.answer_keys = answer_key_fixture(self.accepted)
        trials = build_trials(self.accepted)
        for index, trial in enumerate(trials):
            trial["response"] = {
                "status": "completed",
                "transcript": f"opaque assistant response {index}",
            }
        self.trials = trials

    def test_two_stable_condition_blind_shuffled_sheets(self) -> None:
        package = build_annotation_package(
            self.trials,
            self.accepted,
            self.answer_keys,
            ("annotator_a", "annotator_b"),
            shuffle_seed=20260826,
            expected_accepted_count=8,
            expected_answer_key_count=1,
        )
        repeated = build_annotation_package(
            self.trials,
            self.accepted,
            self.answer_keys,
            ("annotator_a", "annotator_b"),
            shuffle_seed=20260826,
            expected_accepted_count=8,
            expected_answer_key_count=1,
        )
        self.assertEqual(package["sheets"], repeated["sheets"])
        self.assertEqual(len(package["blind_map"]), len(self.trials))
        self.assertEqual(set(package["sheets"]), {"annotator_a", "annotator_b"})
        for rows in package["sheets"].values():
            self.assertEqual(len(rows), len(self.trials))
            for row in rows:
                self.assertFalse(set(row) & FORBIDDEN_PUBLIC_FIELDS)
                self.assertEqual(row["final_target_correct"], "")
                self.assertEqual(row["stale_state_error"], "")
                self.assertEqual(json.loads(row["context_label"]), "museum_food_hotel")
                self.assertIn(json.loads(row["target_value"]), {"Boston", "Seattle"})
                self.assertEqual(
                    json.loads(row["D1_relation_planning_constraint"])["relation"],
                    "activity.location",
                )
                self.assertIsInstance(
                    json.loads(row["root_invariant_constraints"]), list
                )
                public_blob = json.dumps(row, sort_keys=True)
                for condition in CONDITIONS:
                    self.assertNotIn(condition, public_blob)
        order_a = [row["blind_id"] for row in package["sheets"]["annotator_a"]]
        order_b = [row["blind_id"] for row in package["sheets"]["annotator_b"]]
        self.assertNotEqual(order_a, order_b)

    def test_requires_exactly_two_distinct_annotators(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two distinct"):
            build_annotation_package(
                self.trials,
                self.accepted,
                self.answer_keys,
                ("same", "same"),
                shuffle_seed=1,
                expected_accepted_count=8,
                expected_answer_key_count=1,
            )
        with self.assertRaisesRegex(ValueError, "exactly two distinct"):
            build_annotation_package(
                self.trials,
                self.accepted,
                self.answer_keys,
                ("only_one",),
                shuffle_seed=1,
                expected_accepted_count=8,
                expected_answer_key_count=1,
            )

    def test_requires_exact_private_joins_and_uses_new_default_path(self) -> None:
        self.assertEqual(
            DEFAULT_ACCEPTED_MANIFEST.name,
            "accepted_audio.jsonl",
        )
        self.assertEqual(DEFAULT_ACCEPTED_MANIFEST.parent.name, "manifests")
        bad_trials = deepcopy(self.trials)
        bad_trials[0]["accepted_audio_id"] = "missing_audio"
        with self.assertRaisesRegex(ValueError, "does not join exactly once"):
            build_annotation_package(
                bad_trials,
                self.accepted,
                self.answer_keys,
                ("a", "b"),
                shuffle_seed=1,
                expected_accepted_count=8,
                expected_answer_key_count=1,
            )

        leaked_keys = deepcopy(self.answer_keys)
        leaked_keys[0]["safety_note"] = "Use delayed_neutral as the reference."
        with self.assertRaisesRegex(ValueError, "condition identifiers leaked"):
            build_annotation_package(
                self.trials,
                self.accepted,
                leaked_keys,
                ("a", "b"),
                shuffle_seed=1,
                expected_accepted_count=8,
                expected_answer_key_count=1,
            )

    def test_blank_boolean_is_not_coerced_to_false(self) -> None:
        trial = self.trials[0]
        row = {
            "blind_id": "blind_x",
            "annotator_id": "annotator_a",
            "overall_label": "no_evidence",
            "relation_D1": "not_addressed",
            "relation_D2": "not_addressed",
            "relation_D3": "not_addressed",
            "final_target_correct": "",
            "stale_state_error": "false",
            "assistant_started_before_repair": "",
            "notes": "",
        }
        with self.assertRaisesRegex(ValueError, "not coerced to false"):
            annotation_from_sheet_row(
                row,
                {"blind_x": str(trial["eval_trial_id"])},
                expected_annotator_id="annotator_a",
            )


class AnnotationResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trial_id = "trial_001"
        self.blind_id = "blind_001"

    def test_agreement_needs_two_independent_annotations(self) -> None:
        rows = [
            annotation(self.trial_id, self.blind_id, "a", correct=True),
            annotation(self.trial_id, self.blind_id, "b", correct=True),
        ]
        resolved = resolve_annotations((self.trial_id,), rows)
        self.assertEqual(resolved[0]["resolution_method"], "independent_agreement")
        self.assertTrue(resolved[0]["final_target_correct"])

        duplicate_annotator = deepcopy(rows)
        duplicate_annotator[1]["annotator_id"] = "a"
        with self.assertRaisesRegex(ValueError, "not independent"):
            resolve_annotations((self.trial_id,), duplicate_annotator)

    def test_disagreement_requires_one_independent_adjudicator(self) -> None:
        rows = [
            annotation(self.trial_id, self.blind_id, "a", correct=True),
            annotation(self.trial_id, self.blind_id, "b", correct=False),
        ]
        with self.assertRaisesRegex(ValueError, "requires exactly one adjudication"):
            resolve_annotations((self.trial_id,), rows)
        rows.append(
            annotation(
                self.trial_id,
                self.blind_id,
                "c",
                correct=False,
                adjudicator=True,
            )
        )
        resolved = resolve_annotations((self.trial_id,), rows)
        self.assertEqual(resolved[0]["resolution_method"], "adjudicated_disagreement")
        self.assertFalse(resolved[0]["final_target_correct"])
        self.assertEqual(resolved[0]["adjudicator_id"], "c")

    def test_direct_missing_boolean_is_rejected(self) -> None:
        row = annotation(self.trial_id, self.blind_id, "a", correct=True)
        row["final_target_correct"] = None
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            validate_annotation(row)


class PrimaryScoringTests(unittest.TestCase):
    def test_scores_successes_out_of_five_at_audio_level(self) -> None:
        accepted = accepted_fixture(2)
        trials = build_trials(accepted)
        annotations: list[dict[str, object]] = []
        first_id = str(accepted[0]["accepted_audio_id"])
        for trial in trials:
            accepted_id = str(trial["accepted_audio_id"])
            seed = int(trial["generation_seed"])
            correct = accepted_id != first_id or seed in set(SEEDS[:3])
            blind_id = "blind_" + str(trial["eval_trial_id"])[-32:]
            annotations.extend(
                (
                    annotation(str(trial["eval_trial_id"]), blind_id, "a", correct=correct),
                    annotation(str(trial["eval_trial_id"]), blind_id, "b", correct=correct),
                )
            )

        scored = score_primary(
            trials,
            accepted,
            annotations,
            expected_seeds=SEEDS,
            expected_audio_count=2,
            expected_scenario_count=1,
        )
        self.assertEqual(len(scored["audio_scores"]), 2)
        by_id = {row["accepted_audio_id"]: row for row in scored["audio_scores"]}
        self.assertEqual(by_id[first_id]["successes"], 3)
        self.assertEqual(by_id[first_id]["trials"], 5)
        self.assertEqual(by_id[first_id]["final_target_rate"], 0.6)
        other_id = str(accepted[1]["accepted_audio_id"])
        self.assertEqual(by_id[other_id]["successes"], 5)
        endpoint = scored["summary"]["primary_endpoint"]
        self.assertEqual(endpoint["unit"], "accepted_audio_id")
        self.assertEqual(endpoint["primary_cluster"], "scenario_id")
        self.assertFalse(endpoint["seed_trials_are_independent_samples"])
        self.assertEqual(scored["summary"]["counts"]["eval_trials"], 10)
        self.assertEqual(
            scored["summary"]["frozen_contrast_inference"]["status"],
            "not_evaluable",
        )
        self.assertIn("gates", scored["summary"])

    def test_frozen_scenario_bootstrap_is_deterministic(self) -> None:
        scenario_rows: list[dict[str, object]] = []
        for index in range(30):
            scenario_id = f"travel_{index + 1:03d}"
            neutral = 0.40 + 0.01 * (index % 3)
            for condition, rate in (
                ("delayed_neutral", neutral),
                ("delayed_one_dependency", neutral + 0.05),
                ("delayed_three_dependencies", neutral + 0.20),
            ):
                scenario_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "condition": condition,
                        "audio_units": 4,
                        "final_target_rate": rate,
                    }
                )
        first = frozen_contrast_inference(scenario_rows)
        second = frozen_contrast_inference(scenario_rows)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "evaluated")
        self.assertEqual(first["bootstrap"]["replicates"], 10_000)
        self.assertEqual(first["bootstrap"]["seed"], 20260826)
        primary = first["contrasts"]["delayed_three_minus_neutral"]
        secondary = first["contrasts"]["delayed_one_minus_neutral"]
        self.assertAlmostEqual(primary["equal_weight_percentage_point_difference"], 20.0)
        self.assertAlmostEqual(secondary["equal_weight_percentage_point_difference"], 5.0)
        self.assertAlmostEqual(
            primary["bootstrap_percentile_95_ci_percentage_points"][0], 20.0
        )
        self.assertEqual(first["multiplicity"]["method"], "Holm")
        self.assertEqual(first["secondary_model"]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
