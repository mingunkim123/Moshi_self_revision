from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from experiments.self_repair.mechanistic import HARNESS_VERSION
from experiments.self_repair.mechanistic.conversation import (
    DATASET_V2_CONTRACT_SOURCE,
    NATURAL_START_STATUS,
    REQUIRED_EXPERIMENTAL_STARTUP_MODES,
    STARTUP_MODE_COMMON_HANDSHAKE,
    STARTUP_MODE_GREETING_SUPPRESSED,
)
from experiments.self_repair.mechanistic.core import (
    FRAME_SAMPLES,
    MODEL_REPO,
    MODEL_REVISION,
    ContractError,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
    write_jsonl,
)
from experiments.self_repair.mechanistic.runtime import GeneratedSequence, PairedGeneration
from experiments.self_repair.mechanistic.readiness import _conversation_measurement_blockers
from experiments.self_repair.mechanistic.scripts import _cli


def _write_wav(path: Path, *, frames: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(frames * FRAME_SAMPLES, dtype="<i2")
    samples[6 * FRAME_SAMPLES : 8 * FRAME_SAMPLES] = 1_000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(samples.tobytes(order="C"))
    return sha256_file(path)


def _trial_row(
    trial_id: str, condition: str, audio_uri: str, audio_sha256: str,
) -> dict[str, object]:
    prepared_timing = {
        "old_value_offset_ms": 560,
        "new_value_offset_ms": 640,
        "utterance_end_ms": 640,
    }
    capture = {
        "condition": condition,
        "timebase": "prepared_stream_relative",
        "stream_origin_ms": 0,
        "prepared_timing": prepared_timing,
        "prepared_timing_sha256": sha256_value(prepared_timing),
        "utterance_end_ms": 640,
        "primary_window_start_ms": 640,
        "response_capture_ms": 40_000,
        "requested_target_end_ms": 40_640,
        "target_end_sample_count": 508 * FRAME_SAMPLES,
        "target_end_frame_count": 508,
        "actual_target_end_ms": 40_640,
    }
    execution = {
        "input_sample_rate": 24_000,
        "mimi_frame_samples": FRAME_SAMPLES,
        "prefix_silence_ms": 480,
        "response_capture_ms": 40_000,
        "required_model_type": "moshi",
        "required_max_lm_delay": 1,
        "reset_model_stream_between_trials": True,
        "reset_rng_for_each_trial_seed": True,
    }
    input_stimulus = {
        "prepared_stimulus_id": trial_id,
        "sha256": audio_sha256,
        "sample_rate": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "mimi_frame_samples": FRAME_SAMPLES,
        "duration_ms": 960,
        "timeline": "prepared_stream_relative",
    }
    return {
        "schema_version": "1.0.0",
        "trial_id": trial_id,
        "prepared_stimulus_id": trial_id,
        "scenario_id": "scenario-1",
        "condition": condition,
        "direction_id": "a_to_b",
        "speaker_id": "speaker-1",
        "analysis_fold": 1,
        "role": "discovery",
        "data_status": "exploratory_provisional",
        "old_value": "Boston",
        "new_value": "Seattle",
        "audio_uri": audio_uri,
        "audio_sha256": audio_sha256,
        "sample_rate": 24_000,
        "sample_count": 12 * FRAME_SAMPLES,
        "frame_count": 12,
        "input_stimulus": input_stimulus,
        "capture_contract": capture,
        "execution_contract": execution,
        "model_repo": MODEL_REPO,
        "resolved_revision": MODEL_REVISION,
        "conversation_contract": {
            "schema_version": "1.0.0",
            "source": DATASET_V2_CONTRACT_SOURCE,
            "trial_id": trial_id,
            "startup_mode": "natural_model_start",
            "startup_status": NATURAL_START_STATUS,
            "required_startup_modes": list(REQUIRED_EXPERIMENTAL_STARTUP_MODES),
            "file_replay_startup": "prime_once_then_consume_first_mimi_frame",
            "assistant_output_origin_frame": 0,
            "sample_rate": 24_000,
            "frame_samples": FRAME_SAMPLES,
            "prefix_silence_ms": 480,
            "user_start_frame": 6,
            "query_end_ms": 640,
            "query_end_frame": 8,
            "user_end_frame": 8,
            "user_frame_count": 12,
            "user_sample_count": 12 * FRAME_SAMPLES,
            "response_capture_ms": 40_000,
            "response_capture_frames": 500,
            "tail_guard_frames": 25,
            "target_end_frame_count": 508,
            "target_end_sample_count": 508 * FRAME_SAMPLES,
            "appended_zero_frame_count": 496,
            "source_capture_contract_sha256": sha256_value(capture),
            "source_execution_contract_sha256": sha256_value(execution),
        },
    }


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _encoded_row(
    root: Path, source: dict[str, object], manifest_sha256: str, *, synthetic: bool,
) -> dict[str, object]:
    trial_id = str(source["trial_id"])
    user = np.zeros((1, 8, 12), dtype=np.int64)
    conversation = np.zeros((1, 8, 508), dtype=np.int64)
    conversation[..., :12] = user
    silence = np.zeros((1, 8, 508), dtype=np.int64)
    identity = sha256_value({"trial_id": trial_id, "synthetic": synthetic})
    archive = root / "encoded" / f"{trial_id}.npz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive,
        codes=user,
        user_codes=user,
        conversation_codes=conversation,
        assistant_silence_codes=silence,
        artifact_identity_sha256=np.asarray(identity),
    )
    return {
        "schema_version": "1.1.0",
        "trial_id": trial_id,
        "source_manifest_sha256": manifest_sha256,
        "source_row_sha256": sha256_value(source),
        "source_audio_sha256": source["audio_sha256"],
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_identity_sha256": "d" * 64,
        "code_commit": _cli._git_commit(),
        "harness_version": HARNESS_VERSION,
        "artifact_identity_sha256": identity,
        "codes_uri": f"encoded/{trial_id}.npz",
        "archive_sha256": sha256_file(archive),
        "shape": list(user.shape),
        "dtype": str(user.dtype),
        "codes_sha256": _array_sha256(user),
        "user_codes_shape": list(user.shape),
        "user_codes_dtype": str(user.dtype),
        "user_codes_sha256": _array_sha256(user),
        "conversation_codes_shape": list(conversation.shape),
        "conversation_codes_dtype": str(conversation.dtype),
        "conversation_codes_sha256": _array_sha256(conversation),
        "assistant_silence_codes_shape": list(silence.shape),
        "assistant_silence_codes_dtype": str(silence.dtype),
        "assistant_silence_codes_sha256": _array_sha256(silence),
        "user_frame_end_exclusive": 12,
        "conversation_frame_end_exclusive": 508,
        "assistant_silence_frame_end_exclusive": 508,
        "synthetic": synthetic,
    }


def _fixture(tmp_path: Path, *, synthetic: bool = False) -> SimpleNamespace:
    data = tmp_path / "data"
    clean_sha = _write_wav(data / "audio/clean.wav", frames=12)
    repair_sha = _write_wav(data / "audio/repair.wav", frames=12)
    rows = [
        _trial_row("clean-1", "clean_final", "audio/clean.wav", clean_sha),
        _trial_row("repair-1", "repair_immediate", "audio/repair.wav", repair_sha),
    ]
    manifest = tmp_path / "canary.jsonl"
    write_jsonl(manifest, rows)
    manifest_sha = sha256_file(manifest)
    write_json(manifest.with_suffix(".jsonl.selection.json"), {
        "schema_version": "1.0.0",
        "trial_ids": ["clean-1", "repair-1"],
        "trial_count": 2,
        "bounded_max_trials": 4,
        "source_manifest_sha256": "b" * 64,
        "canary_manifest_sha256": manifest_sha,
    })
    encoded_manifest = tmp_path / "encoded_manifest.jsonl"
    write_jsonl(encoded_manifest, [
        _encoded_row(tmp_path, row, manifest_sha, synthetic=synthetic) for row in rows
    ])
    anchors = tmp_path / "anchors.jsonl"
    write_jsonl(anchors, [
        {"trial_id": trial_id, "anchor": anchor, "frame": 7, "time_ms": 640}
        for trial_id in ("clean-1", "repair-1")
        for anchor in ("new_end", "query_end")
    ])
    return SimpleNamespace(
        data=data,
        manifest=manifest,
        encoded_manifest=encoded_manifest,
        anchors=anchors,
        config=Path(__file__).resolve().parents[1] / "config/mechanistic.json",
    )


class _FakeConversationBackend:
    constructed = 0
    calls: list[dict[str, object]] = []
    behavior = "complete"
    fail_once = False

    def __init__(self, **kwargs):
        del kwargs
        type(self).constructed += 1

    def _read_pcm(self, path: Path) -> np.ndarray:
        with wave.open(str(path), "rb") as handle:
            return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(
                np.float32) / 32768.0

    def decode_assistant_silence(self, codes: np.ndarray) -> np.ndarray:
        return np.zeros(int(codes.shape[-1]) * FRAME_SAMPLES, dtype=np.float32)

    def generate_paired_conversation(self, codes, **kwargs) -> PairedGeneration:
        if type(self).fail_once:
            type(self).fail_once = False
            raise RuntimeError("deliberate fake generation failure")
        mode = str(kwargs["startup_mode"])
        target = int(kwargs["target_frame_count"])
        branch = int(kwargs["branch_frame"])
        query_end = int(kwargs["query_end_frame"])
        startup = 22 if mode == STARTUP_MODE_COMMON_HANDSHAKE else 0
        assert int(np.asarray(codes).shape[-1]) == target == 508
        assert (kwargs["conversation_pcm"] is not None) == (
            mode == STARTUP_MODE_COMMON_HANDSHAKE)
        type(self).calls.append({
            "startup_mode": mode,
            "intervention": copy.deepcopy(kwargs["intervention"]),
            "target_frame_count": target,
        })
        total = startup + target
        tokens = np.zeros((1, 9, total), dtype=np.int64)
        tokens[:, 0] = 3
        feedback = np.zeros_like(tokens)
        pieces = [""] * total
        pcm = np.zeros(total * FRAME_SAMPLES, dtype=np.float32)
        if startup:
            tokens[0, 0, 0], pieces[0] = 5, " Hello"
            tokens[0, 0, 1], pieces[1] = 7, "."
            pcm[: 2 * FRAME_SAMPLES] = 0.02
        response = startup + query_end
        if type(self).behavior == "complete":
            tokens[0, 0, response], pieces[response] = 5, " Seattle"
            tokens[0, 0, response + 1], pieces[response + 1] = 7, "."
            pcm[response * FRAME_SAMPLES : (response + 2) * FRAME_SAMPLES] = 0.02
        elif type(self).behavior == "truncated":
            tokens[0, 0, total - 1], pieces[total - 1] = 5, " continuing"
            pcm[(total - 1) * FRAME_SAMPLES :] = 0.02
        elif type(self).behavior != "no_response":
            raise AssertionError(type(self).behavior)

        def sequence() -> GeneratedSequence:
            return GeneratedSequence(
                tokens=tokens.copy(), feedback_tokens=feedback.copy(),
                text_token_ids=[int(value) for value in tokens[0, 0]],
                text_pieces=list(pieces), pcm=pcm.copy(), frame_count=total,
                conversation_frame_count=target, conversation_start_frame=startup,
                frame_samples=FRAME_SAMPLES, pcm_sample_count=int(pcm.size),
            )

        absolute_branch = startup + branch
        return PairedGeneration(
            baseline=sequence(), patched=sequence(), branch_frame=branch,
            shared_prefix_frames=absolute_branch,
            shared_prefix_sha256="1" * 64,
            shared_feedback_sha256="2" * 64,
            first_feedback_divergence_frame=None,
            first_output_divergence_frame=None,
            pre_intervention_identical=True,
            startup_mode=mode,
            startup_frame_count=startup,
            handshake_terminal_frame=1 if startup else None,
            handshake_terminal_piece="." if startup else None,
            handshake_completion_signal=(
                "terminal_punctuation_plus_text_audio_quiet" if startup else None),
            target_frame_count=target,
            lm_step_count=1 + absolute_branch + 2 * (target - branch),
            handshake_probe_lm_step_count=(23 if startup else 0),
            handshake_replay_identical=True if startup else None,
            continuous_mimi_input_verified=True if startup else None,
        )


def _args(fixture: SimpleNamespace, output: Path, *, seeds: str = "17,29,42,101") -> list[str]:
    return [
        "--config", str(fixture.config),
        "--manifest", str(fixture.manifest),
        "--encoded-manifest", str(fixture.encoded_manifest),
        "--anchors", str(fixture.anchors),
        "--input-artifact-root", str(fixture.data),
        "--primary-intervention", "identity_noop",
        "--donor-arms", "none",
        "--seeds", seeds,
        "--output-root", str(output),
    ]


def _completed_reviews(template_path: Path) -> list[dict[str, object]]:
    rows = read_jsonl(template_path)
    output = []
    for row in rows:
        completed = dict(row)
        completed["reviewer_id"] = f"reviewer-{row['reviewer_slot']}"
        for name in _cli._FULL_DUPLEX_REVIEW_FIELDS:
            completed[name] = True
        output.append(completed)
    return output


def test_two_mode_canary_is_blind_resumable_and_review_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "run"
    _FakeConversationBackend.constructed = 0
    _FakeConversationBackend.calls = []
    _FakeConversationBackend.behavior = "complete"
    _FakeConversationBackend.fail_once = False
    monkeypatch.setattr(_cli, "MoshiBackend", _FakeConversationBackend)

    assert _cli.run_full_duplex(_args(fixture, output)) == 0
    assert _FakeConversationBackend.constructed == 1
    assert len(_FakeConversationBackend.calls) == 8
    assert {row["startup_mode"] for row in _FakeConversationBackend.calls} == {
        STARTUP_MODE_COMMON_HANDSHAKE, STARTUP_MODE_GREETING_SUPPRESSED}
    assert not (output / "conversation_canary.json").exists()
    validation = read_jsonl(output / "validation.jsonl")
    assert len(validation) == 8
    assert all(len(row["arms"]) == 2 for row in validation)
    audio_uris = [
        arm[key]
        for row in validation for arm in row["arms"]
        for key in ("full_audio_uri", "primary_audio_uri")
    ]
    assert all("baseline" not in uri and "patched" not in uri for uri in audio_uris)
    assert all((output / uri).is_file() for uri in audio_uris)
    common = next(row for row in validation if row["startup_mode"] == STARTUP_MODE_COMMON_HANDSHAKE)
    suppressed = next(row for row in validation if row["startup_mode"] == STARTUP_MODE_GREETING_SUPPRESSED)
    with wave.open(str(output / common["arms"][0]["full_audio_uri"]), "rb") as handle:
        assert handle.getnframes() == (508 + 22) * FRAME_SAMPLES
    with wave.open(str(output / suppressed["arms"][0]["full_audio_uri"]), "rb") as handle:
        assert handle.getnframes() == 508 * FRAME_SAMPLES

    incomplete = _completed_reviews(output / "blind_review_template.jsonl")[:-1]
    incomplete_path = tmp_path / "incomplete_reviews.jsonl"
    write_jsonl(incomplete_path, incomplete)
    with pytest.raises(ContractError, match="coverage is incomplete"):
        _cli.run_full_duplex([
            *_args(fixture, output), "--resume", "--reviews", str(incomplete_path)])
    assert _FakeConversationBackend.constructed == 1
    assert not (output / "conversation_canary.json").exists()

    reviews = _completed_reviews(output / "blind_review_template.jsonl")
    reviews[1]["natural_flow"] = False
    reviews_path = tmp_path / "reviews.jsonl"
    write_jsonl(reviews_path, reviews)
    with pytest.raises(ContractError, match="require one independent adjudication"):
        _cli.run_full_duplex([
            *_args(fixture, output), "--resume", "--reviews", str(reviews_path)])
    adjudication_template = read_jsonl(output / "adjudication_template.jsonl")
    assert len(adjudication_template) == 1
    adjudication = dict(adjudication_template[0])
    adjudication["adjudicator_id"] = "reviewer-3"
    for name in _cli._FULL_DUPLEX_REVIEW_FIELDS:
        adjudication[name] = True
    adjudication_path = tmp_path / "adjudications.jsonl"
    write_jsonl(adjudication_path, [adjudication])
    assert _cli.run_full_duplex([
        *_args(fixture, output), "--resume", "--reviews", str(reviews_path),
        "--adjudications", str(adjudication_path),
    ]) == 0
    assert _FakeConversationBackend.constructed == 1
    report = read_json(output / "conversation_canary.json")
    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["per_mode"][STARTUP_MODE_COMMON_HANDSHAKE]["trial_count"] == 4
    assert report["per_mode"][STARTUP_MODE_GREETING_SUPPRESSED]["trial_count"] == 4
    assert report["measurements"]["required_mode_trial_count"] == 8
    assert report["tail_detection"]["forced_silence_decode_max_dbfs"] < -45.0
    assert report["canary_purpose"] == "prepaid_conversation_flow_identity_noop"
    assert _conversation_measurement_blockers(report, read_json(fixture.config)) == []
    assert read_json(output / "resume_summary.json")["backend_constructed"] is False


@pytest.mark.parametrize(
    ("behavior", "status", "cap_active"),
    [
        ("no_response", "unevaluable_no_response", False),
        ("truncated", "unevaluable_truncated", True),
    ],
)
def test_no_response_and_cap_activity_are_unevaluable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    behavior: str, status: str, cap_active: bool,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / f"run-{behavior}"
    _FakeConversationBackend.constructed = 0
    _FakeConversationBackend.calls = []
    _FakeConversationBackend.behavior = behavior
    _FakeConversationBackend.fail_once = False
    monkeypatch.setattr(_cli, "MoshiBackend", _FakeConversationBackend)
    assert _cli.run_full_duplex(_args(fixture, output, seeds="17")) == 0
    validation = read_jsonl(output / "validation.jsonl")
    assert all(
        arm["technical_status"] == status
        and arm["combined_cap_active"] is cap_active
        and arm["response_complete"] is False
        for row in validation for arm in row["arms"]
    )
    assert not (output / "conversation_canary.json").exists()


def test_frozen_erasure_and_failed_attempt_are_preserved_then_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    config_sha = sha256_file(fixture.config)
    selection_body = {
        "schema_version": "1.0.0",
        "status": "frozen_discovery_selection",
        "config_sha256": config_sha,
        "component": "resid_post",
        "layer": 2,
        "head": None,
        "anchor": "new_end",
        "direction": "target_minus_stale",
        "donor_arm": "clean_current",
        "relation": "same_scenario_current_value",
        "readout_sha256": "e" * 64,
        "selection_source_cell_id": "f" * 64,
    }
    selection = tmp_path / "selection.json"
    write_json(selection, {
        **selection_body, "selection_sha256": sha256_value(selection_body)})
    output = tmp_path / "retry-run"
    args = [
        "--config", str(fixture.config), "--selection", str(selection),
        "--manifest", str(fixture.manifest),
        "--encoded-manifest", str(fixture.encoded_manifest),
        "--anchors", str(fixture.anchors),
        "--input-artifact-root", str(fixture.data),
        "--primary-intervention", "within_repair_erasure",
        "--donor-arms", "conditional_on_feedback_divergence",
        "--seeds", "17", "--output-root", str(output),
    ]
    _FakeConversationBackend.constructed = 0
    _FakeConversationBackend.calls = []
    _FakeConversationBackend.behavior = "complete"
    _FakeConversationBackend.fail_once = True
    monkeypatch.setattr(_cli, "MoshiBackend", _FakeConversationBackend)
    with pytest.raises(RuntimeError, match="deliberate fake"):
        _cli.run_full_duplex(args)
    failures = read_jsonl(output / "failures.jsonl")
    assert len(failures) == 1
    assert failures[0]["status"] == "failed"
    assert _cli.run_full_duplex([*args, "--resume"]) == 0
    assert len(read_jsonl(output / "failures.jsonl")) == 1
    assert any(
        call["intervention"] == (("resid_post", 2, 7, None),)
        for call in _FakeConversationBackend.calls
    )
    summary = read_json(output / "resume_summary.json")
    assert summary["completed_cells"] == 2
    assert summary["preserved_failed_attempts"] == 1


def test_synthetic_conversation_never_emits_real_readiness_artifact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, synthetic=True)
    output = tmp_path / "synthetic-run"
    args = ["--synthetic", *_args(fixture, output)]
    assert _cli.run_full_duplex(args) == 0
    reviews = _completed_reviews(output / "blind_review_template.jsonl")
    reviews_path = tmp_path / "synthetic_reviews.jsonl"
    write_jsonl(reviews_path, reviews)
    assert _cli.run_full_duplex([
        *args, "--resume", "--reviews", str(reviews_path)]) == 0
    assert not (output / "conversation_canary.json").exists()
    report = read_json(output / "synthetic_conversation_canary.json")
    assert report["analysis_status"] == "synthetic_local_validation"
    assert report["synthetic"] is True
    assert report["synthetic_validation_passed"] is True
    assert report["passed"] is False


def test_stale_encoded_identity_stops_before_backend_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    rows = read_jsonl(fixture.encoded_manifest)
    rows[1]["code_commit"] = "0" * 40
    write_jsonl(fixture.encoded_manifest, rows)
    _FakeConversationBackend.constructed = 0
    monkeypatch.setattr(_cli, "MoshiBackend", _FakeConversationBackend)
    with pytest.raises(ContractError, match="encoded provenance"):
        _cli.run_full_duplex(_args(fixture, tmp_path / "must-not-start"))
    assert _FakeConversationBackend.constructed == 0
