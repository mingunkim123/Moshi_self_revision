from __future__ import annotations

from collections import Counter
from pathlib import Path
import wave

import pytest

from experiments.self_repair.mechanistic.conversation import ConversationContract
from experiments.self_repair.mechanistic.core import (
    ContractError,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_jsonl,
)
from experiments.self_repair.mechanistic.scripts._cli import (
    _load_trials,
    build_mech_manifest,
    validate_mechanistic_contract,
)


def _role_row(trial_id: str, fold: int, role: str) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "analysis_fold": fold,
        "role": role,
    }


def test_v2_roles_are_derived_from_frozen_fold_policy(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        _role_row("d1", 1, "discovery"),
        _role_row("d3", 3, "discovery"),
        _role_row("i4", 4, "internal_validation"),
        _role_row("i5", 5, "internal_validation"),
    ]
    write_jsonl(manifest, rows)

    assert [row["trial_id"] for row in _load_trials(manifest, "discovery")] == ["d1", "d3"]
    assert [row["trial_id"] for row in _load_trials(manifest, "smoke")] == ["d1", "d3"]
    assert [row["trial_id"] for row in _load_trials(manifest, "internal_validation")] == ["i4", "i5"]
    assert [row["trial_id"] for row in _load_trials(manifest, "internal_validation", [5])] == ["i5"]

    with pytest.raises(ContractError, match="cannot use folds"):
        _load_trials(manifest, "discovery", [4])
    rows[0]["role"] = "internal_validation"
    write_jsonl(manifest, rows)
    with pytest.raises(ContractError, match="frozen v2 fold policy"):
        _load_trials(manifest, "discovery")


def test_external_roles_require_the_exact_bound_role_manifest(tmp_path: Path) -> None:
    role_manifest = tmp_path / "roles.jsonl"
    role_row = {"trial_id": "formal-1", "role": "formal_confirmation"}
    write_jsonl(role_manifest, [role_row])
    manifest = tmp_path / "manifest.jsonl"
    manifest_row = {
        "trial_id": "formal-1",
        "role": "formal_confirmation",
        "role_manifest_sha256": sha256_file(role_manifest),
        "role_binding_sha256": sha256_value(role_row),
    }
    write_jsonl(manifest, [manifest_row])
    assert _load_trials(
        manifest, "formal_confirmation", role_manifest=role_manifest
    ) == [manifest_row]

    role_row["role"] = "multivalue_calibration"
    write_jsonl(role_manifest, [role_row])
    with pytest.raises(ContractError, match="hash mismatch"):
        _load_trials(manifest, "formal_confirmation", role_manifest=role_manifest)


def test_real_seed17_manifest_builds_and_validates_all_600_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[4]
    data_root = root / "experiments/self_repair/dataset_v2"
    source = data_root / "evaluation/provisional_eval_trials_seedpilot_b977391.jsonl"
    prepared = data_root / "manifests/provisional_prepared_stimuli.jsonl"
    folds = data_root / "assignments/analysis_folds.jsonl"
    first_wav = next((data_root / "artifacts/provisional_prepared").glob("*.wav"), None)
    if not source.is_file() or not prepared.is_file() or first_wav is None:
        pytest.skip("full frozen seed-17 source assets are not present")

    output = tmp_path / "mechanistic_trials.jsonl"
    assert build_mech_manifest([
        "--source-eval-manifest", str(source),
        "--prepared-manifest", str(prepared),
        "--analysis-folds", str(folds),
        "--audio-root", str(data_root),
        "--seeds", "17",
        "--data-status", "exploratory_provisional",
        "--output", str(output),
    ]) == 0
    rows = read_jsonl(output)
    assert len(rows) == 600
    assert Counter(row["role"] for row in rows) == {
        "discovery": 360,
        "internal_validation": 240,
    }
    contracts = [ConversationContract.from_manifest_row(row) for row in rows]
    assert all(contract.query_end_frame == contract.user_end_frame for contract in contracts)
    assert all(contract.response_capture_frames == 500 for contract in contracts)
    assert all(
        contract.target_end_frame_count == contract.user_end_frame + 500
        for contract in contracts
    )

    monkeypatch.setenv("NO_TORCH_COMPILE", "1")
    monkeypatch.setenv("NO_CUDA_GRAPH", "1")
    assert validate_mechanistic_contract([
        "--config", str(root / "experiments/self_repair/mechanistic/config/mechanistic.json"),
        "--manifest", str(output),
        "--input-artifact-root", str(data_root),
        "--output-root", str(tmp_path / "preflight"),
        "--model-repo", "kyutai/moshiko-pytorch-bf16",
        "--model-revision", "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289",
        "--dry-run",
    ]) == 0


def test_reviewed_multivalue_prepared_only_path_has_a_frozen_capture_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_root = tmp_path / "controls"
    wav = audio_root / "audio/formal-1.wav"
    wav.parent.mkdir(parents=True)
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(b"\0\0" * (12 * 1_920))
    audio_sha = sha256_file(wav)
    prepared_timing = {"utterance_end_ms": 640, "timebase": "prepared_stream_relative"}
    capture = {
        "condition": "repair_immediate",
        "timebase": "prepared_stream_relative",
        "stream_origin_ms": 0,
        "prepared_timing": prepared_timing,
        "prepared_timing_sha256": sha256_value(prepared_timing),
        "primary_window_start_ms": 640,
        "utterance_end_ms": 640,
        "response_capture_ms": 40_000,
        "requested_target_end_ms": 40_640,
        "target_end_sample_count": 508 * 1_920,
        "target_end_frame_count": 508,
        "actual_target_end_ms": 40_640,
    }
    execution = {
        "input_sample_rate": 24_000,
        "mimi_frame_samples": 1_920,
        "prefix_silence_ms": 0,
        "response_capture_ms": 40_000,
        "reset_model_stream_between_trials": True,
        "reset_rng_for_each_trial_seed": True,
        "required_model_type": "moshi",
        "required_max_lm_delay": 1,
    }
    input_stimulus = {
        "prepared_stimulus_id": "formal-1",
        "uri": "audio/formal-1.wav",
        "sha256": audio_sha,
        "duration_ms": 960,
        "sample_rate": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "timeline": "prepared_stream_relative",
        "mimi_frame_samples": 1_920,
    }
    prepared_row = {
        "trial_id": "formal-1",
        "prepared_stimulus_id": "formal-1",
        "scenario_id": "scenario-1",
        "condition": "repair_immediate",
        "speaker_id": "speaker-1",
        "old_value": "Boston",
        "new_value": "Denver",
        "prepared_stimulus": input_stimulus,
        "input_stimulus": input_stimulus,
        "prepared_timing": prepared_timing,
        "capture_contract": capture,
        "execution_contract": execution,
        "conversation_contract_source": "reviewed_multivalue_frozen_capture_contract",
        "model_repo": "kyutai/moshiko-pytorch-bf16",
        "resolved_revision": "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289",
    }
    prepared_manifest = audio_root / "prepared_stimuli.jsonl"
    role_manifest = audio_root / "role_manifest.jsonl"
    write_jsonl(prepared_manifest, [prepared_row])
    write_jsonl(role_manifest, [{"trial_id": "formal-1", "role": "formal_confirmation"}])
    output = tmp_path / "formal_manifest.jsonl"
    assert build_mech_manifest([
        "--prepared-manifest", str(prepared_manifest),
        "--role-manifest", str(role_manifest),
        "--audio-root", str(audio_root),
        "--data-status", "reviewed_multivalue",
        "--output", str(output),
    ]) == 0
    row = read_jsonl(output)[0]
    assert row["direction_id"] == (
        "ordered_value_pair_" + sha256_value(["Boston", "Denver"])[:16]
    )
    contract = ConversationContract.from_manifest_row(row)
    assert contract.user_start_frame == 0
    assert contract.user_end_frame == 8
    assert contract.target_end_frame_count == 508
    assert _load_trials(
        output, "formal_confirmation", role_manifest=role_manifest
    )[0]["trial_id"] == "formal-1"

    root = Path(__file__).resolve().parents[4]
    monkeypatch.setenv("NO_TORCH_COMPILE", "1")
    monkeypatch.setenv("NO_CUDA_GRAPH", "1")
    assert validate_mechanistic_contract([
        "--config", str(root / "experiments/self_repair/mechanistic/config/mechanistic.json"),
        "--manifest", str(output),
        "--input-artifact-root", str(audio_root),
        "--output-root", str(tmp_path / "formal-preflight"),
        "--dry-run",
    ]) == 0
