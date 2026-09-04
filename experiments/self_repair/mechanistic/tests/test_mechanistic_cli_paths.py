from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from experiments.self_repair.mechanistic.core import (
    ContractError,
    MODEL_REPO,
    MODEL_REVISION,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
    write_jsonl,
)
from experiments.self_repair.mechanistic.runtime import ReplayResult, SyntheticBackend
from experiments.self_repair.mechanistic.scripts import _cli


class _FakeTorch:
    @staticmethod
    def as_tensor(value, device=None):
        del device
        return np.asarray(value)


class _FakeTokenizer:
    @staticmethod
    def encode(text: str, out_type=int):
        del out_type
        return [ord(character) for character in text]


class _FakeLMGen:
    def __init__(self) -> None:
        self.lm_model = SimpleNamespace(zero_token_id=0, dep_q=8, delays=[0, 1])
        self.snapshots = 0

    def snapshot_streaming_state(self):
        self.snapshots += 1
        return ("snapshot", self.snapshots)


class _FakeMoshiBackend:
    encode_calls: list[tuple[str, int]] = []
    replay_calls: list[dict[str, object]] = []
    score_calls: list[dict[str, object]] = []

    def __init__(self, **kwargs) -> None:
        assert kwargs == {
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "use_sampling": False,
        }
        self.torch = _FakeTorch()
        self.tokenizer = _FakeTokenizer()
        self.lm_gen = _FakeLMGen()
        self.device = "fake-cuda"

    @classmethod
    def reset_calls(cls) -> None:
        cls.encode_calls = []
        cls.replay_calls = []
        cls.score_calls = []

    def encode_conversation_file(self, path: Path, *, target_frame_count: int):
        type(self).encode_calls.append((path.name, target_frame_count))
        user_frames = 5
        user = np.arange(1 * 8 * user_frames, dtype=np.int64).reshape(1, 8, user_frames)
        conversation = np.zeros((1, 8, target_frame_count), dtype=np.int64)
        conversation[..., :user_frames] = user
        silence = np.full((1, 8, target_frame_count), 7, dtype=np.int64)
        return SimpleNamespace(
            user_codes=user, conversation_codes=conversation,
            assistant_silence_codes=silence, user_frame_count=user_frames,
            target_frame_count=target_frame_count,
        )

    def replay_codes(
        self, codes, *, sites=(), replacement=None, capture_layers=None,
        capture_frames=None, end_frame_exclusive=None, hook_enabled=True,
    ):
        array = np.asarray(codes)
        frames = int(array.shape[-1] if end_frame_exclusive is None else end_frame_exclusive)
        type(self).replay_calls.append({
            "sites": list(sites), "replacement": replacement,
            "capture_layers": capture_layers, "capture_frames": capture_frames,
            "end_frame_exclusive": frames, "hook_enabled": hook_enabled,
        })
        events = {}
        activations = {}
        if hook_enabled:
            layers = list(capture_layers) if capture_layers is not None else [0]
            selected_frames = list(capture_frames) if capture_frames is not None else list(range(frames))
            for site in sites:
                values = []
                for layer in layers:
                    for frame in selected_frames:
                        value = np.full((1, 1, 4), layer * 100 + frame, dtype=np.float32)
                        events[(site, layer, frame)] = value
                        values.append(value)
                if values:
                    activations[site] = np.asarray(values, dtype=np.float32)
        digest = hashlib.sha256()
        for _ in range(frames + 1):
            digest.update(np.zeros((1,), dtype=np.int64).tobytes())
            digest.update(np.zeros((1, 8), dtype=np.int64).tobytes())
        logits = np.arange(1 * 1 * frames * 16, dtype=np.float32).reshape(1, 1, frames, 16)
        return ReplayResult(
            activations=activations, logits=logits, feedback_sha256=digest.hexdigest(),
            frame_count=frames, event_tensors=events, lm_step_count=frames + 1,
        )

    def score_candidates(self, snapshot, candidates, **kwargs):
        type(self).score_calls.append({
            "snapshot": snapshot, "order": list(candidates), **kwargs,
        })
        return {name: -float(len(text)) for name, text in candidates.items()}


def _config(root: Path) -> Path:
    path = root / "config.json"
    write_json(path, {
        "schema_version": "1.0.0",
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "layers": 32,
            "max_lm_delay": 1,
        },
        "audio": {
            "sample_rate": 24_000,
            "mimi_frame_samples": 1_920,
            "frame_ms": 80,
        },
        "open_loop_policy": {
            "primary": "zero_text_and_audio_tokens",
            "text_feedback": "model_zero_token",
            "audio_feedback": "model_zero_token",
            "sampled_tokens_enter_feedback": False,
        },
    })
    return path


def _readouts(root: Path) -> Path:
    path = root / "readouts.json"
    write_json(path, {
        "schema_version": "1.0.0",
        "candidate_scoring": "mean_log_probability_per_token",
        "candidate_branching": "restore_identical_query_snapshot_before_each_candidate",
        "schedule_aggregation": "logmeanexp_over_all_preregistered_schedules",
        "readouts": [
            {"id": "root", "prefix": "The current destination is", "anchor": "query_end"},
        ],
        "emission_schedules": [
            {"id": "immediate", "prefix_start_offset_frames": 0,
             "pad_frames_between_tokens": 0},
            {"id": "padded", "prefix_start_offset_frames": 1,
             "pad_frames_between_tokens": 1},
        ],
    })
    return path


def _trials(root: Path) -> tuple[Path, list[dict[str, object]]]:
    rows: list[dict[str, object]] = [
        {
            "trial_id": "clean", "scenario_id": "scenario",
            "condition": "clean_current", "old_value": "Boston",
            "new_value": "Seattle", "frame_count": 5,
            "sample_count": 5 * 1_920, "analysis_fold": 1, "role": "discovery",
        },
        {
            "trial_id": "repair", "scenario_id": "scenario",
            "condition": "repair_immediate", "old_value": "Boston",
            "new_value": "Seattle", "frame_count": 5,
            "sample_count": 5 * 1_920, "analysis_fold": 1, "role": "discovery",
        },
    ]
    path = root / "manifest.jsonl"
    write_jsonl(path, rows)
    return path, rows


def _anchors(root: Path, *, include_repair: bool = True) -> Path:
    rows = []
    trial_ids = ["clean", "repair"] if include_repair else ["clean"]
    for trial_id in trial_ids:
        rows.extend([
            {"trial_id": trial_id, "anchor": "D1_end", "frame": 1,
             "time_ms": 160.0, "timebase": "prepared_stream_relative"},
            {"trial_id": trial_id, "anchor": "query_end", "frame": 2,
             "time_ms": 240.0, "timebase": "prepared_stream_relative"},
        ])
    path = root / ("anchors.jsonl" if include_repair else "anchors-missing.jsonl")
    write_jsonl(path, rows)
    return path


def _encode_synthetic(root: Path, manifest: Path) -> tuple[Path, Path]:
    output_root = root / "encoded"
    output_manifest = root / "encoded_manifest.jsonl"
    assert _cli.encode_user_audio([
        "--manifest", str(manifest), "--output-root", str(output_root),
        "--output-manifest", str(output_manifest), "--synthetic",
    ]) == 0
    return output_root, output_manifest


def _real_trials(root: Path) -> Path:
    manifest, rows = _trials(root)
    audio_root = root / "audio"
    audio_root.mkdir()
    for index, row in enumerate(rows):
        wav = audio_root / f"{row['trial_id']}.wav"
        with wave.open(str(wav), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24_000)
            handle.writeframes(bytes([index, 0]) * (5 * 1_920))
        row["audio_uri"] = f"audio/{wav.name}"
        row["audio_sha256"] = sha256_file(wav)
    write_jsonl(manifest, rows)
    return manifest


def _bound_readouts(source_path: Path) -> Path:
    source = read_json(source_path)
    source["readouts"] = [
        {**row, "prefix_token_ids": _FakeTokenizer.encode(row["prefix"])}
        for row in source["readouts"]
    ]
    source["candidate_token_ids"] = {
        value: _FakeTokenizer.encode(value) for value in ("Boston", "Seattle")
    }
    source["model_revision"] = MODEL_REVISION
    source["bound_readout_sha256"] = sha256_value(source)
    path = source_path.parent / "readouts.bound.json"
    write_json(path, source)
    return path


def test_encode_stores_all_exact_streams_and_resume_identity(tmp_path: Path) -> None:
    manifest, _ = _trials(tmp_path)
    output_root, output_manifest = _encode_synthetic(tmp_path, manifest)
    rows = read_jsonl(output_manifest)
    assert len(rows) == 2
    assert all(row["user_frame_end_exclusive"] == 5 for row in rows)
    assert all(row["conversation_frame_end_exclusive"] == 5 for row in rows)
    assert all(row["assistant_silence_frame_end_exclusive"] == 5 for row in rows)
    assert all(row["repeat_encode_check"] == "passed" for row in rows)
    for row in rows:
        artifact = output_manifest.parent / row["codes_uri"]
        with np.load(artifact, allow_pickle=False) as archive:
            assert {"codes", "user_codes", "conversation_codes", "assistant_silence_codes"} <= set(
                archive.files)
            np.testing.assert_array_equal(archive["codes"], archive["user_codes"])
            np.testing.assert_array_equal(
                archive["conversation_codes"][..., :5], archive["user_codes"])
    summary = read_json(output_manifest.with_suffix(".jsonl.summary.json"))
    assert summary["passed"] is True
    assert summary["exact_half_open_coverage_count"] == 2
    assert summary["repeated_encode_mismatch_count"] == 0
    before = {path.name: sha256_file(path) for path in output_root.glob("*.npz")}
    assert _cli.encode_user_audio([
        "--manifest", str(manifest), "--output-root", str(output_root),
        "--output-manifest", str(output_manifest), "--synthetic", "--resume",
    ]) == 0
    assert before == {path.name: sha256_file(path) for path in output_root.glob("*.npz")}

    changed = read_jsonl(manifest)
    changed[0]["new_value"] = "Denver"
    write_jsonl(manifest, changed)
    with pytest.raises(ContractError, match="resume identity"):
        _cli.encode_user_audio([
            "--manifest", str(manifest), "--output-root", str(output_root),
            "--output-manifest", str(output_manifest), "--synthetic", "--resume",
        ])


def test_open_loop_checks_are_derived_from_synthetic_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _readouts(tmp_path)
    manifest, _ = _trials(tmp_path)
    _, encoded_manifest = _encode_synthetic(tmp_path, manifest)
    output = tmp_path / "open-loop.json"
    assert _cli.validate_open_loop([
        "--config", str(config), "--encoded-manifest", str(encoded_manifest),
        "--output", str(output), "--synthetic",
    ]) == 0
    report = read_json(output)
    assert report["passed"] is True
    assert set(report["checks"]) == {
        "paired_feedback_identical", "sampled_feedback_absent", "deterministic_replay",
        "identity_patch_noop", "candidate_order_invariant", "delay_mapping_valid",
    }
    assert report["evidence"]["paired_feedback"][0]["compared_frame_span"] == [0, 5]

    class NondeterministicSynthetic(SyntheticBackend):
        calls = 0

        def replay(self, trial, sites):
            result: ReplayResult = super().replay(trial, sites)
            type(self).calls += 1
            if type(self).calls == 2:
                return replace(result, logits=result.logits + 1)
            return result

    monkeypatch.setattr(_cli, "SyntheticBackend", NondeterministicSynthetic)
    with pytest.raises(ContractError, match="deterministic_replay"):
        _cli.validate_open_loop([
            "--config", str(config), "--encoded-manifest", str(encoded_manifest),
            "--output", str(output), "--synthetic",
        ])
    failed = read_json(output)
    assert failed["passed"] is False
    assert failed["checks"]["deterministic_replay"] is False


def test_score_readouts_uses_anchor_schedules_and_both_candidate_orders(tmp_path: Path) -> None:
    config = _config(tmp_path)
    readouts = _readouts(tmp_path)
    manifest, _ = _trials(tmp_path)
    anchors = _anchors(tmp_path)
    output = tmp_path / "scores.jsonl"
    assert _cli.score_readouts([
        "--config", str(config), "--readouts", str(readouts),
        "--manifest", str(manifest), "--anchors", str(anchors),
        "--role", "discovery", "--output", str(output), "--synthetic",
    ]) == 0
    rows = read_jsonl(output)
    assert len(rows) == 2
    assert all(row["anchor_frame"] == 2 for row in rows)
    assert all(row["anchor_end_frame_exclusive"] == 3 for row in rows)
    assert all([schedule["schedule_id"] for schedule in row["schedules"]]
               == ["immediate", "padded"] for row in rows)
    assert all(schedule["forward_order"] == ["target", "stale"]
               and schedule["reverse_order"] == ["stale", "target"]
               for row in rows for schedule in row["schedules"])
    assert all(row["candidate_order_delta"] == 0 for row in rows)

    with pytest.raises(ContractError, match="anchor .* is missing"):
        _cli.score_readouts([
            "--config", str(config), "--readouts", str(readouts),
            "--manifest", str(manifest), "--anchors", str(_anchors(tmp_path, include_repair=False)),
            "--role", "discovery", "--output", str(output), "--synthetic",
        ])


def test_capture_is_bounded_to_requested_sites_layers_and_frames_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    manifest, _ = _trials(tmp_path)
    anchors = _anchors(tmp_path)
    output_root = tmp_path / "capture"
    arguments = [
        "--config", str(config), "--manifest", str(manifest),
        "--anchor-map", str(anchors), "--role", "discovery",
        "--sites", "logits,resid_post", "--layers", "1,3",
        "--anchors", "D1_end,query_end", "--output-root", str(output_root),
        "--synthetic",
    ]
    assert _cli.capture_activations(arguments) == 0
    rows = read_jsonl(output_root / "capture_manifest.jsonl")
    assert len(rows) == 2
    assert all(row["replay_frame_span"] == [0, 3] for row in rows)
    assert all(row["captured_tensor_count"] == 6 for row in rows)
    assert all({item["site"] for item in row["tensors"]} == {"logits", "resid_post"}
               for row in rows)
    assert all({item["layer"] for item in row["tensors"] if item["site"] == "resid_post"}
               == {1, 3} for row in rows)
    assert all({item["frame"] for item in row["tensors"]} == {1, 2} for row in rows)
    for row in rows:
        artifact = output_root / row["feature_uri"]
        with np.load(artifact, allow_pickle=False) as archive:
            assert not any("L000" in key or "F00000000" in key for key in archive.files)

    class MustNotConstruct:
        def __init__(self, *args, **kwargs):
            raise AssertionError("resume constructed a backend despite complete verified artifacts")

    monkeypatch.setattr(_cli, "SyntheticBackend", MustNotConstruct)
    assert _cli.capture_activations([*arguments, "--resume"]) == 0


def test_fake_moshiko_paths_use_exact_gpu_contract_without_cuda_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    source_readouts = _readouts(tmp_path)
    bound_readouts = _bound_readouts(source_readouts)
    manifest = _real_trials(tmp_path)
    anchors = _anchors(tmp_path)
    encoded_root = tmp_path / "real-encoded"
    encoded_manifest = tmp_path / "real-encoded.jsonl"
    _FakeMoshiBackend.reset_calls()
    monkeypatch.setattr(_cli, "MoshiBackend", _FakeMoshiBackend)

    assert _cli.encode_user_audio([
        "--manifest", str(manifest), "--input-artifact-root", str(tmp_path),
        "--output-root", str(encoded_root), "--output-manifest", str(encoded_manifest),
    ]) == 0
    # The deterministic bounded repeat subset is first+last; with two rows every
    # input is encoded once for use and once for the repeat-encode gate.
    assert len(_FakeMoshiBackend.encode_calls) == 4
    assert read_json(encoded_manifest.with_suffix(".jsonl.summary.json"))[
        "repeated_encode_check_count"] == 2

    open_loop_output = tmp_path / "real-open-loop.json"
    assert _cli.validate_open_loop([
        "--config", str(config), "--encoded-manifest", str(encoded_manifest),
        "--output", str(open_loop_output),
    ]) == 0
    assert read_json(open_loop_output)["passed"] is True
    assert any(call["hook_enabled"] is False for call in _FakeMoshiBackend.replay_calls)
    assert any(call["replacement"] for call in _FakeMoshiBackend.replay_calls)

    _FakeMoshiBackend.replay_calls = []
    _FakeMoshiBackend.score_calls = []
    scores = tmp_path / "real-scores.jsonl"
    assert _cli.score_readouts([
        "--config", str(config), "--readouts", str(bound_readouts),
        "--manifest", str(manifest), "--encoded-manifest", str(encoded_manifest),
        "--anchors", str(anchors), "--role", "discovery", "--output", str(scores),
    ]) == 0
    assert all(call["end_frame_exclusive"] == 3 for call in _FakeMoshiBackend.replay_calls)
    assert all(call["hook_enabled"] is False for call in _FakeMoshiBackend.replay_calls)
    assert len(_FakeMoshiBackend.score_calls) == 2 * 2 * 2
    for forward, reverse in zip(
        _FakeMoshiBackend.score_calls[::2], _FakeMoshiBackend.score_calls[1::2], strict=True
    ):
        assert forward["snapshot"] == reverse["snapshot"]
        assert forward["order"] == ["target", "stale"]
        assert reverse["order"] == ["stale", "target"]

    _FakeMoshiBackend.replay_calls = []
    capture_root = tmp_path / "real-capture"
    assert _cli.capture_activations([
        "--config", str(config), "--manifest", str(manifest),
        "--encoded-manifest", str(encoded_manifest), "--anchor-map", str(anchors),
        "--role", "discovery", "--sites", "resid_post", "--layers", "1,3",
        "--anchors", "D1_end,query_end", "--output-root", str(capture_root),
    ]) == 0
    assert all(call["capture_layers"] == [1, 3] for call in _FakeMoshiBackend.replay_calls)
    assert all(call["capture_frames"] == [1, 2] for call in _FakeMoshiBackend.replay_calls)
    assert all(call["end_frame_exclusive"] == 3 for call in _FakeMoshiBackend.replay_calls)
