#!/usr/bin/env python3
"""Execute and resume the frozen v2 Moshi evaluation manifest.

Each completed trial is first persisted as an atomic per-trial record.  The
consolidated JSONL manifest is then checkpointed atomically, so a process failure
can be resumed without rerunning or silently losing a completed model response.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

try:  # Support direct script execution and package-style imports.
    from .audio_utils import duration_ms, read_pcm16_mono, write_pcm16_mono
    from .build_eval_adapter import (
        eval_identity,
        generation_config_hash,
        generation_parameters,
        validate_eval_trials,
    )
    from .common import (
        DATASET_ROOT,
        REPOSITORY_ROOT,
        read_config,
        read_jsonl,
        sha256_file,
        sha256_value,
        write_json,
        write_jsonl,
    )
    from .response_validation import (
        response_evidence_hash,
        validate_stream_events,
        validate_trial_response,
        verify_input_stimulus_file,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI use.
    from audio_utils import duration_ms, read_pcm16_mono, write_pcm16_mono
    from build_eval_adapter import (
        eval_identity,
        generation_config_hash,
        generation_parameters,
        validate_eval_trials,
    )
    from common import (
        DATASET_ROOT,
        REPOSITORY_ROOT,
        read_config,
        read_jsonl,
        sha256_file,
        sha256_value,
        write_json,
        write_jsonl,
    )
    from response_validation import (
        response_evidence_hash,
        validate_stream_events,
        validate_trial_response,
        verify_input_stimulus_file,
    )


RUNNER_VERSION = "2.2.1"
DEFAULT_INPUT = DATASET_ROOT / "evaluation/eval_trials.jsonl"
DEFAULT_OUTPUT = DATASET_ROOT / "evaluation/eval_trials.completed.jsonl"
DEFAULT_RESPONSE_ROOT = DATASET_ROOT / "evaluation/response_artifacts"
IMMUTABLE_TRIAL_EXCLUSIONS = {"response", "stream_events"}


@dataclass(frozen=True)
class BackendOutput:
    """Backend-neutral response returned for exactly one reset trial."""

    audio: np.ndarray
    sample_rate: int
    token_ids: Sequence[int]
    token_pieces: Sequence[str]
    frame_step_ms: float
    eos_reached: bool


class EvalBackend(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def reset_trial(self, seed: int) -> None: ...

    def infer(
        self, input_audio: np.ndarray, input_stimulus: Mapping[str, Any]
    ) -> BackendOutput: ...


BackendFactory = Callable[[dict[str, str], dict[str, Any]], EvalBackend]


class _SilentPrinter:
    def print_header(self) -> None:
        pass

    def print_token(self, token: str) -> None:
        pass

    def log(self, level: str, msg: str) -> None:
        pass


def _snapshot_revision(path: Path) -> str | None:
    # Hugging Face snapshot files are normally symlinks into the same model
    # cache's ``blobs`` directory.  Inspecting ``resolve()`` first therefore
    # discards the immutable ``snapshots/<commit>`` identity.  Preserve that
    # lexical identity, then independently prove that the file resolves either
    # inside the snapshot itself or inside the same model cache's blob store.
    absolute = path.absolute()
    parts = absolute.parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    if index + 1 >= len(parts):
        return None
    revision = parts[index + 1]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        return None
    try:
        resolved = absolute.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file():
        return None
    snapshot_root = Path(*parts[: index + 2])
    model_cache_root = Path(*parts[:index])
    allowed_roots = (snapshot_root.resolve(), (model_cache_root / "blobs").resolve())
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        return None
    return revision


def _verify_clean_git_identity(expected_commit: str) -> str:
    try:
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"],
            cwd=REPOSITORY_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "cannot verify the frozen code_commit; packaged/local source "
            "execution is fail-closed"
        ) from error
    if current_commit != expected_commit:
        raise RuntimeError("checked-out Git commit does not match frozen code_commit")
    if diff.returncode == 1:
        raise RuntimeError("tracked source differs from frozen code_commit")
    if diff.returncode != 0:
        raise RuntimeError("could not verify tracked source cleanliness")
    return current_commit


class MoshiTorchBackend:
    """Thin adapter over the repository's PyTorch ``InferenceState`` entrypoint."""

    def __init__(self, identity: dict[str, str], config: dict[str, Any]) -> None:
        os.environ.setdefault("NO_TORCH_COMPILE", "1")
        try:
            import torch
            import moshi
            from moshi.models import loaders
            from moshi.run_inference import InferenceState, seed_all
        except ImportError as error:  # pragma: no cover - requires model runtime.
            raise RuntimeError(
                "Moshi runtime dependencies are unavailable; install the repository's "
                "PyTorch inference environment before executing v2 evaluation."
            ) from error

        self._torch = torch
        self._seed_all = seed_all
        self._identity = identity
        device = str(config.get("device", "cuda"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested by the frozen config but is unavailable")
        dtype_name = str(config.get("dtype", "bfloat16"))
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        if dtype_name not in dtype_map:
            raise ValueError(f"unsupported frozen dtype: {dtype_name}")
        cfg_coef = float(config.get("cfg_coef", 1.0))
        current_commit = _verify_clean_git_identity(identity["code_commit"])

        seed_all(0)
        checkpoint = loaders.CheckpointInfo.from_hf_repo(
            identity["model_repo"], revision=identity["resolved_revision"]
        )
        if checkpoint.model_type != "moshi":
            raise RuntimeError(
                f"v2 evaluation supports model_type='moshi' only, got {checkpoint.model_type!r}"
            )
        snapshot_revision = _snapshot_revision(checkpoint.moshi_weights)
        if snapshot_revision is None:
            raise RuntimeError(
                "model weights are not from a verifiable Hugging Face snapshot; "
                "local/package weight identity is fail-closed"
            )
        if snapshot_revision != identity["resolved_revision"]:
            raise RuntimeError("loaded Hugging Face snapshot differs from frozen revision")
        mimi = checkpoint.get_mimi(device=device)
        tokenizer = checkpoint.get_text_tokenizer()
        lm = checkpoint.get_moshi(device=device, dtype=dtype_map[dtype_name])
        generation = dict(checkpoint.lm_gen_config)
        unknown_defaults = sorted(
            set(generation)
            - {
                "use_sampling",
                "temp",
                "temp_text",
                "top_k",
                "top_k_text",
                "check",
                "support_out_of_sync",
                "cfg_is_masked_until",
                "cfg_is_no_text",
            }
        )
        if unknown_defaults:
            raise RuntimeError(
                f"checkpoint exposes unsupported generation defaults: {unknown_defaults}"
            )
        generation.update(generation_parameters(config))
        self._state = InferenceState(
            checkpoint,
            mimi,
            tokenizer,
            lm,
            batch_size=1,
            cfg_coef=cfg_coef,
            device=device,
            **generation,
        )
        self._state.printer = _SilentPrinter()
        max_lm_delay = max(self._state.lm_gen.lm_model.delays)
        if max_lm_delay != 1:
            raise RuntimeError(
                f"v2 evaluation requires max LM delay 1, got {max_lm_delay}"
            )
        streaming = config["streaming"]
        if int(mimi.sample_rate) != int(streaming["input_sample_rate"]):
            raise RuntimeError("loaded Mimi sample rate differs from frozen streaming config")
        if int(self._state.frame_size) != int(streaming["mimi_frame_samples"]):
            raise RuntimeError("loaded Mimi frame size differs from frozen streaming config")
        self._mimi = mimi
        self._tokenizer = tokenizer
        self._device = device
        effective_generation = {
            "lm_gen": generation,
            "cfg_coef": cfg_coef,
            "device": device,
            "dtype": dtype_name,
        }
        self._metadata: dict[str, Any] = {
            "name": "moshi-pytorch-inference-state",
            "version": str(getattr(moshi, "__version__", RUNNER_VERSION)),
            "model_repo": identity["model_repo"],
            "resolved_revision": identity["resolved_revision"],
            "snapshot_revision": snapshot_revision,
            "code_commit": current_commit,
            "model_type": checkpoint.model_type,
            "mimi_sample_rate": int(mimi.sample_rate),
            "frame_size": int(self._state.frame_size),
            "max_lm_delay": int(max_lm_delay),
            "effective_generation_config": effective_generation,
            "effective_generation_config_sha256": sha256_value(effective_generation),
        }

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def reset_trial(self, seed: int) -> None:
        self._seed_all(seed)
        self._mimi.reset_streaming()
        self._state.lm_gen.reset_streaming()
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def infer(
        self, input_audio: np.ndarray, input_stimulus: Mapping[str, Any]
    ) -> BackendOutput:
        pcm = np.asarray(input_audio, dtype=np.float32)
        if pcm.ndim != 1:
            raise ValueError(f"unexpected Moshi input shape {pcm.shape}")
        frame_samples = int(input_stimulus["mimi_frame_samples"])
        if pcm.size % frame_samples:
            raise ValueError("input stopped being Mimi-frame aligned")
        in_pcm = self._torch.from_numpy(pcm[None, None, :]).to(device=self._device)
        with self._torch.no_grad():
            outputs = self._state.run(in_pcm)
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        if len(outputs) != 1:
            raise RuntimeError(f"expected one Moshi output, received {len(outputs)}")
        text_tokens, response_pcm = outputs[0]
        token_ids = [int(value) for value in text_tokens.reshape(-1).tolist()]
        pieces = [
            ""
            if token_id in (0, 3)
            else str(self._tokenizer.id_to_piece(token_id)).replace("▁", " ")
            for token_id in token_ids
        ]
        audio = np.asarray(response_pcm[0].numpy(), dtype=np.float32).reshape(-1)
        return BackendOutput(
            audio=audio,
            sample_rate=int(self._mimi.sample_rate),
            token_ids=token_ids,
            token_pieces=pieces,
            frame_step_ms=1000.0 / float(self._mimi.frame_rate),
            eos_reached=any(token_id == self._tokenizer.eos_id() for token_id in token_ids),
        )


def _immutable_trial(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in row.items() if key not in IMMUTABLE_TRIAL_EXCLUSIONS
    }


def _merge_progress(
    base_rows: Sequence[dict[str, Any]],
    progress_rows: Sequence[dict[str, Any]],
    *,
    response_root: Path,
) -> list[dict[str, Any]]:
    validate_eval_trials(base_rows)
    validate_eval_trials(progress_rows)
    if eval_identity(base_rows) != eval_identity(progress_rows):
        raise RuntimeError("existing evaluation output has a different run identity")
    base_by_id = {str(row["eval_trial_id"]): row for row in base_rows}
    progress_by_id = {str(row["eval_trial_id"]): row for row in progress_rows}
    if set(base_by_id) != set(progress_by_id):
        raise RuntimeError("existing evaluation output has a different trial set")
    merged: list[dict[str, Any]] = []
    for base in base_rows:
        trial_id = str(base["eval_trial_id"])
        progress = progress_by_id[trial_id]
        if _immutable_trial(base) != _immutable_trial(progress):
            raise RuntimeError(
                f"existing evaluation output changed immutable input/seed evidence: {trial_id}"
            )
        base_completed = base["response"]["status"] == "completed"
        progress_completed = progress["response"]["status"] == "completed"
        if base_completed and progress_completed and base != progress:
            raise RuntimeError(f"conflicting completed response evidence: {trial_id}")
        winner = progress if progress_completed else base
        if winner["response"]["status"] == "completed":
            validate_trial_response(
                winner, verify_audio=True, response_root=response_root
            )
        merged.append(dict(winner))
    return merged


def _record_name(eval_trial_id: str) -> str:
    return hashlib.sha256(eval_trial_id.encode("utf-8")).hexdigest() + ".json"


def _record_directory(response_root: Path, eval_run_id: str) -> Path:
    run_digest = hashlib.sha256(eval_run_id.encode("utf-8")).hexdigest()[:24]
    return response_root / "records" / run_digest


def _audio_path(response_root: Path, eval_run_id: str, eval_trial_id: str) -> Path:
    run_digest = hashlib.sha256(eval_run_id.encode("utf-8")).hexdigest()[:24]
    trial_digest = hashlib.sha256(eval_trial_id.encode("utf-8")).hexdigest()
    return response_root / "audio" / run_digest / f"{trial_digest}.wav"


def _load_record(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid atomic trial record: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"atomic trial record must contain one object: {path}")
    return value


def ingest_trial_records(
    rows: Sequence[dict[str, Any]], record_dir: Path, *, response_root: Path
) -> tuple[list[dict[str, Any]], int]:
    """Merge verified atomic per-trial records into a manifest in memory."""

    by_id = {str(row["eval_trial_id"]): dict(row) for row in rows}
    ingested = 0
    if not record_dir.exists():
        return [by_id[str(row["eval_trial_id"])] for row in rows], ingested
    for path in sorted(record_dir.glob("*.json")):
        record = _load_record(path)
        trial_id = str(record.get("eval_trial_id", ""))
        if not trial_id or trial_id not in by_id:
            raise RuntimeError(f"atomic record does not belong to this manifest: {path}")
        if path.name != _record_name(trial_id):
            raise RuntimeError(f"atomic record filename does not match trial identity: {path}")
        current = by_id[trial_id]
        if _immutable_trial(current) != _immutable_trial(record):
            raise RuntimeError(f"atomic record changed immutable trial evidence: {trial_id}")
        if record.get("response", {}).get("status") != "completed":
            raise RuntimeError(f"atomic record is not completed: {trial_id}")
        validate_trial_response(
            record, verify_audio=True, response_root=response_root
        )
        if current["response"]["status"] == "completed" and current != record:
            raise RuntimeError(f"atomic record conflicts with completed manifest row: {trial_id}")
        if current["response"]["status"] != "completed":
            ingested += 1
        by_id[trial_id] = record
    return [by_id[str(row["eval_trial_id"])] for row in rows], ingested


def _atomic_write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".wav", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_pcm16_mono(temporary, audio, sample_rate)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _backend_metadata(
    backend: EvalBackend,
    identity: Mapping[str, str],
    execution_contract: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = dict(backend.metadata)
    required = {
        "name",
        "version",
        "model_repo",
        "resolved_revision",
        "snapshot_revision",
        "code_commit",
        "model_type",
        "mimi_sample_rate",
        "frame_size",
        "max_lm_delay",
        "effective_generation_config",
        "effective_generation_config_sha256",
    }
    if set(metadata) != required:
        raise ValueError(f"backend metadata must contain exactly {sorted(required)}")
    if metadata["model_repo"] != identity["model_repo"]:
        raise ValueError("backend model_repo does not match frozen eval identity")
    if metadata["resolved_revision"] != identity["resolved_revision"]:
        raise ValueError("backend resolved_revision does not match frozen eval identity")
    if metadata["snapshot_revision"] != identity["resolved_revision"]:
        raise ValueError("backend snapshot does not match frozen eval identity")
    if metadata["code_commit"] != identity["code_commit"]:
        raise ValueError("backend code commit does not match frozen eval identity")
    if metadata["model_type"] != execution_contract["required_model_type"]:
        raise ValueError("backend model type does not match execution contract")
    if metadata["max_lm_delay"] != execution_contract["required_max_lm_delay"]:
        raise ValueError("backend max LM delay does not match execution contract")
    if metadata["mimi_sample_rate"] != execution_contract["input_sample_rate"]:
        raise ValueError("backend sample rate does not match execution contract")
    if metadata["frame_size"] != execution_contract["mimi_frame_samples"]:
        raise ValueError("backend frame size does not match execution contract")
    effective = metadata["effective_generation_config"]
    if not isinstance(effective, dict) or not effective:
        raise ValueError("backend effective generation config must be non-empty")
    if metadata["effective_generation_config_sha256"] != sha256_value(effective):
        raise ValueError("backend effective generation config hash mismatch")
    return metadata


def _completed_row(
    trial: Mapping[str, Any],
    output: BackendOutput,
    *,
    elapsed_seconds: float,
    audio_path: Path,
    response_root: Path,
    appended_zero_sample_count: int,
    backend_metadata: dict[str, Any],
) -> dict[str, Any]:
    audio = np.asarray(output.audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("backend response audio must be a non-empty mono vector")
    if not np.all(np.isfinite(audio)):
        raise ValueError("backend response audio contains non-finite samples")
    if (
        isinstance(output.sample_rate, bool)
        or not isinstance(output.sample_rate, int)
        or output.sample_rate <= 0
    ):
        raise ValueError("backend response sample_rate must be a positive integer")
    if len(output.token_ids) != len(output.token_pieces) or not output.token_ids:
        raise ValueError("backend must return non-empty, paired token IDs and pieces")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer))
        for token_id in output.token_ids
    ):
        raise ValueError("backend token IDs must be integers")
    if any(not isinstance(piece, str) for piece in output.token_pieces):
        raise ValueError("backend token pieces must be strings")
    if not np.isfinite(float(output.frame_step_ms)) or output.frame_step_ms <= 0:
        raise ValueError("backend frame_step_ms must be finite and positive")
    capture = trial["capture_contract"]
    execution = trial["execution_contract"]
    fed_samples = int(capture["target_end_sample_count"])
    fed_frames = int(capture["target_end_frame_count"])
    frame_samples = int(execution["mimi_frame_samples"])
    if output.sample_rate != execution["input_sample_rate"]:
        raise ValueError("backend response sample rate differs from execution contract")
    if len(output.token_ids) != fed_frames:
        raise RuntimeError(
            f"incomplete model coverage: {len(output.token_ids)} tokens for {fed_frames} fed frames"
        )
    if audio.size != fed_frames * frame_samples or audio.size != fed_samples:
        raise RuntimeError(
            f"incomplete model audio coverage: {audio.size} samples for {fed_samples} fed samples"
        )
    if output.eos_reached is not False:
        raise RuntimeError("early model EOS is forbidden by the capture contract")
    expected_step_ms = frame_samples * 1000.0 / output.sample_rate
    if abs(float(output.frame_step_ms) - expected_step_ms) > 1e-9:
        raise ValueError("backend response frame step differs from execution contract")
    events = [
        {
            "frame_index": index,
            "time_ms": index * float(output.frame_step_ms),
            "token_id": int(token_id),
            "piece": piece,
        }
        for index, (token_id, piece) in enumerate(
            zip(output.token_ids, output.token_pieces)
        )
    ]
    validate_stream_events(events)
    transcript = "".join(event["piece"] for event in events).strip()
    _atomic_write_audio(audio_path, audio, output.sample_rate)
    try:
        audio_uri = audio_path.resolve().relative_to(response_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("response audio path is outside response_root") from error
    elapsed = float(elapsed_seconds)
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_seconds must be finite and non-negative")
    response: dict[str, Any] = {
        "status": "completed",
        "transcript": transcript,
        "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "audio_path": audio_uri,
        "audio_sha256": sha256_file(audio_path),
        "audio_duration_ms": duration_ms(audio, output.sample_rate),
        "audio_sample_rate": output.sample_rate,
        "audio_channels": 1,
        "audio_sample_width_bytes": 2,
        "elapsed_seconds": elapsed,
        "generation_seed": int(trial["generation_seed"]),
        "timebase": "prepared_stream_relative",
        "stream_origin_ms": 0,
        "primary_window_start_ms": capture["primary_window_start_ms"],
        "requested_target_end_ms": capture["requested_target_end_ms"],
        "actual_target_end_ms": capture["actual_target_end_ms"],
        "fed_sample_count": fed_samples,
        "fed_frame_count": fed_frames,
        "output_sample_count": int(audio.size),
        "output_frame_count": int(audio.size // frame_samples),
        "appended_zero_sample_count": appended_zero_sample_count,
        "coverage_complete": True,
        "eos_reached": False,
        "stream_reset": True,
        "rng_reset": True,
        "backend": backend_metadata,
        "runner_source_sha256": execution["runner_source_sha256"],
        "effective_generation_config_sha256": backend_metadata[
            "effective_generation_config_sha256"
        ],
        "stream_events_sha256": sha256_value(events),
    }
    response["evidence_sha256"] = response_evidence_hash(response, events)
    completed = dict(trial)
    completed["response"] = response
    completed["stream_events"] = events
    validate_trial_response(
        completed, verify_audio=True, response_root=response_root
    )
    return completed


def _verify_generation_config(
    rows: Sequence[dict[str, Any]], generation_config: dict[str, Any]
) -> dict[str, str]:
    identity = eval_identity(rows)
    observed_hash = generation_config_hash(generation_config)
    if observed_hash != identity["generation_config_hash"]:
        raise RuntimeError(
            "frozen generation config hash does not match the eval run identity"
        )
    configured_repo = generation_config.get("model_repo")
    if configured_repo is not None and configured_repo != identity["model_repo"]:
        raise RuntimeError("generation config model_repo does not match eval identity")
    return identity


def _extended_trial_input(
    path: Path, trial: Mapping[str, Any]
) -> tuple[np.ndarray, int]:
    audio, sample_rate = read_pcm16_mono(path)
    execution = trial["execution_contract"]
    capture = trial["capture_contract"]
    if sample_rate != execution["input_sample_rate"]:
        raise ValueError(f"prepared stimulus sample rate changed: {path}")
    target_samples = int(capture["target_end_sample_count"])
    if audio.size > target_samples:
        raise ValueError(f"prepared stimulus extends beyond the capture target: {path}")
    appended = target_samples - int(audio.size)
    extended = np.concatenate(
        (audio.astype(np.float32, copy=False), np.zeros(appended, dtype=np.float32))
    )
    if extended.size != target_samples:
        raise AssertionError("extended input size does not match capture target")
    if appended and not np.all(extended[-appended:] == 0.0):
        raise AssertionError("capture extension is not exact digital zero")
    frame_samples = int(execution["mimi_frame_samples"])
    if extended.size % frame_samples:
        raise AssertionError("extended input is not Mimi-frame aligned")
    return extended, appended


def run_evaluation(
    pending_rows: Sequence[dict[str, Any]],
    *,
    generation_config: dict[str, Any],
    output_path: Path,
    artifact_root: Path,
    response_root: Path,
    backend: EvalBackend | None,
    backend_factory: BackendFactory | None = None,
    limit: int | None = None,
    checkpoint_every: int = 25,
    dry_run: bool = False,
    failure_injector: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    validate_eval_trials(pending_rows)
    identity = _verify_generation_config(pending_rows, generation_config)
    execution_contract = pending_rows[0]["execution_contract"]
    if sha256_file(Path(__file__)) != execution_contract["runner_source_sha256"]:
        raise RuntimeError("running source does not match frozen runner_source_sha256")
    streaming = generation_config.get("streaming")
    expected_streaming = {
        key: execution_contract[key]
        for key in (
            "input_sample_rate",
            "mimi_frame_samples",
            "prefix_silence_ms",
            "response_capture_ms",
            "reset_model_stream_between_trials",
            "reset_rng_for_each_trial_seed",
        )
    }
    if streaming != expected_streaming:
        raise RuntimeError("generation config streaming contract differs from eval manifest")
    rows = [dict(row) for row in pending_rows]
    if output_path.exists():
        rows = _merge_progress(
            rows, read_jsonl(output_path), response_root=response_root
        )
    record_dir = _record_directory(response_root, identity["eval_run_id"])
    rows, ingested = ingest_trial_records(
        rows, record_dir, response_root=response_root
    )

    hash_cache: dict[Path, str] = {}
    input_path_by_trial: dict[str, Path] = {}
    for row in rows:
        input_path_by_trial[str(row["eval_trial_id"])] = verify_input_stimulus_file(
            row["input_stimulus"],
            artifact_root=artifact_root,
            hash_cache=hash_cache,
        )
        if row["response"]["status"] == "completed":
            validate_trial_response(
                row, verify_audio=True, response_root=response_root
            )

    pending = [row for row in rows if row["response"]["status"] == "pending"]
    selected = pending[:limit] if limit is not None else pending
    if dry_run:
        return {
            "status": "dry_run_validated",
            "eval_run_id": identity["eval_run_id"],
            "trial_count": len(rows),
            "completed_count": len(rows) - len(pending),
            "pending_count": len(pending),
            "selected_count": len(selected),
            "unique_prepared_files_verified": len(hash_cache),
            "records_ingested": ingested,
        }
    if not selected:
        write_jsonl(output_path, rows)
        return {
            "status": "completed",
            "runner_version": RUNNER_VERSION,
            "eval_run_id": identity["eval_run_id"],
            "trial_count": len(rows),
            "executed_count": 0,
            "skipped_completed_count": len(rows),
            "records_ingested": ingested,
            "remaining_count": 0,
            "unique_prepared_files_verified": len(hash_cache),
            "output_manifest_sha256": sha256_file(output_path),
        }
    if backend is None:
        if backend_factory is None:
            raise ValueError("a model backend is required unless dry_run is true")
        backend = backend_factory(identity, generation_config)
    metadata = _backend_metadata(backend, identity, execution_contract)
    index_by_id = {
        str(row["eval_trial_id"]): index for index, row in enumerate(rows)
    }
    executed = 0
    for trial in selected:
        trial_id = str(trial["eval_trial_id"])
        seed = int(trial["generation_seed"])
        extended_input, appended_zeros = _extended_trial_input(
            input_path_by_trial[trial_id], trial
        )
        backend.reset_trial(seed)
        started = time.perf_counter()
        backend_output = backend.infer(extended_input, trial["input_stimulus"])
        elapsed = time.perf_counter() - started
        completed = _completed_row(
            trial,
            backend_output,
            elapsed_seconds=elapsed,
            audio_path=_audio_path(response_root, identity["eval_run_id"], trial_id),
            response_root=response_root,
            appended_zero_sample_count=appended_zeros,
            backend_metadata=metadata,
        )
        if failure_injector is not None:
            failure_injector("after_audio_before_record", trial_id)
        record_path = record_dir / _record_name(trial_id)
        write_json(record_path, completed)
        if failure_injector is not None:
            failure_injector("after_record_before_manifest", trial_id)
        rows[index_by_id[trial_id]] = completed
        executed += 1
        if executed % checkpoint_every == 0:
            write_jsonl(output_path, rows)
    write_jsonl(output_path, rows)
    remaining = sum(row["response"]["status"] == "pending" for row in rows)
    return {
        "status": "completed" if remaining == 0 else "partially_completed",
        "runner_version": RUNNER_VERSION,
        "eval_run_id": identity["eval_run_id"],
        "trial_count": len(rows),
        "executed_count": executed,
        "skipped_completed_count": len(rows) - len(pending),
        "records_ingested": ingested,
        "remaining_count": remaining,
        "unique_prepared_files_verified": len(hash_cache),
        "output_manifest_sha256": sha256_file(output_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root for artifact-relative prepared stimulus URIs.",
    )
    parser.add_argument("--response-root", type=Path, default=DEFAULT_RESPONSE_ROOT)
    parser.add_argument("--backend", choices=("moshi-pytorch",), default="moshi-pytorch")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: BackendFactory | None = None,
) -> dict[str, Any]:
    args = parse_args(argv)
    rows = read_jsonl(args.input)
    generation_config = read_config(args.generation_config)
    _verify_generation_config(rows, generation_config)
    report = run_evaluation(
        rows,
        generation_config=generation_config,
        output_path=args.output,
        artifact_root=args.artifact_root,
        response_root=args.response_root,
        backend=None,
        backend_factory=backend_factory or MoshiTorchBackend,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
        dry_run=args.dry_run,
    )
    if args.report and not args.dry_run:
        write_json(args.report, report)
    remaining = report.get("remaining_count", report.get("pending_count", 0))
    if args.dry_run:
        print(
            f"{report['status']}: {report['selected_count']} selected; "
            f"{remaining} pending"
        )
    else:
        print(
            f"{report['status']}: {report.get('executed_count', 0)} executed; "
            f"{remaining} pending"
        )
    return report


if __name__ == "__main__":
    main()
