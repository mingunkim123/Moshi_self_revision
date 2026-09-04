from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core import ContractError, FRAME_SAMPLES, MODEL_REPO, MODEL_REVISION, validate_runtime_environment


@dataclass(frozen=True)
class ReplayResult:
    activations: dict[str, np.ndarray]
    logits: np.ndarray
    feedback_sha256: str
    frame_count: int
    event_tensors: dict[tuple[str, int, int], np.ndarray]


class SyntheticBackend:
    """Analytic fixture backend; never valid as empirical model evidence."""

    def __init__(self, *, layers: int = 6, heads: int = 4, hidden: int = 16, seed: int = 17):
        self.layers = layers
        self.heads = heads
        self.hidden = hidden
        self.seed = seed

    def _vector(self, label: str) -> np.ndarray:
        seed = int(hashlib.sha256(label.encode()).hexdigest()[:16], 16) ^ self.seed
        return np.random.default_rng(seed).normal(size=self.hidden)

    def replay(self, trial: Mapping[str, Any], sites: Sequence[str]) -> ReplayResult:
        frames = int(trial.get("frame_count", 12))
        old = str(trial.get("old_value", "Boston"))
        new = str(trial.get("new_value", trial.get("target_value", "Seattle")))
        condition = str(trial.get("condition", "repair"))
        old_v, new_v = self._vector(old), self._vector(new)
        activations: dict[str, np.ndarray] = {}
        for site in sites:
            values = np.empty((self.layers, frames, self.hidden), dtype=np.float32)
            for layer in range(self.layers):
                update = (layer + 1) / self.layers
                target_weight = 1.0 if condition.startswith("clean") else 0.35 + 0.45 * update
                base = target_weight * new_v + (1.0 - target_weight) * old_v
                for frame in range(frames):
                    values[layer, frame] = base + frame * 0.001
            activations[site] = values
        direction = new_v - old_v
        margin = np.array([float(np.dot(activations[sites[-1]][-1, -1], direction))], dtype=np.float32)
        feedback = np.zeros((frames, 9), dtype=np.int64)
        return ReplayResult(activations, margin, hashlib.sha256(feedback.tobytes()).hexdigest(), frames, {})

    def patch(
        self, recipient: Mapping[str, Any], donor: Mapping[str, Any], *, component: str,
        layer: int, head: int | None, anchor_frame: int,
    ) -> dict[str, Any]:
        recipient_run = self.replay(recipient, [component])
        donor_run = self.replay(donor, [component])
        old = self._vector(str(recipient.get("old_value", "Boston")))
        new = self._vector(str(recipient.get("new_value", "Seattle")))
        direction = new - old
        baseline = float(recipient_run.logits[0])
        source = donor_run.activations[component][layer % self.layers, anchor_frame % donor_run.frame_count]
        target = recipient_run.activations[component][layer % self.layers, anchor_frame % recipient_run.frame_count]
        scale = {"resid_post": 1.0, "attn_out": 0.7, "mlp_out": 0.3, "head_z": 0.6,
                 "k_only": 0.35, "v_only": 0.45, "kv": 0.65, "path": 0.5}.get(component, 0.5)
        patched = baseline + scale * float(np.dot(source - target, direction))
        return {"baseline_M": baseline, "patched_M": patched, "delta_M": patched - baseline,
                "feedback_sha256": recipient_run.feedback_sha256}


class MoshiBackend:
    """Eager, batch-one Moshiko backend used by the RunPod CLI commands."""

    def __init__(
        self, *, model_repo: str = MODEL_REPO, model_revision: str = MODEL_REVISION,
        device: str = "cuda", dtype: str = "bfloat16", use_sampling: bool = False,
    ):
        os.environ.setdefault("NO_TORCH_COMPILE", "1")
        os.environ.setdefault("NO_CUDA_GRAPH", "1")
        validate_runtime_environment(require_cuda=device.startswith("cuda"))
        try:
            import torch
            import sphn
            from moshi.models import loaders
            from moshi.run_inference import InferenceState, seed_all
        except ImportError as error:
            raise ContractError("Moshi runtime dependencies are unavailable") from error
        if model_revision != MODEL_REVISION:
            raise ContractError("refusing a non-frozen model revision")
        dtype_value = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype)
        if dtype_value is None:
            raise ContractError(f"unsupported dtype: {dtype}")
        checkpoint = loaders.CheckpointInfo.from_hf_repo(model_repo, revision=model_revision)
        mimi = checkpoint.get_mimi(device=device)
        tokenizer = checkpoint.get_text_tokenizer()
        lm = checkpoint.get_moshi(device=device, dtype=dtype_value)
        if len(lm.transformer.layers) != 32 or lm.transformer.layers[0].self_attn.num_heads != 32:
            raise ContractError("loaded checkpoint is not the expected 32-layer/32-head Moshiko")
        if lm.dim != 4096 or max(lm.delays) != 1:
            raise ContractError("loaded checkpoint hidden size or LM delay differs from contract")
        generation = dict(checkpoint.lm_gen_config)
        generation["use_sampling"] = use_sampling
        self.state = InferenceState(checkpoint, mimi, tokenizer, lm, 1, 1.0, device, **generation)
        self.torch = torch
        self.sphn = sphn
        self.mimi = mimi
        self.tokenizer = tokenizer
        self.lm_gen = self.state.lm_gen
        self.device = device
        self.seed_all = seed_all
        self.metadata = {
            "backend": "moshiko-eager", "model_repo": model_repo, "model_revision": model_revision,
            "layers": 32, "heads": 32, "hidden_size": 4096, "head_dim": 128,
            "sample_rate": int(mimi.sample_rate), "frame_samples": int(self.state.frame_size),
        }

    def reset(self, seed: int = 0) -> None:
        self.seed_all(seed)
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    def encode_file(self, path: Path) -> Any:
        pcm, _ = self.sphn.read(str(path), sample_rate=self.mimi.sample_rate)
        pcm_tensor = self.torch.from_numpy(np.asarray(pcm[:1], dtype=np.float32))[None].to(self.device)
        if pcm_tensor.shape[-1] % FRAME_SAMPLES:
            raise ContractError(f"WAV is not aligned to {FRAME_SAMPLES} samples: {path}")
        self.mimi.reset_streaming()
        codes = []
        with self.torch.no_grad():
            for chunk in pcm_tensor.split(FRAME_SAMPLES, dim=-1):
                codes.append(self.mimi.encode(chunk))
        return self.torch.cat(codes, dim=-1)

    def replay_codes(
        self, codes: Any, *, sites: Sequence[str], replacement: Mapping[tuple[str, int, int], Any] | None = None,
    ) -> ReplayResult:
        torch = self.torch
        captures: dict[str, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
        event_tensors: dict[tuple[str, int, int], np.ndarray] = {}
        frame_index = -1
        replacements = replacement or {}

        def hook(event):
            nonlocal frame_index
            position = max(frame_index, 0)
            tensor = event.tensor
            value = tensor.detach().float().cpu().numpy()
            captures[event.site].append((event.layer, position, value))
            event_tensors[(event.site, event.layer, position)] = value
            candidate = replacements.get((event.site, event.layer, position))
            if candidate is None:
                return None
            if isinstance(candidate, Mapping) and "head" in candidate:
                replacement = tensor.clone()
                replacement[:, int(candidate["head"])] = torch.as_tensor(
                    candidate["tensor"], dtype=tensor.dtype, device=tensor.device)
                return replacement
            return torch.as_tensor(candidate, dtype=tensor.dtype, device=tensor.device).reshape_as(tensor)

        self.reset()
        self.lm_gen.set_mechanistic_hook(hook)
        logits = []
        feedback_hash = hashlib.sha256()
        null_text = torch.full((1,), self.lm_gen.lm_model.zero_token_id, device=self.device, dtype=torch.long)
        null_audio = torch.full(
            (1, self.lm_gen.lm_model.dep_q), self.lm_gen.lm_model.zero_token_id,
            device=self.device, dtype=torch.long)
        try:
            with torch.no_grad():
                for frame_index in range(codes.shape[-1]):
                    frame = codes[:, :, frame_index:frame_index + 1]
                    repeats = 2 if frame_index == 0 else 1
                    for _ in range(repeats):
                        result = self.lm_gen.step_open_loop(
                            frame, feedback_text_token=null_text, feedback_audio_tokens=null_audio)
                        logits.append(result.text_logits.detach().float().cpu().numpy())
                        feedback_hash.update(result.feedback_text.detach().cpu().numpy().tobytes())
                        feedback_hash.update(result.feedback_audio.detach().cpu().numpy().tobytes())
        finally:
            self.lm_gen.set_mechanistic_hook(None)
        packed: dict[str, np.ndarray] = {}
        for site, rows in captures.items():
            packed[site] = np.asarray([row[2] for row in rows], dtype=np.float32)
        return ReplayResult(
            packed, np.concatenate(logits, axis=2), feedback_hash.hexdigest(), codes.shape[-1], event_tensors)

    def score_candidates(self, snapshot, candidates: Mapping[str, str]) -> dict[str, float]:
        torch = self.torch
        scores: dict[str, float] = {}
        null_audio = torch.full(
            (1, self.lm_gen.lm_model.dep_q), self.lm_gen.lm_model.zero_token_id,
            device=self.device, dtype=torch.long)
        user_null = torch.full(
            (1, self.mimi.num_codebooks, 1), self.lm_gen.lm_model.zero_token_id,
            device=self.device, dtype=torch.long)
        for name, text in candidates.items():
            self.lm_gen.restore_streaming_state(snapshot)
            token_ids = list(self.tokenizer.encode(text, out_type=int))
            score = 0.0
            for token_id in token_ids:
                teacher = torch.tensor([token_id], device=self.device, dtype=torch.long)
                result = self.lm_gen.step_open_loop(
                    user_null, feedback_text_token=teacher, feedback_audio_tokens=null_audio)
                log_probs = torch.log_softmax(result.text_logits[0, 0, -1].float(), dim=-1)
                score += float(log_probs[token_id].item())
            scores[name] = score / max(1, len(token_ids))
        self.lm_gen.restore_streaming_state(snapshot)
        return scores

    def generate_codes(
        self,
        codes: Any,
        *,
        seed: int,
        intervention: tuple[str, int, int, int | None] | None = None,
    ) -> tuple[list[int], np.ndarray]:
        """Free-running generation with one frozen within-repair erasure seam."""
        torch = self.torch
        frame_index = -1

        def hook(event):
            if intervention is None:
                return None
            site, layer, frame, head = intervention
            if event.site != site or event.layer != layer or frame_index != frame:
                return None
            replacement = event.tensor.clone()
            if head is None:
                replacement.zero_()
            else:
                replacement[:, head].zero_()
            return replacement

        self.reset(seed)
        self.lm_gen.set_mechanistic_hook(hook if intervention is not None else None)
        text_ids: list[int] = []
        pcm_chunks: list[np.ndarray] = []
        try:
            with torch.no_grad():
                for frame_index in range(codes.shape[-1]):
                    frame_codes = codes[:, :, frame_index:frame_index + 1]
                    repeats = 2 if frame_index == 0 else 1
                    for _ in range(repeats):
                        tokens = self.lm_gen.step(frame_codes)
                        if tokens is None:
                            continue
                        text_ids.extend(int(value) for value in tokens[:, 0].reshape(-1).tolist())
                        if tokens.shape[1] > 1:
                            pcm = self.mimi.decode(tokens[:, 1:]).detach().cpu().numpy().reshape(-1)
                            pcm_chunks.append(pcm)
        finally:
            self.lm_gen.set_mechanistic_hook(None)
        return text_ids, np.concatenate(pcm_chunks) if pcm_chunks else np.zeros(0, dtype=np.float32)
