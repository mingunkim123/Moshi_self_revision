from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .core import (
    ContractError,
    FRAME_SAMPLES,
    MODEL_REPO,
    MODEL_REVISION,
    SAMPLE_RATE,
    validate_runtime_environment,
)
from .conversation import (
    STARTUP_MODE_COMMON_HANDSHAKE,
    STARTUP_MODE_GREETING_SUPPRESSED,
    STARTUP_MODE_NATURAL,
    STARTUP_MODES,
)


FROZEN_GREETING_MAX_FRAMES = 150  # 12,000 ms at 80 ms/frame.
FROZEN_GREETING_QUIET_FRAMES = 20  # 1,600 ms.
FROZEN_PREPARED_LEADIN_FRAMES = 6  # 480 ms, already present in every prepared WAV.
FROZEN_AUDIO_ACTIVITY_THRESHOLD_DBFS = -45.0
EXPECTED_MOSHIKO_DELAYS = (0, 0, *([1] * 7), 0, *([1] * 7))
Intervention = tuple[str, int, int, int | None]


@dataclass(frozen=True)
class ReplayResult:
    activations: dict[str, np.ndarray]
    logits: np.ndarray
    feedback_sha256: str
    frame_count: int
    event_tensors: dict[tuple[str, int, int], np.ndarray]
    lm_step_count: int = 0


@dataclass(frozen=True)
class EncodedConversation:
    user_codes: Any
    conversation_codes: Any
    assistant_silence_codes: Any
    user_frame_count: int
    target_frame_count: int


@dataclass(frozen=True)
class GeneratedSequence:
    """One fixed-horizon assistant stream, including any shared handshake."""

    tokens: Any
    feedback_tokens: Any
    text_token_ids: list[int]
    text_pieces: list[str]
    pcm: np.ndarray
    frame_count: int
    conversation_frame_count: int
    conversation_start_frame: int
    frame_samples: int
    pcm_sample_count: int

    @property
    def conversation_tokens(self) -> Any:
        end = self.conversation_start_frame + self.conversation_frame_count
        return self.tokens[..., self.conversation_start_frame:end]

    @property
    def conversation_feedback_tokens(self) -> Any:
        end = self.conversation_start_frame + self.conversation_frame_count
        return self.feedback_tokens[..., self.conversation_start_frame:end]

    @property
    def conversation_pcm(self) -> np.ndarray:
        start = self.conversation_start_frame * self.frame_samples
        end = start + self.conversation_frame_count * self.frame_samples
        return self.pcm[start:end]


@dataclass(frozen=True)
class PairedGeneration:
    baseline: GeneratedSequence
    patched: GeneratedSequence
    branch_frame: int
    shared_prefix_frames: int
    shared_prefix_sha256: str
    shared_feedback_sha256: str
    first_feedback_divergence_frame: int | None
    first_output_divergence_frame: int | None
    pre_intervention_identical: bool
    startup_mode: str
    startup_frame_count: int
    handshake_terminal_frame: int | None
    handshake_terminal_piece: str | None
    handshake_completion_signal: str | None
    target_frame_count: int
    lm_step_count: int
    handshake_probe_lm_step_count: int = 0
    handshake_replay_identical: bool | None = None
    continuous_mimi_input_verified: bool | None = None


@dataclass(frozen=True)
class _RNGSnapshot:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Any
    torch_cuda: tuple[Any, ...] | None


@dataclass(frozen=True)
class _FrameResult:
    output: Any
    feedback: Any


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ContractError(f"{label} must be an integer")
    return int(value)


def _normalise_interventions(
    value: Intervention | Sequence[Intervention] | None,
) -> tuple[Intervention, ...]:
    """Freeze a legacy intervention or an explicit ordered joint circuit."""

    if value is None:
        return ()
    if (
        isinstance(value, tuple)
        and len(value) == 4
        and isinstance(value[0], str)
    ):
        rows: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = value
    else:
        raise ContractError(
            "intervention must be one (site, layer, frame, head) tuple or an ordered sequence"
        )

    normalised: list[Intervention] = []
    seen_sites: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            raise ContractError(f"intervention[{index}] must contain exactly four fields")
        site, layer, frame, head = row
        if not isinstance(site, str) or not site:
            raise ContractError(f"intervention[{index}] site must be a non-empty string")
        if site in seen_sites:
            raise ContractError(f"duplicate intervention site is not allowed: {site}")
        seen_sites.add(site)
        frozen_layer = _exact_int(layer, f"intervention[{index}] layer")
        frozen_frame = _exact_int(frame, f"intervention[{index}] frame")
        frozen_head = (
            None if head is None else _exact_int(head, f"intervention[{index}] head")
        )
        if frozen_layer < 0 or frozen_frame < 0 or (
            frozen_head is not None and frozen_head < 0
        ):
            raise ContractError("intervention layer/frame/head indices must be non-negative")
        normalised.append((site, frozen_layer, frozen_frame, frozen_head))
    if not normalised:
        raise ContractError("an intervention sequence cannot be empty")
    return tuple(normalised)


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
        return ReplayResult(
            activations, margin, hashlib.sha256(feedback.tobytes()).hexdigest(), frames, {}, frames + 1)

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
        # These flags must be present in the process environment before any
        # model object is constructed.  Silently setting them here cannot undo
        # compilation/graph initialization that may already have happened.
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
        if checkpoint.model_type != "moshi":
            raise ContractError("loaded checkpoint is not a Moshi conversational model")
        if len(lm.transformer.layers) != 32 or lm.transformer.layers[0].self_attn.num_heads != 32:
            raise ContractError("loaded checkpoint is not the expected 32-layer/32-head Moshiko")
        if lm.dim != 4096 or tuple(int(delay) for delay in lm.delays) != EXPECTED_MOSHIKO_DELAYS:
            raise ContractError("loaded checkpoint hidden size or LM delay differs from contract")
        generation = dict(checkpoint.lm_gen_config)
        generation["use_sampling"] = use_sampling
        self.state = InferenceState(checkpoint, mimi, tokenizer, lm, 1, 1.0, device, **generation)
        input_codebooks = int(lm.num_codebooks) - int(lm.dep_q) - 1
        if (
            int(mimi.sample_rate) != SAMPLE_RATE
            or int(self.state.frame_size) != FRAME_SAMPLES
            or float(mimi.frame_rate) != 12.5
            or int(mimi.channels) != 1
            or int(mimi.num_codebooks) != 8
            or int(mimi.cardinality) != 2048
            or int(lm.dep_q) != 8
            or input_codebooks != 8
            or int(lm.card) != 2048
            or int(lm.num_codebooks) != 17
            or int(lm.text_padding_token_id) != 3
            or int(lm.end_of_text_padding_id) != 0
            or int(lm.zero_token_id) != -1
        ):
            raise ContractError(
                "loaded checkpoint does not match the frozen 24 kHz/80 ms, 8+8 audio-stream contract"
            )
        self.torch = torch
        self.sphn = sphn
        self.mimi = mimi
        self.tokenizer = tokenizer
        self.lm_gen = self.state.lm_gen
        self.device = device
        self.seed_all = seed_all
        lm_parameter = next(lm.parameters())
        self.metadata = {
            "backend": "moshiko-eager",
            "model_repo": model_repo,
            "model_revision": model_revision,
            "model_type": str(checkpoint.model_type),
            "dtype": str(lm_parameter.dtype),
            "device": str(lm_parameter.device),
            "layers": len(lm.transformer.layers),
            "heads": int(lm.transformer.layers[0].self_attn.num_heads),
            "hidden_size": int(lm.dim),
            "head_dim": int(lm.dim) // int(lm.transformer.layers[0].self_attn.num_heads),
            "delays": [int(delay) for delay in lm.delays],
            "text_initial_token_id": int(lm.text_initial_token_id),
            "text_padding_token_id": int(lm.text_padding_token_id),
            "end_of_text_padding_id": int(lm.end_of_text_padding_id),
            "audio_initial_token_id": int(lm.initial_token_id),
            "zero_token_id": int(lm.zero_token_id),
            "ungenerated_token_id": int(lm.ungenerated_token_id),
            "text_card": int(lm.text_card),
            "num_codebooks": int(lm.num_codebooks),
            "dep_q": int(lm.dep_q),
            "card": int(lm.card),
            "user_codebooks": input_codebooks,
            "assistant_codebooks": int(lm.dep_q),
            "sample_rate": int(mimi.sample_rate),
            "frame_samples": int(self.state.frame_size),
            "mimi_channels": int(mimi.channels),
            "mimi_cardinality": int(mimi.cardinality),
            "mimi_frame_rate": float(mimi.frame_rate),
            "mimi_num_codebooks": int(mimi.num_codebooks),
        }

    def reset(self, seed: int = 0) -> None:
        self.seed_all(seed)
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    def _read_pcm(self, path: Path) -> np.ndarray:
        pcm, sample_rate = self.sphn.read(str(path), sample_rate=self.mimi.sample_rate)
        if int(sample_rate) != int(self.mimi.sample_rate):
            raise ContractError(f"WAV resampling returned an unexpected sample rate: {path}")
        array = np.asarray(pcm, dtype=np.float32)
        if array.ndim == 1:
            array = array[None]
        if array.ndim != 2 or array.shape[0] != 1:
            raise ContractError(f"WAV must decode as mono audio: {path}")
        return np.ascontiguousarray(array[0])

    def encode_pcm(self, pcm: np.ndarray) -> Any:
        array = np.asarray(pcm, dtype=np.float32)
        if array.ndim != 1:
            raise ContractError("PCM must be a one-dimensional mono stream")
        if not np.isfinite(array).all():
            raise ContractError("PCM contains NaN or infinity")
        array = np.ascontiguousarray(array)
        pcm_tensor = self.torch.from_numpy(np.ascontiguousarray(array))[None, None].to(self.device)
        if pcm_tensor.shape[-1] % FRAME_SAMPLES:
            raise ContractError(f"PCM is not aligned to {FRAME_SAMPLES} samples")
        if pcm_tensor.shape[-1] == 0:
            raise ContractError("cannot Mimi-encode empty PCM")
        self.mimi.reset_streaming()
        codes = []
        with self.torch.no_grad():
            for chunk in pcm_tensor.split(FRAME_SAMPLES, dim=-1):
                encoded = self.mimi.encode(chunk)
                expected_shape = (1, int(self.mimi.num_codebooks), 1)
                if tuple(encoded.shape) != expected_shape:
                    raise ContractError(
                        f"Mimi emitted shape {tuple(encoded.shape)}, expected {expected_shape}"
                    )
                if bool((encoded < 0).any().item()) or bool(
                    (encoded >= int(self.mimi.cardinality)).any().item()
                ):
                    raise ContractError("Mimi emitted an out-of-range audio token")
                codes.append(encoded)
        output = self.torch.cat(codes, dim=-1)
        if output.shape[-1] != pcm_tensor.shape[-1] // FRAME_SAMPLES:
            raise ContractError("Mimi output coverage differs from PCM input coverage")
        return output

    def encode_file(self, path: Path) -> Any:
        return self.encode_pcm(self._read_pcm(path))

    def encode_conversation_file(self, path: Path, *, target_frame_count: int) -> EncodedConversation:
        """Encode user audio and its exact-zero continuation in one causal Mimi stream."""
        pcm = self._read_pcm(path)
        if pcm.size % FRAME_SAMPLES:
            raise ContractError(f"WAV is not aligned to {FRAME_SAMPLES} samples: {path}")
        user_frames = pcm.size // FRAME_SAMPLES
        target_frames = _exact_int(target_frame_count, "target_frame_count")
        if target_frames < user_frames:
            raise ContractError("conversation capture target ends before the prepared WAV")
        target_samples = target_frames * FRAME_SAMPLES
        extended = np.zeros(target_samples, dtype=np.float32)
        extended[:pcm.size] = pcm
        conversation_codes = self.encode_pcm(extended)
        user_codes = conversation_codes[..., :user_frames].clone()
        assistant_silence_codes = self.encode_pcm(np.zeros(target_samples, dtype=np.float32))
        return EncodedConversation(
            user_codes, conversation_codes, assistant_silence_codes, user_frames, target_frames)

    def encode_continuous_conversation_pcm(
        self,
        pcm: np.ndarray,
        *,
        target_frame_count: int,
        startup_frame_count: int,
    ) -> Any:
        """Encode startup silence and the complete conversation in one Mimi stream.

        ``pcm`` is the prepared user WAV (including its frozen 480 ms lead-in).
        Exact-zero continuation is appended through ``target_frame_count`` before
        a startup prefix is prepended.  This prevents a second causal-Mimi
        startup transient at the greeting/request boundary.
        """

        target_frames = _exact_int(target_frame_count, "target_frame_count")
        startup_frames = _exact_int(startup_frame_count, "startup_frame_count")
        if target_frames < 1 or startup_frames < 1:
            raise ContractError("continuous conversation frame counts must be positive")
        array = np.asarray(pcm, dtype=np.float32)
        if array.ndim != 1 or array.size == 0 or array.size % FRAME_SAMPLES:
            raise ContractError("prepared conversation PCM must be mono and frame-aligned")
        if not np.isfinite(array).all():
            raise ContractError("prepared conversation PCM contains NaN or infinity")
        target_samples = target_frames * FRAME_SAMPLES
        if array.size > target_samples:
            raise ContractError("prepared conversation PCM exceeds the target horizon")
        combined = np.zeros((startup_frames + target_frames) * FRAME_SAMPLES, dtype=np.float32)
        start = startup_frames * FRAME_SAMPLES
        combined[start : start + array.size] = array
        codes = self.encode_pcm(combined)
        expected_shape = (1, int(self.mimi.num_codebooks), startup_frames + target_frames)
        if tuple(codes.shape) != expected_shape:
            raise ContractError(
                f"continuous Mimi codes have shape {tuple(codes.shape)}, expected {expected_shape}"
            )
        return codes

    def replay_codes(
        self, codes: Any, *, sites: Sequence[str] = (), replacement: Mapping[tuple[str, int, int], Any] | None = None,
        capture_layers: Sequence[int] | None = None,
        capture_frames: Sequence[int] | None = None,
        end_frame_exclusive: int | None = None,
        hook_enabled: bool = True,
    ) -> ReplayResult:
        torch = self.torch
        codes = torch.as_tensor(codes, device=self.device, dtype=torch.long)
        model = self.lm_gen.lm_model
        expected_input_codebooks = int(model.num_codebooks) - int(model.dep_q) - 1
        if (
            codes.ndim != 3
            or int(codes.shape[0]) != 1
            or int(codes.shape[1]) != expected_input_codebooks
            or int(codes.shape[-1]) < 1
        ):
            raise ContractError(
                "replay codes must have shape "
                f"[1, {expected_input_codebooks}, T] with T >= 1"
            )
        if bool((codes < 0).any().item()) or bool((codes >= int(model.card)).any().item()):
            raise ContractError("replay codes contain an out-of-range Mimi token")
        total_frames = int(codes.shape[-1])
        consumed_frames = (
            total_frames
            if end_frame_exclusive is None
            else _exact_int(end_frame_exclusive, "end_frame_exclusive")
        )
        if not 1 <= consumed_frames <= total_frames:
            raise ContractError(
                f"end_frame_exclusive must be in [1, {total_frames}], got {end_frame_exclusive}"
            )
        captures: dict[str, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
        event_tensors: dict[tuple[str, int, int], np.ndarray] = {}
        frame_index = -1
        replacements = replacement or {}
        requested_sites = frozenset(str(site) for site in sites)
        requested_layers = None if capture_layers is None else frozenset(int(layer) for layer in capture_layers)
        requested_frames = None if capture_frames is None else frozenset(int(frame) for frame in capture_frames)
        if any(not site for site in requested_sites):
            raise ContractError("activation site names must be non-empty")
        if requested_layers is not None and any(layer < 0 for layer in requested_layers):
            raise ContractError("capture layers must be non-negative")
        if requested_frames is not None and any(
            frame < 0 or frame >= consumed_frames for frame in requested_frames
        ):
            raise ContractError("capture frames must fall inside the consumed replay horizon")
        if hook_enabled and not requested_sites and not replacements:
            raise ContractError("at least one activation site must be requested")
        if not hook_enabled and (requested_sites or replacements or capture_layers is not None or capture_frames is not None):
            raise ContractError("hook-off replay cannot request captures or replacements")

        def hook(event):
            nonlocal frame_index
            position = frame_index
            tensor = event.tensor
            candidate = replacements.get((event.site, event.layer, position))
            selected = (
                event.site in requested_sites
                and (requested_layers is None or event.layer in requested_layers)
                and (requested_frames is None or position in requested_frames)
            )
            if selected:
                value = tensor.detach().float().cpu().numpy()
                captures[event.site].append((event.layer, position, value))
                event_tensors[(event.site, event.layer, position)] = value
            if candidate is not None:
                if isinstance(candidate, Mapping) and "head" in candidate:
                    replacement_tensor = tensor.clone()
                    replacement_tensor[:, int(candidate["head"])] = torch.as_tensor(
                        candidate["tensor"], dtype=tensor.dtype, device=tensor.device)
                    return replacement_tensor
                return torch.as_tensor(candidate, dtype=tensor.dtype, device=tensor.device).reshape_as(tensor)
            return None

        self.reset()
        logits = []
        feedback_hash = hashlib.sha256()
        null_text = torch.full((1,), self.lm_gen.lm_model.zero_token_id, device=self.device, dtype=torch.long)
        null_audio = torch.full(
            (1, self.lm_gen.lm_model.dep_q), self.lm_gen.lm_model.zero_token_id,
            device=self.device, dtype=torch.long)
        try:
            with torch.no_grad():
                # LM step 0 consumes initial tokens. The same first Mimi frame is then
                # consumed exactly once at LM step 1, matching InferenceState.run.
                prime = self.lm_gen.step_open_loop(
                    codes[:, :, :1], feedback_text_token=null_text, feedback_audio_tokens=null_audio)
                if prime.output_tokens is not None:
                    raise ContractError("Moshiko prime step unexpectedly emitted an output frame")
                feedback_hash.update(prime.feedback_text.detach().cpu().numpy().tobytes())
                feedback_hash.update(prime.feedback_audio.detach().cpu().numpy().tobytes())
                self.lm_gen.set_mechanistic_hook(hook if hook_enabled else None)
                for frame_index in range(consumed_frames):
                    frame = codes[:, :, frame_index:frame_index + 1]
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
            packed, np.concatenate(logits, axis=2), feedback_hash.hexdigest(), consumed_frames,
            event_tensors, consumed_frames + 1)

    def score_candidates(
        self,
        snapshot,
        candidates: Mapping[str, str],
        *,
        prefix: str = "",
        prefix_start_offset_frames: int = 0,
        pad_frames_between_tokens: int = 0,
    ) -> dict[str, float]:
        """Teacher-force a frozen prefix and score every candidate from the same state."""
        torch = self.torch
        if prefix_start_offset_frames < 0 or pad_frames_between_tokens < 0:
            raise ContractError("readout schedule offsets must be non-negative")
        scores: dict[str, float] = {}
        null_audio = torch.full(
            (1, self.lm_gen.lm_model.dep_q), self.lm_gen.lm_model.zero_token_id,
            device=self.device, dtype=torch.long)
        user_null = torch.full(
            (1, self.mimi.num_codebooks, 1), self.lm_gen.lm_model.zero_token_id,
            device=self.device, dtype=torch.long)
        text_pad = torch.full(
            (1,), self.lm_gen.lm_model.text_padding_token_id,
            device=self.device, dtype=torch.long)

        def advance(feedback_text) -> Any:
            return self.lm_gen.step_open_loop(
                user_null, feedback_text_token=feedback_text, feedback_audio_tokens=null_audio)

        prefix_ids = list(self.tokenizer.encode(prefix, out_type=int)) if prefix else []
        for name, text in candidates.items():
            self.lm_gen.restore_streaming_state(snapshot)
            for _ in range(prefix_start_offset_frames):
                advance(text_pad)
            for prefix_index, token_id in enumerate(prefix_ids):
                teacher = torch.tensor([token_id], device=self.device, dtype=torch.long)
                advance(teacher)
                if prefix_index + 1 < len(prefix_ids):
                    for _ in range(pad_frames_between_tokens):
                        advance(text_pad)
            token_ids = list(self.tokenizer.encode(text, out_type=int))
            score = 0.0
            for token_index, token_id in enumerate(token_ids):
                teacher = torch.tensor([token_id], device=self.device, dtype=torch.long)
                result = advance(teacher)
                log_probs = torch.log_softmax(result.text_logits[0, 0, -1].float(), dim=-1)
                score += float(log_probs[token_id].item())
                if token_index + 1 < len(token_ids):
                    for _ in range(pad_frames_between_tokens):
                        advance(text_pad)
            scores[name] = score / max(1, len(token_ids))
        self.lm_gen.restore_streaming_state(snapshot)
        return scores

    def _snapshot_rng(self) -> _RNGSnapshot:
        torch = self.torch
        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        return _RNGSnapshot(
            python=copy.deepcopy(random.getstate()),
            numpy=copy.deepcopy(np.random.get_state()),
            torch_cpu=torch.random.get_rng_state().clone(),
            torch_cuda=cuda_state,
        )

    def _restore_rng(self, snapshot: _RNGSnapshot) -> None:
        random.setstate(snapshot.python)
        np.random.set_state(snapshot.numpy)
        self.torch.random.set_rng_state(snapshot.torch_cpu)
        if snapshot.torch_cuda is not None:
            if not self.torch.cuda.is_available():
                raise ContractError("CUDA RNG snapshot cannot be restored without CUDA")
            self.torch.cuda.set_rng_state_all(list(snapshot.torch_cuda))

    def _normalise_frame_tokens(self, tokens: Any, *, label: str) -> Any:
        torch = self.torch
        value = torch.as_tensor(tokens)
        if value.ndim == 3:
            if value.shape[0] != 1 or value.shape[-1] != 1:
                raise ContractError(f"{label} must have shape [1, K, 1]")
            value = value[:, :, 0]
        elif value.ndim == 2:
            if value.shape[0] != 1:
                raise ContractError(f"{label} must have batch size one")
        else:
            raise ContractError(f"{label} must have shape [1, K] or [1, K, 1]")
        return value.detach().to(device="cpu", dtype=torch.long).clone()

    def _feedback_tokens(self, detail: Any) -> Any:
        torch = self.torch
        text = torch.as_tensor(detail.feedback_text).reshape(1, 1)
        audio = detail.feedback_audio
        if audio is None:
            combined = text
        else:
            audio_tensor = torch.as_tensor(audio)
            if audio_tensor.ndim == 3 and audio_tensor.shape[-1] == 1:
                audio_tensor = audio_tensor[:, :, 0]
            if audio_tensor.ndim != 2 or audio_tensor.shape[0] != 1:
                raise ContractError("feedback audio must have shape [1, K]")
            combined = torch.cat([text.to(audio_tensor.device), audio_tensor], dim=1)
        return self._normalise_frame_tokens(combined, label="feedback tokens")

    def _advance_generation_frame(
        self,
        input_codes: Any,
        *,
        forced_text: Any | None = None,
        forced_audio: Any | None = None,
        expect_output: bool = True,
    ) -> _FrameResult | None:
        if forced_text is None:
            if forced_audio is not None:
                raise ContractError("forced audio feedback requires forced text feedback")
            output = self.lm_gen.step(input_codes)
            detail = getattr(self.lm_gen, "_last_step_result", None)
        else:
            detail = self.lm_gen.step_open_loop(
                input_codes,
                feedback_text_token=forced_text,
                feedback_audio_tokens=forced_audio,
            )
            output = detail.output_tokens
        if detail is None:
            raise ContractError("LM step did not expose its feedback decision")
        if not expect_output:
            if output is not None:
                raise ContractError("Moshiko delay-prime step unexpectedly emitted output")
            return None
        if output is None:
            raise ContractError("LM emitted no frame after its single delay-prime step")
        return _FrameResult(
            output=self._normalise_frame_tokens(output, label="output tokens"),
            feedback=self._feedback_tokens(detail),
        )

    def _advance_generation_frame_masked(
        self,
        input_codes: Any,
        *,
        forced_text: Any | None,
        forced_audio: Any | None,
        audio_force_mask: Any | None,
        expect_output: bool = True,
    ) -> tuple[_FrameResult | None, int]:
        """Advance one logical frame with an exact per-codebook feedback mask.

        ``LMGen.step_open_loop`` can independently leave text natural, but its
        audio override is all-or-nothing.  At the trailing edge of a delayed
        forced-output interval, only delayed codebooks remain forced.  For that
        one mixed decision we deterministically preview the sampled feedback,
        restore both model and RNG state, and commit a single mixed decision.
        The returned count is the number of physical LM calls (one or two).
        """

        torch = self.torch
        dep_q = int(self.lm_gen.lm_model.dep_q)
        if audio_force_mask is None:
            mask = torch.zeros(dep_q, dtype=torch.bool, device=self.device)
        else:
            mask = torch.as_tensor(
                audio_force_mask, dtype=torch.bool, device=self.device
            ).reshape(-1)
            if tuple(mask.shape) != (dep_q,):
                raise ContractError(f"audio force mask must have shape [{dep_q}]")
        any_audio = bool(mask.any().item())
        all_audio = bool(mask.all().item())
        if any_audio:
            if forced_audio is None:
                raise ContractError("forced audio values are required by the force mask")
            forced_audio_tensor = torch.as_tensor(
                forced_audio, dtype=torch.long, device=self.device
            )
            if tuple(forced_audio_tensor.shape) != (1, dep_q):
                raise ContractError(
                    f"forced audio values must have shape [1, {dep_q}]"
                )
        else:
            forced_audio_tensor = None

        if not any_audio:
            return (
                self._advance_generation_frame(
                    input_codes,
                    forced_text=forced_text,
                    forced_audio=None,
                    expect_output=expect_output,
                ),
                1,
            )
        if all_audio and forced_text is not None:
            return (
                self._advance_generation_frame(
                    input_codes,
                    forced_text=forced_text,
                    forced_audio=forced_audio_tensor,
                    expect_output=expect_output,
                ),
                1,
            )
        if not expect_output:
            raise ContractError("the delay-prime feedback decision cannot require a mixed mask")

        state_snapshot = self.lm_gen.snapshot_streaming_state()
        rng_snapshot = self._snapshot_rng()
        preview: _FrameResult | None = None
        try:
            self._feedback_preview_pass = True
            preview = self._advance_generation_frame(input_codes)
        finally:
            self._feedback_preview_pass = False
            self.lm_gen.restore_streaming_state(state_snapshot)
            self._restore_rng(rng_snapshot)
        if preview is None:  # pragma: no cover - guarded by expect_output.
            raise ContractError("mixed-feedback preview emitted no output frame")
        natural_feedback = preview.feedback.to(device=self.device, dtype=torch.long)
        natural_text = natural_feedback[:, 0]
        natural_audio = natural_feedback[:, 1:]
        if tuple(natural_audio.shape) != (1, dep_q):
            raise ContractError("sampled audio feedback does not match dep_q")
        assert forced_audio_tensor is not None
        mixed_audio = torch.where(mask[None], forced_audio_tensor, natural_audio)
        committed_text = natural_text if forced_text is None else forced_text
        return (
            self._advance_generation_frame(
                input_codes,
                forced_text=committed_text,
                forced_audio=mixed_audio,
                expect_output=True,
            ),
            2,
        )

    def _token_piece(self, token_id: int, *, blank_token_ids: frozenset[int]) -> str:
        if token_id in blank_token_ids:
            return ""
        piece = self.tokenizer.id_to_piece(int(token_id))
        if not isinstance(piece, str):
            raise ContractError(f"tokenizer returned a non-string piece for token {token_id}")
        return piece.replace("▁", " ")

    def _stack_frames(self, frames: Sequence[Any], *, label: str) -> Any:
        if not frames:
            raise ContractError(f"{label} cannot be empty")
        return self.torch.stack(list(frames), dim=-1)

    def _decode_tokens(self, tokens: Any, *, expected_frames: int) -> np.ndarray:
        """Decode one token frame at a time and prove exact PCM coverage."""
        torch = self.torch
        if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[-1] != expected_frames:
            raise ContractError("assistant token tensor does not cover the expected frame horizon")
        if tokens.shape[1] != int(self.lm_gen.lm_model.dep_q) + 1:
            raise ContractError("assistant output codebook count differs from the frozen model")
        if bool((tokens < 0).any().item()):
            raise ContractError("assistant output contains an ungenerated or negative token")
        if bool((tokens[:, 1:] >= int(self.mimi.cardinality)).any().item()):
            raise ContractError("assistant output contains an out-of-range audio token")
        self.mimi.reset_streaming()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for frame in range(expected_frames):
                audio_codes = tokens[:, 1:, frame : frame + 1].to(self.device)
                pcm = self.mimi.decode(audio_codes).detach().float().cpu().numpy().reshape(-1)
                if pcm.size != int(self.state.frame_size):
                    raise ContractError(
                        "Mimi decoder did not emit exactly one PCM frame for one token frame"
                    )
                chunks.append(np.asarray(pcm, dtype=np.float32))
        output = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        expected_samples = expected_frames * int(self.state.frame_size)
        if output.size != expected_samples:
            raise ContractError(
                f"decoded PCM coverage mismatch: {output.size} != {expected_samples}"
            )
        return output

    @staticmethod
    def _token_sha256(tokens: Any) -> str:
        array = np.asarray(tokens.detach().cpu().numpy(), dtype="<i8")
        return hashlib.sha256(array.tobytes(order="C")).hexdigest()

    @staticmethod
    def _first_divergence(left: Any, right: Any) -> int | None:
        left_array = np.asarray(left.detach().cpu().numpy())
        right_array = np.asarray(right.detach().cpu().numpy())
        if left_array.shape != right_array.shape:
            raise ContractError("paired token timelines have different shapes")
        different = np.any(left_array != right_array, axis=(0, 1))
        indices = np.flatnonzero(different)
        return None if indices.size == 0 else int(indices[0])

    def _make_sequence(
        self,
        outputs: Sequence[Any],
        feedback: Sequence[Any],
        *,
        conversation_frame_count: int,
        conversation_start_frame: int,
        blank_token_ids: frozenset[int],
    ) -> GeneratedSequence:
        tokens = self._stack_frames(outputs, label="output token frames")
        feedback_tokens = self._stack_frames(feedback, label="feedback token frames")
        if tokens.shape[-1] != feedback_tokens.shape[-1]:
            raise ContractError("output and feedback timelines have different lengths")
        text_ids = [int(value) for value in tokens[0, 0].tolist()]
        pieces = [self._token_piece(value, blank_token_ids=blank_token_ids) for value in text_ids]
        pcm = self._decode_tokens(tokens, expected_frames=tokens.shape[-1])
        return GeneratedSequence(
            tokens=tokens,
            feedback_tokens=feedback_tokens,
            text_token_ids=text_ids,
            text_pieces=pieces,
            pcm=pcm,
            frame_count=int(tokens.shape[-1]),
            conversation_frame_count=conversation_frame_count,
            conversation_start_frame=conversation_start_frame,
            frame_samples=int(self.state.frame_size),
            pcm_sample_count=int(pcm.size),
        )

    def generate_paired_conversation(
        self,
        conversation_codes: Any,
        *,
        assistant_silence_codes: Any | None,
        conversation_pcm: np.ndarray | None = None,
        seed: int,
        branch_frame: int,
        intervention: Intervention | Sequence[Intervention] | None,
        startup_mode: str = STARTUP_MODE_NATURAL,
        target_frame_count: int | None = None,
        user_start_frame: int = 0,
        query_end_frame: int | None = None,
        user_end_frame: int | None = None,
        handshake_max_frames: int = FROZEN_GREETING_MAX_FRAMES,
        handshake_quiet_frames: int = FROZEN_GREETING_QUIET_FRAMES,
        prepared_leadin_frames: int = FROZEN_PREPARED_LEADIN_FRAMES,
        handshake_silence_threshold_dbfs: float = FROZEN_AUDIO_ACTIVITY_THRESHOLD_DBFS,
        handshake_terminal_punctuation: str = ".?!。？！",
        blank_token_ids: frozenset[int] = frozenset({0, 3}),
    ) -> PairedGeneration:
        """Generate paired arms from one exact state/RNG branch.

        Frame indices in ``branch_frame`` and ``intervention`` are relative to
        ``conversation_codes``.  ``intervention`` accepts the legacy single
        tuple or an ordered sequence of unique-site tuples for a joint circuit.
        The common-handshake mode first measures the
        natural greeting, then starts a fresh deterministic pass whose startup
        silence and full prepared conversation were encoded as one causal Mimi
        stream.  This avoids an encoder reset at the greeting/request boundary.
        The response never stops early: every arm covers ``target_frame_count``.
        """

        torch = self.torch
        if startup_mode not in STARTUP_MODES:
            raise ContractError(f"unsupported startup mode: {startup_mode}")
        codes = torch.as_tensor(conversation_codes, device=self.device, dtype=torch.long)
        if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[-1] < 1:
            raise ContractError("conversation codes must have shape [1, K, T] with T >= 1")
        model = self.lm_gen.lm_model
        expected_input_codebooks = int(model.num_codebooks) - int(model.dep_q) - 1
        if int(codes.shape[1]) != expected_input_codebooks:
            raise ContractError("conversation input codebook count differs from the model")
        if bool((codes < 0).any().item()) or bool((codes >= int(model.card)).any().item()):
            raise ContractError("conversation contains an out-of-range Mimi token")
        target_frames = (
            int(codes.shape[-1])
            if target_frame_count is None
            else _exact_int(target_frame_count, "target_frame_count")
        )
        if target_frames != int(codes.shape[-1]):
            raise ContractError("conversation codes do not exactly cover target_frame_count")
        user_start = _exact_int(user_start_frame, "user_start_frame")
        if not 0 <= user_start < target_frames:
            raise ContractError("user_start_frame is outside the conversation horizon")
        query_end = (
            target_frames
            if query_end_frame is None
            else _exact_int(query_end_frame, "query_end_frame")
        )
        user_end = (
            query_end
            if user_end_frame is None
            else _exact_int(user_end_frame, "user_end_frame")
        )
        if not user_start < query_end <= user_end <= target_frames:
            raise ContractError("user/query boundaries are inconsistent with the target horizon")
        branch = _exact_int(branch_frame, "branch_frame")
        if not 0 <= branch < target_frames:
            raise ContractError("branch_frame must identify a frame inside the target horizon")
        interventions = _normalise_interventions(intervention)
        for _, _, intervention_frame, _ in interventions:
            if intervention_frame != branch:
                raise ContractError(
                    "the state/RNG branch must occur immediately before every intervention frame"
                )

        expected_blank_ids = frozenset(
            {
                int(model.text_padding_token_id),
                int(model.end_of_text_padding_id),
            }
        )
        if blank_token_ids != expected_blank_ids:
            raise ContractError(
                f"blank_token_ids must match the model padding IDs {sorted(expected_blank_ids)}"
            )

        silence = None
        if assistant_silence_codes is not None:
            silence = torch.as_tensor(
                assistant_silence_codes, device=self.device, dtype=torch.long
            )
            if silence.ndim != 3 or silence.shape[0] != 1:
                raise ContractError("assistant silence codes must have shape [1, K, T]")
            if silence.shape[1] != int(self.lm_gen.lm_model.dep_q):
                raise ContractError("assistant silence codebook count differs from the model")
            if bool((silence < 0).any().item()) or bool(
                (silence >= int(model.card)).any().item()
            ):
                raise ContractError("assistant silence contains an out-of-range Mimi token")
        if startup_mode in {STARTUP_MODE_GREETING_SUPPRESSED, STARTUP_MODE_COMMON_HANDSHAKE}:
            if silence is None:
                raise ContractError(f"{startup_mode} requires encoded assistant silence codes")
        if startup_mode == STARTUP_MODE_GREETING_SUPPRESSED and silence.shape[-1] < user_end:
            raise ContractError("assistant silence codes do not cover the suppression interval")
        if startup_mode == STARTUP_MODE_COMMON_HANDSHAKE:
            handshake_max_frames = _exact_int(handshake_max_frames, "handshake_max_frames")
            handshake_quiet_frames = _exact_int(
                handshake_quiet_frames, "handshake_quiet_frames"
            )
            prepared_leadin_frames = _exact_int(
                prepared_leadin_frames, "prepared_leadin_frames"
            )
            if handshake_max_frames != FROZEN_GREETING_MAX_FRAMES:
                raise ContractError("handshake_max_frames must remain frozen at 12 seconds / 150 frames")
            if handshake_quiet_frames != FROZEN_GREETING_QUIET_FRAMES:
                raise ContractError("handshake_quiet_frames must remain frozen at 1.6 seconds / 20 frames")
            if prepared_leadin_frames != FROZEN_PREPARED_LEADIN_FRAMES:
                raise ContractError("prepared lead-in must remain frozen at 480 ms / 6 frames")
            if user_start != prepared_leadin_frames:
                raise ContractError("user_start_frame must equal the frozen 480 ms prepared lead-in")
            if (
                not np.isfinite(handshake_silence_threshold_dbfs)
                or float(handshake_silence_threshold_dbfs)
                != FROZEN_AUDIO_ACTIVITY_THRESHOLD_DBFS
            ):
                raise ContractError("handshake silence threshold must remain frozen at -45 dBFS")
            if not isinstance(handshake_terminal_punctuation, str) or not handshake_terminal_punctuation:
                raise ContractError("handshake terminal punctuation set must be non-empty")
            if silence.shape[-1] < handshake_max_frames:
                raise ContractError("assistant silence codes do not cover handshake_max_frames")
            if conversation_pcm is None:
                raise ContractError(
                    "common handshake requires prepared PCM for continuous causal Mimi encoding"
                )

        text_pad = torch.full(
            (1,), self.lm_gen.lm_model.text_padding_token_id,
            device=self.device, dtype=torch.long,
        )

        output_delays = tuple(
            int(delay) for delay in model.delays[1 : int(model.dep_q) + 1]
        )
        if len(output_delays) != int(model.dep_q) or any(
            delay < 0 or delay > int(max(model.delays)) for delay in output_delays
        ):
            raise ContractError("model output delay vector is inconsistent with dep_q")

        def suppression_feedback(
            decision_step: int,
        ) -> tuple[Any | None, Any | None, Any | None]:
            """Inverse the output delay schedule for the half-open quiet span.

            The first emitted frame is produced after decision step one.  For
            stream ``k``, emitted frame ``f`` reads feedback decision
            ``F[f + delay[k]]``.  A desired aligned silence token ``A[f]`` is
            therefore written at decision ``j`` as ``A[j - delay[k]]``.  Text
            has its own (zero) delay.  At the trailing boundary the mask keeps
            delayed audio forced while immediately releasing zero-delay text
            and audio, so response frame ``user_end`` is not shifted by 80 ms.
            """

            if startup_mode != STARTUP_MODE_GREETING_SUPPRESSED:
                return None, None, None
            assert silence is not None
            text_value = text_pad if 0 <= decision_step < user_end else None
            values: list[Any] = []
            force_mask: list[bool] = []
            for codebook, delay in enumerate(output_delays):
                aligned_frame = decision_step - delay
                should_force = 0 <= aligned_frame < user_end
                # Delay-prime values with negative aligned indices are never
                # emitted.  Fill them with A[0] to keep the initial feedback
                # entirely quiet without requiring a mixed prime call.
                source_frame = 0 if aligned_frame < 0 else aligned_frame
                if should_force or decision_step == 0:
                    values.append(silence[:, codebook, source_frame])
                else:
                    values.append(silence[:, codebook, 0])
                force_mask.append(should_force or decision_step == 0)
            return text_value, torch.stack(values, dim=1), torch.tensor(force_mask)

        shared_outputs: list[Any] = []
        shared_feedback: list[Any] = []
        startup_frames = 0
        handshake_terminal_frame: int | None = None
        handshake_terminal_piece: str | None = None
        lm_steps = 0
        handshake_probe_steps = 0
        handshake_replay_identical: bool | None = None
        continuous_mimi_input_verified: bool | None = None
        startup_codes = None

        def measure_handshake() -> tuple[list[Any], list[Any], int, int, str]:
            assert silence is not None
            probe_outputs: list[Any] = []
            probe_feedback: list[Any] = []
            terminal_frame: int | None = None
            terminal_piece: str | None = None
            seen_lexical = False
            terminal_seen = False
            quiet_frames = 0
            self.reset(seed)
            self.lm_gen.set_mechanistic_hook(None)
            self._advance_generation_frame(silence[:, :, :1], expect_output=False)
            for startup_index in range(handshake_max_frames):
                result = self._advance_generation_frame(
                    silence[:, :, startup_index : startup_index + 1]
                )
                assert result is not None
                probe_outputs.append(result.output)
                probe_feedback.append(result.feedback)

                token_id = int(result.output[0, 0].item())
                piece = self._token_piece(token_id, blank_token_ids=blank_token_ids)
                audio_codes = result.output[:, 1:, None].to(self.device)
                if bool((audio_codes < 0).any().item()) or bool(
                    (audio_codes >= int(self.mimi.cardinality)).any().item()
                ):
                    raise ContractError("handshake output contains an invalid audio token")
                pcm = self.mimi.decode(audio_codes).detach().float().cpu().numpy()
                if pcm.size != int(self.state.frame_size) or not np.isfinite(pcm).all():
                    raise ContractError(
                        "handshake Mimi decode is non-finite or does not cover one PCM frame"
                    )
                rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
                rms_dbfs = 20.0 * np.log10(max(rms, np.finfo(np.float64).tiny))
                piece_has_lexical = any(character.isalnum() for character in piece)
                if piece_has_lexical:
                    seen_lexical = True
                if piece.strip():
                    terminal_candidate = piece.strip().rstrip("\"'”’)]}")
                    ends_utterance = (
                        bool(terminal_candidate)
                        and terminal_candidate[-1] in handshake_terminal_punctuation
                    )
                    if seen_lexical and ends_utterance:
                        terminal_seen = True
                        terminal_frame = startup_index
                        terminal_piece = piece
                    elif piece_has_lexical:
                        terminal_seen = False
                        terminal_frame = None
                        terminal_piece = None
                quiet = (
                    not piece.strip()
                    and rms_dbfs < float(handshake_silence_threshold_dbfs)
                )
                quiet_frames = quiet_frames + 1 if terminal_seen and quiet else 0
                if terminal_seen and quiet_frames >= handshake_quiet_frames:
                    assert terminal_frame is not None and terminal_piece is not None
                    return (
                        probe_outputs,
                        probe_feedback,
                        startup_index + 1,
                        terminal_frame,
                        terminal_piece,
                    )
            raise ContractError(
                "natural startup greeting did not finish with the required quiet gap"
            )

        try:
            with torch.no_grad():
                if startup_mode == STARTUP_MODE_COMMON_HANDSHAKE:
                    (
                        probe_outputs,
                        probe_feedback,
                        startup_frames,
                        handshake_terminal_frame,
                        handshake_terminal_piece,
                    ) = measure_handshake()
                    handshake_probe_steps = 1 + startup_frames
                    raw_pcm = np.asarray(conversation_pcm, dtype=np.float32)
                    continuous_codes = self.encode_continuous_conversation_pcm(
                        raw_pcm,
                        target_frame_count=target_frames,
                        startup_frame_count=startup_frames,
                    ).to(device=self.device, dtype=torch.long)
                    startup_codes = continuous_codes[..., :startup_frames]
                    codes = continuous_codes[..., startup_frames:]
                    if not torch.equal(startup_codes, silence[..., :startup_frames]):
                        raise ContractError(
                            "continuous Mimi startup prefix differs from measured silence prefix"
                        )
                    continuous_mimi_input_verified = True

                    # The measured greeting is only a sizing pass.  Start the
                    # evidential pass from a fresh LM/RNG state and require the
                    # complete greeting token and feedback traces to replay.
                    self.reset(seed)
                    self.lm_gen.set_mechanistic_hook(None)
                    self._advance_generation_frame(startup_codes[:, :, :1], expect_output=False)
                    lm_steps = handshake_probe_steps + 1
                    for startup_index in range(startup_frames):
                        result = self._advance_generation_frame(
                            startup_codes[:, :, startup_index : startup_index + 1]
                        )
                        assert result is not None
                        shared_outputs.append(result.output)
                        shared_feedback.append(result.feedback)
                        lm_steps += 1
                    replay_outputs = self._stack_frames(
                        shared_outputs, label="replayed handshake output"
                    )
                    replay_feedback = self._stack_frames(
                        shared_feedback, label="replayed handshake feedback"
                    )
                    handshake_replay_identical = bool(
                        torch.equal(
                            replay_outputs,
                            self._stack_frames(probe_outputs, label="measured handshake output"),
                        )
                        and torch.equal(
                            replay_feedback,
                            self._stack_frames(probe_feedback, label="measured handshake feedback"),
                        )
                    )
                    if not handshake_replay_identical:
                        raise ContractError(
                            "fresh continuous-input pass did not replay the measured greeting exactly"
                        )
                else:
                    self.reset(seed)
                    self.lm_gen.set_mechanistic_hook(None)

                if startup_mode == STARTUP_MODE_GREETING_SUPPRESSED:
                    forced_text, forced_audio, force_mask = suppression_feedback(0)
                    _, call_count = self._advance_generation_frame_masked(
                        codes[:, :, :1],
                        forced_text=forced_text,
                        forced_audio=forced_audio,
                        audio_force_mask=force_mask,
                        expect_output=False,
                    )
                    lm_steps += call_count
                elif startup_mode != STARTUP_MODE_COMMON_HANDSHAKE:
                    self._advance_generation_frame(codes[:, :, :1], expect_output=False)
                    lm_steps += 1

                for frame in range(branch):
                    forced_text, forced_audio, force_mask = suppression_feedback(frame + 1)
                    result, call_count = self._advance_generation_frame_masked(
                        codes[:, :, frame : frame + 1],
                        forced_text=forced_text,
                        forced_audio=forced_audio,
                        audio_force_mask=force_mask,
                    )
                    assert result is not None
                    shared_outputs.append(result.output)
                    shared_feedback.append(result.feedback)
                    lm_steps += call_count

                state_snapshot = self.lm_gen.snapshot_streaming_state()
                rng_snapshot = self._snapshot_rng()

                def run_suffix(
                    *, patched: bool
                ) -> tuple[list[Any], list[Any], tuple[int, ...], tuple[int, ...], int]:
                    active_frame = -1
                    hit_counts = [0 for _ in interventions]
                    fired_order: list[int] = []

                    def hook(event):
                        if not patched or not interventions:
                            return None
                        matched_index = next(
                            (
                                index
                                for index, (site, layer, frame, _) in enumerate(interventions)
                                if event.site == site
                                and int(event.layer) == layer
                                and active_frame == frame
                            ),
                            None,
                        )
                        if matched_index is None:
                            return None
                        site, layer, frame, head = interventions[matched_index]
                        replacement = event.tensor.clone()
                        if head is None:
                            replacement.zero_()
                        else:
                            head_index = int(head)
                            if replacement.ndim != 4 or head_index >= replacement.shape[1]:
                                raise ContractError("intervention head is outside the event tensor")
                            replacement[:, head_index].zero_()
                        if not getattr(self, "_feedback_preview_pass", False):
                            hit_counts[matched_index] += 1
                            fired_order.append(matched_index)
                        return replacement

                    arm_outputs: list[Any] = []
                    arm_feedback: list[Any] = []
                    # Both arms traverse the identical eager hook path.  The
                    # baseline callback is a no-op; only the patched callback
                    # replaces the selected event.  This prevents a fast-path
                    # versus instrumented-path numeric difference from being
                    # misclassified as an intervention effect.
                    physical_steps = 0
                    self.lm_gen.set_mechanistic_hook(hook if interventions else None)
                    try:
                        for active_frame in range(branch, target_frames):
                            forced_text, forced_audio, force_mask = suppression_feedback(
                                active_frame + 1
                            )
                            result, call_count = self._advance_generation_frame_masked(
                                codes[:, :, active_frame : active_frame + 1],
                                forced_text=forced_text,
                                forced_audio=forced_audio,
                                audio_force_mask=force_mask,
                            )
                            assert result is not None
                            arm_outputs.append(result.output)
                            arm_feedback.append(result.feedback)
                            physical_steps += call_count
                    finally:
                        self.lm_gen.set_mechanistic_hook(None)
                    return (
                        arm_outputs,
                        arm_feedback,
                        tuple(hit_counts),
                        tuple(fired_order),
                        physical_steps,
                    )

                baseline_outputs, baseline_feedback, _, _, baseline_steps = run_suffix(patched=False)
                lm_steps += baseline_steps
                self.lm_gen.restore_streaming_state(state_snapshot)
                self._restore_rng(rng_snapshot)
                (
                    patched_outputs,
                    patched_feedback,
                    hit_counts,
                    fired_order,
                    patched_steps,
                ) = run_suffix(patched=True)
                lm_steps += patched_steps
                if interventions and any(hit_count != 1 for hit_count in hit_counts):
                    raise ContractError(
                        "each intervention site was expected exactly once but observed "
                        f"{dict(zip((row[0] for row in interventions), hit_counts))}"
                    )
                if interventions and fired_order != tuple(range(len(interventions))):
                    raise ContractError(
                        "joint intervention events fired in a different order than requested: "
                        f"observed={list(fired_order)}, "
                        f"expected={list(range(len(interventions)))}"
                    )
        finally:
            self._feedback_preview_pass = False
            self.lm_gen.set_mechanistic_hook(None)

        full_baseline_outputs = [*shared_outputs, *baseline_outputs]
        full_patched_outputs = [*shared_outputs, *patched_outputs]
        full_baseline_feedback = [*shared_feedback, *baseline_feedback]
        full_patched_feedback = [*shared_feedback, *patched_feedback]
        expected_frames = startup_frames + target_frames
        if not all(
            len(rows) == expected_frames
            for rows in (
                full_baseline_outputs,
                full_patched_outputs,
                full_baseline_feedback,
                full_patched_feedback,
            )
        ):
            raise ContractError("paired arms do not cover the exact generation horizon")

        baseline_tokens = self._stack_frames(full_baseline_outputs, label="baseline outputs")
        patched_tokens = self._stack_frames(full_patched_outputs, label="patched outputs")
        baseline_feedback_tokens = self._stack_frames(
            full_baseline_feedback, label="baseline feedback"
        )
        patched_feedback_tokens = self._stack_frames(
            full_patched_feedback, label="patched feedback"
        )
        first_output_divergence = self._first_divergence(baseline_tokens, patched_tokens)
        first_feedback_divergence = self._first_divergence(
            baseline_feedback_tokens, patched_feedback_tokens
        )
        absolute_branch = startup_frames + branch
        for label, divergence in (
            ("output", first_output_divergence),
            ("feedback", first_feedback_divergence),
        ):
            if divergence is not None and divergence < absolute_branch:
                raise ContractError(f"{label} diverged before the state/RNG branch")

        prefix_tokens = self._stack_frames(shared_outputs, label="shared prefix") if shared_outputs else None
        prefix_sha = (
            self._token_sha256(prefix_tokens)
            if prefix_tokens is not None
            else hashlib.sha256(b"").hexdigest()
        )
        prefix_feedback = (
            self._stack_frames(shared_feedback, label="shared feedback")
            if shared_feedback
            else None
        )
        prefix_feedback_sha = (
            self._token_sha256(prefix_feedback)
            if prefix_feedback is not None
            else hashlib.sha256(b"").hexdigest()
        )
        baseline = self._make_sequence(
            full_baseline_outputs,
            full_baseline_feedback,
            conversation_frame_count=target_frames,
            conversation_start_frame=startup_frames,
            blank_token_ids=blank_token_ids,
        )
        patched = self._make_sequence(
            full_patched_outputs,
            full_patched_feedback,
            conversation_frame_count=target_frames,
            conversation_start_frame=startup_frames,
            blank_token_ids=blank_token_ids,
        )
        return PairedGeneration(
            baseline=baseline,
            patched=patched,
            branch_frame=branch,
            shared_prefix_frames=len(shared_outputs),
            shared_prefix_sha256=prefix_sha,
            shared_feedback_sha256=prefix_feedback_sha,
            first_feedback_divergence_frame=first_feedback_divergence,
            first_output_divergence_frame=first_output_divergence,
            pre_intervention_identical=True,
            startup_mode=startup_mode,
            startup_frame_count=startup_frames,
            handshake_terminal_frame=handshake_terminal_frame,
            handshake_terminal_piece=handshake_terminal_piece,
            handshake_completion_signal=(
                "terminal_punctuation_plus_text_audio_quiet"
                if startup_mode == STARTUP_MODE_COMMON_HANDSHAKE
                else None
            ),
            target_frame_count=target_frames,
            lm_step_count=lm_steps,
            handshake_probe_lm_step_count=handshake_probe_steps,
            handshake_replay_identical=handshake_replay_identical,
            continuous_mimi_input_verified=continuous_mimi_input_verified,
        )

    def generate_codes(
        self,
        codes: Any,
        *,
        seed: int,
        intervention: Intervention | Sequence[Intervention] | None = None,
    ) -> tuple[list[int], np.ndarray]:
        """Backward-compatible natural-start wrapper.

        New callers should use :meth:`generate_paired_conversation` so the two
        arms share one pre-intervention model/RNG snapshot.
        """

        interventions = _normalise_interventions(intervention)
        branch = interventions[0][2] if interventions else 0
        result = self.generate_paired_conversation(
            codes,
            assistant_silence_codes=None,
            seed=seed,
            branch_frame=branch,
            intervention=intervention,
            startup_mode=STARTUP_MODE_NATURAL,
            target_frame_count=int(codes.shape[-1]),
            user_start_frame=0,
            query_end_frame=int(codes.shape[-1]),
            user_end_frame=int(codes.shape[-1]),
        )
        selected = result.patched if intervention is not None else result.baseline
        return selected.text_token_ids, selected.pcm
