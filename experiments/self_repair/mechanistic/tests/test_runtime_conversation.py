from __future__ import annotations

import copy
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.self_repair.mechanistic.conversation import (
    STARTUP_MODE_COMMON_HANDSHAKE,
    STARTUP_MODE_GREETING_SUPPRESSED,
    STARTUP_MODE_NATURAL,
)
from experiments.self_repair.mechanistic.core import ContractError, FRAME_SAMPLES
from experiments.self_repair.mechanistic.runtime import (
    FROZEN_AUDIO_ACTIVITY_THRESHOLD_DBFS,
    FROZEN_GREETING_MAX_FRAMES,
    FROZEN_GREETING_QUIET_FRAMES,
    FROZEN_PREPARED_LEADIN_FRAMES,
    MoshiBackend,
)
from moshi.models.lm import LMGen, LMModel


class _Tokenizer:
    def id_to_piece(self, token_id: int) -> str:
        return {0: "", 3: "", 5: "▁hello", 6: "▁there", 7: "."}.get(
            token_id, f"▁token{token_id}"
        )


class _Mimi:
    sample_rate = 24_000
    num_codebooks = 2
    cardinality = 32

    def __init__(self, *, wrong_frame_size: bool = False):
        self.wrong_frame_size = wrong_frame_size
        self.reset_count = 0
        self.encode_offset = 0
        self.encoded_frames: list[tuple[int, bool]] = []

    def reset_streaming(self) -> None:
        self.reset_count += 1
        self.encode_offset = 0

    def encode(self, pcm: torch.Tensor) -> torch.Tensor:
        if tuple(pcm.shape) != (1, 1, FRAME_SAMPLES):
            raise AssertionError(f"unexpected fake Mimi input shape: {tuple(pcm.shape)}")
        active = bool((pcm != 0).any().item())
        self.encoded_frames.append((self.encode_offset, active))
        if active:
            first = 1 + self.encode_offset % 15
            values = torch.tensor([first, first + 8], dtype=torch.long)
        else:
            values = torch.zeros(2, dtype=torch.long)
        self.encode_offset += 1
        return values[None, :, None]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        frame_samples = FRAME_SAMPLES - 1 if self.wrong_frame_size else FRAME_SAMPLES
        active = bool((codes != 0).any().item())
        amplitude = 0.02 if active else 0.0
        return torch.full((1, 1, frame_samples), amplitude, dtype=torch.float32)


class _FakeLMGen:
    """Small deterministic streaming fixture that consumes all four RNGs."""

    def __init__(self, *, startup_script: dict[int, int] | None = None):
        self.lm_model = SimpleNamespace(
            dep_q=2,
            num_codebooks=5,
            card=32,
            delays=[0, 0, 1, 0, 1],
            zero_token_id=-1,
            text_padding_token_id=3,
            end_of_text_padding_id=0,
        )
        self.startup_script = startup_script or {}
        self.hook = None
        self.calls: list[dict[str, object]] = []
        self.reset_streaming()

    def reset_streaming(self) -> None:
        self.offset = 0
        self.accumulator = 0
        self.feedback_history: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._last_step_result = None

    def set_mechanistic_hook(self, hook) -> None:
        self.hook = hook

    def snapshot_streaming_state(self):
        return copy.deepcopy((self.offset, self.accumulator, self.feedback_history))

    def restore_streaming_state(self, snapshot) -> None:
        self.offset, self.accumulator, self.feedback_history = copy.deepcopy(snapshot)

    def _run(self, input_codes, forced_text=None, forced_audio=None):
        input_value = int(torch.as_tensor(input_codes).sum().item())
        intervention_signal = 0
        for event_index, site in enumerate(("k_pre_rope", "v_pre_rope", "resid_post")):
            event_tensor = torch.full(
                (1, 2, 1, 1), input_value + event_index + 1, dtype=torch.float32
            )
            if self.hook is not None:
                replacement = self.hook(
                    SimpleNamespace(site=site, layer=2, tensor=event_tensor)
                )
                if replacement is not None:
                    event_tensor = replacement
            intervention_signal += int(event_tensor.sum().item())
        noise = (
            random.randrange(11)
            + int(np.random.randint(0, 11))
            + int(torch.randint(0, 11, (1,)).item())
        )
        sampled_id = self.startup_script.get(
            self.offset, 4 + ((input_value + self.accumulator + intervention_signal + noise) % 23)
        )
        sampled_text = torch.tensor([sampled_id], dtype=torch.long)
        sampled_audio = torch.full(
            (1, 2), 0 if sampled_id in {0, 3} else 1 + sampled_id % 7, dtype=torch.long
        )
        feedback_text = sampled_text if forced_text is None else torch.as_tensor(forced_text).reshape(1)
        feedback_audio = (
            sampled_audio
            if forced_audio is None
            else torch.as_tensor(forced_audio, dtype=torch.long).reshape(1, 2)
        )
        self.feedback_history.append((feedback_text.clone(), feedback_audio.clone()))
        output_tokens = None
        if self.offset > 0:
            previous_text, previous_audio = self.feedback_history[self.offset - 1]
            current_audio = self.feedback_history[self.offset][1]
            # Exact LMGen schedule for output delays [text=0, cb0=0, cb1=1].
            output_tokens = torch.stack(
                [previous_text, previous_audio[:, 0], current_audio[:, 1]], dim=1
            )[:, :, None]
        text_logits = torch.full((1, 1, 1, 32), float(intervention_signal))
        detail = SimpleNamespace(
            output_tokens=output_tokens,
            text_logits=text_logits,
            sampled_text=sampled_text,
            sampled_audio=sampled_audio,
            feedback_text=feedback_text,
            feedback_audio=feedback_audio,
        )
        self._last_step_result = detail
        self.calls.append(
            {
                "offset": self.offset,
                "input": input_value,
                "forced": forced_text is not None,
                "feedback_text": int(feedback_text.item()),
                "feedback_audio": tuple(int(value) for value in feedback_audio[0].tolist()),
            }
        )
        self.accumulator += intervention_signal + int(feedback_text.item()) + int(
            feedback_audio.sum().item()
        )
        self.offset += 1
        return detail

    def step(self, input_codes):
        return self._run(input_codes).output_tokens

    def step_open_loop(self, input_codes, *, feedback_text_token, feedback_audio_tokens):
        return self._run(input_codes, feedback_text_token, feedback_audio_tokens)


def _backend(
    *,
    startup_script: dict[int, int] | None = None,
    wrong_frame_size: bool = False,
) -> MoshiBackend:
    backend = MoshiBackend.__new__(MoshiBackend)
    backend.torch = torch
    backend.device = "cpu"
    backend.mimi = _Mimi(wrong_frame_size=wrong_frame_size)
    backend.tokenizer = _Tokenizer()
    backend.lm_gen = _FakeLMGen(startup_script=startup_script)
    backend.state = SimpleNamespace(frame_size=FRAME_SAMPLES)

    def seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    backend.seed_all = seed_all
    return backend


def _toy_transformer_backend() -> MoshiBackend:
    torch.manual_seed(41)
    model = LMModel(
        delays=[0, 0, 1, 1, 0, 1],
        n_q=5,
        dep_q=3,
        card=32,
        text_card=48,
        dim=16,
        num_layers=2,
        num_heads=1,
        hidden_scale=1,
        depformer_dim=16,
        depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_weights_per_step_schedule=[0, 1, 1],
        depformer_low_rank_embeddings=8,
        depformer_num_heads=1,
        depformer_gating="silu",
        context=8,
        device="cpu",
        dtype=torch.float32,
    ).eval()
    generator = LMGen(model, use_sampling=True, top_k=8, top_k_text=8)
    generator.streaming_forever(1)
    backend = MoshiBackend.__new__(MoshiBackend)
    backend.torch = torch
    backend.device = "cpu"
    backend.mimi = _Mimi()
    backend.tokenizer = _Tokenizer()
    backend.lm_gen = generator
    backend.state = SimpleNamespace(frame_size=FRAME_SAMPLES)

    def seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    backend.seed_all = seed_all
    return backend


def _codes(frames: int) -> torch.Tensor:
    values = torch.arange(1, frames + 1, dtype=torch.long)
    return torch.stack([values, values + 1])[None]


def _silence(frames: int, *, codebooks: int = 2) -> torch.Tensor:
    return torch.zeros((1, codebooks, frames), dtype=torch.long)


def _prepared_pcm(frames: int) -> np.ndarray:
    assert frames > FROZEN_PREPARED_LEADIN_FRAMES
    pcm = np.zeros(frames * FRAME_SAMPLES, dtype=np.float32)
    pcm[FROZEN_PREPARED_LEADIN_FRAMES * FRAME_SAMPLES :] = 0.1
    return pcm


def test_conversation_encoding_appends_exact_zero_pcm_before_mimi_encoding() -> None:
    backend = MoshiBackend.__new__(MoshiBackend)
    backend.torch = torch
    user_pcm = np.ones(3 * FRAME_SAMPLES, dtype=np.float32)
    encoded_inputs: list[np.ndarray] = []
    backend._read_pcm = lambda _: user_pcm.copy()

    def encode_pcm(pcm: np.ndarray) -> torch.Tensor:
        encoded_inputs.append(np.asarray(pcm).copy())
        frame_count = len(pcm) // FRAME_SAMPLES
        frame_activity = np.asarray(pcm).reshape(frame_count, FRAME_SAMPLES).any(axis=1)
        values = torch.as_tensor(frame_activity, dtype=torch.long)
        return torch.stack([values, values])[None]

    backend.encode_pcm = encode_pcm
    encoded = backend.encode_conversation_file(Path("fixture.wav"), target_frame_count=5)

    assert encoded.user_frame_count == 3
    assert encoded.target_frame_count == 5
    assert encoded.user_codes.shape == (1, 2, 3)
    assert encoded.conversation_codes.shape == (1, 2, 5)
    assert encoded.assistant_silence_codes.shape == (1, 2, 5)
    np.testing.assert_array_equal(encoded_inputs[0][: 3 * FRAME_SAMPLES], user_pcm)
    assert not encoded_inputs[0][3 * FRAME_SAMPLES :].any()
    assert not encoded_inputs[1].any()


def test_natural_pair_has_one_prime_exact_horizon_and_shared_rng() -> None:
    backend = _backend()
    frames = 7
    result = backend.generate_paired_conversation(
        _codes(frames),
        assistant_silence_codes=None,
        seed=91,
        branch_frame=3,
        intervention=None,
        startup_mode=STARTUP_MODE_NATURAL,
        target_frame_count=frames,
        user_start_frame=1,
        query_end_frame=4,
        user_end_frame=5,
    )

    assert result.baseline.frame_count == frames
    assert result.patched.frame_count == frames
    assert result.baseline.pcm_sample_count == frames * FRAME_SAMPLES
    assert result.baseline.conversation_pcm.size == frames * FRAME_SAMPLES
    assert result.baseline.conversation_tokens.shape[-1] == frames
    assert result.baseline.text_token_ids == result.patched.text_token_ids
    torch.testing.assert_close(result.baseline.feedback_tokens, result.patched.feedback_tokens)
    assert result.first_output_divergence_frame is None
    assert result.first_feedback_divergence_frame is None
    assert result.shared_prefix_frames == 3
    assert result.lm_step_count == 1 + 3 + 2 * (frames - 3)
    # One delay-prime plus one shared pass of frames [0, 3), then two suffixes.
    assert len(backend.lm_gen.calls) == result.lm_step_count
    assert backend.lm_gen.calls[0]["input"] == backend.lm_gen.calls[1]["input"]


def test_real_toy_lmgen_restores_streaming_state_and_rng_between_arms() -> None:
    backend = _toy_transformer_backend()
    frames = 5
    user_codes = torch.randint(0, 31, (1, 2, frames), dtype=torch.long)
    result = backend.generate_paired_conversation(
        user_codes,
        assistant_silence_codes=None,
        seed=22,
        branch_frame=2,
        intervention=None,
        startup_mode=STARTUP_MODE_NATURAL,
        target_frame_count=frames,
        user_start_frame=1,
        query_end_frame=3,
        user_end_frame=4,
    )

    torch.testing.assert_close(result.baseline.tokens, result.patched.tokens, rtol=0, atol=0)
    torch.testing.assert_close(
        result.baseline.feedback_tokens, result.patched.feedback_tokens, rtol=0, atol=0
    )
    assert result.baseline.frame_count == frames
    assert result.baseline.pcm_sample_count == frames * FRAME_SAMPLES
    assert result.first_output_divergence_frame is None
    assert result.first_feedback_divergence_frame is None


def test_real_toy_lmgen_hits_one_frozen_intervention_event() -> None:
    backend = _toy_transformer_backend()
    frames = 5
    result = backend.generate_paired_conversation(
        torch.randint(0, 31, (1, 2, frames), dtype=torch.long),
        assistant_silence_codes=None,
        seed=23,
        branch_frame=2,
        intervention=("resid_post", 0, 2, None),
        startup_mode=STARTUP_MODE_NATURAL,
        target_frame_count=frames,
        user_start_frame=1,
        query_end_frame=3,
        user_end_frame=4,
    )

    torch.testing.assert_close(
        result.baseline.tokens[..., :2], result.patched.tokens[..., :2], rtol=0, atol=0
    )
    assert result.pre_intervention_identical
    assert result.baseline.frame_count == frames


def test_intervention_branches_immediately_and_never_changes_prefix() -> None:
    backend = _backend()
    frames = 8
    branch = 4
    result = backend.generate_paired_conversation(
        _codes(frames),
        assistant_silence_codes=None,
        seed=17,
        branch_frame=branch,
        intervention=("resid_post", 2, branch, 0),
        startup_mode=STARTUP_MODE_NATURAL,
        target_frame_count=frames,
        user_start_frame=1,
        query_end_frame=5,
        user_end_frame=6,
    )

    torch.testing.assert_close(
        result.baseline.tokens[..., :branch], result.patched.tokens[..., :branch]
    )
    torch.testing.assert_close(
        result.baseline.feedback_tokens[..., :branch],
        result.patched.feedback_tokens[..., :branch],
    )
    assert result.first_feedback_divergence_frame is not None
    assert result.first_feedback_divergence_frame >= branch
    assert result.first_output_divergence_frame is not None
    assert result.first_output_divergence_frame >= branch
    assert result.pre_intervention_identical
    assert len(result.shared_prefix_sha256) == 64
    assert len(result.shared_feedback_sha256) == 64


def test_ordered_joint_intervention_fires_k_then_v_exactly_once() -> None:
    backend = _backend()
    frames = 7
    branch = 3
    result = backend.generate_paired_conversation(
        _codes(frames),
        assistant_silence_codes=None,
        seed=18,
        branch_frame=branch,
        intervention=[
            ("k_pre_rope", 2, branch, 0),
            ("v_pre_rope", 2, branch, 1),
        ],
        startup_mode=STARTUP_MODE_NATURAL,
        target_frame_count=frames,
        user_start_frame=1,
        query_end_frame=5,
        user_end_frame=5,
    )

    torch.testing.assert_close(
        result.baseline.tokens[..., :branch], result.patched.tokens[..., :branch]
    )
    assert result.pre_intervention_identical
    assert result.first_output_divergence_frame is not None
    assert result.first_output_divergence_frame >= branch


def test_joint_intervention_rejects_duplicate_sites() -> None:
    backend = _backend()
    with pytest.raises(ContractError, match="duplicate intervention site"):
        backend.generate_paired_conversation(
            _codes(6),
            assistant_silence_codes=None,
            seed=0,
            branch_frame=2,
            intervention=[
                ("k_pre_rope", 2, 2, 0),
                ("k_pre_rope", 3, 2, 1),
            ],
            startup_mode=STARTUP_MODE_NATURAL,
            target_frame_count=6,
            user_start_frame=1,
            query_end_frame=4,
            user_end_frame=4,
        )


def test_joint_intervention_fails_if_any_requested_event_does_not_fire() -> None:
    backend = _backend()
    with pytest.raises(ContractError, match="expected exactly once"):
        backend.generate_paired_conversation(
            _codes(6),
            assistant_silence_codes=None,
            seed=0,
            branch_frame=2,
            intervention=[
                ("k_pre_rope", 2, 2, 0),
                ("missing_site", 2, 2, 0),
            ],
            startup_mode=STARTUP_MODE_NATURAL,
            target_frame_count=6,
            user_start_frame=1,
            query_end_frame=4,
            user_end_frame=4,
        )


def test_joint_intervention_rejects_event_order_mismatch() -> None:
    backend = _backend()
    with pytest.raises(ContractError, match="different order"):
        backend.generate_paired_conversation(
            _codes(6),
            assistant_silence_codes=None,
            seed=0,
            branch_frame=2,
            intervention=[
                ("v_pre_rope", 2, 2, 0),
                ("k_pre_rope", 2, 2, 0),
            ],
            startup_mode=STARTUP_MODE_NATURAL,
            target_frame_count=6,
            user_start_frame=1,
            query_end_frame=4,
            user_end_frame=4,
        )


def test_greeting_suppression_forces_pad_and_silence_through_user_end() -> None:
    backend = _backend()
    frames = 7
    result = backend.generate_paired_conversation(
        _codes(frames),
        assistant_silence_codes=_silence(frames),
        seed=3,
        branch_frame=3,
        intervention=("resid_post", 2, 3, None),
        startup_mode=STARTUP_MODE_GREETING_SUPPRESSED,
        target_frame_count=frames,
        user_start_frame=1,
        query_end_frame=3,
        user_end_frame=4,
    )

    assert result.startup_frame_count == 0
    assert result.baseline.text_token_ids[:4] == [3, 3, 3, 3]
    # feedback_tokens are decision-time, not output-aligned.  On the call that
    # emits the last suppressed frame, delay-0 text/audio are already released
    # while delay-1 audio is still forced for that current output frame.
    assert not result.baseline.feedback_tokens[..., :3].ne(
        torch.tensor([[[3], [0], [0]]])
    ).any()
    boundary_feedback = result.baseline.feedback_tokens[0, :, 3]
    assert int(boundary_feedback[0]) != 3
    assert int(boundary_feedback[1]) != 0
    assert int(boundary_feedback[2]) == 0
    assert result.baseline.text_token_ids[4] == int(boundary_feedback[0])


def test_real_toy_lmgen_inverse_delay_schedule_matches_aligned_silence() -> None:
    backend = _toy_transformer_backend()
    frames = 7
    silence = torch.stack(
        [
            torch.arange(1, frames + 1),
            torch.arange(9, 9 + frames),
            torch.arange(17, 17 + frames),
        ]
    )[None]
    result = backend.generate_paired_conversation(
        torch.randint(0, 31, (1, 2, frames), dtype=torch.long),
        assistant_silence_codes=silence,
        seed=31,
        branch_frame=2,
        intervention=None,
        startup_mode=STARTUP_MODE_GREETING_SUPPRESSED,
        target_frame_count=frames,
        user_start_frame=1,
        query_end_frame=3,
        user_end_frame=4,
    )

    torch.testing.assert_close(
        result.baseline.tokens[0, 1:, :4], silence[0, :, :4], rtol=0, atol=0
    )
    assert result.baseline.text_token_ids[:4] == [3, 3, 3, 3]
    torch.testing.assert_close(result.baseline.tokens, result.patched.tokens, rtol=0, atol=0)


def test_common_handshake_waits_for_lexical_greeting_and_audio_quiet_gap() -> None:
    startup_script = {0: 5, 1: 6, 2: 7}
    startup_script.update({offset: 0 for offset in range(3, 24)})
    backend = _backend(
        startup_script=startup_script
    )
    frames = 10
    pcm = _prepared_pcm(frames)
    result = backend.generate_paired_conversation(
        _codes(frames),
        assistant_silence_codes=_silence(FROZEN_GREETING_MAX_FRAMES),
        conversation_pcm=pcm,
        seed=5,
        branch_frame=1,
        intervention=None,
        startup_mode=STARTUP_MODE_COMMON_HANDSHAKE,
        target_frame_count=frames,
        user_start_frame=FROZEN_PREPARED_LEADIN_FRAMES,
        query_end_frame=8,
        user_end_frame=9,
    )

    assert result.startup_frame_count == 23
    assert result.handshake_terminal_frame == 2
    assert result.handshake_terminal_piece == "."
    assert result.handshake_completion_signal == "terminal_punctuation_plus_text_audio_quiet"
    assert result.handshake_probe_lm_step_count == 24
    assert result.handshake_replay_identical is True
    assert result.continuous_mimi_input_verified is True
    assert result.baseline.conversation_start_frame == 23
    assert result.baseline.conversation_frame_count == frames
    assert result.baseline.frame_count == frames + 23
    assert result.baseline.text_token_ids[:23] == [5, 6, 7] + [0] * 20
    assert result.shared_prefix_frames == 23 + 1
    expected_steps = 2 * (1 + 23) + 1 + 2 * (frames - 1)
    assert result.lm_step_count == expected_steps
    assert len(backend.lm_gen.calls) == expected_steps
    assert result.baseline.pcm_sample_count == (frames + 23) * FRAME_SAMPLES
    assert result.baseline.conversation_pcm.size == frames * FRAME_SAMPLES
    assert result.baseline.conversation_tokens.shape[-1] == frames
    # Probe and evidential pass both see the exact same causal-Mimi startup.
    assert backend.lm_gen.calls[1]["input"] == 0
    assert backend.lm_gen.calls[23]["input"] == 0
    assert backend.lm_gen.calls[25]["input"] == 0
    assert backend.lm_gen.calls[47]["input"] == 0
    first_conversation_call = backend.lm_gen.calls[48]
    assert first_conversation_call["input"] == 0  # frozen prepared lead-in, not a reset transient


def test_continuous_mimi_encoding_keeps_state_across_startup_and_prepared_audio() -> None:
    backend = _backend()
    frames = 8
    pcm = _prepared_pcm(frames)
    separately_encoded = backend.encode_pcm(pcm)
    continuous = backend.encode_continuous_conversation_pcm(
        pcm, target_frame_count=frames, startup_frame_count=23
    )

    assert tuple(continuous.shape) == (1, 2, 23 + frames)
    assert not continuous[..., :23].ne(0).any()
    # The active prepared frames have a different causal encoder offset than a
    # separately reset encode; the generation path must use the continuous one.
    assert not torch.equal(continuous[..., 23:], separately_encoded)


def test_common_handshake_fails_closed_without_completed_greeting() -> None:
    # A lexical greeting followed by silence is still incomplete without a
    # terminal punctuation signal; an internal pause can never launch the user.
    startup_script = {0: 5}
    startup_script.update({offset: 0 for offset in range(1, 151)})
    backend = _backend(startup_script=startup_script)
    with pytest.raises(ContractError, match="greeting did not finish"):
        backend.generate_paired_conversation(
            _codes(8),
            assistant_silence_codes=_silence(FROZEN_GREETING_MAX_FRAMES),
            conversation_pcm=_prepared_pcm(8),
            seed=0,
            branch_frame=1,
            intervention=None,
            startup_mode=STARTUP_MODE_COMMON_HANDSHAKE,
            target_frame_count=8,
            user_start_frame=FROZEN_PREPARED_LEADIN_FRAMES,
            query_end_frame=7,
            user_end_frame=7,
        )


def test_decode_coverage_mismatch_fails_closed() -> None:
    backend = _backend(wrong_frame_size=True)
    with pytest.raises(ContractError, match="exactly one PCM frame"):
        backend.generate_paired_conversation(
            _codes(4),
            assistant_silence_codes=None,
            seed=0,
            branch_frame=1,
            intervention=None,
            startup_mode=STARTUP_MODE_NATURAL,
            target_frame_count=4,
            user_start_frame=1,
            query_end_frame=2,
            user_end_frame=3,
        )


def test_replay_can_stop_exactly_after_an_anchor_frame() -> None:
    backend = _backend()
    result = backend.replay_codes(
        _codes(6), sites=["resid_post"], end_frame_exclusive=3
    )
    assert result.frame_count == 3
    assert result.lm_step_count == 4
    assert result.logits.shape[2] == 3
    assert {key[2] for key in result.event_tensors} == {0, 1, 2}


def test_real_toy_lmgen_hook_off_matches_identity_hook_replay() -> None:
    backend = _toy_transformer_backend()
    codes = torch.randint(0, 31, (1, 2, 5), dtype=torch.long)

    hook_off = backend.replay_codes(codes, hook_enabled=False)
    identity_hook = backend.replay_codes(
        codes,
        sites=["resid_post"],
        capture_layers=[0],
        capture_frames=[0, 4],
    )

    np.testing.assert_array_equal(hook_off.logits, identity_hook.logits)
    assert hook_off.feedback_sha256 == identity_hook.feedback_sha256
    assert hook_off.activations == {}
    assert hook_off.event_tensors == {}
    assert set(identity_hook.event_tensors) == {
        ("resid_post", 0, 0),
        ("resid_post", 0, 4),
    }


def test_hook_off_replay_rejects_capture_requests() -> None:
    backend = _backend()
    with pytest.raises(ContractError, match="hook-off replay"):
        backend.replay_codes(_codes(4), sites=["resid_post"], hook_enabled=False)


@pytest.mark.parametrize("value", [0, 7, True, 2.5])
def test_replay_rejects_invalid_exclusive_end(value) -> None:
    backend = _backend()
    with pytest.raises(ContractError, match="end_frame_exclusive"):
        backend.replay_codes(
            _codes(6), sites=["resid_post"], end_frame_exclusive=value
        )


def test_branch_must_equal_intervention_frame() -> None:
    backend = _backend()
    with pytest.raises(ContractError, match="immediately before"):
        backend.generate_paired_conversation(
            _codes(5),
            assistant_silence_codes=None,
            seed=0,
            branch_frame=1,
            intervention=("resid_post", 2, 2, None),
            startup_mode=STARTUP_MODE_NATURAL,
            target_frame_count=5,
            user_start_frame=1,
            query_end_frame=3,
            user_end_frame=4,
        )


@pytest.mark.parametrize("missing", ["NO_TORCH_COMPILE", "NO_CUDA_GRAPH"])
def test_constructor_requires_explicit_eager_environment_flags(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setenv("NO_TORCH_COMPILE", "1")
    monkeypatch.setenv("NO_CUDA_GRAPH", "1")
    monkeypatch.delenv(missing)
    with pytest.raises(ContractError, match=missing):
        MoshiBackend(device="cpu")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"handshake_max_frames": 149}, "150 frames"),
        ({"handshake_quiet_frames": 19}, "20 frames"),
        ({"prepared_leadin_frames": 5}, "480 ms"),
        ({"handshake_silence_threshold_dbfs": -44.0}, "-45 dBFS"),
    ],
)
def test_common_handshake_rejects_nonfrozen_startup_contract(
    override: dict[str, object], message: str
) -> None:
    backend = _backend()
    with pytest.raises(ContractError, match=message):
        backend.generate_paired_conversation(
            _codes(8),
            assistant_silence_codes=_silence(FROZEN_GREETING_MAX_FRAMES),
            conversation_pcm=_prepared_pcm(8),
            seed=0,
            branch_frame=1,
            intervention=None,
            startup_mode=STARTUP_MODE_COMMON_HANDSHAKE,
            target_frame_count=8,
            user_start_frame=FROZEN_PREPARED_LEADIN_FRAMES,
            query_end_frame=7,
            user_end_frame=7,
            **override,
        )


def test_blank_token_ids_are_bound_to_model_padding_ids() -> None:
    backend = _backend()
    with pytest.raises(ContractError, match="padding IDs"):
        backend.generate_paired_conversation(
            _codes(5),
            assistant_silence_codes=None,
            seed=0,
            branch_frame=1,
            intervention=None,
            startup_mode=STARTUP_MODE_NATURAL,
            target_frame_count=5,
            user_start_frame=1,
            query_end_frame=3,
            user_end_frame=4,
            blank_token_ids=frozenset({0}),
        )
