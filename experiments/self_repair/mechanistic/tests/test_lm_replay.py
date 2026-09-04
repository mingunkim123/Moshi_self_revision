from __future__ import annotations

import torch

from moshi.models.lm import LMGen, LMModel


def _generator() -> LMGen:
    torch.manual_seed(11)
    model = LMModel(
        delays=[0, 1, 2, 4], n_q=3, dep_q=3, card=32, text_card=48,
        dim=16, num_layers=2, num_heads=1, hidden_scale=1,
        depformer_dim=16, depformer_multi_linear=True,
        depformer_weights_per_step=True, depformer_weights_per_step_schedule=[0, 1, 1],
        depformer_low_rank_embeddings=8, depformer_num_heads=1,
        depformer_gating="silu", context=4, device="cpu", dtype=torch.float32,
    ).eval()
    return LMGen(model, use_sampling=False)


def test_snapshot_restore_replays_identical_logits_and_feedback() -> None:
    generator = _generator()
    empty_user = torch.empty((1, 0, 1), dtype=torch.long)
    text_feedback = torch.tensor([generator.lm_model.zero_token_id])
    audio_feedback = torch.full((1, 3), generator.lm_model.zero_token_id, dtype=torch.long)
    with generator.streaming(1):
        generator.step_open_loop(empty_user, feedback_text_token=text_feedback, feedback_audio_tokens=audio_feedback)
        snapshot = generator.snapshot_streaming_state()
        first = generator.step_open_loop(
            empty_user, feedback_text_token=text_feedback, feedback_audio_tokens=audio_feedback)
        generator.restore_streaming_state(snapshot)
        second = generator.step_open_loop(
            empty_user, feedback_text_token=text_feedback, feedback_audio_tokens=audio_feedback)
    torch.testing.assert_close(first.text_logits, second.text_logits, rtol=0, atol=0)
    torch.testing.assert_close(first.feedback_text, text_feedback)
    torch.testing.assert_close(first.feedback_audio, audio_feedback)
    assert not torch.equal(first.sampled_text, torch.tensor([-1]))


def test_eager_forward_text_keeps_gradient_path() -> None:
    generator = _generator()
    sequence = generator.lm_model._get_initial_token().clone()
    _, logits = generator.eager_forward_text(sequence)
    logits.sum().backward()
    assert generator.lm_model.text_linear.weight.grad is not None
