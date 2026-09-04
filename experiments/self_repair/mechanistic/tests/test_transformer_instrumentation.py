from __future__ import annotations

import torch

from moshi.modules.transformer import RingKVCache, StreamingTransformer


def _transformer() -> StreamingTransformer:
    torch.manual_seed(4)
    return StreamingTransformer(
        d_model=16, num_heads=4, num_layers=2, dim_feedforward=32,
        causal=True, context=8, positional_embedding="rope", device="cpu", dtype=torch.float32,
    ).eval()


def test_identity_hook_matches_fast_path_and_exposes_sites() -> None:
    model = _transformer()
    x = torch.randn(1, 5, 16)
    baseline = model(x)
    events = []

    def hook(event):
        events.append((event.site, event.layer, tuple(event.tensor.shape), event.tensor.dtype, event.tensor.device.type))
        return None

    model.set_mechanistic_hook(hook)
    observed = model(x)
    torch.testing.assert_close(observed, baseline, rtol=0, atol=0)
    sites = {event[0] for event in events}
    assert {"resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post", "head_z"} <= sites
    assert {event[1] for event in events} == {0, 1}
    assert all(event[3] == torch.float32 and event[4] == "cpu" for event in events)


def test_hook_order_and_attention_tensor_shapes_are_stable() -> None:
    model = _transformer()
    x = torch.randn(1, 3, 16)
    events = []

    def hook(event):
        events.append((event.layer, event.site, tuple(event.tensor.shape)))
        return None

    model.set_mechanistic_hook(hook)
    model(x)
    expected_sites = [
        "resid_pre",
        "q_pre_rope", "k_pre_rope", "v_pre_rope",
        "q_post_rope", "k_post_rope", "v_post_rope",
        "head_z", "attn_out", "resid_mid", "mlp_out", "resid_post",
    ]
    assert [(layer, site) for layer, site, _ in events] == [
        (layer, site) for layer in range(2) for site in expected_sites
    ]
    by_site = {site: shape for layer, site, shape in events if layer == 0}
    for site in (
        "q_pre_rope", "k_pre_rope", "v_pre_rope",
        "q_post_rope", "k_post_rope", "v_post_rope", "head_z",
    ):
        assert by_site[site] == (1, 4, 3, 4)
    for site in ("resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"):
        assert by_site[site] == (1, 3, 16)


def test_attention_hooks_expose_absolute_query_and_kv_positions() -> None:
    model = _transformer()
    observed: dict[tuple[int, str], torch.Tensor | None] = {}

    def hook(event):
        if event.site in {
            "q_pre_rope", "k_pre_rope", "v_pre_rope",
            "q_post_rope", "k_post_rope", "v_post_rope", "head_z",
        }:
            observed[(event.layer, event.site)] = (
                None if event.absolute_positions is None else event.absolute_positions.detach().clone()
            )
        return None

    model.set_mechanistic_hook(hook)
    with model.streaming(1):
        model(torch.randn(1, 1, 16))
        model(torch.randn(1, 1, 16))

    expected = torch.tensor([[1]])
    for layer in range(2):
        for site in {
            "q_pre_rope", "k_pre_rope", "v_pre_rope",
            "q_post_rope", "k_post_rope", "v_post_rope", "head_z",
        }:
            torch.testing.assert_close(observed[(layer, site)], expected)


def test_one_head_patch_changes_only_selected_head_at_seam() -> None:
    model = _transformer()
    x = torch.randn(1, 3, 16)
    original = []
    patched = []

    def hook(event):
        if event.site == "head_z" and event.layer == 0:
            original.append(event.tensor.detach().clone())
            replacement = event.tensor.clone()
            replacement[:, 2] = 0
            patched.append(replacement.detach().clone())
            return replacement
        return None

    model.set_mechanistic_hook(hook)
    model(x)
    assert len(original) == len(patched) == 1
    torch.testing.assert_close(original[0][:, :2], patched[0][:, :2])
    torch.testing.assert_close(original[0][:, 3:], patched[0][:, 3:])
    assert torch.count_nonzero(patched[0][:, 2]) == 0


def test_ring_cache_snapshot_restore_and_wrap_positions() -> None:
    cache = RingKVCache(1, 1, 2, capacity=3, device=torch.device("cpu"), dtype=torch.float32)
    mask = torch.ones(1, dtype=torch.bool)
    for step in range(3):
        value = torch.full((1, 1, 1, 2), float(step))
        cache.complete(value, value + 10, mask)
    snapshot = cache.snapshot()
    before = cache.absolute_positions().clone()
    value = torch.full((1, 1, 1, 2), 99.0)
    cache.complete(value, value, mask)
    assert cache.absolute_positions().tolist() == [[3, 1, 2]]
    cache.restore(snapshot)
    torch.testing.assert_close(cache.absolute_positions(), before)
    assert cache.absolute_positions().tolist() == [[0, 1, 2]]
