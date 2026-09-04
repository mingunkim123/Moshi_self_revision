from __future__ import annotations

import numpy as np

from experiments.self_repair.mechanistic.runtime import SyntheticBackend


def test_synthetic_replay_is_deterministic_and_open_loop() -> None:
    backend = SyntheticBackend()
    trial = {"trial_id": "repair", "condition": "repair", "old_value": "Boston",
             "new_value": "Seattle", "frame_count": 12}
    first = backend.replay(trial, ["resid_post"])
    second = backend.replay(trial, ["resid_post"])
    np.testing.assert_array_equal(first.activations["resid_post"], second.activations["resid_post"])
    np.testing.assert_array_equal(first.logits, second.logits)
    assert first.feedback_sha256 == second.feedback_sha256


def test_analytic_patch_has_expected_direction() -> None:
    backend = SyntheticBackend()
    repair = {"trial_id": "repair", "condition": "repair", "old_value": "Boston",
              "new_value": "Seattle", "frame_count": 12}
    clean = {**repair, "trial_id": "clean", "condition": "clean_current"}
    result = backend.patch(repair, clean, component="resid_post", layer=5, head=None, anchor_frame=8)
    assert result["delta_M"] > 0
    self_result = backend.patch(repair, repair, component="resid_post", layer=5, head=None, anchor_frame=8)
    assert self_result["delta_M"] == 0
