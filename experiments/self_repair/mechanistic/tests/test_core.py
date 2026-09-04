from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from experiments.self_repair.mechanistic.core import (
    AtomicCellStore,
    ContractError,
    PatchCell,
    anchor_rows,
    apply_probe,
    fit_ridge_probe,
    holm_adjust,
    package_tree,
    sha256_value,
    verify_archive,
    write_json,
)


def _cell() -> PatchCell:
    digest = "a" * 64
    return PatchCell(digest, "donor", "recipient", "resid_post", 2, None, (3,), (4,), digest)


def test_atomic_cell_resume_and_conflict(tmp_path: Path) -> None:
    store = AtomicCellStore(tmp_path)
    assert store.record(_cell(), {"status": "completed", "delta_M": 1.0})
    assert not store.record(_cell(), {"status": "completed", "delta_M": 1.0})
    with pytest.raises(ContractError, match="conflicting"):
        store.record(_cell(), {"status": "completed", "delta_M": 2.0})
    rows = store.merge(tmp_path / "results.jsonl")
    assert len(rows) == 1
    assert rows[0]["cell_id"] == _cell().cell_id


def test_anchor_mapping_and_delay_trace() -> None:
    trials = [{"trial_id": "t1", "prepared_stimulus_id": "p1", "frame_count": 20}]
    prepared = [{
        "prepared_stimulus_id": "p1", "frame_count": 20,
        "prepared_timing": {"old_value_offset_ms": 160, "repair_cue_offset_ms": 240,
                            "new_value_offset_ms": 400, "utterance_end_ms": 800},
        "alignment": {"unit_spans": [{"unit_id": "D1", "offset_ms": 560}]},
    }]
    anchors, trace = anchor_rows(trials, prepared)
    by_name = {row["anchor"]: row["frame"] for row in anchors}
    assert by_name == {"old_end": 2, "cue_end": 3, "new_end": 5, "D1_end": 7, "query_end": 10}
    assert trace[0]["lm_input_offset"] == 1
    assert trace[-1]["frame"] == 19


def test_probe_and_holm_are_deterministic() -> None:
    x = np.asarray([[-2, 0], [-1, 0], [1, 0], [2, 0]], dtype=float)
    labels = ["Boston", "Boston", "Seattle", "Seattle"]
    model = fit_ridge_probe(x, labels, alpha=0.1)
    assert apply_probe(model, x) == labels
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_public_private_packaging_is_separated(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_json(run / "reports/summary.json", {"status": "synthetic"})
    (run / "audio").mkdir(parents=True)
    (run / "audio/trial.wav").write_bytes(b"RIFF-private")
    fake_secret = "sk_" + "a" * 16
    (run / "reports/secret.json").write_text(f'{{"api_key":"{fake_secret}"}}', encoding="utf-8")
    public, private = tmp_path / "public.tar.gz", tmp_path / "private.tar.gz"
    hashes = package_tree(run, public, private)
    verify_archive(public, public=True)
    verify_archive(private, public=False)
    assert set(hashes) == {"public_sha256", "private_sha256"}
    assert all(len(value) == 64 for value in hashes.values())
    with tarfile.open(public, "r:gz") as archive:
        assert "reports/secret.json" not in archive.getnames()
    with tarfile.open(private, "r:gz") as archive:
        assert {"audio/trial.wav", "reports/secret.json"} <= set(archive.getnames())
