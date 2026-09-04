from __future__ import annotations

from pathlib import Path

import pytest

from experiments.self_repair.mechanistic.blinding import BlindAssignmentStore
from experiments.self_repair.mechanistic.core import ContractError, read_json, write_json


def test_blinding_is_opaque_and_resume_stable(tmp_path: Path) -> None:
    run_id = "a" * 64
    store = BlindAssignmentStore(tmp_path, run_identity_sha256=run_id)
    first = store.assign(cell_key={"trial_id": "t1", "seed": 17}, arms=["baseline", "patched"])
    resumed = BlindAssignmentStore(tmp_path, run_identity_sha256=run_id).assign(
        cell_key={"trial_id": "t1", "seed": 17}, arms=["baseline", "patched"])
    assert resumed == first
    assert set(first.arm_to_label.values()) == {"arm_01", "arm_02"}
    assert all("baseline" not in stem and "patched" not in stem for stem in first.arm_to_audio_stem.values())
    assert len(set(first.arm_to_audio_stem.values())) == 2


def test_blinding_rejects_identity_or_arm_drift(tmp_path: Path) -> None:
    store = BlindAssignmentStore(tmp_path, run_identity_sha256="a" * 64)
    store.assign(cell_key={"trial": "t"}, arms=["baseline", "patched"])
    with pytest.raises(ContractError, match="another run"):
        BlindAssignmentStore(tmp_path, run_identity_sha256="b" * 64)
    with pytest.raises(ContractError, match="arm set differs"):
        store.assign(cell_key={"trial": "t"}, arms=["baseline", "other"])


def test_blinding_rejects_corrupt_private_map(tmp_path: Path) -> None:
    write_json(tmp_path / "private_blind_map.json", {
        "run_identity_sha256": "a" * 64, "secret_hex": "bad", "assignments": {},
    })
    with pytest.raises(ContractError, match="malformed"):
        BlindAssignmentStore(tmp_path, run_identity_sha256="a" * 64)
