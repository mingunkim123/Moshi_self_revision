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
    frame_for_ms,
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
    assert store.contains(_cell())
    assert store.get(_cell())["status"] == "completed"
    assert not store.record(_cell(), {"status": "completed", "delta_M": 1.0})
    with pytest.raises(ContractError, match="conflicting"):
        store.record(_cell(), {"status": "completed", "delta_M": 2.0})
    rows = store.merge(tmp_path / "results.jsonl")
    assert len(rows) == 1
    assert rows[0]["cell_id"] == _cell().cell_id


def test_atomic_cell_resume_rejects_tampered_identity(tmp_path: Path) -> None:
    store = AtomicCellStore(tmp_path)
    cell = _cell()
    store.record(cell, {"status": "completed", "delta_M": 1.0})
    path = store.cells / f"{cell.cell_id}.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["recipient_trial_id"] = "tampered"
    write_json(path, row)
    with pytest.raises(ContractError, match="identity mismatch"):
        store.contains(cell)


def test_failed_attempt_is_preserved_but_does_not_block_same_identity_retry(
    tmp_path: Path,
) -> None:
    store = AtomicCellStore(tmp_path)
    cell = _cell()
    assert store.record(cell, {
        "status": "failed", "failure_type": "OutOfMemoryError",
        "failure_message": "fixture OOM",
    })
    assert not store.contains(cell)
    assert store.get(cell) is None
    assert len(store.failure_rows()) == 1
    unresolved = store.merge(tmp_path / "results.before-retry.jsonl")
    assert [row["status"] for row in unresolved] == ["failed"]

    assert store.record(cell, {"status": "completed", "delta_M": 1.0})
    assert store.contains(cell)
    final = store.merge(tmp_path / "results.after-retry.jsonl")
    assert [row["status"] for row in final] == ["completed"]
    failures = store.merge_failures(tmp_path / "failures.jsonl")
    assert len(failures) == 1
    assert failures[0]["failure_type"] == "OutOfMemoryError"


def test_atomic_store_rejects_invalid_status(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="status"):
        AtomicCellStore(tmp_path).record(_cell(), {"status": "partial"})


@pytest.mark.parametrize(
    ("offset_ms", "expected_frame"),
    [
        (0.001, 0),
        (79.999, 0),
        (80, 0),
        (80.001, 1),
        (159.999, 1),
        (160, 1),
        (160.001, 2),
    ],
)
def test_semantic_end_anchor_uses_last_overlapping_frame(
    offset_ms: float, expected_frame: int,
) -> None:
    assert frame_for_ms(offset_ms) == expected_frame


@pytest.mark.parametrize("invalid", [-0.001, float("nan"), float("inf")])
def test_semantic_end_anchor_rejects_invalid_times(invalid: float) -> None:
    with pytest.raises(ContractError):
        frame_for_ms(invalid)


def test_anchor_mapping_and_delay_trace() -> None:
    trials = [{"trial_id": "t1", "prepared_stimulus_id": "p1", "frame_count": 20}]
    prepared = [{
        "prepared_stimulus_id": "p1", "frame_count": 20,
        "prepared_timing": {"old_value_offset_ms": 160, "repair_cue_offset_ms": 240,
                            "new_value_offset_ms": 400, "utterance_end_ms": 800},
        "preparation": {"prefix_ms_actual": 480},
        "alignment": {"unit_spans": [{"unit_id": "D1", "offset_ms": 560}]},
    }]
    anchors, trace = anchor_rows(trials, prepared)
    by_name = {row["anchor"]: row["frame"] for row in anchors}
    assert by_name == {"old_end": 1, "cue_end": 2, "new_end": 4, "D1_end": 12,
                       "query_end": 9}
    by_anchor = {row["anchor"]: row for row in anchors}
    assert by_anchor["old_end"]["time_ms"] == 160
    assert by_anchor["old_end"]["timebase"] == "prepared_stream_relative"
    assert by_anchor["D1_end"]["time_ms"] == 1040
    assert by_anchor["D1_end"]["timebase"] == "alignment_content_relative_plus_prefix"

    assert len(trace) == 21
    prime = trace[0]
    assert prime["trace_kind"] == "lm_prime"
    assert prime["submitted_audio_frame"] == 0
    assert prime["consumed_audio_frame"] is None
    assert prime["lm_step"] == 0
    assert prime["hidden_absolute_position"] == 0
    assert prime["max_lm_delay"] == 1
    assert all(slot["uses_initial_token"] for slot in prime["delay_slots"])

    first = trace[1]
    assert first["consumed_audio_frame"] == 0
    assert first["lm_step"] == first["hidden_absolute_position"] == 1
    assert first["delay_slots"][0] == {
        "delay_slot": 0, "stream": "user_audio", "user_codebooks": [0],
        "source_audio_frame": 0, "uses_initial_token": False,
    }
    assert first["delay_slots"][1]["source_audio_frame"] is None
    assert first["delay_slots"][1]["uses_initial_token"] is True

    second = trace[2]
    assert second["consumed_audio_frame"] == 1
    assert second["lm_step"] == second["hidden_absolute_position"] == 2
    assert second["delay_slots"][0]["source_audio_frame"] == 1
    assert second["delay_slots"][1]["source_audio_frame"] == 0
    assert trace[-1]["consumed_audio_frame"] == 19
    assert trace[-1]["lm_step"] == trace[-1]["hidden_absolute_position"] == 20


def test_anchor_boundaries_are_not_clipped() -> None:
    trials = [{"trial_id": "t1", "prepared_stimulus_id": "p1"}]

    def prepared(offset_ms: float) -> list[dict[str, object]]:
        return [{
            "prepared_stimulus_id": "p1",
            "frame_count": 2,
            "prepared_timing": {"utterance_end_ms": offset_ms},
        }]

    anchors, _ = anchor_rows(trials, prepared(160))
    assert anchors[0]["frame"] == 1
    assert anchors[0]["plus_one_frame"] is None

    with pytest.raises(ContractError, match="outside the encoded sequence"):
        anchor_rows(trials, prepared(160.001))
    with pytest.raises(ContractError, match="outside the encoded sequence"):
        anchor_rows(trials, prepared(0))


def test_alignment_unit_span_requires_prefix_metadata() -> None:
    trials = [{"trial_id": "t1", "prepared_stimulus_id": "p1"}]
    prepared = [{
        "prepared_stimulus_id": "p1",
        "frame_count": 20,
        "prepared_timing": {"utterance_end_ms": 800},
        "alignment": {"unit_spans": [{"unit_id": "D1", "offset_ms": 560}]},
    }]
    with pytest.raises(ContractError, match="prefix_ms_actual is required"):
        anchor_rows(trials, prepared)


def test_query_end_fallback_maps_to_last_encoded_frame() -> None:
    trials = [{"trial_id": "t1", "prepared_stimulus_id": "p1"}]
    prepared = [{
        "prepared_stimulus_id": "p1",
        "frame_count": 2,
        "prepared_timing": {"old_value_offset_ms": 80},
    }]
    anchors, _ = anchor_rows(trials, prepared)
    by_name = {row["anchor"]: row for row in anchors}
    assert by_name["old_end"]["frame"] == 0
    assert by_name["old_end"]["minus_one_frame"] is None
    assert by_name["query_end"]["frame"] == 1
    assert by_name["query_end"]["timebase"] == "encoded_stream_end"


def test_unprepared_content_timing_is_not_accepted_as_stream_time() -> None:
    trials = [{"trial_id": "t1", "prepared_stimulus_id": "p1"}]
    prepared = [{
        "prepared_stimulus_id": "p1",
        "frame_count": 2,
        "timing": {"utterance_end_ms": 80},
    }]
    with pytest.raises(ContractError, match="missing prepared_timing"):
        anchor_rows(trials, prepared)


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


def test_packaging_rejects_outputs_inside_run_root(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_json(run / "reports/summary.json", {"status": "synthetic"})
    with pytest.raises(ContractError, match="outside run_root"):
        package_tree(run, run / "public.tar.gz", tmp_path / "private.tar.gz")


def test_public_archive_verifier_rechecks_member_content(tmp_path: Path) -> None:
    archive_path = tmp_path / "tampered-public.tar.gz"
    source = tmp_path / "summary.json"
    source.write_text('{"api_key":"sk_' + "a" * 16 + '"}', encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname="reports/summary.json")
    with pytest.raises(ContractError, match="sensitive content"):
        verify_archive(archive_path, public=True)
