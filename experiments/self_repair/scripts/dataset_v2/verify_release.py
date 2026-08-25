#!/usr/bin/env python3
"""Verify a dataset v2 release directory without trusting its manifest alone."""

from __future__ import annotations

import argparse
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

try:  # Direct CLI execution.
    from build_release import (
        CONFIG_FILES,
        DOC_FILES,
        FORMAT_VERSION,
        FULL_AUDIO_RELEASE,
        FULL_EVIDENCE_FILES,
        FULL_PUBLIC_FILES,
        OVERALL_LABELS,
        PUBLIC_EVAL_ROW_FIELDS,
        PUBLIC_RESPONSE_EVIDENCE_FIELDS,
        PUBLIC_RUN_CONTRACT_EVIDENCE_FIELDS,
        RELATION_LABELS,
        REQUIRED_APPROVAL_GATES,
        SCHEMA_FILES,
        SCHEMA_VERSION,
        TEXT_DEVELOPMENT,
        TEXT_MANIFEST_FILES,
        ReleaseError,
        _counts,
        _dataset_config,
        _public_artifact,
        _scan_output_tree,
        _validate_answer_keys,
        _validate_assignments,
        _validate_blueprints,
        _validate_gate_reports,
        _validate_scripts,
        _validate_policy_evidence,
        accepted_audio_id,
        prepared_stimulus_id,
        read_json,
        read_jsonl,
        sha256_file,
        sha256_value,
        validate_downstream_alignment_evidence,
    )
except ImportError:  # pragma: no cover - package-style imports in external callers.
    from .build_release import (
        CONFIG_FILES,
        DOC_FILES,
        FORMAT_VERSION,
        FULL_AUDIO_RELEASE,
        FULL_EVIDENCE_FILES,
        FULL_PUBLIC_FILES,
        OVERALL_LABELS,
        PUBLIC_EVAL_ROW_FIELDS,
        PUBLIC_RESPONSE_EVIDENCE_FIELDS,
        PUBLIC_RUN_CONTRACT_EVIDENCE_FIELDS,
        RELATION_LABELS,
        REQUIRED_APPROVAL_GATES,
        SCHEMA_FILES,
        SCHEMA_VERSION,
        TEXT_DEVELOPMENT,
        TEXT_MANIFEST_FILES,
        ReleaseError,
        _counts,
        _dataset_config,
        _public_artifact,
        _scan_output_tree,
        _validate_answer_keys,
        _validate_assignments,
        _validate_blueprints,
        _validate_gate_reports,
        _validate_scripts,
        _validate_policy_evidence,
        accepted_audio_id,
        prepared_stimulus_id,
        read_json,
        read_jsonl,
        sha256_file,
        sha256_value,
        validate_downstream_alignment_evidence,
    )


CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")


def _allowed_paths(kind: str) -> set[str]:
    paths = {
        "VERSION",
        "RELEASE_MANIFEST.json",
        "CHECKSUMS.sha256",
        *DOC_FILES,
        *CONFIG_FILES,
        *SCHEMA_FILES,
        *TEXT_MANIFEST_FILES.values(),
    }
    if kind == TEXT_DEVELOPMENT:
        paths.add("DEVELOPMENT_SNAPSHOT_NOTICE.md")
    elif kind == FULL_AUDIO_RELEASE:
        paths.add("LICENSE")
        paths.update(FULL_PUBLIC_FILES.values())
        paths.update(FULL_EVIDENCE_FILES.values())
    else:
        raise ReleaseError(f"unknown release kind in manifest: {kind!r}")
    return paths


def _regular_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ReleaseError(f"release contains a symlink: {relative}")
        if path.is_file():
            files.add(relative)
        elif not path.is_dir():
            raise ReleaseError(f"release contains a non-regular object: {relative}")
    return files


def _safe_checksum_path(value: str) -> str:
    if "\\" in value or "\x00" in value:
        raise ReleaseError(f"unsafe checksum path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ReleaseError(f"unsafe checksum path: {value!r}")
    return path.as_posix()


def _checksums(root: Path) -> dict[str, str]:
    path = root / "CHECKSUMS.sha256"
    if not path.is_file() or path.is_symlink():
        raise ReleaseError("missing regular CHECKSUMS.sha256")
    result: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        match = CHECKSUM_RE.fullmatch(line)
        if match is None:
            raise ReleaseError(f"CHECKSUMS.sha256:{line_number}: invalid checksum line")
        digest, relative = match.groups()
        relative = _safe_checksum_path(relative)
        if relative == "CHECKSUMS.sha256":
            raise ReleaseError("CHECKSUMS.sha256 must not checksum itself")
        if relative in result:
            raise ReleaseError(f"duplicate checksum path: {relative}")
        result[relative] = digest
    if list(result) != sorted(result):
        raise ReleaseError("CHECKSUMS.sha256 is not in canonical path order")
    return result


def _verify_file_set_and_hashes(
    root: Path, manifest: Mapping[str, Any], kind: str
) -> tuple[int, set[str]]:
    actual = _regular_files(root)
    allowed = _allowed_paths(kind)
    unexpected = sorted(actual - allowed)
    missing = sorted(allowed - actual)
    if unexpected or missing:
        raise ReleaseError(f"release file-set mismatch; missing={missing}, unexpected={unexpected}")

    checksums = _checksums(root)
    expected_checksum_paths = actual - {"CHECKSUMS.sha256"}
    if set(checksums) != expected_checksum_paths:
        raise ReleaseError("checksum paths do not match the exact release file set")
    for relative, expected in checksums.items():
        observed = sha256_file(root / relative)
        if observed != expected:
            raise ReleaseError(f"checksum mismatch: {relative}")

    payload = manifest.get("payload_files")
    if not isinstance(payload, Mapping):
        raise ReleaseError("RELEASE_MANIFEST payload_files is missing")
    expected_payload = actual - {"CHECKSUMS.sha256", "RELEASE_MANIFEST.json"}
    if set(payload) != expected_payload:
        raise ReleaseError("manifest payload_files does not match the exact payload")
    for relative in sorted(expected_payload):
        metadata = payload[relative]
        if not isinstance(metadata, Mapping):
            raise ReleaseError(f"payload metadata is invalid: {relative}")
        path = root / relative
        if metadata.get("sha256") != sha256_file(path):
            raise ReleaseError(f"manifest payload hash mismatch: {relative}")
        if metadata.get("size_bytes") != path.stat().st_size:
            raise ReleaseError(f"manifest payload size mismatch: {relative}")
    return len(actual), actual


def _validate_public_eval(
    rows: Sequence[dict[str, Any]],
    accepted_by_id: Mapping[str, Mapping[str, Any]],
    prepared_by_accepted: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    expected: int,
) -> tuple[set[str], str, str]:
    if len(rows) != expected:
        raise ReleaseError(f"public eval count must be {expected}, found {len(rows)}")
    trial_ids: set[str] = set()
    cells: set[tuple[str, int]] = set()
    seeds = tuple(config["evaluation"]["generation_seeds"])
    run_ids: set[str] = set()
    matrix_hashes: set[str] = set()
    execution_hashes: set[str] = set()
    input_hash_by_audio: dict[str, str] = {}
    capture_hash_by_audio: dict[str, str] = {}
    accepted_ids = set(accepted_by_id)
    for index, row in enumerate(rows):
        if set(row) != PUBLIC_EVAL_ROW_FIELDS:
            raise ReleaseError(f"public eval row {index}: fields are invalid")
        if "response" in row or "stream_events" in row:
            raise ReleaseError(f"public eval row {index} leaks model response payload")
        trial_id = row.get("eval_trial_id")
        accepted_id = row.get("accepted_audio_id")
        seed = row.get("generation_seed")
        if not isinstance(trial_id, str) or not trial_id or trial_id in trial_ids:
            raise ReleaseError(f"public eval row {index}: duplicate/invalid eval_trial_id")
        trial_ids.add(trial_id)
        if accepted_id not in accepted_ids or seed not in seeds:
            raise ReleaseError(f"public eval row {index}: invalid audio/seed cell")
        cell = (str(accepted_id), int(seed))
        if cell in cells:
            raise ReleaseError(f"public eval row {index}: duplicate audio/seed cell")
        cells.add(cell)
        accepted_row = accepted_by_id[str(accepted_id)]
        prepared_row = prepared_by_accepted.get(str(accepted_id))
        if (
            prepared_row is None
            or row.get("condition") != accepted_row.get("condition")
            or row.get("condition") != prepared_row.get("condition")
            or row.get("prepared_stimulus_id") != prepared_row.get("prepared_stimulus_id")
        ):
            raise ReleaseError(f"public eval row {index}: prepared input lineage mismatch")
        contract_evidence = row.get("run_contract_evidence")
        if (
            not isinstance(contract_evidence, Mapping)
            or set(contract_evidence) != PUBLIC_RUN_CONTRACT_EVIDENCE_FIELDS
        ):
            raise ReleaseError(f"public eval row {index}: run contract evidence is invalid")
        for field in PUBLIC_RUN_CONTRACT_EVIDENCE_FIELDS:
            value = contract_evidence.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ReleaseError(f"public eval row {index}: invalid {field}")
        matrix_hashes.add(str(contract_evidence["matrix_contract_sha256"]))
        execution_hashes.add(str(contract_evidence["execution_contract_sha256"]))
        prior_input_hash = input_hash_by_audio.setdefault(
            str(accepted_id), str(contract_evidence["input_stimulus_sha256"])
        )
        prior_capture_hash = capture_hash_by_audio.setdefault(
            str(accepted_id), str(contract_evidence["capture_contract_sha256"])
        )
        if (
            prior_input_hash != contract_evidence["input_stimulus_sha256"]
            or prior_capture_hash != contract_evidence["capture_contract_sha256"]
        ):
            raise ReleaseError(f"public eval row {index}: run contract changes across seeds")
        if row.get("response_status") != "completed":
            raise ReleaseError(f"public eval row {index}: response completion status is missing")
        evidence = row.get("response_evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != PUBLIC_RESPONSE_EVIDENCE_FIELDS:
            raise ReleaseError(f"public eval row {index}: response evidence fields are invalid")
        for field in (
            "audio_sha256",
            "transcript_sha256",
            "stream_events_sha256",
            "evidence_sha256",
            "runner_source_sha256",
            "effective_generation_config_sha256",
        ):
            value = evidence.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ReleaseError(f"public eval row {index}: invalid response evidence {field}")
        duration = evidence.get("audio_duration_ms")
        event_count = evidence.get("stream_event_count")
        first_ms = evidence.get("first_stream_event_ms")
        last_ms = evidence.get("last_stream_event_ms")
        audio_sample_rate = evidence.get("audio_sample_rate")
        audio_channels = evidence.get("audio_channels")
        audio_sample_width = evidence.get("audio_sample_width_bytes")
        integer_fields = (
            "fed_sample_count",
            "fed_frame_count",
            "output_sample_count",
            "output_frame_count",
            "appended_zero_sample_count",
        )
        integer_values = {field: evidence.get(field) for field in integer_fields}
        mimi_frame_samples = evidence.get("mimi_frame_samples")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or duration <= 0
            or not isinstance(event_count, int)
            or isinstance(event_count, bool)
            or event_count < 1
            or not isinstance(audio_sample_rate, int)
            or isinstance(audio_sample_rate, bool)
            or audio_sample_rate <= 0
            or audio_channels != 1
            or audio_sample_width != 2
            or evidence.get("timebase") != "prepared_stream_relative"
            or evidence.get("stream_origin_ms") != 0
            or evidence.get("coverage_complete") is not True
            or evidence.get("eos_reached") is not False
            or not isinstance(mimi_frame_samples, int)
            or isinstance(mimi_frame_samples, bool)
            or mimi_frame_samples <= 0
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in integer_values.values()
            )
            or not isinstance(first_ms, (int, float))
            or isinstance(first_ms, bool)
            or not isinstance(last_ms, (int, float))
            or isinstance(last_ms, bool)
            or not math.isfinite(float(first_ms))
            or not math.isfinite(float(last_ms))
            or first_ms < 0
            or last_ms < first_ms
        ):
            raise ReleaseError(f"public eval row {index}: invalid response evidence timeline")
        fed_samples = int(integer_values["fed_sample_count"])
        fed_frames = int(integer_values["fed_frame_count"])
        output_samples = int(integer_values["output_sample_count"])
        output_frames = int(integer_values["output_frame_count"])
        if (
            fed_samples != output_samples
            or fed_frames != output_frames
            or fed_samples != fed_frames * mimi_frame_samples
            or output_samples != output_frames * mimi_frame_samples
            or event_count != fed_frames
            or abs(float(duration) - output_samples * 1000.0 / audio_sample_rate) > 0.05
        ):
            raise ReleaseError(f"public eval row {index}: response coverage evidence is inconsistent")
        for field in (
            "primary_window_start_ms",
            "requested_target_end_ms",
            "actual_target_end_ms",
        ):
            value = evidence.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ReleaseError(f"public eval row {index}: invalid response timing evidence")
        if (
            float(evidence["primary_window_start_ms"])
            > float(evidence["requested_target_end_ms"])
            or float(evidence["requested_target_end_ms"])
            > float(evidence["actual_target_end_ms"])
            or float(last_ms) > float(evidence["actual_target_end_ms"])
        ):
            raise ReleaseError(f"public eval row {index}: response timing order is invalid")
        if evidence["runner_source_sha256"] != contract_evidence["runner_source_sha256"]:
            raise ReleaseError(f"public eval row {index}: runner source binding mismatch")
        run_ids.add(str(row.get("eval_run_id", "")))
    expected_cells = {(accepted_id, seed) for accepted_id in accepted_ids for seed in seeds}
    if (
        cells != expected_cells
        or len(run_ids) != 1
        or "" in run_ids
        or len(matrix_hashes) != 1
        or len(execution_hashes) != 1
        or set(input_hash_by_audio) != accepted_ids
        or set(capture_hash_by_audio) != accepted_ids
    ):
        raise ReleaseError("public eval matrix or run identity is incomplete")
    return trial_ids, next(iter(matrix_hashes)), next(iter(execution_hashes))


def _validate_public_annotations(rows: Sequence[dict[str, Any]], trial_ids: set[str]) -> None:
    if len(rows) != len(trial_ids):
        raise ReleaseError("resolved public annotation count does not match eval trials")
    observed: set[str] = set()
    for index, row in enumerate(rows):
        forbidden = {"blind_id", "annotator_id", "primary_annotator_ids", "adjudicator_id"}
        if set(row) & forbidden:
            raise ReleaseError(f"public annotation row {index} leaks private annotation identity")
        trial_id = row.get("eval_trial_id")
        if trial_id not in trial_ids or trial_id in observed:
            raise ReleaseError(f"public annotation row {index}: invalid/duplicate eval_trial_id")
        observed.add(str(trial_id))
        if row.get("resolution_method") not in (
            "independent_agreement",
            "adjudicated_disagreement",
        ):
            raise ReleaseError(f"public annotation row {index}: invalid resolution method")
        overall = row.get("overall_label")
        if overall not in OVERALL_LABELS:
            raise ReleaseError(f"public annotation row {index}: invalid overall label")
        relations = row.get("relation_labels")
        if not isinstance(relations, Mapping) or set(relations) != {"D1", "D2", "D3"}:
            raise ReleaseError(f"public annotation row {index}: incomplete relation labels")
        if any(value not in RELATION_LABELS for value in relations.values()):
            raise ReleaseError(f"public annotation row {index}: invalid relation label value")
        expected_final = overall in {"target_only", "recovered"}
        if row.get("final_target_correct") is not expected_final:
            raise ReleaseError(
                f"public annotation row {index}: final target flag contradicts overall label"
            )
        expected_stale = any(value in {"old_bound", "both"} for value in relations.values())
        if row.get("stale_state_error") is not expected_stale:
            raise ReleaseError(
                f"public annotation row {index}: stale flag contradicts relation labels"
            )
        early = row.get("assistant_started_before_repair")
        if early is not None and not isinstance(early, bool):
            raise ReleaseError(f"public annotation row {index}: invalid early-response flag")
    if observed != trial_ids:
        raise ReleaseError("resolved public annotations do not cover every eval trial")


def _verify_audio_inventory(
    accepted: Sequence[dict[str, Any]],
    prepared: Sequence[dict[str, Any]],
    inventory: Sequence[dict[str, Any]],
) -> None:
    expected = {
        (
            "accepted_utterance",
            str(row["accepted_audio_id"]),
            str(row["accepted_utterance"]["uri"]),
            str(row["accepted_utterance"]["sha256"]),
        )
        for row in accepted
    }
    expected.update(
        {
            (
                "prepared_stimulus",
                str(row["prepared_stimulus_id"]),
                str(row["prepared_stimulus"]["uri"]),
                str(row["prepared_stimulus"]["sha256"]),
            )
            for row in prepared
        }
    )
    observed = {
        (
            str(row.get("artifact_role", "")),
            str(row.get("owner_id", "")),
            str(row.get("uri", "")),
            str(row.get("sha256", "")),
        )
        for row in inventory
    }
    if len(inventory) != len(observed) or observed != expected:
        raise ReleaseError("audio checksum inventory does not exactly match public audio manifests")


def _verify_semantics(root: Path, manifest: Mapping[str, Any], kind: str) -> None:
    config = _dataset_config(root)
    counts = _counts(config)
    for key, expected in counts.items():
        if manifest.get("counts", {}).get(key) != expected:
            raise ReleaseError(f"release manifest count mismatch: {key}")
    blueprints = read_jsonl(root / TEXT_MANIFEST_FILES["blueprints"])
    scripts = read_jsonl(root / TEXT_MANIFEST_FILES["scripts"])
    answer_keys = read_jsonl(root / TEXT_MANIFEST_FILES["answer_keys"])
    folds = read_jsonl(root / TEXT_MANIFEST_FILES["analysis_folds"])
    bundles = read_jsonl(root / TEXT_MANIFEST_FILES["speaker_bundles"])
    targets = read_jsonl(root / TEXT_MANIFEST_FILES["rendition_targets"])
    recording = read_jsonl(root / TEXT_MANIFEST_FILES["recording_order"])
    _validate_blueprints(blueprints, config, counts)
    _validate_scripts(scripts, blueprints, config, counts)
    _validate_answer_keys(answer_keys, scripts, counts)
    _validate_assignments(folds, bundles, targets, recording, scripts, config, counts)

    if kind == TEXT_DEVELOPMENT:
        if manifest.get("release_eligible") is not False:
            raise ReleaseError("text development snapshot is incorrectly marked release eligible")
        if manifest.get("status") != "development_snapshot_not_public_release":
            raise ReleaseError("text development snapshot status is not explicit")
        if manifest.get("approval") is not None or manifest.get("audio_delivery") is not None:
            raise ReleaseError("text development snapshot asserts full-release metadata")
        return

    if manifest.get("release_eligible") is not True or manifest.get("status") != "approved_full_audio_release":
        raise ReleaseError("full release is not explicitly marked approved/release eligible")
    approval = manifest.get("approval")
    if not isinstance(approval, Mapping) or approval.get("status") != "approved":
        raise ReleaseError("full release approval summary is missing")
    gates = approval.get("gates")
    if not isinstance(gates, Mapping) or any(
        gates.get(gate) != "passed" for gate in REQUIRED_APPROVAL_GATES
    ):
        raise ReleaseError("one or more full-release approval gates are not passed")
    if approval.get("approved_source_hashes_sha256") != manifest.get("source_lineage_sha256"):
        raise ReleaseError("approval/source-lineage binding is inconsistent")
    privacy = manifest.get("privacy")
    if not isinstance(privacy, Mapping) or any(value is not False for value in privacy.values()):
        raise ReleaseError("full release privacy exclusions are incomplete")

    evidence_paths = {
        logical: root / relative for logical, relative in FULL_EVIDENCE_FILES.items()
    }
    evidence, selection_policy_hash, timing_policy_hash = _validate_policy_evidence(
        evidence_paths, config
    )
    source_lineage = manifest.get("source_lineage")
    if not isinstance(source_lineage, Mapping):
        raise ReleaseError("full release source lineage is missing")
    lineage_hashes = {
        str(logical): entry.get("sha256")
        for logical, entry in source_lineage.items()
        if isinstance(entry, Mapping)
    }
    if len(lineage_hashes) != len(source_lineage) or manifest.get(
        "source_lineage_sha256"
    ) != sha256_value(lineage_hashes):
        raise ReleaseError("full release source-lineage digest is inconsistent")

    def source_digest(logical: str) -> str:
        entry = source_lineage.get(logical)
        digest = entry.get("sha256") if isinstance(entry, Mapping) else None
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ReleaseError(f"full release source lineage is missing {logical}")
        return digest

    for logical, relative in FULL_EVIDENCE_FILES.items():
        if source_digest(logical) != sha256_file(root / relative):
            raise ReleaseError(f"packaged release evidence hash mismatch: {logical}")

    accepted = read_jsonl(root / FULL_PUBLIC_FILES["accepted_audio"])
    prepared = read_jsonl(root / FULL_PUBLIC_FILES["prepared_stimuli"])
    target_by_id = {str(row["rendition_target_id"]): row for row in targets}
    script_by_id = {str(row["script_id"]): row for row in scripts}
    target_ids = set(target_by_id)
    accepted_ids = {str(row.get("accepted_audio_id", "")) for row in accepted}
    if len(accepted) != counts["rendition_targets"] or len(accepted_ids) != len(accepted):
        raise ReleaseError("public accepted-audio count/identity is invalid")
    if {str(row.get("rendition_target_id", "")) for row in accepted} != target_ids:
        raise ReleaseError("public accepted audio does not match rendition targets")
    accepted_by_id: dict[str, dict[str, Any]] = {}
    accepted_artifact_uris: set[str] = set()
    accepted_artifact_hashes: set[str] = set()
    for row in accepted:
        accepted_id = str(row["accepted_audio_id"])
        target_id = str(row["rendition_target_id"])
        if row.get("schema_version") != SCHEMA_VERSION or row.get("lifecycle_status") != "accepted":
            raise ReleaseError(f"public accepted audio {accepted_id}: lifecycle/schema mismatch")
        if accepted_id != accepted_audio_id(target_id):
            raise ReleaseError(f"public accepted audio {accepted_id}: non-canonical ID")
        target = target_by_id[target_id]
        for field in (
            "script_id",
            "text_bundle_id",
            "matched_audio_bundle_id",
            "scenario_id",
            "direction_id",
            "condition",
            "source_track_id",
            "speaker_id",
            "voice",
            "analysis_fold",
            "inferential_role",
        ):
            if row.get(field) != target.get(field):
                raise ReleaseError(f"public accepted audio {accepted_id}: {field} mismatch")
        selection = row.get("selection")
        if not isinstance(selection, Mapping) or selection.get("policy_hash") != selection_policy_hash:
            raise ReleaseError(f"public accepted audio {accepted_id}: selection policy mismatch")
        selected_candidate_id = row.get("selected_candidate_id")
        if (
            not isinstance(selected_candidate_id, str)
            or not re.fullmatch(re.escape(target_id) + r"__cand\d{2}", selected_candidate_id)
            or selection.get("selected_candidate_id") != selected_candidate_id
        ):
            raise ReleaseError(f"public accepted audio {accepted_id}: candidate lineage mismatch")
        alignment_gate = evidence["selection_policy"].get("alignment_gate")
        if selection.get("alignment_gate_hash") != sha256_value(alignment_gate):
            raise ReleaseError(f"public accepted audio {accepted_id}: alignment gate hash mismatch")
        qc = row.get("qc")
        if not isinstance(qc, Mapping) or qc.get("automatic_status", qc.get("status")) != "passed":
            raise ReleaseError(f"public accepted audio {accepted_id}: automatic QC gate failed")
        if qc.get("outcome_blind") is not True:
            raise ReleaseError(f"public accepted audio {accepted_id}: selected QC is not outcome-blind")
        selected_evidence = row.get("selected_evidence")
        canonical_digest = selection.get("selected_canonical_sha256")
        expected_selected_evidence = {
            "candidate_id": selected_candidate_id,
            "canonical_audio_sha256": canonical_digest,
            "timing_sha256": sha256_value(row.get("timing")),
            "alignment_sha256": sha256_value(row.get("alignment")),
            "qc_sha256": sha256_value(qc),
        }
        if selected_evidence != expected_selected_evidence:
            raise ReleaseError(f"public accepted audio {accepted_id}: selected evidence hashes mismatch")
        alignment_source = dict(row)
        alignment_source["canonical_candidate"] = {"sha256": canonical_digest}
        alignment_errors = validate_downstream_alignment_evidence(
            alignment_source,
            script_by_id[str(row["script_id"])],
            dict(alignment_gate),
            actual_canonical_audio_sha256=str(canonical_digest),
        )
        if alignment_errors:
            raise ReleaseError(
                f"public accepted audio {accepted_id}: alignment evidence is invalid: "
                + "; ".join(alignment_errors)
            )
        license_value = row.get("license")
        if (
            not isinstance(license_value, Mapping)
            or license_value.get("redistribution_allowed") is not True
            or not license_value.get("identifier")
            or not license_value.get("scope")
        ):
            raise ReleaseError(f"public accepted audio {accepted_id}: license gate failed")
        artifact = row.get("accepted_utterance")
        artifact = _public_artifact(artifact, f"public accepted audio {accepted_id}")
        if artifact.get("source_canonical_sha256") != canonical_digest:
            raise ReleaseError(f"public accepted audio {accepted_id}: source canonical hash mismatch")
        artifact_uri = str(artifact["uri"])
        artifact_hash = str(artifact["sha256"])
        if artifact_uri in accepted_artifact_uris or artifact_hash in accepted_artifact_hashes:
            raise ReleaseError("public accepted audio artifacts are not one-to-one")
        accepted_artifact_uris.add(artifact_uri)
        accepted_artifact_hashes.add(artifact_hash)
        accepted_by_id[accepted_id] = row
    prepared_ids = {str(row.get("accepted_audio_id", "")) for row in prepared}
    if len(prepared) != counts["rendition_targets"] or prepared_ids != accepted_ids:
        raise ReleaseError("public prepared stimuli do not match accepted audio")
    prepared_artifact_uris: set[str] = set()
    prepared_artifact_hashes: set[str] = set()
    for row in prepared:
        prepared_id = str(row.get("prepared_stimulus_id", ""))
        accepted_id = str(row.get("accepted_audio_id", ""))
        if row.get("schema_version") != SCHEMA_VERSION or row.get("lifecycle_status") != "prepared":
            raise ReleaseError(f"public prepared stimulus {prepared_id}: lifecycle/schema mismatch")
        preparation_hash = row.get("preparation_hash")
        if not isinstance(preparation_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", preparation_hash
        ):
            raise ReleaseError(f"public prepared stimulus {prepared_id}: invalid preparation hash")
        if prepared_id != prepared_stimulus_id(accepted_id, preparation_hash):
            raise ReleaseError(f"public prepared stimulus {prepared_id}: non-canonical ID")
        accepted_row = accepted_by_id[accepted_id]
        for field in (
            "rendition_target_id",
            "script_id",
            "text_bundle_id",
            "matched_audio_bundle_id",
            "scenario_id",
            "direction_id",
            "condition",
            "source_track_id",
            "speaker_id",
            "voice",
            "analysis_fold",
            "inferential_role",
        ):
            if row.get(field) != accepted_row.get(field):
                raise ReleaseError(f"public prepared stimulus {prepared_id}: {field} mismatch")
        artifact = row.get("prepared_stimulus")
        artifact = _public_artifact(artifact, f"public prepared stimulus {prepared_id}")
        artifact_uri = str(artifact["uri"])
        artifact_hash = str(artifact["sha256"])
        if artifact_uri in prepared_artifact_uris or artifact_hash in prepared_artifact_hashes:
            raise ReleaseError("public prepared artifacts are not one-to-one")
        prepared_artifact_uris.add(artifact_uri)
        prepared_artifact_hashes.add(artifact_hash)
    for row in accepted:
        artifact = row.get("accepted_utterance")
        if not isinstance(artifact, Mapping) or PurePosixPath(str(artifact.get("uri", "/"))).is_absolute():
            raise ReleaseError("public accepted manifest contains an absolute artifact URI")
    for row in prepared:
        artifact = row.get("prepared_stimulus")
        if not isinstance(artifact, Mapping) or PurePosixPath(str(artifact.get("uri", "/"))).is_absolute():
            raise ReleaseError("public prepared manifest contains an absolute artifact URI")
    inventory = read_jsonl(root / FULL_PUBLIC_FILES["audio_inventory"])
    _verify_audio_inventory(accepted, prepared, inventory)
    trials = read_jsonl(root / FULL_PUBLIC_FILES["eval_trials"])
    prepared_by_accepted = {
        str(row["accepted_audio_id"]): row for row in prepared
    }
    trial_ids, matrix_contract_hash, execution_contract_hash = _validate_public_eval(
        trials,
        accepted_by_id,
        prepared_by_accepted,
        config,
        counts["eval_trials"],
    )
    annotations = read_jsonl(root / FULL_PUBLIC_FILES["annotations"])
    _validate_public_annotations(annotations, trial_ids)
    run_ids = {str(row.get("eval_run_id", "")) for row in trials}
    if len(run_ids) != 1 or "" in run_ids:
        raise ReleaseError("public eval run identity is incomplete")
    evaluation_gate = manifest.get("gate_results", {}).get("evaluation", {})
    if (
        evaluation_gate.get("matrix_contract_sha256") != matrix_contract_hash
        or evaluation_gate.get("execution_contract_sha256") != execution_contract_hash
        or evaluation_gate.get("responses_completed") != len(trials)
    ):
        raise ReleaseError("public eval contract evidence disagrees with release manifest")
    evidence_gate = _validate_gate_reports(
        evidence,
        config=config,
        selection_policy_hash=selection_policy_hash,
        timing_policy_hash=timing_policy_hash,
        accepted_manifest_sha256=source_digest("accepted_audio"),
        eval_manifest_sha256=source_digest("eval_trials"),
        annotation_manifest_sha256=source_digest("annotations"),
        accepted_count=len(accepted),
        eval_trial_count=len(trials),
        eval_run_id=next(iter(run_ids)),
    )
    manifest_evidence_gate = manifest.get("gate_results", {}).get("release_evidence")
    if manifest_evidence_gate != evidence_gate:
        raise ReleaseError("release evidence gate summary is inconsistent")
    annotation_gate = manifest.get("gate_results", {}).get("annotations", {})
    if annotation_gate.get("primary_annotations") != counts["primary_annotations"]:
        raise ReleaseError("full release does not attest the exact primary-annotation count")
    if annotation_gate.get("resolved_annotations") != counts["eval_trials"]:
        raise ReleaseError("full release resolved-annotation gate is incomplete")


def verify_release(release_dir: Path) -> dict[str, Any]:
    """Raise ``ReleaseError`` on any mismatch and return a concise success report."""

    root = release_dir.resolve()
    if release_dir.is_symlink() or not root.is_dir():
        raise ReleaseError(f"release path must be a regular directory: {release_dir}")
    manifest_path = root / "RELEASE_MANIFEST.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ReleaseError("RELEASE_MANIFEST must be an object")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ReleaseError("unknown release manifest format_version")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseError("release schema_version mismatch")
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    if manifest.get("dataset_version") != version or version != SCHEMA_VERSION:
        raise ReleaseError("VERSION and release manifest disagree")
    kind = str(manifest.get("release_kind", ""))
    file_count, files = _verify_file_set_and_hashes(root, manifest, kind)
    _scan_output_tree(root)
    _verify_semantics(root, manifest, kind)
    config = _dataset_config(root)
    config_hashes = manifest.get("config_hashes")
    if not isinstance(config_hashes, Mapping):
        raise ReleaseError("release config hashes are missing")
    if config_hashes.get("dataset_config_file_sha256") != sha256_file(
        root / "config/dataset.yaml"
    ):
        raise ReleaseError("dataset config file hash mismatch")
    if config_hashes.get("dataset_config_canonical_sha256") != sha256_value(config):
        raise ReleaseError("canonical dataset config hash mismatch")
    return {
        "status": "passed",
        "release_kind": kind,
        "dataset_version": version,
        "git_commit": manifest.get("git_commit"),
        "file_count": file_count,
        "checksum_count": len(files) - 1,
        "release_eligible": manifest.get("release_eligible"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_release(args.release_dir)
    print(
        f"Verified {report['release_kind']} dataset {report['dataset_version']} "
        f"({report['file_count']} files)"
    )


if __name__ == "__main__":
    main()
