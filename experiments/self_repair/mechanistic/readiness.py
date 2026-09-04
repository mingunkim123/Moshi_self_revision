"""Fail-closed workload estimates and paid-run readiness decisions.

This module is intentionally independent from the command-line harness.  A caller
can build a scan specification, inspect the exact arithmetic returned by
``estimate_workload``, and only start a paid scan when ``assess_readiness`` says
``GO``.  Byte counts are reservations, not compression predictions: compressed
NPZ/tar output is never used to justify a smaller volume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


class ReadinessError(ValueError):
    """Raised when a manifest or scan specification is not auditable."""


AUTHORIZATION_TYPE = "mechanistic_paid_scan_authorization"
AUTHORIZATION_STAGES = (
    "static_plan",
    "budget",
    "evidence_binding",
    "model_contract",
    "open_loop",
    "conversation_canary",
    "gpu_canary",
    "paid_scan",
)
TARGET_BINDING_FIELDS = (
    "code_commit",
    "code_sha256",
    "model_repo",
    "model_revision",
    "model_sha256",
    "manifest_sha256",
    "data_sha256",
    "encoded_manifest_sha256",
    "config_sha256",
    "scan_spec_sha256",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def validate_target_binding(binding: Mapping[str, Any]) -> dict[str, str]:
    """Validate and normalize the immutable identity of a proposed paid scan."""

    if not isinstance(binding, Mapping):
        raise ReadinessError("target binding must be an object")
    missing = [name for name in TARGET_BINDING_FIELDS if name not in binding]
    unknown = sorted(set(binding) - set(TARGET_BINDING_FIELDS))
    if missing or unknown:
        raise ReadinessError(f"target binding fields mismatch; missing={missing}, unknown={unknown}")
    result = {name: str(binding[name]) for name in TARGET_BINDING_FIELDS}
    if re.fullmatch(r"[0-9a-f]{40}", result["code_commit"]) is None:
        raise ReadinessError("target binding code_commit is not a lowercase 40-hex commit")
    if not result["model_repo"]:
        raise ReadinessError("target binding model_repo is empty")
    if re.fullmatch(r"[0-9a-f]{40}", result["model_revision"]) is None:
        raise ReadinessError("target binding model_revision is not a lowercase 40-hex revision")
    for name in TARGET_BINDING_FIELDS:
        if name in {"code_commit", "model_repo", "model_revision"}:
            continue
        if re.fullmatch(r"[0-9a-f]{64}", result[name]) is None:
            raise ReadinessError(f"target binding {name} is not a lowercase SHA-256")
    if result["code_sha256"] != _sha256_value({"git_commit": result["code_commit"]}):
        raise ReadinessError("target binding code_sha256 does not bind code_commit")
    if result["model_sha256"] != _sha256_value({
        "repo": result["model_repo"], "revision": result["model_revision"]
    }):
        raise ReadinessError("target binding model_sha256 does not bind model repo/revision")
    return result


def target_binding_sha256(binding: Mapping[str, Any]) -> str:
    return _sha256_value(validate_target_binding(binding))


def build_authorization_artifact(
    binding: Mapping[str, Any], evidence: Mapping[str, Any], assessment: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a content-addressed GO/NO_GO artifact; only GO can authorize work."""

    normalized_binding = validate_target_binding(binding)
    if not isinstance(evidence, Mapping) or not isinstance(assessment, Mapping):
        raise ReadinessError("evidence and assessment must be objects")
    binding_hash = target_binding_sha256(normalized_binding)
    if evidence.get("target_binding_sha256") != binding_hash:
        raise ReadinessError("evidence target binding does not match authorization target")
    decision = "GO" if assessment.get("decision") == "GO" else "NO_GO"
    body: dict[str, Any] = {
        "schema_version": "1.0.0",
        "artifact_type": AUTHORIZATION_TYPE,
        "decision": decision,
        "target_binding": normalized_binding,
        "target_binding_sha256": binding_hash,
        "evidence": dict(evidence),
        "evidence_sha256": _sha256_value(evidence),
        "assessment": dict(assessment),
        "assessment_sha256": _sha256_value(assessment),
    }
    body["authorization_sha256"] = _sha256_value(body)
    return body


def verify_authorization_artifact(
    artifact: Mapping[str, Any], expected_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify hashes, all staged gates, and exact target identity before GPU setup."""

    if not isinstance(artifact, Mapping):
        raise ReadinessError("readiness authorization must be an object")
    if artifact.get("schema_version") != "1.0.0" or artifact.get("artifact_type") != AUTHORIZATION_TYPE:
        raise ReadinessError("readiness authorization type/schema mismatch")
    if artifact.get("decision") != "GO":
        raise ReadinessError("readiness authorization decision is not GO")
    body = dict(artifact)
    observed_authorization_hash = body.pop("authorization_sha256", None)
    if observed_authorization_hash != _sha256_value(body):
        raise ReadinessError("readiness authorization SHA-256 mismatch")
    normalized_expected = validate_target_binding(expected_binding)
    normalized_observed = validate_target_binding(artifact.get("target_binding", {}))
    if normalized_observed != normalized_expected:
        raise ReadinessError("readiness authorization targets a different model/code/data/config/scan")
    binding_hash = target_binding_sha256(normalized_expected)
    if artifact.get("target_binding_sha256") != binding_hash:
        raise ReadinessError("readiness target binding SHA-256 mismatch")
    evidence = artifact.get("evidence")
    assessment = artifact.get("assessment")
    if not isinstance(evidence, Mapping) or artifact.get("evidence_sha256") != _sha256_value(evidence):
        raise ReadinessError("readiness evidence SHA-256 mismatch")
    if not isinstance(assessment, Mapping) or artifact.get("assessment_sha256") != _sha256_value(assessment):
        raise ReadinessError("readiness assessment SHA-256 mismatch")
    if assessment.get("decision") != "GO" or assessment.get("blockers") != []:
        raise ReadinessError("readiness assessment is not an unblocked GO")
    stages = assessment.get("stages")
    if not isinstance(stages, Sequence) or not stages or any(
        not isinstance(stage, Mapping) or stage.get("decision") != "GO" for stage in stages
    ):
        raise ReadinessError("readiness assessment contains a non-GO or invalid stage")
    if [stage.get("stage") for stage in stages] != list(AUTHORIZATION_STAGES):
        raise ReadinessError("readiness assessment does not contain the exact ordered gate set")
    if evidence.get("target_binding_sha256") != binding_hash:
        raise ReadinessError("readiness evidence targets a different scan binding")
    return normalized_observed


MODEL_CHECKS = (
    "exact_model_revision",
    "model_type_moshi",
    "shape_contract",
    "mimi_contract",
    "dtype_contract",
    "device_contract",
    "hook_off_identity",
    "identity_patch_noop",
)

OPEN_LOOP_CHECKS = (
    "paired_feedback_identical",
    "sampled_feedback_absent",
    "deterministic_replay",
    "identity_patch_noop",
    "candidate_order_invariant",
    "delay_mapping_valid",
)

CONVERSATION_CHECKS = (
    "initial_greeting_measured",
    "turn_taking_reviewed",
    "human_flow_review_passed",
    "response_capture_complete",
    "output_coverage_complete",
    "text_tail_checked",
    "audio_tail_checked",
    "no_tail_truncation",
)

GPU_CANARY_CHECKS = (
    "bounded_grid",
    "finite_outputs",
    "no_failed_cells",
    "resume_no_duplicates",
    "peak_vram_measured",
    "activation_bytes_measured",
    "runtime_measured",
)


FROZEN_AUDIO_ACTIVITY_POLICY = {
    "version": "1.0.0",
    "detector": "frame_rms_dbfs",
    "frame_samples": 1_920,
    "threshold_dbfs": -45.0,
    "calibration": "forced_silence_decode_max_must_remain_below_threshold",
}
FROZEN_AUDIO_ACTIVITY_POLICY_SHA256 = (
    "3d6389a42d452d4efa25ea7feb87bb4e82134bb6829356bd5e041ba55a8a289f"
)


@dataclass(frozen=True)
class BudgetLimits:
    """Hard ceilings applied before a paid scan can receive a GO decision."""

    max_cells: int = 10_000
    max_cells_per_recipient: int = 2_048
    max_model_frames: int = 50_000_000
    max_generation_runs: int = 5_000
    max_generated_audio_hours: float = 100.0
    max_storage_bytes: int = 100 * 1024**3

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "BudgetLimits":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ReadinessError("budget limits must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ReadinessError(f"unknown budget limits: {sorted(unknown)}")
        limits = cls(**dict(value))
        for name, item in asdict(limits).items():
            integer_limit = name != "max_generated_audio_hours"
            valid_type = isinstance(item, int) if integer_limit else isinstance(item, (int, float))
            if isinstance(item, bool) or not valid_type or item <= 0:
                raise ReadinessError(f"budget limit {name} must be finite and positive")
            if not math.isfinite(float(item)):
                raise ReadinessError(f"budget limit {name} must be finite and positive")
        return limits


@dataclass(frozen=True)
class WorkloadEstimate:
    manifest_trial_count: int
    manifest_frame_count: int
    manifest_audio_hours: float
    selected_trial_count: int
    selected_frame_count: int
    recipient_trial_count: int
    cell_count: int
    cells_per_recipient: int
    replay_pass_count: int
    replay_frame_count: int
    readout_frame_count: int
    generation_trial_count: int
    generation_count: int
    generation_frame_count: int
    generated_audio_frame_count: int
    generated_audio_hours: float
    encoded_tensor_bytes: int
    activation_tensor_bytes: int
    cell_record_reserved_bytes: int
    generated_wav_reserved_bytes: int
    fixed_reserved_bytes: int
    total_storage_reserved_bytes: int
    scan_breakdown: tuple[dict[str, Any], ...]

    @property
    def total_model_frames(self) -> int:
        return self.replay_frame_count + self.readout_frame_count + self.generation_frame_count

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["total_model_frames"] = self.total_model_frames
        result["scan_breakdown"] = list(self.scan_breakdown)
        return result


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReadinessError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ReadinessError(f"{label} must be {qualifier}")
    return value


def _unique_sequence(value: Any, label: str, *, allow_empty: bool = False) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReadinessError(f"{label} must be a sequence")
    result = tuple(value)
    if not result and not allow_empty:
        raise ReadinessError(f"{label} must not be empty")
    try:
        unique_count = len(set(result))
    except TypeError as error:
        raise ReadinessError(f"{label} entries must be scalar values") from error
    if unique_count != len(result):
        raise ReadinessError(f"{label} must not contain duplicates")
    return result


def _manifest_rows(
    manifest: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int, int]:
    if not manifest:
        raise ReadinessError("manifest must contain at least one trial")
    audio = config.get("audio")
    if not isinstance(audio, Mapping):
        raise ReadinessError("config.audio is required")
    sample_rate = _positive_int(audio.get("sample_rate"), "config.audio.sample_rate")
    frame_samples = _positive_int(audio.get("mimi_frame_samples"), "config.audio.mimi_frame_samples")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(manifest):
        if not isinstance(source, Mapping):
            raise ReadinessError(f"manifest row {index} must be an object")
        row = dict(source)
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise ReadinessError(f"manifest row {index} has no trial_id")
        if trial_id in seen:
            raise ReadinessError(f"duplicate trial_id: {trial_id}")
        seen.add(trial_id)
        frame_count = _positive_int(row.get("frame_count"), f"{trial_id}.frame_count")
        sample_count = _positive_int(row.get("sample_count"), f"{trial_id}.sample_count")
        if row.get("sample_rate") != sample_rate:
            raise ReadinessError(f"{trial_id}: sample rate differs from config")
        if sample_count != frame_count * frame_samples:
            raise ReadinessError(f"{trial_id}: sample_count is not exactly frame_count * frame_samples")
        conversation_contract = row.get("conversation_contract")
        if conversation_contract is not None:
            if not isinstance(conversation_contract, Mapping):
                raise ReadinessError(f"{trial_id}.conversation_contract must be an object")
            target_frames = _positive_int(
                conversation_contract.get("target_end_frame_count"),
                f"{trial_id}.conversation_contract.target_end_frame_count",
            )
            if target_frames < frame_count:
                raise ReadinessError(f"{trial_id}: conversation target ends before prepared input")
            target_samples = _positive_int(
                conversation_contract.get("target_end_sample_count"),
                f"{trial_id}.conversation_contract.target_end_sample_count",
            )
            if target_samples != target_frames * frame_samples:
                raise ReadinessError(f"{trial_id}: conversation target frame/sample counts disagree")
            user_frames = _positive_int(
                conversation_contract.get("user_frame_count"),
                f"{trial_id}.conversation_contract.user_frame_count",
            )
            user_end = _positive_int(
                conversation_contract.get("user_end_frame"),
                f"{trial_id}.conversation_contract.user_end_frame",
            )
            response_frames = _positive_int(
                conversation_contract.get("response_capture_frames"),
                f"{trial_id}.conversation_contract.response_capture_frames",
            )
            appended_frames = _positive_int(
                conversation_contract.get("appended_zero_frame_count"),
                f"{trial_id}.conversation_contract.appended_zero_frame_count",
                allow_zero=True,
            )
            if user_frames != frame_count or user_end > user_frames:
                raise ReadinessError(f"{trial_id}: conversation user frame boundaries disagree")
            if target_frames != user_end + response_frames:
                raise ReadinessError(
                    f"{trial_id}: target horizon must be semantic user_end + response capture")
            if appended_frames != target_frames - frame_count:
                raise ReadinessError(f"{trial_id}: exact-zero continuation length disagrees")
        else:
            target_frames = frame_count
        row["_model_frame_count"] = target_frames
        rows.append(row)
    return rows, sample_rate, frame_samples


def _select(rows: Sequence[dict[str, Any]], selector: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    selector = selector or {}
    supported = {
        "trial_ids", "roles", "folds", "conditions", "scenario_ids", "speaker_ids",
        "exclude_clean",
    }
    unknown = set(selector) - supported
    if unknown:
        raise ReadinessError(f"unknown selector fields: {sorted(unknown)}")
    selected = list(rows)
    fields = {
        "trial_ids": "trial_id",
        "roles": "role",
        "folds": "analysis_fold",
        "conditions": "condition",
        "scenario_ids": "scenario_id",
        "speaker_ids": "speaker_id",
    }
    for key, row_key in fields.items():
        if key in selector:
            allowed = set(_unique_sequence(selector[key], f"selector.{key}"))
            selected = [row for row in selected if row.get(row_key) in allowed]
    if selector.get("exclude_clean"):
        selected = [row for row in selected if not str(row.get("condition", "")).startswith("clean")]
    return selected


def _component_width(component: Any, model_heads: int) -> tuple[str, int]:
    if isinstance(component, str):
        return component, model_heads if component == "head_z" else 1
    if not isinstance(component, Mapping):
        raise ReadinessError("scan components must be strings or objects")
    name = component.get("name")
    if not isinstance(name, str) or not name:
        raise ReadinessError("component.name is required")
    if "heads" in component:
        heads = _unique_sequence(component["heads"], f"component {name}.heads")
        width = len(heads)
    else:
        width = _positive_int(component.get("instances", 1), f"component {name}.instances")
    return name, width


def _capture_bytes(
    rows: Sequence[dict[str, Any]], captures: Any, hidden_size: int
) -> int:
    if captures is None:
        return 0
    if isinstance(captures, (str, bytes)) or not isinstance(captures, Sequence):
        raise ReadinessError("storage.captures must be a sequence")
    total = 0
    for index, capture in enumerate(captures):
        if not isinstance(capture, Mapping):
            raise ReadinessError(f"storage.captures[{index}] must be an object")
        chosen = _select(rows, capture.get("selector"))
        if not chosen:
            raise ReadinessError(f"storage.captures[{index}] selects no trials")
        layers = _unique_sequence(capture.get("layers"), f"storage.captures[{index}].layers")
        anchors = _unique_sequence(capture.get("anchors"), f"storage.captures[{index}].anchors")
        sites = _unique_sequence(capture.get("sites"), f"storage.captures[{index}].sites")
        dtype_bytes = _positive_int(capture.get("dtype_bytes"), f"storage.captures[{index}].dtype_bytes")
        elements = _positive_int(
            capture.get("elements_per_site", hidden_size),
            f"storage.captures[{index}].elements_per_site",
        )
        total += len(chosen) * len(layers) * len(anchors) * len(sites) * elements * dtype_bytes
    return total


def _execution_contract(
    scan_spec: Mapping[str, Any], scans: Sequence[Any], selected: list[dict[str, Any]],
    *, model_heads: int,
) -> list[dict[str, Any]]:
    """Bind cost arithmetic to the literal paid-scan CLI grid when declared."""

    execution = scan_spec.get("execution")
    if execution is None:
        return selected
    if not isinstance(execution, Mapping):
        raise ReadinessError("scan_spec.execution must be an object")
    required = {
        "kind", "role", "layers", "anchors", "donors", "controls", "components",
        "limit_scenarios", "selection_sha256",
    }
    missing = sorted(required - set(execution))
    unknown = sorted(set(execution) - required)
    if missing or unknown:
        raise ReadinessError(
            f"scan execution fields mismatch; missing={missing}, unknown={unknown}")
    if len(scans) != 1 or not isinstance(scans[0], Mapping):
        raise ReadinessError("an executable paid scan spec must contain exactly one scan grid")
    scan = scans[0]
    kind = execution.get("kind")
    if kind not in {"residual", "component", "kv", "path"}:
        raise ReadinessError("scan execution kind is unsupported")
    role = execution.get("role")
    if not isinstance(role, str) or not role:
        raise ReadinessError("scan execution role must be a non-empty string")
    selector = scan_spec.get("trial_selector", {})
    if not isinstance(selector, Mapping) or list(selector.get("roles", [])) != [role]:
        raise ReadinessError("trial_selector.roles must exactly bind the scan execution role")

    list_pairs = (
        ("layers", scan.get("layers"), execution.get("layers")),
        ("anchors", scan.get("anchors"), execution.get("anchors")),
    )
    for label, declared, invoked in list_pairs:
        if list(_unique_sequence(declared, f"scan {label}")) != list(
            _unique_sequence(invoked, f"execution.{label}")):
            raise ReadinessError(f"scan {label} differs from execution.{label}")
    execution_components = list(
        _unique_sequence(execution.get("components"), "execution.components"))
    scan_components = scan.get("components")
    if isinstance(scan_components, (str, bytes)) or not isinstance(scan_components, Sequence):
        raise ReadinessError("scan components must be a sequence")
    scan_component_names = [_component_width(component, model_heads)[0] for component in scan_components]
    if scan_component_names != execution_components:
        raise ReadinessError("scan component names differ from execution.components")
    allowed_components = {
        "residual": {"resid_post"},
        "component": {"attn_out", "mlp_out", "head_z"},
        "kv": {"k_only", "v_only", "kv"},
        "path": {"path"},
    }[str(kind)]
    if not execution_components or any(name not in allowed_components for name in execution_components):
        raise ReadinessError(f"scan components are invalid for execution kind {kind}")
    if kind in {"residual", "path"} and set(execution_components) != allowed_components:
        raise ReadinessError(f"execution kind {kind} requires exactly {sorted(allowed_components)}")

    active_field = "controls" if kind == "component" else "donors"
    active_arms = list(_unique_sequence(execution.get(active_field), f"execution.{active_field}"))
    declared_arms = list(_unique_sequence(scan.get("donor_arms"), "scan.donor_arms"))
    if declared_arms != active_arms:
        raise ReadinessError(
            f"scan.donor_arms must exactly equal the kind-active execution.{active_field}")

    limit = execution.get("limit_scenarios")
    if limit is not None:
        limit_count = _positive_int(limit, "execution.limit_scenarios")
        allowed = sorted({str(row.get("scenario_id", "")) for row in selected})[:limit_count]
        if not allowed or any(not scenario_id for scenario_id in allowed):
            raise ReadinessError("execution.limit_scenarios requires non-empty scenario IDs")
        selected = [row for row in selected if str(row.get("scenario_id", "")) in allowed]
    return selected


def estimate_workload(
    manifest: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    scan_spec: Mapping[str, Any],
) -> WorkloadEstimate:
    """Return exact integer arithmetic for the declared workload.

    ``full_replays_per_cell`` means full recipient-length passes.  If an
    implementation uses donor lengths or suffix replay instead, its scan spec must
    express the resulting exact frame count with ``replay_frames_per_cell``.
    """

    if not isinstance(config, Mapping):
        raise ReadinessError("config must be an object")
    if not isinstance(scan_spec, Mapping):
        raise ReadinessError("scan_spec must be an object")
    rows, sample_rate, frame_samples = _manifest_rows(manifest, config)
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ReadinessError("config.model is required")
    model_heads = _positive_int(model.get("heads"), "config.model.heads")
    hidden_size = _positive_int(model.get("hidden_size"), "config.model.hidden_size")
    scans = scan_spec.get("scans")
    if isinstance(scans, (str, bytes)) or not isinstance(scans, Sequence) or not scans:
        raise ReadinessError("scan_spec.scans must be a non-empty sequence")
    selected = _select(rows, scan_spec.get("trial_selector"))
    if not selected:
        raise ReadinessError("trial_selector selects no trials")
    selected = _execution_contract(scan_spec, scans, selected, model_heads=model_heads)
    recipients = _select(selected, scan_spec.get("recipient_selector"))
    if not recipients:
        raise ReadinessError("recipient_selector selects no trials")
    if scan_spec.get("execution") is not None:
        executed_recipients = [
            row for row in selected if not str(row.get("condition", "")).startswith("clean")]
        if not executed_recipients:
            executed_recipients = selected[:1]
        if [row["trial_id"] for row in recipients] != [row["trial_id"] for row in executed_recipients]:
            raise ReadinessError(
                "recipient_selector differs from the recipients materialized by the paid scan CLI")
    total_cells = total_passes = total_replay_frames = total_readout_frames = 0
    cells_by_recipient = {str(row["trial_id"]): 0 for row in recipients}
    breakdown: list[dict[str, Any]] = []
    for index, scan in enumerate(scans):
        if not isinstance(scan, Mapping):
            raise ReadinessError(f"scans[{index}] must be an object")
        name = scan.get("name")
        if not isinstance(name, str) or not name:
            raise ReadinessError(f"scans[{index}].name is required")
        scan_recipients = _select(recipients, scan.get("recipient_selector"))
        if not scan_recipients:
            raise ReadinessError(f"scan {name} selects no recipients")
        layers = _unique_sequence(scan.get("layers"), f"scan {name}.layers")
        anchors = _unique_sequence(scan.get("anchors"), f"scan {name}.anchors")
        donors = _unique_sequence(scan.get("donor_arms"), f"scan {name}.donor_arms")
        components = scan.get("components")
        if isinstance(components, (str, bytes)) or not isinstance(components, Sequence) or not components:
            raise ReadinessError(f"scan {name}.components must be a non-empty sequence")
        component_instances = sum(_component_width(component, model_heads)[1] for component in components)
        combinations_per_recipient = len(layers) * len(anchors) * len(donors) * component_instances
        cells = len(scan_recipients) * combinations_per_recipient
        full_replays = _positive_int(
            scan.get("full_replays_per_cell", 0), f"scan {name}.full_replays_per_cell", allow_zero=True)
        readout_steps = _positive_int(
            scan.get("readout_steps_per_cell", 0), f"scan {name}.readout_steps_per_cell", allow_zero=True)
        if "replay_frames_per_cell" in scan:
            replay_frames = cells * _positive_int(
                scan["replay_frames_per_cell"], f"scan {name}.replay_frames_per_cell", allow_zero=True)
        else:
            replay_frames = sum(int(row["_model_frame_count"]) for row in scan_recipients) * combinations_per_recipient * full_replays
        passes = cells * full_replays
        readout_frames = cells * readout_steps
        expected = scan.get("expected_cell_count")
        if expected is not None and _positive_int(expected, f"scan {name}.expected_cell_count") != cells:
            raise ReadinessError(f"scan {name}: expected_cell_count does not match computed grid ({cells})")
        total_cells += cells
        for row in scan_recipients:
            cells_by_recipient[str(row["trial_id"])] += combinations_per_recipient
        total_passes += passes
        total_replay_frames += replay_frames
        total_readout_frames += readout_frames
        breakdown.append({
            "name": name,
            "recipient_trials": len(scan_recipients),
            "component_instances": component_instances,
            "layers": len(layers),
            "anchors": len(anchors),
            "donor_arms": len(donors),
            "active_arm_source": (
                "execution.controls"
                if isinstance(scan_spec.get("execution"), Mapping)
                and scan_spec["execution"].get("kind") == "component"
                else "execution.donors"
                if isinstance(scan_spec.get("execution"), Mapping)
                else "scan.donor_arms"
            ),
            "cells_per_recipient": combinations_per_recipient,
            "cell_count": cells,
            "replay_pass_count": passes,
            "replay_frame_count": replay_frames,
            "readout_frame_count": readout_frames,
        })

    generation = scan_spec.get("generation", {})
    if not isinstance(generation, Mapping):
        raise ReadinessError("scan_spec.generation must be an object")
    generation_trials = _select(selected, generation.get("trial_selector")) if generation else []
    if generation:
        seeds = _unique_sequence(generation.get("seeds"), "generation.seeds")
        branches = _unique_sequence(generation.get("branches"), "generation.branches")
        configured_conversation = config.get("conversation", {})
        required_startup_modes = (
            configured_conversation.get("required_modes", ())
            if isinstance(configured_conversation, Mapping) else ())
        if required_startup_modes:
            startup_modes = _unique_sequence(generation.get("startup_modes"), "generation.startup_modes")
            if set(startup_modes) != set(required_startup_modes):
                raise ReadinessError(
                    "generation.startup_modes must exactly cover config.conversation.required_modes")
        else:
            startup_modes = ("fixed_file_replay",)
        response_capture_ms = generation.get("response_capture_ms")
        if isinstance(response_capture_ms, bool) or not isinstance(response_capture_ms, (int, float)):
            raise ReadinessError("generation.response_capture_ms must be numeric")
        if not math.isfinite(float(response_capture_ms)) or response_capture_ms < 0:
            raise ReadinessError("generation.response_capture_ms must be finite and non-negative")
        tail_frames = math.ceil(float(response_capture_ms) * sample_rate / (1000 * frame_samples))
        for row in generation_trials:
            contract = row.get("conversation_contract")
            if isinstance(contract, Mapping):
                if tail_frames != int(contract["response_capture_frames"]):
                    raise ReadinessError(
                        f"{row['trial_id']}: generation response horizon differs from conversation contract")
        startup = configured_conversation.get("startup", {}) if isinstance(configured_conversation, Mapping) else {}
        natural_max_ms = startup.get("natural_max_ms", 0) if isinstance(startup, Mapping) else 0
        if isinstance(natural_max_ms, bool) or not isinstance(natural_max_ms, (int, float)):
            raise ReadinessError("config.conversation.startup.natural_max_ms must be numeric")
        if not math.isfinite(float(natural_max_ms)) or natural_max_ms < 0:
            raise ReadinessError("config.conversation.startup.natural_max_ms must be finite and non-negative")
        handshake_frames = math.ceil(float(natural_max_ms) * sample_rate / (1000 * frame_samples))
        multiplicity = len(seeds) * len(branches) * len(startup_modes)
        generation_count = len(generation_trials) * multiplicity
        generated_frames = 0
        for row in generation_trials:
            if row.get("conversation_contract") is not None:
                base_frames = int(row["_model_frame_count"])
            else:
                base_frames = int(row["frame_count"]) + tail_frames
            for startup_mode in startup_modes:
                # The common-handshake arm is budgeted at its hard startup cap,
                # never at an optimistic observed greeting duration.
                extra = handshake_frames if startup_mode == "common_handshake_then_request" else 0
                generated_frames += (base_frames + extra) * len(seeds) * len(branches)
        expected = generation.get("expected_generation_count")
        if expected is not None and _positive_int(
            expected, "generation.expected_generation_count", allow_zero=True
        ) != generation_count:
            raise ReadinessError(
                f"generation.expected_generation_count does not match computed grid ({generation_count})")
    else:
        seeds = branches = ()
        generation_count = generated_frames = 0

    storage = scan_spec.get("storage", {})
    if not isinstance(storage, Mapping):
        raise ReadinessError("scan_spec.storage must be an object")
    codebooks = _positive_int(storage.get("user_codebooks", 8), "storage.user_codebooks")
    code_dtype_bytes = _positive_int(storage.get("code_dtype_bytes", 8), "storage.code_dtype_bytes")
    sample_width = _positive_int(storage.get("audio_sample_width_bytes", 2), "storage.audio_sample_width_bytes")
    wav_header_bytes = _positive_int(storage.get("wav_header_bytes", 44), "storage.wav_header_bytes", allow_zero=True)
    cell_bytes = _positive_int(storage.get("result_bytes_per_cell", 2048), "storage.result_bytes_per_cell")
    fixed_bytes = _positive_int(storage.get("fixed_reserved_bytes", 0), "storage.fixed_reserved_bytes", allow_zero=True)
    # The canonical cache retains the prepared user prefix, continuous user+tail
    # codes, and the independently encoded assistant-silence trace.  Counting all
    # three arrays prevents compressed output from hiding the true reservation.
    encoded_code_frames = sum(
        int(row["frame_count"]) + 2 * int(row["_model_frame_count"]) for row in selected)
    encoded_bytes = encoded_code_frames * codebooks * code_dtype_bytes
    activation_bytes = _capture_bytes(selected, storage.get("captures"), hidden_size)
    cell_record_bytes = total_cells * cell_bytes
    generated_wav_bytes = generated_frames * frame_samples * sample_width + generation_count * wav_header_bytes
    total_storage = encoded_bytes + activation_bytes + cell_record_bytes + generated_wav_bytes + fixed_bytes

    manifest_frames = sum(int(row["frame_count"]) for row in rows)
    selected_frames = sum(int(row["frame_count"]) for row in selected)
    return WorkloadEstimate(
        manifest_trial_count=len(rows),
        manifest_frame_count=manifest_frames,
        manifest_audio_hours=manifest_frames * frame_samples / sample_rate / 3600,
        selected_trial_count=len(selected),
        selected_frame_count=selected_frames,
        recipient_trial_count=len(recipients),
        cell_count=total_cells,
        cells_per_recipient=max(cells_by_recipient.values()),
        replay_pass_count=total_passes,
        replay_frame_count=total_replay_frames,
        readout_frame_count=total_readout_frames,
        generation_trial_count=len(generation_trials),
        generation_count=generation_count,
        generation_frame_count=generated_frames,
        generated_audio_frame_count=generated_frames,
        generated_audio_hours=generated_frames * frame_samples / sample_rate / 3600,
        encoded_tensor_bytes=encoded_bytes,
        activation_tensor_bytes=activation_bytes,
        cell_record_reserved_bytes=cell_record_bytes,
        generated_wav_reserved_bytes=generated_wav_bytes,
        fixed_reserved_bytes=fixed_bytes,
        total_storage_reserved_bytes=total_storage,
        scan_breakdown=tuple(breakdown),
    )


def _evidence_blockers(
    stage: str, evidence: Mapping[str, Any] | None, required_checks: Sequence[str]
) -> list[dict[str, str]]:
    if evidence is None:
        return [{"stage": stage, "code": f"missing_{stage}_evidence", "message": f"{stage} evidence is missing"}]
    if not isinstance(evidence, Mapping):
        return [{
            "stage": stage,
            "code": f"invalid_{stage}_evidence",
            "message": f"{stage} evidence must be an object",
        }]
    blockers: list[dict[str, str]] = []
    if evidence.get("passed") is not True:
        blockers.append({"stage": stage, "code": f"{stage}_not_passed", "message": f"{stage}.passed is not true"})
    checks = evidence.get("checks")
    if not isinstance(checks, Mapping):
        blockers.append({"stage": stage, "code": f"missing_{stage}_checks", "message": f"{stage}.checks is missing"})
        return blockers
    for name in required_checks:
        if checks.get(name) is not True:
            blockers.append({
                "stage": stage,
                "code": f"{stage}_{name}_not_proven",
                "message": f"{stage} check {name!r} is missing or false",
            })
    return blockers


def _finite_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReadinessError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ReadinessError(f"{label} must be finite and >= {minimum}")
    return result


def _validate_audio_activity_policy(
    value: Any, *, mimi_frame_samples: Any,
) -> tuple[float, str]:
    """Validate the frozen policy and recompute its content hash.

    Merely copying the expected digest into a different policy object must never
    be enough to authorize a paid run.  The digest covers every policy field
    except the digest itself, and the policy shape is closed so an unreviewed
    detector option cannot hitchhike on the authorization.
    """

    if not isinstance(value, Mapping):
        raise ReadinessError("config.conversation.response.audio_activity must be an object")
    expected_fields = set(FROZEN_AUDIO_ACTIVITY_POLICY) | {"policy_sha256"}
    missing = sorted(expected_fields - set(value))
    unknown = sorted(set(value) - expected_fields)
    if missing or unknown:
        raise ReadinessError(
            f"audio activity policy fields mismatch; missing={missing}, unknown={unknown}")
    for name, expected in FROZEN_AUDIO_ACTIVITY_POLICY.items():
        observed = value.get(name)
        if isinstance(expected, float):
            if isinstance(observed, bool) or not isinstance(observed, (int, float)):
                raise ReadinessError(f"audio activity policy {name} must be numeric")
            if not math.isfinite(float(observed)) or float(observed) != expected:
                raise ReadinessError(
                    f"audio activity policy {name} must equal frozen value {expected!r}")
        elif observed != expected:
            raise ReadinessError(
                f"audio activity policy {name} must equal frozen value {expected!r}")
    if value["frame_samples"] != mimi_frame_samples:
        raise ReadinessError("audio activity frame size differs from Mimi")
    declared_sha = value.get("policy_sha256")
    if not isinstance(declared_sha, str) or re.fullmatch(r"[0-9a-f]{64}", declared_sha) is None:
        raise ReadinessError("audio activity policy must have a lowercase SHA-256")
    policy_body = {name: item for name, item in value.items() if name != "policy_sha256"}
    computed_sha = _sha256_value(policy_body)
    if computed_sha != declared_sha:
        raise ReadinessError("audio activity policy SHA-256 does not match its canonical contents")
    if computed_sha != FROZEN_AUDIO_ACTIVITY_POLICY_SHA256:
        raise ReadinessError("audio activity policy differs from the frozen canonical policy")
    return float(value["threshold_dbfs"]), declared_sha


def _conversation_measurement_blockers(
    evidence: Mapping[str, Any] | None, config: Mapping[str, Any]
) -> list[dict[str, str]]:
    stage = "conversation_canary"
    if not isinstance(evidence, Mapping):
        return []  # The generic evidence validator already reports this.
    conversation = config.get("conversation", {})
    gates = config.get("gates", {})
    if not isinstance(conversation, Mapping) or not isinstance(gates, Mapping):
        return [{
            "stage": stage,
            "code": "missing_conversation_gate_config",
            "message": "config.conversation and config.gates must be objects",
        }]
    required_modes = conversation.get("required_modes", ())
    try:
        modes = _unique_sequence(required_modes, "config.conversation.required_modes")
        minimum_trials = _positive_int(
            gates.get("conversation_canary_min_trials_per_mode"),
            "config.gates.conversation_canary_min_trials_per_mode",
        )
        maximum_truncated = _positive_int(
            gates.get("conversation_canary_truncated_max", 0),
            "config.gates.conversation_canary_truncated_max",
            allow_zero=True,
        )
        minimum_coverage = _finite_number(
            gates.get("conversation_canary_coverage_min"),
            "config.gates.conversation_canary_coverage_min",
        )
        audio = config.get("audio", {})
        response = conversation.get("response", {})
        if not isinstance(audio, Mapping) or not isinstance(response, Mapping):
            raise ReadinessError("config.audio and config.conversation.response must be objects")
        frame_ms = _finite_number(audio.get("frame_ms"), "config.audio.frame_ms", minimum=1e-12)
        text_quiet_ms = _finite_number(
            response.get("trailing_text_quiet_ms"),
            "config.conversation.response.trailing_text_quiet_ms",
            minimum=1e-12,
        )
        tail_guard_ms = _finite_number(
            response.get("tail_guard_ms"),
            "config.conversation.response.tail_guard_ms",
            minimum=1e-12,
        )
        text_quiet_frames = int(round(text_quiet_ms / frame_ms))
        tail_guard_frames = int(round(tail_guard_ms / frame_ms))
        if (not math.isclose(text_quiet_frames * frame_ms, text_quiet_ms, abs_tol=1e-9)
                or not math.isclose(tail_guard_frames * frame_ms, tail_guard_ms, abs_tol=1e-9)):
            raise ReadinessError("text quiet and tail guard durations must be Mimi-frame aligned")
        audio_activity = response.get("audio_activity")
        audio_threshold_dbfs, audio_policy_sha = _validate_audio_activity_policy(
            audio_activity, mimi_frame_samples=audio.get("mimi_frame_samples"))
    except ReadinessError as error:
        return [{"stage": stage, "code": "invalid_conversation_gate_config", "message": str(error)}]
    if minimum_coverage > 1:
        return [{
            "stage": stage,
            "code": "invalid_conversation_gate_config",
            "message": "conversation_canary_coverage_min must be <= 1",
        }]
    per_mode = evidence.get("per_mode")
    if not isinstance(per_mode, Mapping):
        return [{
            "stage": stage,
            "code": "missing_conversation_mode_measurements",
            "message": "conversation_canary.per_mode is missing",
        }]
    blockers: list[dict[str, str]] = []
    summed_trials = summed_truncated = summed_cap_active = summed_covered = 0
    summed_text_checked = summed_audio_checked = summed_human_reviewed = 0
    for mode in modes:
        measurement = per_mode.get(mode)
        if not isinstance(measurement, Mapping):
            blockers.append({
                "stage": stage,
                "code": "missing_required_startup_mode",
                "message": f"no conversation canary measurements for startup mode {mode!r}",
            })
            continue
        try:
            trials = _positive_int(measurement.get("trial_count"), f"{mode}.trial_count")
            truncated = _positive_int(
                measurement.get("truncated_count"), f"{mode}.truncated_count", allow_zero=True)
            cap_active = _positive_int(
                measurement.get("cap_active_count"), f"{mode}.cap_active_count", allow_zero=True)
            covered = _positive_int(
                measurement.get("exact_output_coverage_count"),
                f"{mode}.exact_output_coverage_count",
                allow_zero=True,
            )
            complete = _positive_int(
                measurement.get("response_complete_count"),
                f"{mode}.response_complete_count",
                allow_zero=True,
            )
            text_checked = _positive_int(
                measurement.get("text_tail_checked_count"),
                f"{mode}.text_tail_checked_count",
                allow_zero=True,
            )
            audio_checked = _positive_int(
                measurement.get("audio_tail_checked_count"),
                f"{mode}.audio_tail_checked_count",
                allow_zero=True,
            )
            human_reviewed = _positive_int(
                measurement.get("human_flow_review_pass_count"),
                f"{mode}.human_flow_review_pass_count",
                allow_zero=True,
            )
        except ReadinessError as error:
            blockers.append({"stage": stage, "code": "invalid_mode_measurement", "message": str(error)})
            continue
        summed_trials += trials
        summed_truncated += truncated
        summed_cap_active += cap_active
        summed_covered += covered
        summed_text_checked += text_checked
        summed_audio_checked += audio_checked
        summed_human_reviewed += human_reviewed
        if trials < minimum_trials:
            blockers.append({
                "stage": stage,
                "code": "insufficient_trials_per_startup_mode",
                "message": f"{mode}: trial_count={trials} is below {minimum_trials}",
            })
        if truncated > maximum_truncated:
            blockers.append({
                "stage": stage,
                "code": "tail_truncation_observed",
                "message": f"{mode}: truncated_count={truncated} exceeds {maximum_truncated}",
            })
        if cap_active:
            blockers.append({
                "stage": stage,
                "code": "cap_active_response_observed",
                "message": f"{mode}: {cap_active} responses were still active at the capture cap",
            })
        if covered > trials or complete > trials:
            blockers.append({
                "stage": stage,
                "code": "impossible_conversation_counts",
                "message": f"{mode}: coverage/complete count exceeds trial_count",
            })
        elif covered / trials < minimum_coverage:
            blockers.append({
                "stage": stage,
                "code": "incomplete_output_coverage",
                "message": f"{mode}: exact output coverage is {covered}/{trials}",
            })
        if complete != trials:
            blockers.append({
                "stage": stage,
                "code": "incomplete_response_boundaries",
                "message": f"{mode}: complete response boundaries are {complete}/{trials}",
            })
        if text_checked != trials or audio_checked != trials:
            blockers.append({
                "stage": stage,
                "code": "tail_diagnostics_incomplete",
                "message": f"{mode}: text/audio tail checks are {text_checked}/{audio_checked} of {trials}",
            })
        if human_reviewed != trials:
            blockers.append({
                "stage": stage,
                "code": "human_flow_review_incomplete",
                "message": f"{mode}: human natural-flow approvals are {human_reviewed}/{trials}",
            })
    aggregate = evidence.get("measurements")
    if not isinstance(aggregate, Mapping):
        blockers.append({
            "stage": stage,
            "code": "missing_conversation_aggregate_measurements",
            "message": "conversation_canary.measurements is missing",
        })
    else:
        expected = {
            "required_mode_trial_count": summed_trials,
            "truncated_count": summed_truncated,
            "cap_active_count": summed_cap_active,
            "exact_output_coverage_count": summed_covered,
            "text_tail_checked_count": summed_text_checked,
            "audio_tail_checked_count": summed_audio_checked,
            "human_flow_review_pass_count": summed_human_reviewed,
        }
        for name, value in expected.items():
            if aggregate.get(name) != value:
                blockers.append({
                    "stage": stage,
                    "code": "conversation_aggregate_mismatch",
                    "message": f"conversation aggregate {name} does not equal per-mode sum {value}",
                })
    tail_detection = evidence.get("tail_detection")
    expected_tail = {
        "text_quiet_frames": text_quiet_frames,
        "tail_guard_frames": tail_guard_frames,
        "audio_activity_policy_version": FROZEN_AUDIO_ACTIVITY_POLICY["version"],
        "audio_activity_detector": "frame_rms_dbfs",
        "audio_activity_frame_samples": audio.get("mimi_frame_samples"),
        "audio_activity_threshold_dbfs": audio_threshold_dbfs,
        "audio_activity_calibration": FROZEN_AUDIO_ACTIVITY_POLICY["calibration"],
        "audio_activity_policy_sha256": audio_policy_sha,
    }
    if not isinstance(tail_detection, Mapping):
        blockers.append({
            "stage": stage,
            "code": "missing_frozen_tail_detection_provenance",
            "message": "conversation_canary.tail_detection is missing",
        })
    elif any(tail_detection.get(name) != value for name, value in expected_tail.items()):
        blockers.append({
            "stage": stage,
            "code": "tail_detection_provenance_mismatch",
            "message": "conversation tail detector differs from the frozen config",
        })
    else:
        try:
            forced_silence_max = _finite_number(
                tail_detection.get("forced_silence_decode_max_dbfs"),
                "conversation_canary.tail_detection.forced_silence_decode_max_dbfs",
                minimum=-math.inf,
            )
        except ReadinessError as error:
            blockers.append({
                "stage": stage,
                "code": "missing_forced_silence_calibration_measurement",
                "message": str(error),
            })
        else:
            if forced_silence_max >= audio_threshold_dbfs:
                blockers.append({
                    "stage": stage,
                    "code": "forced_silence_exceeds_audio_threshold",
                    "message": (
                        f"forced-silence decode max {forced_silence_max} dBFS is not below "
                        f"the frozen {audio_threshold_dbfs} dBFS threshold"),
                })
    return blockers


def _gpu_measurement_blockers(evidence: Mapping[str, Any] | None) -> list[dict[str, str]]:
    stage = "gpu_canary"
    if not isinstance(evidence, Mapping):
        return []  # The generic evidence validator already reports this.
    measurements = evidence.get("measurements")
    if not isinstance(measurements, Mapping):
        return [{
            "stage": stage,
            "code": "missing_gpu_measurements",
            "message": "gpu_canary.measurements is missing",
        }]
    blockers: list[dict[str, str]] = []
    try:
        cells = _positive_int(measurements.get("completed_cells"), "gpu_canary.completed_cells")
        failed = _positive_int(
            measurements.get("failed_cells"), "gpu_canary.failed_cells", allow_zero=True)
        duplicates = _positive_int(
            measurements.get("duplicate_cells"), "gpu_canary.duplicate_cells", allow_zero=True)
        frames = _positive_int(measurements.get("model_frame_count"), "gpu_canary.model_frame_count")
        peak_vram = _positive_int(measurements.get("peak_vram_bytes"), "gpu_canary.peak_vram_bytes")
        total_vram = _positive_int(
            measurements.get("device_total_vram_bytes"), "gpu_canary.device_total_vram_bytes")
        activation_bytes = _positive_int(
            measurements.get("activation_bytes"), "gpu_canary.activation_bytes")
        elapsed = _finite_number(measurements.get("elapsed_seconds"), "gpu_canary.elapsed_seconds", minimum=1e-12)
        cell_seconds = _finite_number(
            measurements.get("mean_cell_seconds"), "gpu_canary.mean_cell_seconds", minimum=1e-12)
        frame_seconds = _finite_number(
            measurements.get("seconds_per_model_frame"),
            "gpu_canary.seconds_per_model_frame",
            minimum=1e-12,
        )
    except ReadinessError as error:
        return [{"stage": stage, "code": "invalid_gpu_measurement", "message": str(error)}]
    if failed:
        blockers.append({
            "stage": stage, "code": "gpu_canary_failed_cells", "message": f"failed_cells={failed}"})
    if duplicates:
        blockers.append({
            "stage": stage,
            "code": "gpu_canary_duplicate_cells",
            "message": f"duplicate_cells={duplicates}",
        })
    if peak_vram > total_vram:
        blockers.append({
            "stage": stage,
            "code": "gpu_canary_vram_impossible",
            "message": "peak_vram_bytes exceeds device_total_vram_bytes",
        })
    if not math.isclose(cell_seconds, elapsed / cells, rel_tol=1e-6, abs_tol=1e-9):
        blockers.append({
            "stage": stage,
            "code": "gpu_canary_cell_timing_mismatch",
            "message": "mean_cell_seconds is inconsistent with elapsed_seconds/completed_cells",
        })
    if not math.isclose(frame_seconds, elapsed / frames, rel_tol=1e-6, abs_tol=1e-12):
        blockers.append({
            "stage": stage,
            "code": "gpu_canary_frame_timing_mismatch",
            "message": "seconds_per_model_frame is inconsistent with elapsed_seconds/model_frame_count",
        })
    if activation_bytes <= 0:
        blockers.append({
            "stage": stage,
            "code": "gpu_canary_activation_bytes_missing",
            "message": "no activation bytes were measured",
        })
    return blockers


def assess_readiness(
    manifest: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    scan_spec: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any] | None = None,
    limits: Mapping[str, Any] | None = None,
    target_binding_sha256: str | None = None,
) -> dict[str, Any]:
    """Return staged GO/NO_GO decisions; missing evidence always blocks paid work."""

    try:
        estimate = estimate_workload(manifest, config, scan_spec)
        budget_limits = BudgetLimits.from_mapping(limits)
    except ReadinessError as error:
        blocker = {"stage": "static_plan", "code": "invalid_workload_contract", "message": str(error)}
        return {
            "schema_version": "1.0.0",
            "decision": "NO_GO",
            "estimate": None,
            "limits": None,
            "stages": [{"stage": "static_plan", "decision": "NO_GO", "blockers": [blocker]}],
            "blockers": [blocker],
        }

    stages: list[dict[str, Any]] = []
    all_blockers: list[dict[str, str]] = []

    def add_stage(name: str, blockers: list[dict[str, str]]) -> None:
        all_blockers.extend(blockers)
        stages.append({"stage": name, "decision": "NO_GO" if blockers else "GO", "blockers": blockers})

    add_stage("static_plan", [])
    budget_blockers: list[dict[str, str]] = []
    comparisons = (
        ("cell_count", estimate.cell_count, budget_limits.max_cells),
        ("cells_per_recipient", estimate.cells_per_recipient, budget_limits.max_cells_per_recipient),
        ("total_model_frames", estimate.total_model_frames, budget_limits.max_model_frames),
        ("generation_count", estimate.generation_count, budget_limits.max_generation_runs),
        ("generated_audio_hours", estimate.generated_audio_hours, budget_limits.max_generated_audio_hours),
        ("total_storage_reserved_bytes", estimate.total_storage_reserved_bytes, budget_limits.max_storage_bytes),
    )
    for metric, observed, maximum in comparisons:
        if observed > maximum:
            budget_blockers.append({
                "stage": "budget",
                "code": f"explosive_{metric}",
                "message": f"{metric}={observed} exceeds hard limit {maximum}",
            })
    add_stage("budget", budget_blockers)

    if evidence is None:
        evidence = {}
    elif not isinstance(evidence, Mapping):
        invalid = {
            "stage": "model_contract",
            "code": "invalid_evidence_bundle",
            "message": "evidence must be an object",
        }
        add_stage("model_contract", [invalid])
        add_stage("open_loop", _evidence_blockers("open_loop", None, OPEN_LOOP_CHECKS))
        add_stage("conversation_canary", _evidence_blockers("conversation_canary", None, CONVERSATION_CHECKS))
        add_stage("gpu_canary", _evidence_blockers("gpu_canary", None, GPU_CANARY_CHECKS))
        paid = {
            "stage": "paid_scan",
            "code": "prerequisite_stage_not_ready",
            "message": "one or more static, budget, model, open-loop, or conversation gates are NO_GO",
        }
        add_stage("paid_scan", [paid])
        return {
            "schema_version": "1.0.0",
            "decision": "NO_GO",
            "estimate": estimate.to_dict(),
            "limits": asdict(budget_limits),
            "stages": stages,
            "blockers": all_blockers,
        }
    binding_blockers: list[dict[str, str]] = []
    if target_binding_sha256 is not None and evidence.get("target_binding_sha256") != target_binding_sha256:
        binding_blockers.append({
            "stage": "evidence_binding",
            "code": "evidence_target_binding_mismatch",
            "message": "evidence was not produced for this model/code/data/config/scan binding",
        })
    add_stage("evidence_binding", binding_blockers)
    add_stage("model_contract", _evidence_blockers("model_contract", evidence.get("model_contract"), MODEL_CHECKS))
    add_stage("open_loop", _evidence_blockers("open_loop", evidence.get("open_loop"), OPEN_LOOP_CHECKS))
    conversation_blockers = _evidence_blockers(
        "conversation_canary", evidence.get("conversation_canary"), CONVERSATION_CHECKS)
    conversation_blockers.extend(_conversation_measurement_blockers(evidence.get("conversation_canary"), config))
    add_stage(
        "conversation_canary",
        conversation_blockers,
    )
    gpu_blockers = _evidence_blockers("gpu_canary", evidence.get("gpu_canary"), GPU_CANARY_CHECKS)
    gpu_blockers.extend(_gpu_measurement_blockers(evidence.get("gpu_canary")))
    add_stage("gpu_canary", gpu_blockers)
    prerequisites_go = not all_blockers
    paid_blockers = [] if prerequisites_go else [{
        "stage": "paid_scan",
        "code": "prerequisite_stage_not_ready",
        "message": "one or more static, budget, model, open-loop, or conversation gates are NO_GO",
    }]
    add_stage("paid_scan", paid_blockers)
    gpu_measurements = evidence.get("gpu_canary", {}).get("measurements", {})
    runtime_projection = None
    if isinstance(gpu_measurements, Mapping):
        cell_seconds = gpu_measurements.get("mean_cell_seconds")
        frame_seconds = gpu_measurements.get("seconds_per_model_frame")
        if isinstance(cell_seconds, (int, float)) and isinstance(frame_seconds, (int, float)):
            runtime_projection = {
                "basis": "bounded_gpu_canary_wall_clock",
                "estimated_gpu_hours_by_cell": estimate.cell_count * float(cell_seconds) / 3600,
                "estimated_gpu_hours_by_model_frame": estimate.total_model_frames * float(frame_seconds) / 3600,
                "canary_peak_vram_bytes": gpu_measurements.get("peak_vram_bytes"),
                "canary_activation_bytes": gpu_measurements.get("activation_bytes"),
            }
    return {
        "schema_version": "1.0.0",
        "decision": "GO" if not all_blockers else "NO_GO",
        "estimate": estimate.to_dict(),
        "runtime_projection": runtime_projection,
        "limits": asdict(budget_limits),
        "stages": stages,
        "blockers": all_blockers,
    }
