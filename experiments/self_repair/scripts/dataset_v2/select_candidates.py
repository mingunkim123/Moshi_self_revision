#!/usr/bin/env python3
"""Select aligned candidates and materialize immutable accepted utterances."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from alignment_evidence import (
    validate_alignment_gate_contract,
    validate_downstream_alignment_evidence,
)
from audio_utils import (
    duration_ms,
    normalize_audio,
    read_pcm16_mono,
    write_pcm16_mono,
)
from common import (
    CONDITIONS,
    DATASET_ROOT,
    DEFAULT_SCRIPTS,
    portable_path,
    read_config,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
    write_jsonl,
)
from ids import accepted_audio_id, safe_component
from timing import validate_timing


SELECTION_VERSION = "2.3.0"
DEFAULT_INPUT = DATASET_ROOT / "manifests/qc_candidates.jsonl"
GLOBAL_TARGET_SCOPE = "global"
SPEAKER_TARGET_SCOPE = "speaker_specific"
TARGET_FIELDS = {"latency_ms", "post_duration_ms"}
REQUIRED_WEIGHT_KEYS = {
    "latency_error",
    "post_duration_error",
    "alignment_confidence_penalty",
    "clipping_penalty",
    "noise_penalty",
}
FROZEN_TIE_BREAK = (
    "selection_score",
    "alignment_confidence_desc",
    "canonical_sha256",
    "candidate_id",
)
MODEL_OUTCOME_KEYS = {
    "model_output",
    "model_response",
    "moshi_output",
    "moshi_response",
    "prediction",
    "evaluation_result",
    "eval_result",
    "final_target_correct",
    "stale_state_error",
    "relation_rebinding_accuracy",
    "early_stale_response",
    "recovery",
    "scorer_output",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select canonical candidates under a frozen, outcome-blind policy."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Frozen JSON selection policy; its complete canonical hash is recorded.",
    )
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_ROOT / "manifests/accepted_audio.jsonl",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DATASET_ROOT / "manifests/candidate_selection_decisions.jsonl",
    )
    parser.add_argument(
        "--audio-root", type=Path, default=DATASET_ROOT / "artifacts/accepted"
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _walk_keys(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield str(key), child_path
            yield from _walk_keys(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{path}[{index}]")


def assert_outcome_blind(row: dict[str, Any]) -> None:
    forbidden = [path for key, path in _walk_keys(row) if key.casefold() in MODEL_OUTCOME_KEYS]
    if forbidden:
        raise ValueError(
            f"{row.get('candidate_id')}: model-outcome fields are forbidden in selection: "
            + ", ".join(forbidden[:8])
        )


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _timing_target_scope(policy: dict[str, Any]) -> str:
    # Policies created before speaker-specific targets were added did not carry an
    # explicit scope.  Treating those policies as global preserves their exact
    # historical behavior and hash.
    scope = policy.get("timing_target_scope", GLOBAL_TARGET_SCOPE)
    if scope not in {GLOBAL_TARGET_SCOPE, SPEAKER_TARGET_SCOPE}:
        raise ValueError(
            "timing_target_scope must be 'global' or 'speaker_specific'"
        )
    return str(scope)


def _validate_condition_targets(targets: Any, label: str) -> None:
    if not isinstance(targets, dict) or set(targets) != set(CONDITIONS):
        raise ValueError(f"{label} must contain exactly {CONDITIONS}")
    for condition, target in targets.items():
        if not isinstance(target, dict) or set(target) != TARGET_FIELDS:
            raise ValueError(
                f"{label}.{condition}: target must contain exactly "
                "latency_ms/post_duration_ms"
            )
        latency = target["latency_ms"]
        if condition == "clean_final":
            if latency is not None:
                raise ValueError(
                    f"{label}.clean_final latency_ms target must be null"
                )
        elif not _is_finite_number(latency) or float(latency) < 0:
            raise ValueError(
                f"{label}.{condition}: latency_ms target must be finite and non-negative"
            )
        post = target["post_duration_ms"]
        if not _is_finite_number(post) or float(post) < 0:
            raise ValueError(
                f"{label}.{condition}: post_duration_ms target must be finite and non-negative"
            )


def validate_selection_policy(policy: dict[str, Any]) -> str:
    forbidden = [
        path for key, path in _walk_keys(policy) if key.casefold() in MODEL_OUTCOME_KEYS
    ]
    if forbidden:
        raise ValueError(
            "selection policy must be outcome-blind; forbidden fields: "
            + ", ".join(forbidden[:8])
        )
    if policy.get("schema_version") != "2.0.0":
        raise ValueError("selection policy schema_version must be 2.0.0")
    if policy.get("status") != "frozen":
        raise ValueError("selection policy status must be 'frozen'")
    if not isinstance(policy.get("policy_version"), str) or not policy["policy_version"]:
        raise ValueError("selection policy requires a non-empty policy_version")
    if not isinstance(policy.get("frozen_at"), str) or "T" not in policy["frozen_at"]:
        raise ValueError("selection policy requires an ISO-like frozen_at timestamp")
    weights = policy.get("weights")
    if not isinstance(weights, dict) or set(weights) != REQUIRED_WEIGHT_KEYS:
        raise ValueError(f"selection weights must be exactly {sorted(REQUIRED_WEIGHT_KEYS)}")
    for key, value in weights.items():
        if not _is_finite_number(value) or float(value) < 0:
            raise ValueError(f"selection weight {key} must be finite and non-negative")
    scales = policy.get("scales_ms")
    if not isinstance(scales, dict) or set(scales) != {"latency", "post_duration"}:
        raise ValueError("scales_ms must contain exactly latency and post_duration")
    if any(
        not _is_finite_number(value)
        or float(value) <= 0
        for value in scales.values()
    ):
        raise ValueError("selection scales must be finite and positive")
    scope = _timing_target_scope(policy)
    if scope == GLOBAL_TARGET_SCOPE:
        unexpected = [
            key for key in ("speaker_ids", "targets_by_speaker") if key in policy
        ]
        if unexpected:
            raise ValueError(
                "global timing policy must not contain speaker target fields: "
                + ", ".join(unexpected)
            )
        _validate_condition_targets(
            policy.get("targets_by_condition"), "targets_by_condition"
        )
    else:
        if "targets_by_condition" in policy:
            raise ValueError(
                "speaker_specific timing policy must not contain targets_by_condition"
            )
        speaker_ids = policy.get("speaker_ids")
        if not isinstance(speaker_ids, list) or not speaker_ids:
            raise ValueError(
                "speaker_specific timing policy requires a non-empty speaker_ids list"
            )
        if not all(isinstance(value, str) and value for value in speaker_ids):
            raise ValueError("speaker_ids must contain non-empty strings")
        for speaker_id in speaker_ids:
            safe_component(speaker_id, "speaker_id")
        if len(speaker_ids) != len(set(speaker_ids)):
            raise ValueError("speaker_ids must be unique")
        if speaker_ids != sorted(speaker_ids):
            raise ValueError("speaker_ids must be in lexicographic order")
        targets_by_speaker = policy.get("targets_by_speaker")
        if not isinstance(targets_by_speaker, dict):
            raise ValueError("targets_by_speaker must be an object")
        declared = set(speaker_ids)
        provided = set(targets_by_speaker)
        missing = sorted(declared - provided)
        extra = sorted(provided - declared)
        if missing or extra:
            raise ValueError(
                "targets_by_speaker keys must exactly match speaker_ids; "
                f"missing={missing}, extra={extra}"
            )
        for speaker_id in speaker_ids:
            _validate_condition_targets(
                targets_by_speaker[speaker_id],
                f"targets_by_speaker.{speaker_id}",
            )
    if tuple(policy.get("tie_break", ())) != FROZEN_TIE_BREAK:
        raise ValueError(f"tie_break must be exactly {FROZEN_TIE_BREAK}")
    if policy.get("require_alignment_review_complete") is not True:
        raise ValueError("require_alignment_review_complete must be true")
    alignment_gate_errors = validate_alignment_gate_contract(
        policy.get("alignment_gate")
    )
    if alignment_gate_errors:
        raise ValueError("invalid alignment_gate: " + "; ".join(alignment_gate_errors))
    tail_ms = policy.get("tail_after_utterance_ms")
    if not _is_finite_number(tail_ms) or abs(float(tail_ms) - 200.0) > 1e-9:
        raise ValueError("tail_after_utterance_ms must be frozen at exactly 200 ms")
    if "policy_hash" in policy:
        declared_hash = policy["policy_hash"]
        hash_input = {key: value for key, value in policy.items() if key != "policy_hash"}
        calculated_hash = sha256_value(hash_input)
        if declared_hash != calculated_hash:
            raise ValueError("selection policy_hash does not match its canonical contents")
        return calculated_hash
    return sha256_value(policy)


def _timing_target_for_row(
    row: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, Any], str, str | None]:
    condition = str(row.get("condition", ""))
    if condition not in CONDITIONS:
        raise ValueError(f"{row.get('candidate_id')}: invalid condition {condition!r}")
    scope = _timing_target_scope(policy)
    if scope == GLOBAL_TARGET_SCOPE:
        return policy["targets_by_condition"][condition], scope, None
    speaker_id = row.get("speaker_id")
    if not isinstance(speaker_id, str) or not speaker_id:
        raise ValueError(
            f"{row.get('candidate_id')}: speaker_specific timing target requires speaker_id"
        )
    targets_by_speaker = policy["targets_by_speaker"]
    if speaker_id not in targets_by_speaker:
        raise ValueError(
            f"{row.get('candidate_id')}: speaker_id {speaker_id!r} is not declared "
            "by the speaker_specific timing policy"
        )
    return targets_by_speaker[speaker_id][condition], scope, speaker_id


def _artifact(row: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    candidate_id = str(row.get("candidate_id", ""))
    artifact = row.get("canonical_candidate")
    if not isinstance(artifact, dict):
        raise ValueError(f"{candidate_id}: missing canonical_candidate artifact")
    path = Path(str(artifact.get("uri", "")))
    if not path.is_file():
        raise ValueError(f"{candidate_id}: canonical candidate does not exist: {path}")
    actual_hash = sha256_file(path)
    if artifact.get("sha256") != actual_hash:
        raise ValueError(f"{candidate_id}: canonical candidate hash mismatch")
    return artifact, path


def _alignment_confidence(row: dict[str, Any]) -> float:
    alignment = row.get("alignment")
    confidence = alignment.get("confidence") if isinstance(alignment, dict) else None
    value = confidence.get("aggregate") if isinstance(confidence, dict) else None
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError(f"{row.get('candidate_id')}: invalid alignment confidence")
    return float(value)


def _script_index(scripts: Iterable[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for script in scripts or ():
        script_id = str(script.get("script_id", ""))
        if not script_id or script_id in result:
            raise ValueError(f"missing or duplicate script_id: {script_id!r}")
        result[script_id] = script
    return result


def _qc_status(row: dict[str, Any]) -> str | None:
    qc = row.get("qc")
    if not isinstance(qc, dict):
        return None
    # Only qc_candidates.py emits automatic_status.  A generic lifecycle
    # ``status=passed`` on a pre-QC row must not satisfy this gate.
    value = qc.get("automatic_status")
    return str(value) if value is not None else None


def _clipping_penalty(row: dict[str, Any]) -> float:
    qc = row.get("qc") if isinstance(row.get("qc"), dict) else {}
    value = qc.get("clipping", qc.get("clipping_detected", False))
    return 1.0 if bool(value) else 0.0


def _noise_penalty(row: dict[str, Any]) -> float:
    qc = row.get("qc") if isinstance(row.get("qc"), dict) else {}
    value = qc.get("noise_penalty")
    if isinstance(value, (int, float)):
        if not 0 <= float(value) <= 1:
            raise ValueError(f"{row.get('candidate_id')}: qc.noise_penalty must be in [0, 1]")
        return float(value)
    return 1.0 if bool(qc.get("noise", qc.get("noise_flag", False))) else 0.0


def _score(row: dict[str, Any], policy: dict[str, Any]) -> tuple[float, dict[str, float]]:
    condition = str(row.get("condition", ""))
    if condition not in CONDITIONS:
        raise ValueError(f"{row.get('candidate_id')}: invalid condition {condition!r}")
    timing = row.get("timing")
    artifact = row.get("canonical_candidate")
    if not isinstance(timing, dict) or not isinstance(artifact, dict):
        raise ValueError(f"{row.get('candidate_id')}: timing/canonical artifact missing")
    timing_errors = validate_timing(condition, timing, float(artifact["duration_ms"]))
    if timing_errors:
        raise ValueError(f"{row.get('candidate_id')}: " + "; ".join(timing_errors))
    targets, _, _ = _timing_target_for_row(row, policy)
    if targets["latency_ms"] is None:
        latency_error = 0.0
    else:
        latency_error = abs(float(timing["actual_latency_ms"]) - float(targets["latency_ms"]))
    post_error = abs(
        float(timing["post_final_value_duration_ms"])
        - float(targets["post_duration_ms"])
    )
    confidence = _alignment_confidence(row)
    components = {
        "latency_error": latency_error / float(policy["scales_ms"]["latency"]),
        "post_duration_error": post_error / float(policy["scales_ms"]["post_duration"]),
        "alignment_confidence_penalty": 1.0 - confidence,
        "clipping_penalty": _clipping_penalty(row),
        "noise_penalty": _noise_penalty(row),
    }
    score = sum(float(policy["weights"][key]) * value for key, value in components.items())
    if not math.isfinite(score):
        raise ValueError(f"{row.get('candidate_id')}: non-finite selection score")
    return score, components


def select_candidate_rows(
    rows: Iterable[dict[str, Any]],
    policy: dict[str, Any],
    scripts: Iterable[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy_hash = validate_selection_policy(policy)
    scripts_by_id = _script_index(scripts)
    target_scope = _timing_target_scope(policy)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    hash_targets: dict[str, str] = {}
    speaker_by_target: dict[str, str] = {}
    for row in rows:
        assert_outcome_blind(row)
        candidate_id = str(row.get("candidate_id", ""))
        target_id = str(row.get("rendition_target_id", ""))
        if not candidate_id or candidate_id in seen_ids:
            raise ValueError(f"missing or duplicate candidate_id: {candidate_id!r}")
        if not target_id:
            raise ValueError(f"{candidate_id}: missing rendition_target_id")
        if row.get("lifecycle_status") != "canonical_candidate":
            raise ValueError(f"{candidate_id}: expected canonical_candidate lifecycle")
        speaker_id = row.get("speaker_id")
        if target_scope == SPEAKER_TARGET_SCOPE:
            # Resolve before QC/ranking so an undeclared speaker cannot be hidden
            # by an otherwise ineligible candidate.
            _timing_target_for_row(row, policy)
        if isinstance(speaker_id, str) and speaker_id:
            prior_speaker = speaker_by_target.get(target_id)
            if prior_speaker is not None and prior_speaker != speaker_id:
                raise ValueError(
                    f"{target_id}: candidates mix speaker IDs "
                    f"{prior_speaker!r} and {speaker_id!r}"
                )
            speaker_by_target[target_id] = speaker_id
        artifact, _ = _artifact(row)
        canonical_hash = str(artifact["sha256"])
        prior_target = hash_targets.get(canonical_hash)
        if prior_target is not None and prior_target != target_id:
            raise ValueError(
                f"canonical hash {canonical_hash} is reused across targets "
                f"{prior_target!r} and {target_id!r}"
            )
        hash_targets[canonical_hash] = target_id
        seen_ids.add(candidate_id)
        by_target[target_id].append(row)

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for target_id in sorted(by_target):
        candidates = sorted(by_target[target_id], key=lambda item: str(item["candidate_id"]))
        representative_by_hash: dict[str, str] = {}
        ranked: list[tuple[tuple[Any, ...], dict[str, Any], float, dict[str, float]]] = []
        excluded: dict[str, str] = {}
        for row in candidates:
            candidate_id = str(row["candidate_id"])
            canonical_hash = str(row["canonical_candidate"]["sha256"])
            if canonical_hash in representative_by_hash:
                excluded[candidate_id] = (
                    "duplicate_canonical_hash_of:" + representative_by_hash[canonical_hash]
                )
                continue
            representative_by_hash[canonical_hash] = candidate_id
            evidence_errors = validate_downstream_alignment_evidence(
                row,
                scripts_by_id.get(str(row.get("script_id"))),
                policy["alignment_gate"],
                actual_canonical_audio_sha256=canonical_hash,
            )
            if evidence_errors:
                excluded[candidate_id] = "alignment_evidence_invalid:" + "; ".join(
                    evidence_errors
                )
                continue
            if _qc_status(row) != "passed":
                excluded[candidate_id] = "automatic_audio_qc_not_passed"
                continue
            try:
                score, components = _score(row, policy)
            except (KeyError, TypeError, ValueError) as error:
                excluded[candidate_id] = "invalid_selection_inputs:" + str(error)
                continue
            confidence = _alignment_confidence(row)
            rank_key = (score, -confidence, canonical_hash, candidate_id)
            ranked.append((rank_key, row, score, components))
        if not ranked:
            reasons = ", ".join(f"{key}={value}" for key, value in sorted(excluded.items()))
            raise ValueError(f"{target_id}: no eligible candidate ({reasons})")
        ranked.sort(key=lambda item: item[0])
        winning_key, winner, winning_score, winning_components = ranked[0]
        winner_id = str(winner["candidate_id"])
        _, _, target_speaker_id = _timing_target_for_row(winner, policy)
        winner_item = dict(winner)
        winner_item["selection"] = {
            "selection_version": SELECTION_VERSION,
            "status": "selected_pending_materialization",
            "policy_version": policy["policy_version"],
            "policy_hash": policy_hash,
            "policy_frozen_at": policy["frozen_at"],
            "alignment_gate_hash": sha256_value(policy["alignment_gate"]),
            "timing_target_scope": target_scope,
            "timing_target_speaker_id": target_speaker_id,
            "selection_score": winning_score,
            "score_components": winning_components,
            "tie_break": list(FROZEN_TIE_BREAK),
            "tie_break_values": [
                winning_score,
                -winning_key[1],
                winning_key[2],
                winning_key[3],
            ],
            "candidate_pool_size": len(candidates),
            "unique_canonical_hash_count": len(representative_by_hash),
            "outcome_blind": True,
        }
        selected.append(winner_item)
        for rank, (_, row, score, components) in enumerate(ranked, 1):
            candidate_id = str(row["candidate_id"])
            is_selected = candidate_id == winner_id
            decisions.append(
                {
                    "schema_version": "2.0.0",
                    "selection_version": SELECTION_VERSION,
                    "rendition_target_id": target_id,
                    "candidate_id": candidate_id,
                    "canonical_sha256": row["canonical_candidate"]["sha256"],
                    "selected": is_selected,
                    "rank": rank,
                    "selection_score": score,
                    "score_components": components,
                    "reason": "selected" if is_selected else "lower_frozen_policy_rank",
                    "policy_hash": policy_hash,
                    "timing_target_scope": target_scope,
                    "timing_target_speaker_id": (
                        str(row["speaker_id"])
                        if target_scope == SPEAKER_TARGET_SCOPE
                        else None
                    ),
                    "outcome_blind": True,
                }
            )
        for candidate_id, reason in sorted(excluded.items()):
            row = next(item for item in candidates if item["candidate_id"] == candidate_id)
            decisions.append(
                {
                    "schema_version": "2.0.0",
                    "selection_version": SELECTION_VERSION,
                    "rendition_target_id": target_id,
                    "candidate_id": candidate_id,
                    "canonical_sha256": row["canonical_candidate"]["sha256"],
                    "selected": False,
                    "rank": None,
                    "selection_score": None,
                    "score_components": None,
                    "reason": reason,
                    "policy_hash": policy_hash,
                    "timing_target_scope": target_scope,
                    "timing_target_speaker_id": (
                        str(row["speaker_id"])
                        if target_scope == SPEAKER_TARGET_SCOPE
                        else None
                    ),
                    "outcome_blind": True,
                }
            )
    return selected, sorted(decisions, key=lambda item: str(item["candidate_id"]))


def materialize_accepted_rows(
    selected_rows: Iterable[dict[str, Any]],
    config: dict[str, Any],
    policy: dict[str, Any],
    audio_root: Path,
    scripts: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    policy_hash = validate_selection_policy(policy)
    scripts_by_id = _script_index(scripts)
    audio_config = config["audio"]
    sample_rate = int(audio_config["canonical_sample_rate"])
    target_rms = float(audio_config["target_active_rms_dbfs"])
    peak_limit = float(audio_config["peak_limit_dbfs"])
    tail_ms = float(policy["tail_after_utterance_ms"])
    output: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for selected in selected_rows:
        assert_outcome_blind(selected)
        target_id = str(selected["rendition_target_id"])
        candidate_id = str(selected["candidate_id"])
        if target_id in seen_targets:
            raise ValueError(f"duplicate selected rendition target: {target_id}")
        seen_targets.add(target_id)
        selection = selected.get("selection")
        if not isinstance(selection, dict) or selection.get("policy_hash") != policy_hash:
            raise ValueError(f"{candidate_id}: selection policy hash mismatch")
        artifact, source_path = _artifact(selected)
        evidence_errors = validate_downstream_alignment_evidence(
            selected,
            scripts_by_id.get(str(selected.get("script_id"))),
            policy["alignment_gate"],
            actual_canonical_audio_sha256=str(artifact["sha256"]),
        )
        if evidence_errors:
            raise ValueError(
                f"{candidate_id}: alignment evidence is invalid: "
                + "; ".join(evidence_errors)
            )
        audio, observed_rate = read_pcm16_mono(source_path)
        if observed_rate != sample_rate:
            raise ValueError(f"{candidate_id}: canonical sample rate must be {sample_rate}")
        utterance_end_ms = selected.get("timing", {}).get("utterance_end_ms")
        if not isinstance(utterance_end_ms, (int, float)) or utterance_end_ms < 0:
            raise ValueError(f"{candidate_id}: missing aligned utterance_end_ms")
        target_samples = round((float(utterance_end_ms) + tail_ms) * sample_rate / 1000.0)
        if target_samples < 1:
            raise ValueError(f"{candidate_id}: invalid accepted target length")
        if audio.size < target_samples:
            trimmed = np.pad(audio, (0, target_samples - audio.size))
            tail_action = "zero_extended_to_fixed_tail"
        else:
            trimmed = audio[:target_samples].copy()
            tail_action = "trimmed_to_fixed_tail" if audio.size > target_samples else "already_fixed_tail"
        normalized, normalization = normalize_audio(trimmed, target_rms, peak_limit)

        accepted_id = accepted_audio_id(target_id)
        target_path = audio_root / f"{accepted_id}.wav"
        if target_path.exists():
            raise FileExistsError(f"immutable accepted utterance already exists: {target_path}")
        write_pcm16_mono(target_path, normalized, sample_rate)
        accepted_duration = duration_ms(normalized, sample_rate)
        tail_actual = accepted_duration - float(utterance_end_ms)
        item = dict(selected)
        item["candidate_id"] = None
        item["selected_candidate_id"] = candidate_id
        item["accepted_audio_id"] = accepted_id
        item["lifecycle_status"] = "accepted"
        item["accepted_utterance"] = {
            "uri": str(target_path.resolve()),
            "sha256": sha256_file(target_path),
            "duration_ms": accepted_duration,
            "sample_rate": sample_rate,
            "channels": 1,
            "sample_width_bytes": 2,
            "timeline": "content_relative",
            "source_canonical_sha256": artifact["sha256"],
        }
        item["selection"] = {
            **selection,
            "status": "materialized_accepted",
            "selected_candidate_id": candidate_id,
            "selected_canonical_uri": str(source_path.resolve()),
            "selected_canonical_sha256": artifact["sha256"],
            "materialization_mode": "gain_normalized_transformed_copy",
            "normalization": {
                **normalization,
                "target_active_rms_dbfs": target_rms,
                "peak_limit_dbfs": peak_limit,
            },
            "tail_policy": {
                "utterance_end_ms": float(utterance_end_ms),
                "fixed_tail_ms": tail_ms,
                "tail_after_utterance_ms_actual": tail_actual,
                "target_samples": target_samples,
                "action": tail_action,
                "leading_coordinate_shift_samples": 0,
                "frame_padding_applied": False,
                "prefix_silence_applied": False,
            },
        }
        qc = dict(selected.get("qc") or {})
        qc["accepted_normalization"] = item["selection"]["normalization"]
        qc["accepted_tail_ms"] = tail_actual
        item["qc"] = qc
        output.append(item)
    return sorted(output, key=lambda item: str(item["accepted_audio_id"]))


def main() -> None:
    args = parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise SystemExit("selection policy must be a JSON object")
    rows = read_jsonl(args.input)
    scripts = read_jsonl(args.scripts)
    selected, decisions = select_candidate_rows(rows, policy, scripts)
    accepted = materialize_accepted_rows(
        selected, read_config(), policy, args.audio_root, scripts
    )
    write_jsonl(args.output, accepted)
    write_jsonl(args.decisions, decisions)
    report = {
        "schema_version": "2.0.0",
        "selection_version": SELECTION_VERSION,
        "policy_hash": validate_selection_policy(policy),
        "candidate_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_count": sum(not row["selected"] for row in decisions),
        "duplicate_hash_rejection_count": sum(
            str(row["reason"]).startswith("duplicate_canonical_hash_of:")
            for row in decisions
        ),
        "output": portable_path(args.output),
        "decisions": portable_path(args.decisions),
    }
    if args.report:
        write_json(args.report, report)
    print(f"Selected and materialized {len(accepted)} accepted utterances -> {args.output}")


if __name__ == "__main__":
    main()
