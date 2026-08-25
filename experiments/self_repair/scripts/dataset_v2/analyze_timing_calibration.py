#!/usr/bin/env python3
"""Summarize the private Edge timing calibration without selecting audio.

This report is deliberately restricted to the non-release
``edge_private_calibration_r1`` track.  It describes QC-passed alignment
measurements and proposes *provisional* timing bands; it never materializes an
accepted rendition and cannot make the Edge track release eligible.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from common import CONDITIONS, DATASET_ROOT, DEFAULT_SCRIPTS, read_config, read_jsonl, sha256_value, write_json


REPORT_VERSION = "2.0.0"
DELAYED_CONDITIONS = (
    "delayed_neutral",
    "delayed_one_dependency",
    "delayed_three_dependencies",
)
DEFAULT_INPUT = DATASET_ROOT / "calibration/edge_private_qc_candidates.jsonl"
DEFAULT_OUTPUT = DATASET_ROOT / "reports/edge_private_timing_calibration.json"
CENTRAL_INTERVAL_QUANTILES = (0.10, 0.90)
MINIMUM_CELL_OBSERVATIONS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze QC-augmented aligned candidates from the private Edge "
            "calibration track; no accepted audio is selected."
        )
    )
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _rounded(value: float) -> float:
    return round(float(value), 3)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def describe(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "status": "no_eligible_observations"}
    return {
        "count": len(ordered),
        "status": "observed",
        "min": _rounded(ordered[0]),
        "p10": _rounded(_percentile(ordered, 0.10)),
        "p25": _rounded(_percentile(ordered, 0.25)),
        "median": _rounded(statistics.median(ordered)),
        "mean": _rounded(statistics.fmean(ordered)),
        "p75": _rounded(_percentile(ordered, 0.75)),
        "p90": _rounded(_percentile(ordered, 0.90)),
        "max": _rounded(ordered[-1]),
        "population_sd": _rounded(statistics.pstdev(ordered)),
    }


def _timing_errors(condition: str, timing: Any) -> list[str]:
    if not isinstance(timing, dict):
        return ["timing_missing"]
    end = _number(timing.get("utterance_end_ms"))
    new_offset = _number(timing.get("new_value_offset_ms"))
    post = _number(timing.get("post_final_value_duration_ms"))
    errors: list[str] = []
    if end is None or end <= 0:
        errors.append("utterance_end_invalid")
    if new_offset is None or new_offset < 0:
        errors.append("new_value_offset_invalid")
    if post is None or post < 0:
        errors.append("post_final_value_duration_invalid")
    if end is not None and new_offset is not None and post is not None:
        if abs(post - (end - new_offset)) > 1.0:
            errors.append("post_final_value_duration_inconsistent")

    latency = _number(timing.get("actual_latency_ms"))
    if condition == "clean_final":
        if timing.get("actual_latency_ms") is not None:
            errors.append("clean_actual_latency_must_be_null")
    else:
        old_offset = _number(timing.get("old_value_offset_ms"))
        cue_onset = _number(timing.get("repair_cue_onset_ms"))
        if latency is None or latency < 0:
            errors.append("actual_latency_invalid")
        if old_offset is None or cue_onset is None:
            errors.append("latency_boundary_missing")
        elif latency is not None and abs(latency - (cue_onset - old_offset)) > 1.0:
            errors.append("actual_latency_inconsistent")
    return errors


def _candidate_voice(row: dict[str, Any]) -> str:
    top_level = row.get("voice")
    synthesis = row.get("synthesis")
    nested = synthesis.get("voice") if isinstance(synthesis, dict) else None
    values = {str(value) for value in (top_level, nested) if isinstance(value, str) and value}
    if len(values) != 1:
        raise ValueError(
            f"{row.get('candidate_id')}: candidate must declare one consistent voice"
        )
    return next(iter(values))


def _validate_private_scope(
    candidates: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[str, str, dict[str, str]]:
    calibration = config.get("engineering_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("engineering_calibration config is missing")
    if calibration.get("release_eligible") is not False:
        raise ValueError("private timing analyzer requires release_eligible=false")
    source_track_id = str(calibration.get("source_track_id", ""))
    provider = str(calibration.get("provider", ""))
    if not source_track_id or provider != "edge_private_smoke":
        raise ValueError("private timing analyzer is restricted to edge_private_smoke")
    speakers = calibration.get("speakers")
    if not isinstance(speakers, list) or not speakers:
        raise ValueError("engineering_calibration speakers are missing")
    voice_by_speaker = {
        str(item["speaker_id"]): str(item["voice"])
        for item in speakers
        if isinstance(item, dict) and item.get("speaker_id") and item.get("voice")
    }
    if len(voice_by_speaker) != len(speakers):
        raise ValueError("engineering_calibration speakers must be unique and complete")

    for row in candidates:
        candidate_id = str(row.get("candidate_id", ""))
        if row.get("source_track_id") != source_track_id:
            raise ValueError(f"{candidate_id}: candidate is outside the private source track")
        if row.get("inferential_role") != "engineering_calibration_only":
            raise ValueError(f"{candidate_id}: candidate is not engineering calibration only")
        if row.get("lifecycle_status") != "canonical_candidate":
            raise ValueError(f"{candidate_id}: expected lifecycle_status=canonical_candidate")
        for field in (
            "selected_candidate_id",
            "accepted_audio_id",
            "accepted_utterance",
            "prepared_stimulus",
        ):
            if row.get(field) is not None:
                raise ValueError(f"{candidate_id}: {field} must remain null in calibration")
        synthesis = row.get("synthesis")
        observed_provider = synthesis.get("provider") if isinstance(synthesis, dict) else None
        if observed_provider != provider:
            raise ValueError(f"{candidate_id}: unexpected provider {observed_provider!r}")
        speaker_id = str(row.get("speaker_id", ""))
        if speaker_id not in voice_by_speaker:
            raise ValueError(f"{candidate_id}: unexpected calibration speaker {speaker_id!r}")
        if _candidate_voice(row) != voice_by_speaker[speaker_id]:
            raise ValueError(f"{candidate_id}: voice does not match calibration speaker")
    return source_track_id, provider, voice_by_speaker


def _script_index(scripts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for script in scripts:
        script_id = str(script.get("script_id", ""))
        if not script_id or script_id in result:
            raise ValueError(f"missing or duplicate script_id: {script_id!r}")
        result[script_id] = script
    return result


def _index_candidates(
    candidates: list[dict[str, Any]], scripts: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, str, str, str]]]:
    by_id: dict[str, dict[str, Any]] = {}
    target_identity: dict[str, tuple[str, str, str, str]] = {}
    for row in candidates:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in by_id:
            raise ValueError(f"missing or duplicate candidate_id: {candidate_id!r}")
        script_id = str(row.get("script_id", ""))
        if script_id not in scripts:
            raise ValueError(f"{candidate_id}: unknown script_id {script_id!r}")
        condition = str(row.get("condition", ""))
        if condition != scripts[script_id].get("condition") or condition not in CONDITIONS:
            raise ValueError(f"{candidate_id}: candidate/script condition mismatch")
        target_id = str(row.get("rendition_target_id", ""))
        if not target_id:
            raise ValueError(f"{candidate_id}: rendition_target_id is missing")
        identity = (script_id, str(row["speaker_id"]), _candidate_voice(row), condition)
        if target_id in target_identity and target_identity[target_id] != identity:
            raise ValueError(f"{candidate_id}: rendition target identity changed across attempts")
        target_identity[target_id] = identity
        by_id[candidate_id] = row
    return by_id, target_identity


def _artifact_duration(row: dict[str, Any]) -> float | None:
    qc = row.get("qc")
    if isinstance(qc, dict) and isinstance(qc.get("metrics"), dict):
        value = _number(qc["metrics"].get("duration_ms"))
        if value is not None and value > 0:
            return value
    artifact = row.get("canonical_candidate")
    if isinstance(artifact, dict):
        value = _number(artifact.get("duration_ms"))
        if value is not None and value > 0:
            return value
    return None


def _failure_reasons(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    qc_messages: list[str] = []
    qc = row.get("qc")
    automatic_status = qc.get("automatic_status") if isinstance(qc, dict) else None
    if automatic_status == "failed":
        reasons.append("automatic_qc_failed")
    elif automatic_status != "passed":
        reasons.append("automatic_qc_not_passed")
    if isinstance(qc, dict) and isinstance(qc.get("errors"), list):
        qc_messages = sorted(str(value) for value in qc["errors"] if str(value))

    timing_errors = _timing_errors(str(row["condition"]), row.get("timing"))
    if timing_errors == ["timing_missing"]:
        reasons.append("timing_missing")
    elif timing_errors:
        reasons.append("timing_invalid")
    if _artifact_duration(row) is None:
        reasons.append("canonical_duration_missing")
    return sorted(set(reasons)), [*qc_messages, *timing_errors]


def _metric_values(
    eligible: list[dict[str, Any]], speaker_id: str, condition: str
) -> dict[str, dict[str, Any]]:
    rows = [
        row
        for row in eligible
        if row["speaker_id"] == speaker_id and row["condition"] == condition
    ]
    latency = [
        float(row["timing"]["actual_latency_ms"])
        for row in rows
        if _number(row["timing"].get("actual_latency_ms")) is not None
    ]
    return {
        "actual_latency_ms": describe(latency),
        "post_final_value_duration_ms": describe(
            float(row["timing"]["post_final_value_duration_ms"]) for row in rows
        ),
        "utterance_duration_ms": describe(
            float(row["timing"]["utterance_end_ms"]) for row in rows
        ),
        "canonical_file_duration_ms": describe(
            value for row in rows if (value := _artifact_duration(row)) is not None
        ),
    }


def _balance_cell(
    targets: list[tuple[str, dict[str, Any], str]], condition: str, speaker_id: str | None
) -> dict[str, Any]:
    selected = [
        (target_id, script, voice_speaker)
        for target_id, script, voice_speaker in targets
        if script["condition"] == condition
        and (speaker_id is None or voice_speaker == speaker_id)
    ]
    identities: Counter[str] = Counter()
    binding_counts: Counter[str] = Counter()
    positions: Counter[str] = Counter()
    for target_id, script, _ in selected:
        pre_units = [str(value) for value in script.get("pre_repair_units", [])]
        if len(pre_units) != len(set(pre_units)):
            raise ValueError(f"{target_id}: duplicate pre-repair units")
        for unit_id in pre_units:
            if unit_id.startswith("D"):
                binding_counts["dependent"] += 1
            elif unit_id.startswith("N"):
                binding_counts["neutral"] += 1
            else:
                raise ValueError(f"{target_id}: unknown pre-repair unit {unit_id!r}")
            identities[unit_id] += 1
        if condition == "delayed_one_dependency":
            dependent = [value for value in pre_units if value.startswith("D")]
            if len(dependent) != 1:
                raise ValueError(f"{target_id}: delayed-one must have one dependent unit")
            actual_position = pre_units.index(dependent[0]) + 1
            declared_position = script.get("one_dependency_pre_position")
            if declared_position != actual_position:
                raise ValueError(f"{target_id}: dependent pre-position is inconsistent")
            positions[str(actual_position)] += 1
    return {
        "condition": condition,
        "speaker_id": speaker_id,
        "unique_target_count": len(selected),
        "pre_repair_unit_counts_by_binding": {
            key: binding_counts.get(key, 0) for key in ("dependent", "neutral")
        },
        "pre_repair_unit_counts_by_identity": {
            key: identities[key] for key in sorted(identities)
        },
        "dependent_pre_position_counts": {
            key: positions.get(key, 0) for key in ("1", "2", "3")
        },
    }


def _balance_report(
    target_identity: dict[str, tuple[str, str, str, str]],
    scripts: dict[str, dict[str, Any]],
    speaker_ids: list[str],
) -> dict[str, Any]:
    targets = [
        (target_id, scripts[identity[0]], identity[1])
        for target_id, identity in sorted(target_identity.items())
    ]
    by_condition = [_balance_cell(targets, condition, None) for condition in CONDITIONS]
    by_voice_condition = [
        _balance_cell(targets, condition, speaker_id)
        for speaker_id in speaker_ids
        for condition in CONDITIONS
    ]
    one_dependency = next(
        item for item in by_condition if item["condition"] == "delayed_one_dependency"
    )
    position_counts = list(one_dependency["dependent_pre_position_counts"].values())
    identity_counts = [
        one_dependency["pre_repair_unit_counts_by_identity"].get(key, 0)
        for key in ("D1", "D2", "D3")
    ]
    return {
        "basis": "unique_rendition_targets_not_candidate_attempts",
        "unique_target_count": len(targets),
        "by_condition": by_condition,
        "by_voice_condition": by_voice_condition,
        "delayed_one_dependency_balance": {
            "dependent_identity_counts": dict(zip(("D1", "D2", "D3"), identity_counts)),
            "dependent_pre_position_counts": dict(
                one_dependency["dependent_pre_position_counts"]
            ),
            "identity_count_range": max(identity_counts, default=0) - min(identity_counts, default=0),
            "pre_position_count_range": max(position_counts, default=0) - min(position_counts, default=0),
        },
    }


def _intersection(intervals: list[tuple[float, float]], label: str) -> dict[str, Any]:
    if not intervals:
        return {"status": "insufficient_data", "interval_kind": label}
    lower = max(value[0] for value in intervals)
    upper = min(value[1] for value in intervals)
    gap = lower - upper
    if gap > 0:
        return {
            "status": "no_common_overlap",
            "interval_kind": label,
            "lower_bound_ms": _rounded(lower),
            "upper_bound_ms": _rounded(upper),
            "separation_gap_ms": _rounded(gap),
            "provisional_target_ms": None,
            "provisional_tolerance_ms": None,
        }
    width = upper - lower
    return {
        "status": "common_overlap" if width > 0 else "point_overlap",
        "interval_kind": label,
        "lower_bound_ms": _rounded(lower),
        "upper_bound_ms": _rounded(upper),
        "width_ms": _rounded(width),
        "provisional_target_ms": _rounded((lower + upper) / 2.0),
        "provisional_tolerance_ms": _rounded(width / 2.0),
    }


def _overlap_for_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [
        f"{cell['speaker_id']}::{cell['condition']}"
        for cell in cells
        if len(cell["values"]) < MINIMUM_CELL_OBSERVATIONS
    ]
    public_cells = [
        {
            "speaker_id": cell["speaker_id"],
            "voice": cell["voice"],
            "condition": cell["condition"],
            "observation_count": len(cell["values"]),
            "observed_range_ms": (
                [_rounded(min(cell["values"])), _rounded(max(cell["values"]))]
                if cell["values"]
                else None
            ),
            "central_80_interval_ms": (
                [
                    _rounded(_percentile(sorted(cell["values"]), CENTRAL_INTERVAL_QUANTILES[0])),
                    _rounded(_percentile(sorted(cell["values"]), CENTRAL_INTERVAL_QUANTILES[1])),
                ]
                if cell["values"]
                else None
            ),
        }
        for cell in cells
    ]
    if missing:
        return {
            "status": "insufficient_data",
            "minimum_observations_per_cell": MINIMUM_CELL_OBSERVATIONS,
            "missing_or_undersampled_cells": missing,
            "cells": public_cells,
            "central_80_common_overlap": {"status": "insufficient_data"},
            "observed_range_common_overlap": {"status": "insufficient_data"},
        }
    central = [
        (
            _percentile(sorted(cell["values"]), CENTRAL_INTERVAL_QUANTILES[0]),
            _percentile(sorted(cell["values"]), CENTRAL_INTERVAL_QUANTILES[1]),
        )
        for cell in cells
    ]
    observed = [(min(cell["values"]), max(cell["values"])) for cell in cells]
    central_overlap = _intersection(central, "central_80_p10_p90")
    observed_overlap = _intersection(observed, "observed_min_max")
    usable = central_overlap["status"] == "common_overlap"
    return {
        "status": "provisional_common_overlap" if usable else "no_usable_central_overlap",
        "minimum_observations_per_cell": MINIMUM_CELL_OBSERVATIONS,
        "missing_or_undersampled_cells": [],
        "cells": public_cells,
        "central_80_common_overlap": central_overlap,
        "observed_range_common_overlap": observed_overlap,
    }


def _overlap_recommendation(
    eligible: list[dict[str, Any]],
    metric: str,
    voice_by_speaker: dict[str, str],
    coverage_complete: bool,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for speaker_id, voice in voice_by_speaker.items():
        for condition in DELAYED_CONDITIONS:
            values = [
                float(row["timing"][metric])
                for row in eligible
                if row["speaker_id"] == speaker_id
                and row["condition"] == condition
                and _number(row["timing"].get(metric)) is not None
            ]
            cells.append(
                {
                    "speaker_id": speaker_id,
                    "voice": voice,
                    "condition": condition,
                    "values": sorted(values),
                }
            )
    global_result = _overlap_for_cells(cells)
    by_voice = [
        {
            "speaker_id": speaker_id,
            "voice": voice_by_speaker[speaker_id],
            **_overlap_for_cells(
                [cell for cell in cells if cell["speaker_id"] == speaker_id]
            ),
        }
        for speaker_id in voice_by_speaker
    ]
    central = global_result["central_80_common_overlap"]
    if central.get("status") == "common_overlap":
        recommendation = {
            "mode": "single_global_provisional_target",
            "target_ms": central["provisional_target_ms"],
            "tolerance_ms": central["provisional_tolerance_ms"],
        }
    elif all(
        item["central_80_common_overlap"].get("status") == "common_overlap"
        for item in by_voice
    ):
        recommendation = {
            "mode": "voice_specific_provisional_targets",
            "targets": [
                {
                    "speaker_id": item["speaker_id"],
                    "voice": item["voice"],
                    "target_ms": item["central_80_common_overlap"]["provisional_target_ms"],
                    "tolerance_ms": item["central_80_common_overlap"]["provisional_tolerance_ms"],
                }
                for item in by_voice
            ],
        }
    else:
        recommendation = {
            "mode": "no_provisional_target",
            "target_ms": None,
            "tolerance_ms": None,
            "required_action": "revise timing design or collect more private calibration observations",
        }
    recommendation.update(
        {
            "provisional": True,
            "private_engineering_only": True,
            "production_timing_freeze_eligible": False,
            "coverage_complete": coverage_complete,
            "required_production_follow_up": (
                "repeat and validate with the approved release provider and independent alignment"
            ),
        }
    )
    return {
        "metric": metric,
        "conditions": list(DELAYED_CONDITIONS),
        "estimator": {
            "cell_interval": "linear-interpolated p10 to p90",
            "common_overlap": "maximum cell lower bound to minimum cell upper bound",
            "target": "common-overlap midpoint",
            "tolerance": "common-overlap half-width",
            "minimum_observations_per_voice_condition_cell": MINIMUM_CELL_OBSERVATIONS,
        },
        "global_across_voices_and_conditions": global_result,
        "by_voice_across_conditions": by_voice,
        "recommendation": recommendation,
    }


def analyze_calibration(
    scripts: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    source_track_id, provider, voice_by_speaker = _validate_private_scope(candidates, config)
    scripts_by_id = _script_index(scripts)
    ordered_scripts = [scripts_by_id[key] for key in sorted(scripts_by_id)]
    candidates_by_id, target_identity = _index_candidates(candidates, scripts_by_id)
    ordered_rows = [candidates_by_id[key] for key in sorted(candidates_by_id)]

    failure_ids: defaultdict[str, list[str]] = defaultdict(list)
    detail_counts: Counter[str] = Counter()
    candidate_reasons: dict[str, list[str]] = {}
    eligible: list[dict[str, Any]] = []
    for row in ordered_rows:
        reasons, details = _failure_reasons(row)
        candidate_id = str(row["candidate_id"])
        candidate_reasons[candidate_id] = reasons
        for reason in reasons:
            failure_ids[reason].append(candidate_id)
        detail_counts.update(details)
        if not reasons:
            eligible.append(row)

    speaker_order = list(voice_by_speaker)
    per_voice_condition: list[dict[str, Any]] = []
    for speaker_id in speaker_order:
        for condition in CONDITIONS:
            all_cell = [
                row
                for row in ordered_rows
                if row["speaker_id"] == speaker_id and row["condition"] == condition
            ]
            eligible_cell = [
                row
                for row in eligible
                if row["speaker_id"] == speaker_id and row["condition"] == condition
            ]
            cell_failures: Counter[str] = Counter(
                reason for row in all_cell for reason in candidate_reasons[str(row["candidate_id"])]
            )
            per_voice_condition.append(
                {
                    "speaker_id": speaker_id,
                    "voice": voice_by_speaker[speaker_id],
                    "condition": condition,
                    "candidate_count": len(all_cell),
                    "eligible_candidate_count": len(eligible_cell),
                    "excluded_candidate_count": len(all_cell) - len(eligible_cell),
                    "failure_reason_counts": dict(sorted(cell_failures.items())),
                    "metrics": _metric_values(eligible, speaker_id, condition),
                }
            )

    calibration = config["engineering_calibration"]
    attempts = int(calibration["candidates_per_target"])
    expected_selected_scenarios = 3
    expected_targets = expected_selected_scenarios * 2 * len(CONDITIONS) * len(voice_by_speaker)
    expected_candidates = expected_targets * attempts
    attempts_by_target = Counter(str(row["rendition_target_id"]) for row in ordered_rows)
    scenario_ids = sorted(
        {str(scripts_by_id[identity[0]]["scenario_id"]) for identity in target_identity.values()}
    )
    all_expected_cells_observed = all(
        item["candidate_count"] > 0 for item in per_voice_condition
    )
    collection_complete = (
        len(target_identity) == expected_targets
        and len(ordered_rows) == expected_candidates
        and len(scenario_ids) == expected_selected_scenarios
        and all(value == attempts for value in attempts_by_target.values())
        and all_expected_cells_observed
    )
    usable_complete = collection_complete and len(eligible) == len(ordered_rows)

    failed_candidates = sorted(
        candidate_id for candidate_id, reasons in candidate_reasons.items() if reasons
    )
    report = {
        "schema_version": "2.0.0",
        "report_version": REPORT_VERSION,
        "report_kind": "edge_private_timing_calibration",
        "status": "provisional_private_engineering_only",
        "provisional": True,
        "private": True,
        "release_eligible": False,
        "accepted_release_audio_selected": False,
        "source_scope": {
            "source_track_id": source_track_id,
            "provider": provider,
            "inferential_role": "engineering_calibration_only",
            "provider_release_eligibility_inferred": False,
        },
        "provenance": {
            "script_manifest_content_sha256": sha256_value(ordered_scripts),
            "candidate_manifest_content_sha256": sha256_value(ordered_rows),
            "configuration_content_sha256": sha256_value(config),
            "uses_observed_candidate_timing_only": True,
            "uses_config_illustrative_target_latency": False,
            "config_illustrative_target_latency_ms_ignored": config.get("timing", {}).get(
                "illustrative_target_latency_ms"
            ),
        },
        "coverage": {
            "expected_selected_scenario_count": expected_selected_scenarios,
            "observed_scenario_ids": scenario_ids,
            "expected_speaker_count": len(voice_by_speaker),
            "observed_speaker_ids": sorted(
                {str(row["speaker_id"]) for row in ordered_rows}
            ),
            "expected_condition_count": len(CONDITIONS),
            "observed_conditions": sorted(
                {str(row["condition"]) for row in ordered_rows}
            ),
            "expected_unique_target_count": expected_targets,
            "observed_unique_target_count": len(target_identity),
            "expected_candidate_count": expected_candidates,
            "observed_candidate_count": len(ordered_rows),
            "eligible_candidate_count": len(eligible),
            "candidates_per_target_expected": attempts,
            "targets_with_expected_candidate_count": sum(
                value == attempts for value in attempts_by_target.values()
            ),
            "all_expected_voice_condition_cells_observed": all_expected_cells_observed,
            "collection_complete": collection_complete,
            "all_candidates_usable": len(eligible) == len(ordered_rows),
            "usable_private_calibration_complete": usable_complete,
        },
        "failures": {
            "excluded_candidate_count": len(failed_candidates),
            "excluded_candidate_ids": failed_candidates,
            "by_reason": [
                {
                    "reason": reason,
                    "count": len(failure_ids[reason]),
                    "candidate_ids": sorted(failure_ids[reason]),
                }
                for reason in sorted(failure_ids)
            ],
            "detail_counts": [
                {"detail": detail, "count": count}
                for detail, count in sorted(detail_counts.items())
            ],
        },
        "per_voice_condition": per_voice_condition,
        "design_balance": _balance_report(
            target_identity, scripts_by_id, speaker_order
        ),
        "empirical_common_overlap_recommendations": {
            "delayed_actual_latency": _overlap_recommendation(
                eligible,
                "actual_latency_ms",
                voice_by_speaker,
                usable_complete,
            ),
            "delayed_post_final_value_duration": _overlap_recommendation(
                eligible,
                "post_final_value_duration_ms",
                voice_by_speaker,
                usable_complete,
            ),
        },
        "production_gate": {
            "timing_policy_frozen": False,
            "release_audio_ready": False,
            "required_before_freeze": [
                "approved production TTS provider",
                "independent forced alignment",
                "replication of the empirical overlap on production voices",
            ],
        },
    }
    return report


def main() -> None:
    args = parse_args()
    config = read_config(args.config) if args.config else read_config()
    report = analyze_calibration(read_jsonl(args.scripts), read_jsonl(args.input), config)
    write_json(args.output, report)
    overlap = report["empirical_common_overlap_recommendations"]["delayed_actual_latency"]
    mode = overlap["recommendation"]["mode"]
    print(
        f"Analyzed {report['coverage']['observed_candidate_count']} private candidates; "
        f"latency recommendation={mode} -> {args.output}"
    )


if __name__ == "__main__":
    main()
