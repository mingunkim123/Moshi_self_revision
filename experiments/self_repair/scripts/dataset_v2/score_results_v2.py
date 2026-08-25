#!/usr/bin/env python3
"""Resolve v2 annotations and compute the preregistered audio-level seed counts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

try:
    from .build_eval_adapter import eval_identity, validate_eval_trials
    from .common import DATASET_ROOT, DEFAULT_CONFIG, read_config, read_jsonl, write_json, write_jsonl
    from .make_annotation_sheet_v2 import RELATIONS, resolve_annotations
except ImportError:  # pragma: no cover - exercised by direct CLI use.
    from build_eval_adapter import eval_identity, validate_eval_trials
    from common import DATASET_ROOT, DEFAULT_CONFIG, read_config, read_jsonl, write_json, write_jsonl
    from make_annotation_sheet_v2 import RELATIONS, resolve_annotations


SCHEMA_VERSION = "2.0.0"
DEFAULT_EVAL_TRIALS = DATASET_ROOT / "evaluation/eval_trials.jsonl"
DEFAULT_ANNOTATIONS = DATASET_ROOT / "annotations/annotations.jsonl"
DEFAULT_ACCEPTED = DATASET_ROOT / "manifests/accepted_audio.jsonl"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "evaluation/scored"
DELAYED_CONDITIONS = (
    "delayed_neutral",
    "delayed_one_dependency",
    "delayed_three_dependencies",
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260826


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sample")
    if probability < 0.0 or probability > 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - fraction)
        + float(sorted_values[upper]) * fraction
    )


def _paired_t_normal_approx_p(values: Sequence[float]) -> tuple[float, float | None]:
    """Two-sided normal approximation to the paired-scenario t statistic."""

    if len(values) < 2:
        return 1.0, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance == 0.0:
        if mean == 0.0:
            return 1.0, 0.0
        # The mathematical statistic is signed infinity.  Store null rather than a
        # non-standard JSON Infinity token; the zero p-value preserves the result.
        return 0.0, None
    statistic = mean / math.sqrt(variance / len(values))
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return min(1.0, max(0.0, p_value)), statistic


def _paired_sign_test_p(values: Sequence[float]) -> float:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    nonzero = positive + negative
    if nonzero == 0:
        return 1.0
    tail = min(positive, negative)
    probability = sum(math.comb(nonzero, count) for count in range(tail + 1)) / (2**nonzero)
    return min(1.0, 2.0 * probability)


def _holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    family_size = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (family_size - index) * value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def frozen_contrast_inference(
    scenario_scores: Sequence[dict[str, Any]],
    *,
    required_scenario_count: int = 30,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    strict: bool = True,
) -> dict[str, Any]:
    """Run the frozen equal-scenario delayed-condition contrasts.

    This intentionally stops at the preregistered cluster bootstrap and paired
    sensitivity checks.  The secondary GLMM requires an external statistical runtime.
    """

    if bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("the frozen analysis requires exactly 10,000 bootstrap replicates")
    if bootstrap_seed != BOOTSTRAP_SEED:
        raise ValueError("the frozen analysis requires bootstrap seed 20260826")
    rates: dict[str, dict[str, float]] = defaultdict(dict)
    duplicate_cells: set[tuple[str, str]] = set()
    invalid_audio_cells: list[tuple[str, str, Any]] = []
    for row in scenario_scores:
        scenario_id = str(row.get("scenario_id", ""))
        condition = str(row.get("condition", ""))
        if condition not in DELAYED_CONDITIONS:
            continue
        cell = (scenario_id, condition)
        if condition in rates[scenario_id]:
            duplicate_cells.add(cell)
        rate = row.get("final_target_rate")
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            raise ValueError(f"scenario cell {cell} has an invalid final_target_rate")
        if row.get("audio_units") != 4:
            invalid_audio_cells.append((scenario_id, condition, row.get("audio_units")))
        rates[scenario_id][condition] = float(rate)
    if duplicate_cells:
        raise ValueError(f"duplicate scenario/condition cells: {sorted(duplicate_cells)}")
    if strict and invalid_audio_cells:
        raise ValueError(
            "frozen contrasts require four direction × speaker audio units per "
            f"scenario/condition; examples={invalid_audio_cells[:5]}"
        )

    complete_scenarios = sorted(
        scenario_id
        for scenario_id, condition_rates in rates.items()
        if set(condition_rates) == set(DELAYED_CONDITIONS)
    )
    all_scenarios = sorted(rates)
    complete = (
        len(all_scenarios) == required_scenario_count
        and complete_scenarios == all_scenarios
    )
    if not complete:
        result = {
            "status": "not_evaluable",
            "reason": "incomplete delayed-condition scenario matrix",
            "required_scenario_clusters": required_scenario_count,
            "observed_scenario_clusters": len(all_scenarios),
            "complete_scenario_clusters": len(complete_scenarios),
        }
        if strict:
            raise ValueError(
                "frozen contrasts require a complete delayed-condition matrix for "
                f"{required_scenario_count} scenarios"
            )
        return result

    contrast_specs = {
        "delayed_three_minus_neutral": (
            "delayed_three_dependencies",
            "delayed_neutral",
            "primary",
        ),
        "delayed_one_minus_neutral": (
            "delayed_one_dependency",
            "delayed_neutral",
            "key_secondary",
        ),
    }
    differences: dict[str, list[float]] = {}
    for name, (left, right, _) in contrast_specs.items():
        differences[name] = [
            rates[scenario_id][left] - rates[scenario_id][right]
            for scenario_id in complete_scenarios
        ]

    rng = random.Random(bootstrap_seed)
    bootstrap_values = {name: [] for name in contrast_specs}
    scenario_count = len(complete_scenarios)
    for _ in range(bootstrap_replicates):
        sampled_indexes = [rng.randrange(scenario_count) for _ in range(scenario_count)]
        for name, values in differences.items():
            bootstrap_values[name].append(
                sum(values[index] for index in sampled_indexes) / scenario_count
            )

    raw_t_p: dict[str, float] = {}
    statistics: dict[str, float | None] = {}
    for name, values in differences.items():
        raw_t_p[name], statistics[name] = _paired_t_normal_approx_p(values)
    adjusted = _holm_adjust(raw_t_p)

    contrasts: dict[str, dict[str, Any]] = {}
    for name, (_, _, role) in contrast_specs.items():
        values = differences[name]
        point = sum(values) / len(values)
        boot = sorted(bootstrap_values[name])
        contrasts[name] = {
            "role": role,
            "scenario_clusters": len(values),
            "equal_weight_rate_difference": point,
            "equal_weight_percentage_point_difference": point * 100.0,
            "bootstrap_percentile_95_ci_rate": [
                _percentile(boot, 0.025),
                _percentile(boot, 0.975),
            ],
            "bootstrap_percentile_95_ci_percentage_points": [
                _percentile(boot, 0.025) * 100.0,
                _percentile(boot, 0.975) * 100.0,
            ],
            "paired_t_normal_approx": {
                "statistic": statistics[name],
                "raw_two_sided_p": raw_t_p[name],
                "holm_adjusted_two_sided_p": adjusted[name],
            },
            "paired_exact_sign_test_two_sided_p": _paired_sign_test_p(values),
        }
    return {
        "status": "evaluated",
        "estimand": "equal-weight mean of scenario-level condition-rate differences",
        "primary_cluster": "scenario_id",
        "bootstrap": {
            "method": "scenario_cluster_percentile",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
        },
        "multiplicity": {
            "method": "Holm",
            "family": list(contrast_specs),
            "family_alpha": 0.05,
            "applied_to": "paired_t_normal_approx_two_sided_p",
        },
        "contrasts": contrasts,
        "secondary_model": {
            "status": "not_run",
            "method": "preregistered GLMM",
            "reason": "requires an external statistical runtime and actual timing covariates",
        },
    }


def _accepted_index(
    accepted_audio: Iterable[dict[str, Any]], expected_audio_count: int
) -> dict[str, dict[str, Any]]:
    rows = list(accepted_audio)
    if len(rows) != expected_audio_count:
        raise ValueError(
            f"expected {expected_audio_count} accepted audio rows, found {len(rows)}"
        )
    required = {
        "accepted_audio_id",
        "rendition_target_id",
        "matched_audio_bundle_id",
        "text_bundle_id",
        "scenario_id",
        "direction_id",
        "condition",
        "source_track_id",
        "speaker_id",
    }
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"accepted audio row {index} is missing fields: {missing}")
        accepted_id = row["accepted_audio_id"]
        if not isinstance(accepted_id, str) or not accepted_id:
            raise ValueError(f"accepted audio row {index} has an invalid accepted_audio_id")
        if accepted_id in indexed:
            raise ValueError(f"duplicate accepted_audio_id: {accepted_id}")
        if row.get("lifecycle_status") not in ("accepted", "prepared"):
            raise ValueError(
                f"accepted audio {accepted_id}: lifecycle_status must be accepted/prepared"
            )
        indexed[accepted_id] = row
    return indexed


def _annotation_disagreement_counts(
    annotations: Sequence[dict[str, Any]],
) -> tuple[int, int]:
    primary_by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        if not row.get("adjudicator"):
            primary_by_trial[str(row["eval_trial_id"])].append(row)
    disagreements = 0
    for rows in primary_by_trial.values():
        if len(rows) != 2:
            continue  # resolve_annotations will produce the precise validation error.
        signatures = {
            (
                row["overall_label"],
                tuple(row["relation_labels"][relation] for relation in RELATIONS),
                row["final_target_correct"],
                row["stale_state_error"],
                row.get("assistant_started_before_repair"),
            )
            for row in rows
        }
        disagreements += int(len(signatures) > 1)
    return disagreements, len(primary_by_trial)


def _gate_metrics(
    scenario_scores: Sequence[dict[str, Any]],
    eval_trials: Sequence[dict[str, Any]],
    resolved: Mapping[str, dict[str, Any]],
    accepted: Mapping[str, dict[str, Any]],
    *,
    expected_scenario_count: int,
) -> dict[str, Any]:
    condition_scenario_rates: dict[str, list[float]] = defaultdict(list)
    for row in scenario_scores:
        condition_scenario_rates[str(row["condition"])].append(
            float(row["final_target_rate"])
        )
    equal_weight_rates = {
        condition: sum(rates) / len(rates)
        for condition, rates in condition_scenario_rates.items()
        if rates
    }
    clean_rate = equal_weight_rates.get("clean_final")
    immediate_rate = equal_weight_rates.get("immediate_repair")

    # Only unintelligible output is technically unscorable.  no_speech,
    # clarification, and no_evidence remain conservative failures rather than being
    # dropped from the denominator.
    scorable = sum(
        row["overall_label"] != "unintelligible" for row in resolved.values()
    )
    total = len(resolved)
    coverage = scorable / total if total else None
    core_evaluable = (
        clean_rate is not None
        and immediate_rate is not None
        and coverage is not None
        and len(condition_scenario_rates.get("clean_final", []))
        == expected_scenario_count
        and len(condition_scenario_rates.get("immediate_repair", []))
        == expected_scenario_count
    )
    core_passed = (
        clean_rate >= 0.80 and immediate_rate >= 0.70 and coverage >= 0.90
        if core_evaluable
        else None
    )

    early_values: dict[str, list[bool]] = {condition: [] for condition in DELAYED_CONDITIONS}
    early_missing = Counter()
    for trial in eval_trials:
        metadata = accepted[str(trial["accepted_audio_id"])]
        condition = str(metadata["condition"])
        if condition not in early_values:
            continue
        decision = resolved[str(trial["eval_trial_id"])]
        value = decision.get("assistant_started_before_repair")
        if value is None:
            early_missing[condition] += 1
        else:
            early_values[condition].append(bool(value))
    early_rates = {
        condition: (
            sum(values) / len(values) if values else None
        )
        for condition, values in early_values.items()
    }
    causal_evaluable = all(
        early_rates[condition] is not None and early_missing[condition] == 0
        for condition in DELAYED_CONDITIONS
    )
    if causal_evaluable:
        pairwise_differences = {
            f"{first}_minus_{second}": early_rates[first] - early_rates[second]
            for index, first in enumerate(DELAYED_CONDITIONS)
            for second in DELAYED_CONDITIONS[index + 1 :]
        }
        maximum_difference = max(abs(value) for value in pairwise_differences.values())
        causal_passed: bool | None = maximum_difference <= 0.05
    else:
        pairwise_differences = {}
        maximum_difference = None
        causal_passed = None

    return {
        "core_capability": {
            "status": "evaluated" if core_evaluable else "not_evaluable",
            "passed": core_passed,
            "thresholds": {
                "clean_final_target_rate_min": 0.80,
                "immediate_repair_target_rate_min": 0.70,
                "scorable_primary_window_rate_min": 0.90,
            },
            "metrics": {
                "clean_final_equal_scenario_rate": clean_rate,
                "immediate_repair_equal_scenario_rate": immediate_rate,
                "scorable_primary_window_rate": coverage,
                "scorable_primary_windows": scorable,
                "primary_windows": total,
            },
            "scorable_definition": (
                "overall_label != unintelligible; no_speech, clarification, and "
                "no_evidence stay in the denominator as failures"
            ),
        },
        "early_assistant_causal_interpretation": {
            "status": "evaluated" if causal_evaluable else "not_evaluable",
            "passed": causal_passed,
            "maximum_allowed_pairwise_rate_difference": 0.05,
            "rates": early_rates,
            "missing_by_condition": {
                condition: early_missing[condition] for condition in DELAYED_CONDITIONS
            },
            "pairwise_rate_differences": pairwise_differences,
            "maximum_absolute_pairwise_rate_difference": maximum_difference,
            "missing_values_imputed": False,
        },
        "claim_policy": "gate failure retains results but limits causal/capability claims",
    }


def score_primary(
    eval_trials: Sequence[dict[str, Any]],
    accepted_audio: Iterable[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
    *,
    expected_seeds: Sequence[int],
    expected_audio_count: int = 600,
    expected_scenario_count: int = 30,
) -> dict[str, Any]:
    """Aggregate five seeds within audio; never treat seeds as independent samples."""

    seeds = tuple(expected_seeds)
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("the v2 primary endpoint requires exactly five unique generation seeds")
    accepted = _accepted_index(accepted_audio, expected_audio_count)
    validate_eval_trials(
        eval_trials,
        expected_audio_ids=set(accepted),
        expected_seeds=seeds,
    )
    identity = eval_identity(eval_trials)
    trial_ids = {str(row["eval_trial_id"]) for row in eval_trials}
    resolved_rows = resolve_annotations(trial_ids, annotations)
    resolved = {str(row["eval_trial_id"]): row for row in resolved_rows}
    if len(resolved) != len(eval_trials):
        raise ValueError("resolved annotation count does not match eval trial count")

    trials_by_audio: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in eval_trials:
        trials_by_audio[str(trial["accepted_audio_id"])].append(trial)

    audio_scores: list[dict[str, Any]] = []
    for accepted_id in sorted(accepted):
        metadata = accepted[accepted_id]
        trials = trials_by_audio.get(accepted_id, [])
        observed_seeds = [int(row["generation_seed"]) for row in trials]
        if len(trials) != 5 or set(observed_seeds) != set(seeds):
            raise ValueError(
                f"accepted audio {accepted_id}: expected one trial for each of five seeds"
            )
        decisions = [resolved[str(row["eval_trial_id"])] for row in trials]
        successes = sum(int(row["final_target_correct"]) for row in decisions)
        stale_errors = sum(int(row["stale_state_error"]) for row in decisions)
        relation_counts = {
            relation: Counter(
                str(row["relation_labels"][relation]) for row in decisions
            )
            for relation in RELATIONS
        }
        audio_scores.append(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_run_id": identity["eval_run_id"],
                "accepted_audio_id": accepted_id,
                "rendition_target_id": metadata["rendition_target_id"],
                "matched_audio_bundle_id": metadata["matched_audio_bundle_id"],
                "text_bundle_id": metadata["text_bundle_id"],
                "scenario_id": metadata["scenario_id"],
                "direction_id": metadata["direction_id"],
                "condition": metadata["condition"],
                "source_track_id": metadata["source_track_id"],
                "speaker_id": metadata["speaker_id"],
                "generation_seeds": sorted(observed_seeds),
                "successes": successes,
                "trials": 5,
                "final_target_rate": successes / 5,
                "stale_errors": stale_errors,
                "stale_trials": 5,
                "stale_state_error_rate": stale_errors / 5,
                "relation_label_counts": {
                    relation: dict(sorted(counts.items()))
                    for relation, counts in relation_counts.items()
                },
                "primary_sampling_unit": "accepted_audio_id",
                "primary_cluster": "scenario_id",
            }
        )

    scenarios = {str(row["scenario_id"]) for row in audio_scores}
    if len(scenarios) != expected_scenario_count:
        raise ValueError(
            f"expected {expected_scenario_count} scenario clusters, found {len(scenarios)}"
        )

    scenario_cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    condition_cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audio_scores:
        scenario_cells[(str(row["scenario_id"]), str(row["condition"]))].append(row)
        condition_cells[str(row["condition"])].append(row)

    scenario_scores: list[dict[str, Any]] = []
    for (scenario_id, condition), rows in sorted(scenario_cells.items()):
        successes = sum(int(row["successes"]) for row in rows)
        trials = sum(int(row["trials"]) for row in rows)
        stale_errors = sum(int(row["stale_errors"]) for row in rows)
        scenario_scores.append(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_run_id": identity["eval_run_id"],
                "scenario_id": scenario_id,
                "condition": condition,
                "audio_units": len(rows),
                "successes": successes,
                "trials": trials,
                "final_target_rate": sum(
                    float(row["final_target_rate"]) for row in rows
                )
                / len(rows),
                "stale_errors": stale_errors,
                "stale_state_error_rate": stale_errors / trials,
                "cluster_role": "primary_resampling_cluster",
            }
        )

    condition_summary: dict[str, dict[str, Any]] = {}
    for condition, rows in sorted(condition_cells.items()):
        successes = sum(int(row["successes"]) for row in rows)
        trials = sum(int(row["trials"]) for row in rows)
        stale_errors = sum(int(row["stale_errors"]) for row in rows)
        condition_summary[condition] = {
            "scenario_clusters": len({row["scenario_id"] for row in rows}),
            "audio_units": len(rows),
            "successes": successes,
            "trials": trials,
            "final_target_rate": successes / trials,
            "stale_errors": stale_errors,
            "stale_state_error_rate": stale_errors / trials,
            "inference_note": "descriptive total; uncertainty must cluster on scenario_id",
        }

    disagreements, annotated_trial_count = _annotation_disagreement_counts(annotations)
    inference = frozen_contrast_inference(
        scenario_scores,
        required_scenario_count=expected_scenario_count,
        strict=expected_scenario_count == 30,
    )
    gates = _gate_metrics(
        scenario_scores,
        eval_trials,
        resolved,
        accepted,
        expected_scenario_count=expected_scenario_count,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "eval_identity": identity,
        "primary_endpoint": {
            "name": "final_target_correct",
            "unit": "accepted_audio_id",
            "representation": "binomial_successes_out_of_5",
            "seed_denominator": 5,
            "primary_cluster": "scenario_id",
            "seed_trials_are_independent_samples": False,
        },
        "counts": {
            "scenario_clusters": len(scenarios),
            "accepted_audio_units": len(audio_scores),
            "eval_trials": len(eval_trials),
            "resolved_annotations": len(resolved_rows),
        },
        "annotation_quality": {
            "primary_annotations_expected_per_trial": 2,
            "adjudication_required_on_disagreement": True,
            "trials_with_two_primary_annotations": annotated_trial_count,
            "disagreements": disagreements,
            "raw_exact_agreement_rate": (
                (annotated_trial_count - disagreements) / annotated_trial_count
                if annotated_trial_count
                else None
            ),
        },
        "frozen_contrast_inference": inference,
        "gates": gates,
        "conditions": condition_summary,
    }
    return {
        "audio_scores": audio_scores,
        "scenario_scores": scenario_scores,
        "resolved_annotations": resolved_rows,
        "summary": summary,
    }


def write_scores(output_dir: Path, scored: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "audio_binomial_scores.jsonl", scored["audio_scores"])
    write_jsonl(output_dir / "scenario_cluster_scores.jsonl", scored["scenario_scores"])
    write_jsonl(output_dir / "resolved_annotations.jsonl", scored["resolved_annotations"])
    write_json(output_dir / "score_summary.json", scored["summary"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-trials", type=Path, default=DEFAULT_EVAL_TRIALS)
    parser.add_argument("--accepted-manifest", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.dataset_config)
    scored = score_primary(
        read_jsonl(args.eval_trials),
        read_jsonl(args.accepted_manifest),
        read_jsonl(args.annotations),
        expected_seeds=config["evaluation"]["generation_seeds"],
        expected_audio_count=int(config["counts"]["rendition_targets_per_track"]),
        expected_scenario_count=int(config["counts"]["scenarios"]),
    )
    write_scores(args.output_dir, scored)
    print(
        f"Scored {len(scored['audio_scores'])} audio units across "
        f"{scored['summary']['counts']['scenario_clusters']} scenario clusters -> "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
