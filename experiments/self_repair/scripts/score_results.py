#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from common import (
    EXPERIMENT_ROOT,
    parse_bool,
    read_csv,
    write_json,
)


ALLOWED_LABELS = {
    "target_only",
    "stale_only",
    "both",
    "recovered",
    "clarification",
    "irrelevant",
    "no_speech",
    "unintelligible",
}
BOOLEAN_FIELDS = [
    "final_target_correct",
    "early_stale_before_repair",
    "stale_after_repair",
    "recovered",
    "intelligible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate blinded annotations and compute self-repair metrics."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=EXPERIMENT_ROOT / "annotations/annotations.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXPERIMENT_ROOT / "data/manifest.prepared.csv",
    )
    parser.add_argument(
        "--annotation-key",
        type=Path,
        default=EXPERIMENT_ROOT / "annotations/annotation_key.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=EXPERIMENT_ROOT / "results/metrics.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=EXPERIMENT_ROOT / "results/metrics.md",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    return parser.parse_args()


def majority(values: list[Any]) -> Any | None:
    if not values:
        return None
    counts = Counter(values)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def resolve_annotations(
    path: Path, key_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unblinding_key = {row["blind_id"]: row for row in read_csv(key_path)}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    completed_rows = 0
    for row in read_csv(path):
        label = row.get("label", "").strip()
        if not label:
            continue
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Unknown label {label!r}")
        completed_rows += 1
        blind_id = row["blind_id"]
        if blind_id not in unblinding_key:
            raise ValueError(f"Unknown blind_id {blind_id} in {path}")
        groups[blind_id].append(row)

    resolved = []
    unresolved = []
    agreement_numerator = 0
    agreement_denominator = 0
    for blind_id, rows in groups.items():
        unblinded = unblinding_key[blind_id]
        adjudicated = [row for row in rows if parse_bool(row.get("adjudicator"), False)]
        if adjudicated:
            chosen_label = adjudicated[-1]["label"]
            source_rows = [adjudicated[-1]]
        else:
            labels = [row["label"] for row in rows]
            chosen_label = majority(labels)
            source_rows = rows
            if len(labels) >= 2:
                agreement_denominator += 1
                agreement_numerator += int(len(set(labels)) == 1)
        if chosen_label is None:
            unresolved.append(
                {
                    "blind_id": blind_id,
                    "trial_id": unblinded["trial_id"],
                    "seed": int(unblinded["seed"]),
                }
            )
            continue
        record: dict[str, Any] = {
            "blind_id": blind_id,
            "trial_id": unblinded["trial_id"],
            "seed": int(unblinded["seed"]),
            "label": chosen_label,
        }
        for field in BOOLEAN_FIELDS:
            parsed = [
                parse_bool(row.get(field), None)
                for row in source_rows
                if row.get(field, "").strip()
            ]
            record[field] = majority(parsed)
        resolved.append(record)
    diagnostics = {
        "completed_annotation_rows": completed_rows,
        "resolved_outputs": len(resolved),
        "unresolved_outputs": unresolved,
        "exact_label_agreement": (
            agreement_numerator / agreement_denominator
            if agreement_denominator
            else None
        ),
        "agreement_output_count": agreement_denominator,
    }
    return resolved, diagnostics


def bootstrap_ci(
    speaker_values: dict[str, float],
    rng: np.random.Generator,
    samples: int,
) -> list[float] | None:
    if not speaker_values:
        return None
    values = np.asarray(list(speaker_values.values()), dtype=np.float64)
    if len(values) == 1:
        value = float(values[0])
        return [value, value]
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def condition_speaker_means(
    records: list[dict[str, Any]],
    value: str,
) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record["condition_id"], record["speaker_id"])].append(
            float(record[value])
        )
    output: dict[str, dict[str, float]] = defaultdict(dict)
    for (condition_id, speaker_id), values in grouped.items():
        output[condition_id][speaker_id] = mean(values)
    return dict(output)


def paired_difference(
    first: dict[str, float], second: dict[str, float]
) -> dict[str, float]:
    speakers = sorted(set(first) & set(second))
    return {speaker: first[speaker] - second[speaker] for speaker in speakers}


def metric_summary(
    values: dict[str, float],
    rng: np.random.Generator,
    samples: int,
) -> dict[str, Any]:
    return {
        "mean": mean(values.values()) if values else None,
        "ci95": bootstrap_ci(values, rng, samples),
        "speaker_count": len(values),
    }


def derive_correct(label: str, target: str) -> bool:
    if "|" in target:
        return label == "both"
    return label in {"target_only", "recovered"}


def format_percent(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    manifest = {row["trial_id"]: row for row in read_csv(args.manifest)}
    resolved, diagnostics = resolve_annotations(args.annotations, args.annotation_key)
    records = []
    for annotation in resolved:
        trial_id = annotation["trial_id"]
        if trial_id not in manifest:
            raise ValueError(f"Annotation references unknown trial_id {trial_id}")
        item = dict(manifest[trial_id])
        item.update(annotation)
        if item["final_target_correct"] is None:
            item["final_target_correct"] = derive_correct(item["label"], item["target"])
        item["stale_error"] = item["label"] == "stale_only"
        item["no_valid_response"] = item["label"] in {
            "irrelevant",
            "no_speech",
            "unintelligible",
        }
        for field in (
            "early_stale_before_repair",
            "stale_after_repair",
            "recovered",
        ):
            if item[field] is None:
                item[field] = item["label"] == "recovered" if field == "recovered" else False
        records.append(item)
    if not records:
        raise ValueError("No completed, resolvable annotations were found")

    rng = np.random.default_rng(args.bootstrap_seed)
    target_means = condition_speaker_means(records, "final_target_correct")
    stale_means = condition_speaker_means(records, "stale_error")
    early_means = condition_speaker_means(records, "early_stale_before_repair")
    recovery_means = condition_speaker_means(records, "recovered")
    condition_ids = sorted(target_means)
    per_condition = {}
    for condition_id in condition_ids:
        per_condition[condition_id] = {
            "target_selection_rate": metric_summary(
                target_means[condition_id], rng, args.bootstrap_samples
            ),
            "strict_sier": metric_summary(
                stale_means.get(condition_id, {}), rng, args.bootstrap_samples
            ),
            "early_stale_rate": metric_summary(
                early_means.get(condition_id, {}), rng, args.bootstrap_samples
            ),
            "recovery_rate": metric_summary(
                recovery_means.get(condition_id, {}), rng, args.bootstrap_samples
            ),
        }

    crg_pairs = {}
    trial_condition = {row["condition_id"]: row for row in manifest.values()}
    for condition_id in condition_ids:
        condition = trial_condition[condition_id]
        clean_id = condition.get("clean_match_id", "")
        if clean_id and clean_id in target_means:
            differences = paired_difference(
                target_means[clean_id], target_means[condition_id]
            )
            crg_pairs[f"{clean_id}_minus_{condition_id}"] = metric_summary(
                differences, rng, args.bootstrap_samples
            )

    available = set(target_means)
    if {"K1", "K2", "K3", "K4"}.issubset(available):
        core_prefix = "K"
    elif {"E1", "E2", "E3", "E4"}.issubset(available):
        core_prefix = "E"
    else:
        core_prefix = None
    core_ids = (
        {name: f"{core_prefix}{number}" for name, number in {
            "clean_1": 1,
            "clean_2": 2,
            "repair_1": 3,
            "repair_2": 4,
            "long_1": 5,
            "long_2": 6,
        }.items()}
        if core_prefix
        else {}
    )
    direction_difference = metric_summary(
        paired_difference(
            target_means.get(core_ids.get("repair_1", ""), {}),
            target_means.get(core_ids.get("repair_2", ""), {}),
        ),
        rng,
        args.bootstrap_samples,
    )
    short_vs_long = {}
    gap_pairs = (
        (
            (core_ids["repair_1"], core_ids["long_1"]),
            (core_ids["repair_2"], core_ids["long_2"]),
        )
        if core_prefix
        else ()
    )
    for short_id, long_id in gap_pairs:
        short_vs_long[f"{short_id}_minus_{long_id}"] = metric_summary(
            paired_difference(
                target_means.get(short_id, {}), target_means.get(long_id, {})
            ),
            rng,
            args.bootstrap_samples,
        )

    def metric(condition_id: str, key: str) -> float | None:
        return per_condition.get(condition_id, {}).get(key, {}).get("mean")

    clean_rates = [
        metric(core_ids.get("clean_1", ""), "target_selection_rate"),
        metric(core_ids.get("clean_2", ""), "target_selection_rate"),
    ]
    repair_rates = [
        metric(core_ids.get("repair_1", ""), "target_selection_rate"),
        metric(core_ids.get("repair_2", ""), "target_selection_rate"),
    ]
    repair_siers = [
        metric(core_ids.get("repair_1", ""), "strict_sier"),
        metric(core_ids.get("repair_2", ""), "strict_sier"),
    ]
    valid_clean = [value for value in clean_rates if value is not None]
    valid_repair = [value for value in repair_rates if value is not None]
    valid_sier = [value for value in repair_siers if value is not None]
    core_crg_names = (
        {
            f"{core_ids['clean_1']}_minus_{core_ids['repair_1']}",
            f"{core_ids['clean_2']}_minus_{core_ids['repair_2']}",
        }
        if core_prefix
        else set()
    )
    core_crgs = [
        summary["mean"]
        for name, summary in crg_pairs.items()
        if name in core_crg_names and summary["mean"] is not None
    ]
    decisions = {
        "core_prefix": core_prefix,
        "clean_gate_pass": len(valid_clean) == 2 and min(valid_clean) >= 0.80,
        "repair_signal": (
            len(valid_clean) == 2
            and min(valid_clean) >= 0.80
            and (
                (core_crgs and max(core_crgs) >= 0.10)
                or (valid_sier and max(valid_sier) >= 0.10)
            )
        ),
        "core_example_too_easy": (
            len(valid_repair) == 2
            and min(valid_repair) >= 0.95
            and valid_sier
            and max(valid_sier) <= 0.05
            and core_crgs
            and max(core_crgs) <= 0.05
        ),
        "note": "These are preregistered pilot decision thresholds, not population claims.",
    }

    output = {
        "diagnostics": diagnostics,
        "per_condition": per_condition,
        "correction_robustness_gap": crg_pairs,
        "direction_difference": direction_difference,
        "short_vs_long_gap": short_vs_long,
        "decisions": decisions,
    }
    write_json(args.output_json, output)

    lines = [
        "# Moshi Korean self-repair pilot metrics",
        "",
        f"Resolved outputs: {diagnostics['resolved_outputs']}",
        f"Exact annotator agreement: {format_percent(diagnostics['exact_label_agreement'])}",
        "",
        "| Condition | Target selection | Strict SIER | Early stale | Recovery | Speakers |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition_id in condition_ids:
        summary = per_condition[condition_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition_id,
                    format_percent(summary["target_selection_rate"]["mean"]),
                    format_percent(summary["strict_sier"]["mean"]),
                    format_percent(summary["early_stale_rate"]["mean"]),
                    format_percent(summary["recovery_rate"]["mean"]),
                    str(summary["target_selection_rate"]["speaker_count"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Correction robustness gap", ""])
    for name, summary in crg_pairs.items():
        ci = summary["ci95"]
        ci_text = "NA" if ci is None else f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] pp"
        mean_text = "NA" if summary["mean"] is None else f"{100 * summary['mean']:.1f} pp"
        lines.append(f"- {name}: {mean_text}, 95% CI {ci_text}")
    lines.extend(
        [
            "",
            "## Pilot decisions",
            "",
            f"- Core condition prefix: {decisions['core_prefix']}",
            f"- Clean gate pass: {decisions['clean_gate_pass']}",
            f"- Repair-specific signal: {decisions['repair_signal']}",
            f"- Core example too easy: {decisions['core_example_too_easy']}",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Metrics JSON: {args.output_json}")
    print(f"Metrics report: {args.output_md}")


if __name__ == "__main__":
    main()
