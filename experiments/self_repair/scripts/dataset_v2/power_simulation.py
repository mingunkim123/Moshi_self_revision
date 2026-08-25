#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import DATASET_ROOT, write_json


DEFAULT_REPORT = DATASET_ROOT / "reports/power_mde_simulation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario-cluster sensitivity simulation for the primary contrast.")
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def expit(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def simulate_power(
    *,
    simulations: int,
    seed: int,
    effect_log_odds: float,
    scenario_count: int = 30,
    axes_per_scenario: int = 4,
    seeds_per_axis: int = 5,
    baseline_probability: float = 0.80,
    scenario_intercept_sd: float = 0.60,
    axis_intercept_sd: float = 0.30,
    scenario_condition_slope_sd: float = 0.25,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    base_logit = math.log(baseline_probability / (1.0 - baseline_probability))
    scenario = rng.normal(0.0, scenario_intercept_sd, (simulations, scenario_count, 1))
    axes = rng.normal(0.0, axis_intercept_sd, (simulations, scenario_count, axes_per_scenario))
    condition_slope = rng.normal(0.0, scenario_condition_slope_sd, (simulations, scenario_count, 1))
    neutral_p = expit(base_logit + scenario + axes)
    dep3_p = expit(base_logit + scenario + axes + effect_log_odds + condition_slope)
    neutral = rng.binomial(seeds_per_axis, neutral_p) / seeds_per_axis
    dep3 = rng.binomial(seeds_per_axis, dep3_p) / seeds_per_axis
    neutral_scenario = neutral.mean(axis=2)
    dep3_scenario = dep3.mean(axis=2)
    differences = dep3_scenario - neutral_scenario
    estimates = differences.mean(axis=1)
    standard_errors = differences.std(axis=1, ddof=1) / math.sqrt(scenario_count)
    # Two-sided t(29), alpha=.05. The primary implementation uses the same
    # equal-weight scenario contrast and a scenario-cluster bootstrap CI.
    critical_t = 2.045229642
    rejected = np.abs(estimates) > critical_t * standard_errors
    return {
        "effect_log_odds": effect_log_odds,
        "mean_neutral_rate": float(neutral_scenario.mean()),
        "mean_dep3_rate": float(dep3_scenario.mean()),
        "mean_absolute_percentage_point_difference": float(abs(estimates.mean()) * 100.0),
        "power": float(rejected.mean()),
        "monte_carlo_se": float(math.sqrt(rejected.mean() * (1.0 - rejected.mean()) / simulations)),
    }


def build_report(simulations: int, seed: int) -> dict[str, Any]:
    effects = [round(-0.05 * step, 2) for step in range(0, 31)]
    rows = [
        simulate_power(
            simulations=simulations,
            seed=seed + index * 1009,
            effect_log_odds=effect,
        )
        for index, effect in enumerate(effects)
    ]
    detectable = [row for row in rows if row["power"] >= 0.80]
    mde = min(detectable, key=lambda row: row["mean_absolute_percentage_point_difference"]) if detectable else None
    return {
        "schema_version": "2.0.0",
        "status": "design_sensitivity_not_observed_data",
        "simulation_seed": seed,
        "simulations_per_effect": simulations,
        "design": {
            "scenario_clusters": 30,
            "directions_x_speakers_per_scenario_condition": 4,
            "generation_seeds_per_rendition": 5,
            "baseline_probability": 0.80,
            "scenario_random_intercept_sd_logit": 0.60,
            "direction_speaker_axis_intercept_sd_logit": 0.30,
            "scenario_condition_slope_sd_logit": 0.25,
            "test_proxy": "two-sided equal-weight paired scenario contrast, t(29), alpha=0.05",
            "primary_estimator": "equal-weight scenario-cluster contrast with 10000-replicate cluster bootstrap CI",
        },
        "effect_grid": rows,
        "mde_at_80_percent_power": mde,
        "interpretation": (
            "The MDE is conditional on explicit variance assumptions and is a sensitivity analysis, "
            "not evidence about Moshi. Re-run if pilot ICC or baseline differs materially."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.simulations < 1000:
        raise ValueError("use at least 1000 simulations")
    report = build_report(args.simulations, args.seed)
    write_json(args.output, report)
    mde = report["mde_at_80_percent_power"]
    if mde:
        print(
            f"80% power MDE proxy: {mde['mean_absolute_percentage_point_difference']:.1f} pp "
            f"(power={mde['power']:.3f}) -> {args.output}"
        )
    else:
        print(f"No effect reached 80% power -> {args.output}")


if __name__ == "__main__":
    main()
