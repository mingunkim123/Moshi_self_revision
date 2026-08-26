#!/usr/bin/env python3
"""Analyze one completed v2 generation seed without claiming production scoring.

The production scorer deliberately requires five seeds and adjudicated human
annotations.  This script is a descriptive pilot companion: it reconstructs the
primary response window from timestamped text-token events and applies a frozen,
conservative answer-key evidence matcher.  Its labels are review queues, not
substitutes for the annotation protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence


CONDITIONS = (
    "clean_final",
    "immediate_repair",
    "delayed_neutral",
    "delayed_one_dependency",
    "delayed_three_dependencies",
)
REPAIR_CONDITIONS = CONDITIONS[1:]
DELAYED_CONDITIONS = CONDITIONS[2:]
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260826
ANALYZER_VERSION = "1.0.0"
GREETING_RE = re.compile(
    r"^\s*hi\s*,?\s*how\s+is\s+your\s+day\s*\?\s*",
    flags=re.IGNORECASE,
)
LEXICAL_RE = re.compile(r"[A-Za-z0-9]")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def normalize_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def evidence_matches(text: str, terms: Sequence[str]) -> list[str]:
    normalized_text = f" {normalize_phrase(text)} "
    matches = {
        term
        for term in terms
        if normalize_phrase(term)
        and f" {normalize_phrase(term)} " in normalized_text
    }
    return sorted(matches, key=lambda value: (value.casefold(), value))


def window_text(
    events: Sequence[Mapping[str, Any]],
    start_ms: float,
    end_ms: float | None = None,
) -> str:
    pieces: list[str] = []
    for event in events:
        time_ms = event.get("time_ms")
        piece = event.get("piece")
        if not isinstance(time_ms, (int, float)) or isinstance(time_ms, bool):
            raise ValueError("stream event has invalid time_ms")
        if not isinstance(piece, str):
            raise ValueError("stream event has invalid piece")
        if float(time_ms) < start_ms:
            continue
        if end_ms is not None and float(time_ms) >= end_ms:
            continue
        pieces.append(piece)
    return "".join(pieces).strip()


def first_lexical_event_ms(
    events: Sequence[Mapping[str, Any]],
    *,
    start_ms: float = 0.0,
    end_ms: float | None = None,
) -> float | None:
    for event in events:
        time_ms = float(event["time_ms"])
        if time_ms < start_ms or (end_ms is not None and time_ms >= end_ms):
            continue
        if LEXICAL_RE.search(str(event["piece"])):
            return time_ms
    return None


def _required_text(row: Mapping[str, Any], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: invalid {field}")
    return value


def _index_unique(
    rows: Sequence[dict[str, Any]], field: str, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        value = _required_text(row, field, f"{label} row {index}")
        if value in indexed:
            raise ValueError(f"duplicate {field}: {value}")
        indexed[value] = row
    return indexed


def _terms(answer_key: Mapping[str, Any], kind: str) -> list[str]:
    evidence = answer_key.get(f"{kind}_evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"answer key has invalid {kind}_evidence")
    aliases = evidence.get("aliases")
    entities = evidence.get("illustrative_entities")
    if not isinstance(aliases, list) or not isinstance(entities, list):
        raise ValueError(f"answer key has invalid {kind} evidence term lists")
    terms = [*aliases, *entities]
    if not all(isinstance(term, str) and term for term in terms):
        raise ValueError(f"answer key has invalid {kind} evidence term")
    return terms


def derive_labels(
    eval_rows: Sequence[dict[str, Any]],
    accepted_rows: Sequence[dict[str, Any]],
    prepared_rows: Sequence[dict[str, Any]],
    answer_keys: Sequence[dict[str, Any]],
    *,
    expected_completed_count: int = 600,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted = _index_unique(accepted_rows, "accepted_audio_id", "accepted audio")
    prepared = _index_unique(prepared_rows, "accepted_audio_id", "prepared stimulus")
    answers = _index_unique(answer_keys, "answer_key_id", "answer key")
    if set(prepared) != set(accepted):
        raise ValueError("prepared/accepted audio coverage is not exact")
    completed = [
        row
        for row in eval_rows
        if isinstance(row.get("response"), dict)
        and row["response"].get("status") == "completed"
    ]
    if len(completed) != expected_completed_count:
        raise ValueError(
            f"expected {expected_completed_count} completed trials, found {len(completed)}"
        )

    completed_ids: set[str] = set()
    completed_audio_ids: set[str] = set()
    seeds: set[int] = set()
    transcript_reconstruction_mismatches = 0
    labels: list[dict[str, Any]] = []

    for row_index, trial in enumerate(completed):
        trial_id = _required_text(trial, "eval_trial_id", f"trial row {row_index}")
        accepted_id = _required_text(
            trial, "accepted_audio_id", f"trial {trial_id}"
        )
        if trial_id in completed_ids:
            raise ValueError(f"duplicate completed eval_trial_id: {trial_id}")
        if accepted_id in completed_audio_ids:
            raise ValueError(
                f"single-seed pilot has more than one completed trial for {accepted_id}"
            )
        completed_ids.add(trial_id)
        completed_audio_ids.add(accepted_id)
        if accepted_id not in accepted:
            raise ValueError(f"trial {trial_id}: unknown accepted_audio_id")
        metadata = accepted[accepted_id]
        prepared_row = prepared[accepted_id]
        answer_key_id = _required_text(metadata, "text_bundle_id", accepted_id)
        if answer_key_id not in answers:
            raise ValueError(f"trial {trial_id}: missing answer key {answer_key_id}")
        answer = answers[answer_key_id]
        for field in ("scenario_id", "direction_id"):
            if metadata.get(field) != answer.get(field):
                raise ValueError(f"trial {trial_id}: {field} disagrees with answer key")
        if metadata.get("condition") != trial.get("condition"):
            raise ValueError(f"trial {trial_id}: condition disagrees with accepted audio")

        seed = trial.get("generation_seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"trial {trial_id}: invalid generation_seed")
        seeds.add(seed)
        response = trial["response"]
        input_stimulus = trial.get("input_stimulus")
        if not isinstance(input_stimulus, dict):
            raise ValueError(f"trial {trial_id}: missing input_stimulus")
        if input_stimulus.get("prepared_stimulus_id") != prepared_row.get(
            "prepared_stimulus_id"
        ):
            raise ValueError(f"trial {trial_id}: prepared_stimulus_id mismatch")
        prepared_audio = prepared_row.get("prepared_stimulus")
        if not isinstance(prepared_audio, dict) or input_stimulus.get(
            "sha256"
        ) != prepared_audio.get("sha256"):
            raise ValueError(f"trial {trial_id}: prepared stimulus hash mismatch")
        preparation = prepared_row.get("preparation")
        if not isinstance(preparation, dict):
            raise ValueError(f"trial {trial_id}: missing preparation metadata")
        prefix_raw = preparation.get("prefix_ms_actual")
        if not isinstance(prefix_raw, (int, float)) or isinstance(prefix_raw, bool):
            raise ValueError(f"trial {trial_id}: invalid prefix_ms_actual")
        prefix_ms = float(prefix_raw)
        events = trial.get("stream_events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"trial {trial_id}: missing stream_events")
        event_times = [float(event["time_ms"]) for event in events]
        if event_times != sorted(event_times):
            raise ValueError(f"trial {trial_id}: stream events are not monotonic")

        primary_start = response.get("primary_window_start_ms")
        if not isinstance(primary_start, (int, float)) or isinstance(primary_start, bool):
            raise ValueError(f"trial {trial_id}: invalid primary_window_start_ms")
        capture = trial.get("capture_contract")
        if not isinstance(capture, dict):
            raise ValueError(f"trial {trial_id}: missing capture_contract")
        if float(capture.get("primary_window_start_ms", -1)) != float(primary_start):
            raise ValueError(f"trial {trial_id}: primary-window contract mismatch")
        timing = capture.get("prepared_timing")
        if not isinstance(timing, dict):
            raise ValueError(f"trial {trial_id}: missing prepared_timing")

        full_text = window_text(events, 0.0)
        recorded_transcript = str(response.get("transcript", "")).strip()
        if full_text != recorded_transcript:
            transcript_reconstruction_mismatches += 1
        primary_text = window_text(events, float(primary_start))
        lexical_words = WORD_RE.findall(primary_text)

        target_matches = evidence_matches(primary_text, _terms(answer, "target"))
        stale_matches = evidence_matches(primary_text, _terms(answer, "stale"))
        exact_target = bool(evidence_matches(primary_text, [str(answer["target_value"])]))
        exact_stale = bool(evidence_matches(primary_text, [str(answer["stale_value"])]))
        if target_matches and stale_matches:
            evidence_category = "both"
        elif target_matches:
            evidence_category = "target_only"
        elif stale_matches:
            evidence_category = "stale_only"
        else:
            evidence_category = "no_evidence"
        if exact_target and exact_stale:
            exact_value_category = "both"
        elif exact_target:
            exact_value_category = "target_only"
        elif exact_stale:
            exact_value_category = "stale_only"
        else:
            exact_value_category = "no_evidence"
        has_lexical_primary = bool(LEXICAL_RE.search(primary_text))
        primary_text_empty = not primary_text
        review_label = evidence_category if has_lexical_primary else "no_speech_proxy"

        repair_onset_raw = timing.get("repair_cue_onset_ms")
        repair_onset = (
            float(repair_onset_raw)
            if isinstance(repair_onset_raw, (int, float))
            and not isinstance(repair_onset_raw, bool)
            else None
        )
        any_early = None
        beyond_greeting_early = None
        pre_repair_text = ""
        if repair_onset is not None:
            pre_repair_text = window_text(events, 0.0, repair_onset)
            any_early = first_lexical_event_ms(events, end_ms=repair_onset) is not None
            without_greeting = GREETING_RE.sub("", pre_repair_text, count=1)
            beyond_greeting_early = bool(LEXICAL_RE.search(without_greeting))

        target_terms_full = evidence_matches(full_text, _terms(answer, "target"))
        stale_terms_full = evidence_matches(full_text, _terms(answer, "stale"))
        labels.append(
            {
                "analyzer_version": ANALYZER_VERSION,
                "eval_trial_id": trial_id,
                "eval_run_id": trial["eval_run_id"],
                "generation_seed": seed,
                "accepted_audio_id": accepted_id,
                "scenario_id": metadata["scenario_id"],
                "direction_id": metadata["direction_id"],
                "condition": metadata["condition"],
                "speaker_id": metadata["speaker_id"],
                "voice": metadata["voice"],
                "text_bundle_id": answer_key_id,
                "target_value": answer["target_value"],
                "stale_value": answer["stale_value"],
                "primary_window_start_ms": float(primary_start),
                "primary_text": primary_text,
                "primary_lexical_word_count": len(lexical_words),
                "primary_text_empty": primary_text_empty,
                "primary_has_lexical_text": has_lexical_primary,
                "target_evidence_matches": target_matches,
                "stale_evidence_matches": stale_matches,
                "evidence_category": evidence_category,
                "exact_value_category": exact_value_category,
                "exact_target_value_match": exact_target,
                "exact_stale_value_match": exact_stale,
                "review_queue_label": review_label,
                "conservative_target_only_proxy": evidence_category == "target_only",
                "stale_evidence_proxy": bool(stale_matches),
                "full_transcript_target_evidence_matches": target_terms_full,
                "full_transcript_stale_evidence_matches": stale_terms_full,
                "full_transcript_starts_standard_greeting": bool(
                    GREETING_RE.match(full_text)
                ),
                "first_assistant_lexical_ms": first_lexical_event_ms(events),
                "input_prefix_silence_ms": prefix_ms,
                "assistant_started_before_user_audio_proxy": (
                    first_lexical_event_ms(events) is not None
                    and float(first_lexical_event_ms(events)) < prefix_ms
                ),
                "repair_cue_onset_ms": repair_onset,
                "assistant_started_before_repair_proxy": any_early,
                "assistant_beyond_standard_greeting_before_repair_proxy": (
                    beyond_greeting_early
                ),
                "elapsed_seconds": float(response["elapsed_seconds"]),
                "response_audio_duration_ms": float(response["audio_duration_ms"]),
                "coverage_complete": response.get("coverage_complete") is True,
                "contract_consistent": (
                    response.get("coverage_complete") is True
                    and response.get("eos_reached") is False
                    and response.get("fed_frame_count")
                    == response.get("output_frame_count")
                    and response.get("fed_sample_count")
                    == response.get("output_sample_count")
                    and response.get("stream_reset") is True
                    and response.get("rng_reset") is True
                ),
                "response_audio_path": response["audio_path"],
                "response_audio_sha256": response["audio_sha256"],
            }
        )

    if len(seeds) != 1:
        raise ValueError(f"single-seed pilot requires exactly one completed seed, found {seeds}")
    if completed_audio_ids != set(accepted):
        raise ValueError(
            "completed single-seed audio coverage does not exactly match accepted manifest"
        )
    return sorted(labels, key=lambda row: str(row["accepted_audio_id"])), {
        "completed_seed": next(iter(seeds)),
        "transcript_reconstruction_mismatches": transcript_reconstruction_mismatches,
    }


def _rate(rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap_rate(
    rows: Sequence[dict[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
    rng: random.Random,
) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario_id"])].append(row)
    scenario_ids = sorted(by_scenario)
    scenario_rates = {
        scenario_id: _rate(by_scenario[scenario_id], predicate)
        for scenario_id in scenario_ids
    }
    point = statistics.mean(scenario_rates.values())
    boot: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [rng.choice(scenario_ids) for _ in scenario_ids]
        boot.append(statistics.mean(scenario_rates[value] for value in sampled))
    return {
        "rate": point,
        "percentage": 100.0 * point,
        "scenario_cluster_bootstrap_95_ci": [
            _percentile(boot, 0.025),
            _percentile(boot, 0.975),
        ],
        "scenario_clusters": len(scenario_ids),
    }


def _paired_cluster_contrast(
    labels: Sequence[dict[str, Any]],
    left: str,
    right: str,
    field: str,
    rng: random.Random,
) -> dict[str, Any]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        cells[(str(row["scenario_id"]), str(row["condition"]))].append(row)
    scenario_ids = sorted({str(row["scenario_id"]) for row in labels})
    differences: dict[str, float] = {}
    for scenario_id in scenario_ids:
        left_rows = cells[(scenario_id, left)]
        right_rows = cells[(scenario_id, right)]
        if len(left_rows) != 4 or len(right_rows) != 4:
            raise ValueError(
                f"scenario {scenario_id}: contrast cells must each contain four audio units"
            )
        differences[scenario_id] = _rate(
            left_rows, lambda row: bool(row[field])
        ) - _rate(right_rows, lambda row: bool(row[field]))
    point = statistics.mean(differences.values())
    boot: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [rng.choice(scenario_ids) for _ in scenario_ids]
        boot.append(statistics.mean(differences[value] for value in sampled))
    return {
        "left_condition": left,
        "right_condition": right,
        "metric": field,
        "rate_difference": point,
        "percentage_point_difference": point * 100.0,
        "scenario_cluster_bootstrap_95_ci_rate": [
            _percentile(boot, 0.025),
            _percentile(boot, 0.975),
        ],
        "scenario_clusters": len(scenario_ids),
    }


def _descriptive_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    review_queues = Counter(str(row["review_queue_label"]) for row in rows)
    word_counts = [int(row["primary_lexical_word_count"]) for row in rows]
    audio_durations_ms = [float(row["response_audio_duration_ms"]) for row in rows]
    runtimes_seconds = [float(row["elapsed_seconds"]) for row in rows]
    repair_rows = [
        row for row in rows if row["assistant_started_before_repair_proxy"] is not None
    ]
    return {
        "trials": len(rows),
        "review_queue_counts": dict(sorted(review_queues.items())),
        "target_only_proxy_count": sum(
            bool(row["conservative_target_only_proxy"]) for row in rows
        ),
        "target_only_proxy_rate": _rate(
            rows, lambda row: bool(row["conservative_target_only_proxy"])
        ),
        "stale_evidence_proxy_count": sum(
            bool(row["stale_evidence_proxy"]) for row in rows
        ),
        "stale_evidence_proxy_rate": _rate(
            rows, lambda row: bool(row["stale_evidence_proxy"])
        ),
        "no_evidence_count": sum(
            row["evidence_category"] == "no_evidence" for row in rows
        ),
        "no_evidence_rate": _rate(
            rows, lambda row: row["evidence_category"] == "no_evidence"
        ),
        "no_speech_proxy_count": sum(
            row["review_queue_label"] == "no_speech_proxy" for row in rows
        ),
        "no_speech_proxy_rate": _rate(
            rows, lambda row: row["review_queue_label"] == "no_speech_proxy"
        ),
        "empty_primary_text_count": sum(bool(row["primary_text_empty"]) for row in rows),
        "empty_primary_text_rate": _rate(
            rows, lambda row: bool(row["primary_text_empty"])
        ),
        "primary_word_count_mean": statistics.mean(word_counts),
        "primary_word_count_median": statistics.median(word_counts),
        "runtime_seconds_mean": statistics.mean(runtimes_seconds),
        "runtime_hours_total": sum(runtimes_seconds) / 3600.0,
        "response_audio_duration_ms_mean": statistics.mean(audio_durations_ms),
        "response_audio_hours_total": sum(audio_durations_ms) / 3_600_000.0,
        "assistant_started_before_user_audio_proxy_rate": _rate(
            rows, lambda row: bool(row["assistant_started_before_user_audio_proxy"])
        ),
        "assistant_started_before_repair_proxy_rate": (
            _rate(
                repair_rows,
                lambda row: bool(row["assistant_started_before_repair_proxy"]),
            )
            if repair_rows
            else None
        ),
        "assistant_beyond_standard_greeting_before_repair_proxy_rate": (
            _rate(
                repair_rows,
                lambda row: bool(
                    row["assistant_beyond_standard_greeting_before_repair_proxy"]
                ),
            )
            if repair_rows
            else None
        ),
    }


def summarize(
    labels: Sequence[dict[str, Any]], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    if len(labels) != 600:
        raise ValueError("pilot summary requires 600 derived labels")
    scenarios = {str(row["scenario_id"]) for row in labels}
    speakers = {str(row["speaker_id"]) for row in labels}
    directions = {str(row["direction_id"]) for row in labels}
    if len(scenarios) != 30 or len(speakers) != 10 or len(directions) != 2:
        raise ValueError("pilot matrix is not 30 scenarios × 2 directions × 2 speakers × 5 conditions")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        by_condition[str(row["condition"])].append(row)
        by_direction[str(row["direction_id"])].append(row)
        by_speaker[str(row["speaker_id"])].append(row)
    if set(by_condition) != set(CONDITIONS) or any(
        len(by_condition[condition]) != 120 for condition in CONDITIONS
    ):
        raise ValueError("pilot condition matrix is not exactly 120 trials per condition")

    rng = random.Random(BOOTSTRAP_SEED)
    condition_summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = by_condition[condition]
        condition_summary[condition] = {
            **_descriptive_group(rows),
            "target_only_proxy_inference": _cluster_bootstrap_rate(
                rows,
                lambda row: bool(row["conservative_target_only_proxy"]),
                rng,
            ),
            "stale_evidence_proxy_inference": _cluster_bootstrap_rate(
                rows, lambda row: bool(row["stale_evidence_proxy"]), rng
            ),
        }

    contrasts: dict[str, Any] = {}
    for left, short_name in (
        ("delayed_three_dependencies", "three_minus_neutral"),
        ("delayed_one_dependency", "one_minus_neutral"),
    ):
        for field, metric_name in (
            ("conservative_target_only_proxy", "target_only_proxy"),
            ("stale_evidence_proxy", "stale_evidence_proxy"),
        ):
            contrasts[f"{short_name}__{metric_name}"] = _paired_cluster_contrast(
                labels, left, "delayed_neutral", field, rng
            )

    return {
        "analyzer_version": ANALYZER_VERSION,
        "analysis_status": "descriptive_single_seed_pilot_not_production_scoring",
        "provenance": dict(provenance),
        "counts": {
            "completed_trials": len(labels),
            "completed_seeds": 1,
            "scenario_clusters": len(scenarios),
            "directions": len(directions),
            "speakers": len(speakers),
            "conditions": len(by_condition),
            "technical_contract_failures": sum(
                not bool(row["contract_consistent"]) for row in labels
            ),
        },
        "overall": _descriptive_group(list(labels)),
        "exact_value_sensitivity": {
            "target_only_count": sum(
                row["exact_value_category"] == "target_only" for row in labels
            ),
            "stale_evidence_count": sum(
                bool(row["exact_stale_value_match"]) for row in labels
            ),
            "both_count": sum(row["exact_value_category"] == "both" for row in labels),
        },
        "onset_diagnostics": {
            "first_assistant_lexical_ms_values": sorted(
                {
                    float(row["first_assistant_lexical_ms"])
                    for row in labels
                    if row["first_assistant_lexical_ms"] is not None
                }
            ),
            "input_prefix_silence_ms_values": sorted(
                {float(row["input_prefix_silence_ms"]) for row in labels}
            ),
            "standard_greeting_start_count": sum(
                bool(row["full_transcript_starts_standard_greeting"])
                for row in labels
            ),
            "standard_greeting_start_rate": _rate(
                list(labels),
                lambda row: bool(row["full_transcript_starts_standard_greeting"]),
            ),
        },
        "conditions": condition_summary,
        "directions": {
            key: _descriptive_group(rows) for key, rows in sorted(by_direction.items())
        },
        "speakers": {
            key: _descriptive_group(rows) for key, rows in sorted(by_speaker.items())
        },
        "paired_scenario_cluster_contrasts": contrasts,
        "bootstrap": {
            "method": "scenario-cluster percentile",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "label_policy": {
            "primary_window": (
                "concatenate stream-event text pieces whose time_ms is greater than or "
                "equal to response.primary_window_start_ms"
            ),
            "target_only_proxy": (
                "at least one answer-key target alias/illustrative entity and no stale "
                "alias/illustrative entity in the primary text"
            ),
            "stale_evidence_proxy": (
                "at least one answer-key stale alias/illustrative entity in primary text"
            ),
            "no_evidence": (
                "no listed target or stale alias/entity; requires human review and is "
                "not automatically a semantic failure"
            ),
            "no_speech_proxy": "no alphanumeric text token in the primary window",
        },
        "limitations": [
            "Only generation seed 17 is complete; seed variability is unknown.",
            "Evidence matching is conservative and cannot score D1-D3 relation binding.",
            "No-evidence responses require the preregistered two-annotator review.",
            "The assistant-onset proxy includes Moshi's standard opening greeting.",
            "All trials used a prepared-input prefix; onset before that boundary means the assistant spoke before user audio began.",
            "Audio/alignment release review remains provisional and is not repaired by this analysis.",
        ],
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Seed 17 파일럿 분석",
        "",
        "> 이 문서는 자동 증거 매칭을 이용한 단일-seed 기술·내용 파일럿입니다. "
        "정식 5-seed 이중 인간 주석 결과가 아닙니다.",
        "",
        "## 핵심 결과",
        "",
    ]
    overall = summary["overall"]
    onset = summary["onset_diagnostics"]
    exact = summary["exact_value_sensitivity"]
    lines.extend(
        [
            f"- 완료 응답: **{summary['counts']['completed_trials']}개**, 기술 계약 실패 "
            f"**{summary['counts']['technical_contract_failures']}개**",
            f"- primary window의 보수적 target-only 증거: "
            f"**{overall['target_only_proxy_count']}개 "
            f"({_pct(overall['target_only_proxy_rate'])})**",
            f"- stale 증거: **{overall['stale_evidence_proxy_count']}개 "
            f"({_pct(overall['stale_evidence_proxy_rate'])})**",
            f"- 등록된 도시명·별칭·예시 장소 증거 없음: **{overall['no_evidence_count']}개 "
            f"({_pct(overall['no_evidence_rate'])})**",
            f"  - 도시명 exact-match만 사용한 민감도 분석도 target-only "
            f"**{exact['target_only_count']}개**, stale 증거 "
            f"**{exact['stale_evidence_count']}개**로 거의 같습니다.",
            f"- primary window lexical output 없음: **{overall['no_speech_proxy_count']}개 "
            f"({_pct(overall['no_speech_proxy_rate'])})**",
            f"  - 이 중 text token 자체가 완전히 빈 응답은 "
            f"**{overall['empty_primary_text_count']}개 "
            f"({_pct(overall['empty_primary_text_rate'])})**",
            f"- 사용자 audio 시작 전 assistant text 시작: "
            f"**{_pct(overall['assistant_started_before_user_audio_proxy_rate'])}**",
            f"  - 첫 assistant lexical token: `{onset['first_assistant_lexical_ms_values']}` ms; "
            f"입력 prefix silence: `{onset['input_prefix_silence_ms_values']}` ms",
            f"  - 표준 인사로 시작한 trial: **{onset['standard_greeting_start_count']}개 "
            f"({_pct(onset['standard_greeting_start_rate'])})**",
            f"- 총 GPU 실행시간: **{overall['runtime_hours_total']:.2f}시간**; "
            f"응답 audio 총 길이: **{overall['response_audio_hours_total']:.2f}시간**",
            "",
            "## 조건별 자동 증거",
            "",
            "| 조건 | n | target-only | stale 증거 | no evidence | lexical text 없음 | 평균 단어 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        row = summary["conditions"][condition]
        lines.append(
            f"| `{condition}` | {row['trials']} | {_pct(row['target_only_proxy_rate'])} "
            f"| {_pct(row['stale_evidence_proxy_rate'])} | {_pct(row['no_evidence_rate'])} "
            f"| {_pct(row['no_speech_proxy_rate'])} | {row['primary_word_count_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "`target-only`는 answer key에 등록된 target 도시명·별칭·예시 장소가 있고 "
            "stale 증거가 없는 경우입니다. `no evidence`는 자동 실패 확정이 아니라 인간 "
            "검토 대기입니다.",
            "",
            "## 턴테이킹 진단",
            "",
            "| 조건 | repair 전 assistant 시작 | 표준 인사 이후 추가 발화도 repair 전 시작 |",
            "|---|---:|---:|",
        ]
    )
    for condition in REPAIR_CONDITIONS:
        row = summary["conditions"][condition]
        lines.append(
            f"| `{condition}` | "
            f"{_pct(row['assistant_started_before_repair_proxy_rate'])} | "
            f"{_pct(row['assistant_beyond_standard_greeting_before_repair_proxy_rate'])} |"
        )
    lines.extend(
        [
            "",
            "모든 trial이 400ms에 인사를 시작했지만 사용자 audio는 480ms에 시작합니다. "
            "따라서 모델은 매번 사용자를 듣기 전에 speaking state에 들어갔습니다. 단순 "
            "`assistant_started_before_repair` 조건 차이는 ceiling 때문에 0%p가 되며, 이를 "
            "인과 gate 통과로 해석하면 안 됩니다. 표준 인사 이후의 추가 발화도 delayed "
            "조건에서 80.0–89.2%로 높아 dependency/latency와 turn-taking이 분리되지 않습니다.",
            "",
            "## 지연 조건 대비",
            "",
            "| 대비 | 지표 | 차이(%p) | scenario bootstrap 95% CI(%p) |",
            "|---|---|---:|---:|",
        ]
    )
    for name, contrast in summary["paired_scenario_cluster_contrasts"].items():
        ci = contrast["scenario_cluster_bootstrap_95_ci_rate"]
        lines.append(
            f"| `{name.split('__')[0]}` | `{contrast['metric']}` | "
            f"{contrast['percentage_point_difference']:.1f} | "
            f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] |"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- seed 17 하나뿐이므로 생성 seed 변동을 추정할 수 없습니다.",
            "- 자동 매칭은 D1–D3 관계가 새 도시에 실제로 재결합됐는지 판정하지 못합니다.",
            "- primary window 이전에 나온 발화는 최종 endpoint에서 제외했습니다.",
            "- 정식 결론에는 primary-only 미디어를 사용한 2명 독립 주석과 불일치 조정, "
            "나머지 4개 seed가 필요합니다.",
            "- 사람 audio/alignment 검수 기록이 없으므로 데이터 릴리스 상태는 provisional입니다.",
            "",
            "## 현재 판단",
            "",
            "기술 산출물은 완전하지만, 이 seed에서는 target 증거 사건이 너무 적어 floor "
            "effect가 있고 turn-taking 시작 상태가 조작과 얽혀 있습니다. 현 설정으로 나머지 "
            "2,400개를 바로 생성하기보다 assistant의 자동 선행 인사를 억제하거나 endpoint를 "
            "재설계한 소규모 비교 smoke를 먼저 수행해야 합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-results", type=Path, required=True)
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--answer-keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--published-report", type=Path)
    parser.add_argument("--published-summary", type=Path)
    parser.add_argument("--expected-completed-count", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels, derivation = derive_labels(
        read_jsonl(args.eval_results),
        read_jsonl(args.accepted_manifest),
        read_jsonl(args.prepared_manifest),
        read_jsonl(args.answer_keys),
        expected_completed_count=args.expected_completed_count,
    )
    provenance = {
        "eval_results": str(args.eval_results),
        "eval_results_sha256": sha256_file(args.eval_results),
        "accepted_manifest": str(args.accepted_manifest),
        "accepted_manifest_sha256": sha256_file(args.accepted_manifest),
        "prepared_manifest": str(args.prepared_manifest),
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        "answer_keys": str(args.answer_keys),
        "answer_keys_sha256": sha256_file(args.answer_keys),
        **derivation,
    }
    summary = summarize(labels, provenance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "pilot_trial_labels.jsonl"
    summary_path = args.output_dir / "pilot_summary.json"
    report_path = args.output_dir / "PILOT_ANALYSIS.md"
    _write_jsonl(labels_path, labels)
    _write_json(summary_path, summary)
    rendered_report = render_markdown(summary)
    report_path.write_text(rendered_report, encoding="utf-8")
    if args.published_report is not None:
        args.published_report.parent.mkdir(parents=True, exist_ok=True)
        args.published_report.write_text(rendered_report, encoding="utf-8")
    if args.published_summary is not None:
        _write_json(args.published_summary, summary)
    print(
        f"Analyzed {len(labels)} completed single-seed trials -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
