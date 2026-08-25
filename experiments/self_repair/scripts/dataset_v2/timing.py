from __future__ import annotations

from typing import Any, Iterable


MAX_CANONICAL_CANDIDATE_TAIL_MS = 1500


REPAIR_EVENT_ORDER = (
    "old_value_onset_ms",
    "old_value_offset_ms",
    "repair_cue_onset_ms",
    "new_value_onset_ms",
    "new_value_offset_ms",
    "repeated_old_onset_ms",
    "repeated_old_offset_ms",
    "repair_cue_offset_ms",
    "closing_prompt_onset_ms",
    "closing_prompt_offset_ms",
    "utterance_end_ms",
)

CLEAN_EVENT_ORDER = (
    "new_value_onset_ms",
    "new_value_offset_ms",
    "closing_prompt_onset_ms",
    "closing_prompt_offset_ms",
    "utterance_end_ms",
)


def derived_timing(condition: str, events: dict[str, float | None]) -> dict[str, float | None]:
    result = dict(events)
    new_onset = _number(events, "new_value_onset_ms")
    new_offset = _number(events, "new_value_offset_ms")
    end = _number(events, "utterance_end_ms")
    result["post_final_value_duration_ms"] = end - new_offset
    if condition == "clean_final":
        result["actual_latency_ms"] = None
        result["post_cue_duration_ms"] = None
    else:
        old_offset = _number(events, "old_value_offset_ms")
        cue_onset = _number(events, "repair_cue_onset_ms")
        cue_offset = _number(events, "repair_cue_offset_ms")
        result["actual_latency_ms"] = cue_onset - old_offset
        result["post_cue_duration_ms"] = end - cue_offset
    result["post_repair_duration_ms"] = result["post_final_value_duration_ms"]
    return result


def validate_timing(condition: str, events: dict[str, Any], duration_ms: float) -> list[str]:
    errors: list[str] = []
    required = CLEAN_EVENT_ORDER if condition == "clean_final" else REPAIR_EVENT_ORDER
    null_only = (
        "repair_cue_onset_ms",
        "repair_cue_offset_ms",
        "repeated_old_onset_ms",
        "repeated_old_offset_ms",
        "actual_latency_ms",
        "post_cue_duration_ms",
    )
    if condition == "clean_final":
        null_only = ("old_value_onset_ms", "old_value_offset_ms", *null_only)
        for key in null_only:
            if events.get(key) is not None:
                errors.append(f"clean timing field {key} must be null")
    for key in required:
        value = events.get(key)
        if not isinstance(value, (int, float)):
            errors.append(f"{key} must be numeric")
        elif value < 0 or value > duration_ms + 1:
            errors.append(f"{key}={value} outside audio duration {duration_ms}")
    numeric_order = [float(events[key]) for key in required if isinstance(events.get(key), (int, float))]
    if len(numeric_order) == len(required) and any(left > right for left, right in zip(numeric_order, numeric_order[1:])):
        errors.append(f"timing events are not monotonic for {condition}")
    if isinstance(events.get("utterance_end_ms"), (int, float)):
        trailing_silence_ms = duration_ms - float(events["utterance_end_ms"])
        if trailing_silence_ms < -1:
            errors.append("utterance_end_ms occurs after canonical file duration")
        elif trailing_silence_ms > MAX_CANONICAL_CANDIDATE_TAIL_MS:
            errors.append(
                "canonical candidate has more than "
                f"{MAX_CANONICAL_CANDIDATE_TAIL_MS} ms trailing silence"
            )
    if not errors:
        expected = derived_timing(condition, events)
        for key in (
            "actual_latency_ms",
            "post_final_value_duration_ms",
            "post_repair_duration_ms",
            "post_cue_duration_ms",
        ):
            observed, target = events.get(key), expected[key]
            if observed is None and target is None:
                continue
            if not isinstance(observed, (int, float)) or target is None or abs(float(observed) - float(target)) > 1:
                errors.append(f"{key} is inconsistent with aligned events")
    return errors


def shift_events(events: dict[str, Any], prefix_ms: float) -> dict[str, Any]:
    if prefix_ms < 0:
        raise ValueError("prefix_ms must be non-negative")
    shifted: dict[str, Any] = {}
    for key, value in events.items():
        if key.endswith("_ms") and isinstance(value, (int, float)) and key not in {
            "actual_latency_ms",
            "post_final_value_duration_ms",
            "post_repair_duration_ms",
            "post_cue_duration_ms",
        }:
            shifted[key] = value + prefix_ms
        else:
            shifted[key] = value
    return shifted


def unit_ages_at_repair(unit_offsets_ms: Iterable[float], repair_cue_onset_ms: float) -> list[float]:
    ages = [repair_cue_onset_ms - float(offset) for offset in unit_offsets_ms]
    if any(age < 0 for age in ages):
        raise ValueError("pre-repair unit offset occurs after repair cue onset")
    return ages


def _number(events: dict[str, Any], key: str) -> float:
    value = events.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)
