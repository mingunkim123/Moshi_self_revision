from __future__ import annotations

import re
from typing import Final

from common import CONDITIONS


SAFE_COMPONENT: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def safe_component(value: str, label: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def scenario_id(index: int) -> str:
    if index < 1 or index > 999:
        raise ValueError("scenario index must be in 1..999")
    return f"travel_{index:03d}"


def text_bundle_id(scenario: str, direction: str) -> str:
    safe_component(scenario, "scenario_id")
    if direction not in ("a_to_b", "b_to_a"):
        raise ValueError(f"invalid direction_id: {direction!r}")
    return f"{scenario}__{direction}"


def script_id(bundle: str, condition: str) -> str:
    safe_component(bundle, "text_bundle_id")
    if condition not in CONDITIONS:
        raise ValueError(f"invalid condition: {condition!r}")
    return f"{bundle}__{condition}"


def matched_audio_bundle_id(bundle: str, source_track: str, speaker: str) -> str:
    for value, label in ((bundle, "text_bundle_id"), (source_track, "source_track_id"), (speaker, "speaker_id")):
        safe_component(value, label)
    return f"{bundle}__{source_track}__{speaker}"


def rendition_target_id(script: str, source_track: str, speaker: str) -> str:
    for value, label in ((script, "script_id"), (source_track, "source_track_id"), (speaker, "speaker_id")):
        safe_component(value, label)
    condition = script.rsplit("__", 1)[-1]
    if condition not in CONDITIONS:
        raise ValueError(f"script_id has invalid condition suffix: {script!r}")
    return f"{script}__{source_track}__{speaker}"


def candidate_id(target: str, candidate_index: int) -> str:
    safe_component(target, "rendition_target_id")
    if candidate_index < 1 or candidate_index > 99:
        raise ValueError("candidate index must be in 1..99")
    return f"{target}__cand{candidate_index:02d}"


def accepted_audio_id(target: str) -> str:
    safe_component(target, "rendition_target_id")
    return f"{target}__accepted"


def prepared_stimulus_id(accepted: str, preparation_hash: str) -> str:
    safe_component(accepted, "accepted_audio_id")
    if not re.fullmatch(r"[0-9a-f]{64}", preparation_hash):
        raise ValueError("preparation_hash must be a lowercase SHA-256")
    return f"{accepted}__prepared_{preparation_hash[:12]}"


def eval_run_id(model_repo: str, resolved_revision: str, generation_config_hash: str, code_commit: str) -> str:
    for value, label in (
        (model_repo, "model_repo"),
        (resolved_revision, "resolved_revision"),
        (code_commit, "code_commit"),
    ):
        safe_component(value.casefold().replace("/", "-"), label)
    if not re.fullmatch(r"[0-9a-f]{64}", generation_config_hash):
        raise ValueError("generation_config_hash must be a lowercase SHA-256")
    repo = re.sub(r"[^a-z0-9_-]+", "-", model_repo.casefold()).strip("-")
    revision = re.sub(r"[^a-z0-9_-]+", "-", resolved_revision.casefold()).strip("-")
    commit = re.sub(r"[^a-z0-9_-]+", "-", code_commit.casefold()).strip("-")
    return f"eval__{repo[:24]}__{revision[:24]}__{generation_config_hash[:12]}__{commit[:12]}"


def eval_trial_id(accepted: str, run_id: str, seed: int) -> str:
    safe_component(accepted, "accepted_audio_id")
    safe_component(run_id, "eval_run_id")
    if seed < 0:
        raise ValueError("generation seed must be non-negative")
    return f"{accepted}__{run_id}__seed_{seed}"
