#!/usr/bin/env python3
"""Fail-closed readiness and request-volume check for v2 production TTS.

The command performs no provider or GPU calls. It records only the presence of
credentials, never their values, and writes its report inside the ignored private
release-evidence directory by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from common import (
    DATASET_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_SCRIPTS,
    REPOSITORY_ROOT,
    read_config,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
)
from synthesize_candidates import render_ssml


SCHEMA_VERSION = "2.0.0"
PREFLIGHT_VERSION = "2.0.0"
PROVIDER = "azure_speech_s0"
GIB = 1024**3
DEFAULT_TARGETS = DATASET_ROOT / "assignments/rendition_targets.jsonl"
DEFAULT_AUTHORITY = DATASET_ROOT / "release_evidence/production_authority.json"
DEFAULT_REPORT = DATASET_ROOT / "release_evidence/production_preflight.json"
DEFAULT_ARTIFACT_ROOT = DATASET_ROOT / "artifacts"
AUTHORITY_FIELDS = {
    "schema_version",
    "status",
    "provider",
    "purpose",
    "paid_tier_confirmed",
    "moshi_evaluation_terms_approved",
    "public_redistribution",
    "redistribution_terms_approved",
    "human_text_signoff",
    "budget_cap",
    "budget_currency",
    "azure_rate_per_million_characters",
    "pricing_verified_at",
    "terms_reviewed_at",
    "voice_inventory_verified_at",
    "alignment_environment",
    "storage_execution_mode",
    "runpod_audio_upload_approved",
    "artifact_store_uri",
    "local_minimum_free_gib",
    "remote_minimum_free_gib",
    "approver_id",
}
_UNSET = object()


def azure_billable_character_count(ssml: str) -> int:
    """Count code points using Azure's documented SSML exclusions.

    Azure excludes only the outer ``speak`` and ``voice`` tags. Their contents,
    whitespace, punctuation, and all other markup remain billable.
    """

    if not isinstance(ssml, str) or not ssml:
        raise ValueError("SSML must be a non-empty string")
    without_excluded = re.sub(r"</?speak(?:\s[^>]*)?>", "", ssml)
    without_excluded = re.sub(r"</?voice(?:\s[^>]*)?>", "", without_excluded)
    if re.search(r"</?(?:speak|voice)(?:\s|>)", without_excluded):
        raise ValueError("could not remove the excluded speak/voice tags exactly")
    return len(without_excluded)


def request_budget(
    scripts: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    counts = config.get("counts")
    tracks = config.get("source_tracks")
    if not isinstance(counts, Mapping) or not isinstance(tracks, Mapping):
        raise ValueError("dataset config is missing counts/source_tracks")
    expected_scripts = int(counts["scripts"])
    expected_targets = int(counts["rendition_targets_per_track"])
    initial_attempts = int(counts["initial_candidates_per_target"])
    maximum_attempts = int(counts["maximum_candidates_per_target"])
    if len(scripts) != expected_scripts:
        raise ValueError(f"expected {expected_scripts} scripts, found {len(scripts)}")
    if len(targets) != expected_targets:
        raise ValueError(f"expected {expected_targets} targets, found {len(targets)}")
    if not (1 <= initial_attempts <= maximum_attempts <= 5):
        raise ValueError("candidate attempt counts must satisfy 1 <= initial <= maximum <= 5")

    script_map = {str(row.get("script_id")): row for row in scripts}
    if len(script_map) != len(scripts) or "None" in script_map:
        raise ValueError("script IDs must be present and unique")
    target_ids = [str(row.get("rendition_target_id")) for row in targets]
    if len(set(target_ids)) != len(target_ids) or "None" in target_ids:
        raise ValueError("rendition target IDs must be present and unique")
    source_tracks = {str(row.get("source_track_id")) for row in targets}
    if len(source_tracks) != 1:
        raise ValueError("production preflight requires exactly one source track")
    source_track = next(iter(source_tracks))
    track = tracks.get(source_track)
    if not isinstance(track, Mapping):
        raise ValueError(f"source track {source_track!r} is absent from config")
    speakers = track.get("speakers")
    if not isinstance(speakers, list):
        raise ValueError(f"source track {source_track!r} has no speaker inventory")
    voice_by_speaker = {
        str(item["speaker_id"]): str(item["voice"])
        for item in speakers
        if isinstance(item, Mapping) and "speaker_id" in item and "voice" in item
    }
    if len(voice_by_speaker) != int(track.get("speaker_count", -1)):
        raise ValueError("configured speaker inventory is incomplete or duplicated")

    per_request: list[dict[str, Any]] = []
    for target in targets:
        target_id = str(target["rendition_target_id"])
        script_id = str(target.get("script_id"))
        script = script_map.get(script_id)
        if script is None:
            raise ValueError(f"{target_id}: unknown script {script_id!r}")
        for field in ("scenario_id", "direction_id", "condition", "text_bundle_id"):
            if target.get(field) != script.get(field):
                raise ValueError(f"{target_id}: {field} does not match script")
        speaker_id = str(target.get("speaker_id"))
        voice = str(target.get("voice"))
        if voice_by_speaker.get(speaker_id) != voice:
            raise ValueError(f"{target_id}: speaker/voice is not in the frozen inventory")
        transcript = script.get("transcript")
        if not isinstance(transcript, str) or not transcript:
            raise ValueError(f"{target_id}: script transcript is empty")
        ssml = render_ssml(script, voice)
        per_request.append(
            {
                "rendition_target_id": target_id,
                "script_id": script_id,
                "speaker_id": speaker_id,
                "voice": voice,
                "transcript_characters": len(transcript),
                "azure_billable_characters": azure_billable_character_count(ssml),
                "ssml_sha256": sha256_value(ssml),
            }
        )

    one_pass_plain = sum(row["transcript_characters"] for row in per_request)
    one_pass_billable = sum(row["azure_billable_characters"] for row in per_request)
    return {
        "provider": PROVIDER,
        "source_track_id": source_track,
        "script_count": len(scripts),
        "rendition_target_count": len(targets),
        "speaker_count": len(voice_by_speaker),
        "one_candidate_per_target": {
            "request_count": len(targets),
            "plain_transcript_characters": one_pass_plain,
            "azure_billable_characters": one_pass_billable,
        },
        "initial_policy": {
            "attempts_per_target": initial_attempts,
            "request_count": len(targets) * initial_attempts,
            "plain_transcript_characters": one_pass_plain * initial_attempts,
            "azure_billable_characters": one_pass_billable * initial_attempts,
        },
        "hard_maximum": {
            "attempts_per_target": maximum_attempts,
            "request_count": len(targets) * maximum_attempts,
            "plain_transcript_characters": one_pass_plain * maximum_attempts,
            "azure_billable_characters": one_pass_billable * maximum_attempts,
        },
        "request_projection_sha256": sha256_value(per_request),
        "pricing_formula": "azure_billable_characters / 1_000_000 * current_portal_rate",
        "pricing_note": (
            "The portal rate, taxes, currency, and subscription discounts are deliberately "
            "not frozen in source control. Verify them immediately before approval."
        ),
    }


def validate_authority(
    value: Any, *, now: datetime | None = None
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return ["production authority must be a JSON object"]
    if set(value) != AUTHORITY_FIELDS:
        return [
            "production authority must contain exactly: "
            + ", ".join(sorted(AUTHORITY_FIELDS))
        ]
    exact = {
        "schema_version": SCHEMA_VERSION,
        "status": "approved",
        "provider": PROVIDER,
        "purpose": "controlled_evaluation",
        "paid_tier_confirmed": True,
        "moshi_evaluation_terms_approved": True,
        "human_text_signoff": True,
        "budget_currency": "USD",
        "alignment_environment": "runpod_linux_mfa",
        "storage_execution_mode": "local_audio_then_runpod_evaluation",
        "runpod_audio_upload_approved": True,
    }
    for field, expected in exact.items():
        if value.get(field) != expected:
            errors.append(f"{field} must equal {expected!r}")
    public = value.get("public_redistribution")
    if not isinstance(public, bool):
        errors.append("public_redistribution must be boolean")
    if value.get("redistribution_terms_approved") is not public:
        errors.append(
            "redistribution_terms_approved must exactly match public_redistribution"
        )
    for field in ("budget_cap", "azure_rate_per_million_characters"):
        number = value.get(field)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or number <= 0
        ):
            errors.append(f"{field} must be finite and positive")
    storage_minimums = {
        "local_minimum_free_gib": 12,
        "remote_minimum_free_gib": 40,
    }
    for field, lower_bound in storage_minimums.items():
        minimum = value.get(field)
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, (int, float))
            or not math.isfinite(float(minimum))
            or minimum < lower_bound
        ):
            errors.append(f"{field} must be at least {lower_bound}")
    for field in (
        "pricing_verified_at",
        "terms_reviewed_at",
        "voice_inventory_verified_at",
        "artifact_store_uri",
        "approver_id",
    ):
        if not isinstance(value.get(field), str) or not str(value[field]).strip():
            errors.append(f"{field} must be a non-empty string")
    current = now or datetime.now(timezone.utc)
    freshness_hours = {
        "pricing_verified_at": 24,
        "voice_inventory_verified_at": 24,
        "terms_reviewed_at": 24 * 30,
    }
    for field, maximum_hours in freshness_hours.items():
        raw = value.get(field)
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"{field} must be an ISO-8601 timestamp")
            continue
        if parsed.tzinfo is None:
            errors.append(f"{field} must include a timezone")
            continue
        age_hours = (current - parsed.astimezone(timezone.utc)).total_seconds() / 3600
        if age_hours < -5 / 60:
            errors.append(f"{field} must not be in the future")
        elif age_hours > maximum_hours:
            errors.append(f"{field} is stale; maximum age is {maximum_hours} hours")
    uri = value.get("artifact_store_uri")
    if isinstance(uri, str) and not re.match(r"^(?:s3|gs|azure|file)://", uri):
        errors.append("artifact_store_uri must use an explicit approved storage scheme")
    return errors


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tracked_tree_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def build_report(
    *,
    config_path: Path,
    scripts_path: Path,
    targets_path: Path,
    authority_path: Path,
    artifact_root: Path,
    environment: Mapping[str, str] | None = None,
    free_bytes: int | None = None,
    azure_sdk_version: str | None | object = _UNSET,
    mfa_executable: str | None | object = _UNSET,
    tracked_tree_clean: bool | None = None,
) -> dict[str, Any]:
    config = read_config(config_path)
    scripts = read_jsonl(scripts_path)
    targets = read_jsonl(targets_path)
    budget = request_budget(scripts, targets, config)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any, action: str) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "evidence": evidence, "action": action}
        )

    track_id = budget["source_track_id"]
    track_status = config["source_tracks"][track_id].get("status")
    check(
        "source_track_release_status",
        track_status == "release_approved",
        {"source_track_id": track_id, "status": track_status},
        "Set release_approved only after provider, purpose, and redistribution review.",
    )

    authority: dict[str, Any] | None = None
    authority_errors: list[str]
    if authority_path.is_file():
        try:
            loaded = json.loads(authority_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            authority_errors = [f"cannot read authority JSON: {error}"]
        else:
            authority_errors = validate_authority(loaded)
            if not authority_errors:
                authority = dict(loaded)
    else:
        authority_errors = ["production authority file is missing"]
    check(
        "production_authority",
        not authority_errors,
        {
            "path": authority_path.as_posix(),
            "errors": authority_errors,
            "sha256": sha256_file(authority_path) if authority_path.is_file() else None,
        },
        "Record the user's explicit provider, budget, legal, human-review, RunPod, and storage approvals.",
    )

    rate = float(authority["azure_rate_per_million_characters"]) if authority else None
    cap = float(authority["budget_cap"]) if authority else None
    initial_characters = budget["initial_policy"]["azure_billable_characters"]
    maximum_characters = budget["hard_maximum"]["azure_billable_characters"]
    initial_cost = initial_characters / 1_000_000 * rate if rate is not None else None
    maximum_cost = maximum_characters / 1_000_000 * rate if rate is not None else None
    check(
        "initial_candidate_budget",
        bool(cap is not None and initial_cost is not None and initial_cost <= cap),
        {
            "currency": authority.get("budget_currency") if authority else None,
            "portal_rate_per_million_characters": rate,
            "initial_estimated_cost_before_tax": initial_cost,
            "hard_maximum_estimated_cost_before_tax": maximum_cost,
            "approved_cap": cap,
        },
        "Verify the current Azure portal rate and approve a cap covering the initial 3-candidate policy.",
    )

    env = environment if environment is not None else os.environ
    credential_presence = {
        "AZURE_SPEECH_KEY": bool(env.get("AZURE_SPEECH_KEY") or env.get("SPEECH_KEY")),
        "AZURE_SPEECH_REGION": bool(env.get("AZURE_SPEECH_REGION")),
    }
    check(
        "azure_credentials_present",
        all(credential_presence.values()),
        credential_presence,
        "Set Azure credentials locally; never paste or commit their values.",
    )

    sdk_version = (
        _package_version("azure-cognitiveservices-speech")
        if azure_sdk_version is _UNSET
        else azure_sdk_version
    )
    check(
        "azure_sdk_installed",
        bool(sdk_version),
        {"distribution": "azure-cognitiveservices-speech", "version": sdk_version},
        "Install requirements-azure-tts.txt in the production environment.",
    )

    mfa = shutil.which("mfa") if mfa_executable is _UNSET else mfa_executable
    remote_mfa_approved = bool(
        authority
        and authority.get("alignment_environment") == "runpod_linux_mfa"
        and authority.get("runpod_audio_upload_approved") is True
    )
    check(
        "independent_alignment_environment",
        bool(mfa) or remote_mfa_approved,
        {"local_mfa_executable": mfa, "approved_remote_mfa": remote_mfa_approved},
        "Install MFA locally or approve the frozen RunPod/Linux MFA environment and upload.",
    )

    runpod_presence = bool(env.get("RUNPOD_API_KEY"))
    check(
        "runpod_access_present",
        runpod_presence,
        {"RUNPOD_API_KEY": runpod_presence},
        "Set RunPod access locally; never paste or commit the API key.",
    )

    artifact_root.mkdir(parents=True, exist_ok=True)
    available = free_bytes if free_bytes is not None else shutil.disk_usage(artifact_root).free
    configured_local_gib = float(
        config.get("release", {})
        .get("storage_plan", {})
        .get("local_audio_production", {})
        .get("minimum_free_gib", 12)
    )
    required_gib = (
        float(authority["local_minimum_free_gib"])
        if authority
        else configured_local_gib
    )
    check(
        "artifact_storage_capacity",
        available >= required_gib * GIB,
        {"available_bytes": available, "available_gib": available / GIB, "required_gib": required_gib},
        "Provide an approved artifact root with the required free capacity.",
    )

    clean = tracked_tree_clean if tracked_tree_clean is not None else _tracked_tree_clean()
    check(
        "tracked_git_tree_clean",
        clean,
        {"clean_against_head": clean},
        "Commit tracked production code/config changes before freezing run identity.",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "preflight_version": PREFLIGHT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if all(item["passed"] for item in checks) else "blocked",
        "provider_calls_made": False,
        "credential_values_recorded": False,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "scripts_sha256": sha256_file(scripts_path),
            "targets_sha256": sha256_file(targets_path),
        },
        "request_budget": budget,
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Write a blocked diagnostic report and exit zero without weakening any gate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        config_path=args.config,
        scripts_path=args.scripts,
        targets_path=args.targets,
        authority_path=args.authority,
        artifact_root=args.artifact_root,
    )
    write_json(args.report, report)
    print(
        f"Production preflight {report['status']}: "
        f"{len(report['failed_checks'])} failed checks; report -> {args.report}"
    )
    if report["status"] == "ready" or args.allow_blocked:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
