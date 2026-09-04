"""Standalone cost/readiness commands used before importing the GPU backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.self_repair.mechanistic.core import (  # noqa: E402
    AtomicCellStore,
    ContractError,
    MODEL_REPO,
    MODEL_REVISION,
    PatchCell,
    canonical_json,
    read_json,
    read_jsonl,
    require_relative_uri,
    sha256_file,
    sha256_value,
    write_json,
    write_jsonl,
)
from experiments.self_repair.mechanistic.readiness import (  # noqa: E402
    CONVERSATION_CHECKS,
    GPU_CANARY_CHECKS,
    MODEL_CHECKS,
    OPEN_LOOP_CHECKS,
    ReadinessError,
    assess_readiness,
    build_authorization_artifact,
    estimate_workload,
    target_binding_sha256,
    verify_authorization_artifact,
)


def _parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description)


def _validate_schema(value: Mapping[str, Any], filename: str) -> None:
    try:
        import jsonschema
    except ImportError as error:
        raise ContractError("jsonschema is required for readiness artifact validation") from error
    schema = read_json(SCRIPT_DIR.parent / "schemas" / filename)
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.path))
    if errors:
        location = ".".join(str(item) for item in errors[0].path) or "<root>"
        raise ContractError(f"{filename} violation at {location}: {errors[0].message}")


def _git_commit() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ContractError("HEAD is not a 40-character Git commit")
    return value


def _require_clean_tree() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT,
        text=True,
    )
    if status.strip():
        raise ContractError("tracked Git working tree is dirty; refusing to bind paid work")
    untracked_code = subprocess.check_output(
        [
            "git", "ls-files", "--others", "--exclude-standard", "--",
            "experiments/self_repair/mechanistic", "moshi/moshi",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
    )
    if untracked_code.strip():
        paths = untracked_code.splitlines()[:5]
        raise ContractError(
            f"untracked executable/source files are not bound to code_commit: {paths}")


def _data_identity(
    manifest_rows: Sequence[Mapping[str, Any]], encoded_rows: Sequence[Mapping[str, Any]], *, model_revision: str,
) -> str:
    source = []
    seen_source: set[str] = set()
    for row in manifest_rows:
        trial_id = str(row.get("trial_id", ""))
        audio_sha = str(row.get("audio_sha256", ""))
        if not trial_id or re.fullmatch(r"[0-9a-f]{64}", audio_sha) is None or trial_id in seen_source:
            raise ContractError("manifest data identity has missing/duplicate trial IDs or invalid audio hashes")
        seen_source.add(trial_id)
        source.append({
            "trial_id": trial_id,
            "audio_sha256": audio_sha,
            "sample_count": int(row.get("sample_count", -1)),
            "frame_count": int(row.get("frame_count", -1)),
            "conversation_contract": row.get("conversation_contract"),
        })
    encoded = []
    seen_encoded: set[str] = set()
    source_by_id = {str(row["trial_id"]): row for row in manifest_rows}
    for row in encoded_rows:
        trial_id = str(row.get("trial_id", ""))
        codes_sha = str(row.get("codes_sha256", ""))
        if not trial_id or re.fullmatch(r"[0-9a-f]{64}", codes_sha) is None or trial_id in seen_encoded:
            raise ContractError("encoded data identity has missing/duplicate trial IDs or invalid code hashes")
        seen_encoded.add(trial_id)
        source_row = source_by_id.get(trial_id)
        if source_row is None:
            raise ContractError(f"encoded data has an unknown trial: {trial_id}")
        if row.get("source_audio_sha256") != source_row.get("audio_sha256"):
            raise ContractError(f"encoded source audio hash mismatch: {trial_id}")
        if row.get("model_revision") != model_revision:
            raise ContractError(f"encoded model revision mismatch: {trial_id}")
        contract = source_row.get("conversation_contract")
        if isinstance(contract, Mapping):
            for hash_name in ("conversation_codes_sha256", "assistant_silence_codes_sha256"):
                if re.fullmatch(r"[0-9a-f]{64}", str(row.get(hash_name, ""))) is None:
                    raise ContractError(f"{trial_id}: encoded conversation cache lacks {hash_name}")
            expected_user_frames = int(contract["user_frame_count"])
            expected_target_frames = int(contract["target_end_frame_count"])
            shapes = {
                "shape": expected_user_frames,
                "conversation_codes_shape": expected_target_frames,
                "assistant_silence_codes_shape": expected_target_frames,
            }
            for shape_name, expected_frames in shapes.items():
                shape = row.get(shape_name)
                if (not isinstance(shape, list) or len(shape) != 3
                        or int(shape[-1]) != expected_frames):
                    raise ContractError(
                        f"{trial_id}: {shape_name} does not cover {expected_frames} frozen frames")
        encoded.append({
            "trial_id": trial_id,
            "source_audio_sha256": row.get("source_audio_sha256"),
            "codes_sha256": codes_sha,
            "conversation_codes_sha256": row.get("conversation_codes_sha256"),
            "assistant_silence_codes_sha256": row.get("assistant_silence_codes_sha256"),
            "shape": row.get("shape"),
            "conversation_codes_shape": row.get("conversation_codes_shape"),
            "assistant_silence_codes_shape": row.get("assistant_silence_codes_shape"),
            "model_revision": row.get("model_revision"),
        })
    if seen_source != seen_encoded:
        missing = sorted(seen_source - seen_encoded)[:5]
        extra = sorted(seen_encoded - seen_source)[:5]
        raise ContractError(f"encoded manifest coverage mismatch; missing={missing}, extra={extra}")
    return sha256_value({
        "source": sorted(source, key=lambda row: row["trial_id"]),
        "encoded": sorted(encoded, key=lambda row: row["trial_id"]),
    })


def build_target_binding_from_files(
    *, config_path: Path, manifest_path: Path, encoded_manifest_path: Path,
    scan_spec_path: Path, code_commit: str | None = None, require_clean: bool = True,
) -> dict[str, str]:
    """Build the exact identity checked again by every non-canary scan."""

    if require_clean:
        _require_clean_tree()
    commit = code_commit or _git_commit()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ContractError("code commit is not lowercase 40-hex")
    config = read_json(config_path)
    scan_spec = read_json(scan_spec_path)
    if not isinstance(config, Mapping) or not isinstance(scan_spec, Mapping):
        raise ContractError("config and scan spec must be JSON objects")
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ContractError("config.model is missing")
    model_repo = str(model.get("repo", ""))
    model_revision = str(model.get("revision", ""))
    if model_repo != MODEL_REPO or model_revision != MODEL_REVISION:
        raise ContractError("paid scan binding is not the frozen Moshiko model identity")
    manifest_rows = read_jsonl(manifest_path)
    encoded_rows = read_jsonl(encoded_manifest_path)
    data_sha = _data_identity(manifest_rows, encoded_rows, model_revision=model_revision)
    return {
        "code_commit": commit,
        "code_sha256": sha256_value({"git_commit": commit}),
        "model_repo": model_repo,
        "model_revision": model_revision,
        "model_sha256": sha256_value({"repo": model_repo, "revision": model_revision}),
        "manifest_sha256": sha256_file(manifest_path),
        "data_sha256": data_sha,
        "encoded_manifest_sha256": sha256_file(encoded_manifest_path),
        "config_sha256": sha256_file(config_path),
        "scan_spec_sha256": sha256_file(scan_spec_path),
    }


SCAN_EXECUTION_FIELDS = (
    "kind", "role", "layers", "anchors", "donors", "controls", "components",
    "limit_scenarios", "selection_sha256",
)


def validate_scan_execution(scan_spec: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    """Require the authorized scan spec to describe the literal CLI grid."""

    expected = scan_spec.get("execution")
    if not isinstance(expected, Mapping):
        raise ContractError("scan spec must contain an execution object")
    unknown = sorted(set(expected) - set(SCAN_EXECUTION_FIELDS))
    missing = [name for name in SCAN_EXECUTION_FIELDS if name not in expected]
    if unknown or missing:
        raise ContractError(f"scan execution fields mismatch; missing={missing}, unknown={unknown}")
    normalized_expected = dict(expected)
    normalized_actual = dict(actual)
    for name in ("layers", "anchors", "donors", "controls", "components"):
        if not isinstance(normalized_expected[name], list):
            raise ContractError(f"scan execution {name} must be a JSON list")
        normalized_actual[name] = list(normalized_actual[name])
    if normalized_expected != normalized_actual:
        raise ContractError(
            "paid scan CLI differs from authorized execution: "
            f"expected={canonical_json(normalized_expected)} actual={canonical_json(normalized_actual)}")


def _load_limits(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ContractError("budget limits must be a JSON object")
    return value


def estimate(argv: Sequence[str]) -> int:
    parser = _parser("Compute the exact declared cell/frame/storage workload without loading a model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scan-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config, manifest, spec = read_json(args.config), read_jsonl(args.manifest), read_json(args.scan_spec)
    estimate_value = estimate_workload(manifest, config, spec)
    report = {
        "schema_version": "1.0.0",
        "analysis_status": "static_exact_declared_workload",
        "config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.manifest),
        "scan_spec_sha256": sha256_file(args.scan_spec),
        "estimate": estimate_value.to_dict(),
    }
    write_json(args.output, report)
    print(canonical_json(report["estimate"]))
    return 0


def select_canary_manifest(argv: Sequence[str]) -> int:
    parser = _parser("Select a deterministic, bounded clean/repair GPU canary subset.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-trials", type=int, default=4)
    parser.add_argument(
        "--role",
        help="Optional immutable role filter, e.g. discovery or formal_confirmation.",
    )
    args = parser.parse_args(argv)
    if args.max_trials < 2 or args.max_trials > 8:
        raise ContractError("GPU canary max-trials must be in [2, 8]")
    if args.role is not None and not args.role.strip():
        raise ContractError("GPU canary role filter must be non-empty")
    source_rows = read_jsonl(args.manifest)
    rows = sorted(
        (
            row for row in source_rows
            if args.role is None or str(row.get("role", "")) == args.role
        ),
        key=lambda row: str(row.get("trial_id", "")),
    )
    if not rows:
        raise ContractError(
            "GPU canary role filter selects no trials" if args.role else
            "GPU canary source manifest is empty"
        )

    def group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
        fields = (
            str(row.get("scenario_id", "")),
            str(row.get("direction_id", "")),
            str(row.get("speaker_id", "")),
            str(row.get("current_value", row.get("new_value", ""))),
        )
        if any(not field for field in fields):
            raise ContractError(
                f"canary trial {row.get('trial_id')!r} lacks scenario/direction/speaker/current value")
        return fields

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)
    candidates: list[tuple[int, tuple[str, str, str, str], list[dict[str, Any]]]] = []
    for key, group_rows in grouped.items():
        clean = [row for row in group_rows if str(row.get("condition", "")).startswith("clean")]
        repair = [row for row in group_rows if not str(row.get("condition", "")).startswith("clean")]
        if clean and repair:
            preferred = int(
                any(row.get("condition") == "clean_final" for row in clean)
                and any(row.get("condition") == "delayed_three_dependencies" for row in repair)
            )
            candidates.append((preferred, key, group_rows))
    chosen: list[dict[str, Any]] | None = None
    chosen_key: tuple[str, str, str, str] | None = None
    for _, key, group_rows in sorted(candidates, key=lambda item: (-item[0], item[1])):
        clean = sorted(
            (row for row in group_rows if str(row.get("condition", "")).startswith("clean")),
            key=lambda row: (row.get("condition") != "clean_final", str(row.get("trial_id", ""))),
        )
        repair = sorted(
            (row for row in group_rows if not str(row.get("condition", "")).startswith("clean")),
            key=lambda row: (
                row.get("condition") != "delayed_three_dependencies",
                str(row.get("trial_id", "")),
            ),
        )
        selected = [clean[0], repair[0]]
        remaining = sorted(
            (row for row in group_rows if row not in selected),
            key=lambda row: str(row.get("trial_id", "")),
        )
        chosen = (selected + remaining)[:args.max_trials]
        chosen_key = key
        break
    if chosen is None:
        raise ContractError(
            "no scenario/direction/speaker/current-value group contains both clean and repair trials")
    assert chosen_key is not None
    write_jsonl(args.output, chosen)
    write_json(args.output.with_suffix(args.output.suffix + ".selection.json"), {
        "schema_version": "1.0.0",
        "source_manifest_sha256": sha256_file(args.manifest),
        "canary_manifest_sha256": sha256_file(args.output),
        "matched_group": {
            "scenario_id": chosen_key[0],
            "direction_id": chosen_key[1],
            "speaker_id": chosen_key[2],
            "current_value": chosen_key[3],
        },
        "clean_trial_id": str(chosen[0]["trial_id"]),
        "repair_trial_id": str(chosen[1]["trial_id"]),
        "clean_condition": str(chosen[0].get("condition", "")),
        "repair_condition": str(chosen[1].get("condition", "")),
        "trial_ids": [str(row["trial_id"]) for row in chosen],
        "trial_count": len(chosen),
        "bounded_max_trials": args.max_trials,
        "role_filter": args.role,
    })
    print(f"selected {len(chosen)} bounded GPU canary trials -> {args.output}")
    return 0


def _normalize_block(source: Mapping[str, Any], names: Sequence[str], aliases: Mapping[str, str] | None = None) -> dict[str, Any]:
    checks = source.get("checks", {})
    aliases = aliases or {}
    normalized = {
        name: bool(checks.get(name, checks.get(aliases.get(name, ""), False)))
        for name in names
    } if isinstance(checks, Mapping) else {name: False for name in names}
    result = dict(source)
    result["checks"] = normalized
    result["passed"] = bool(source.get("passed", all(normalized.values()))) and all(normalized.values())
    return result


def assemble_evidence(argv: Sequence[str]) -> int:
    parser = _parser("Bind static/model/open-loop/conversation/GPU canary evidence to one target scan.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoded-manifest", type=Path, required=True)
    parser.add_argument("--scan-spec", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--model-run-identity", type=Path, required=True)
    parser.add_argument("--open-loop", type=Path, required=True)
    parser.add_argument("--conversation-canary", type=Path, required=True)
    parser.add_argument("--gpu-canary", type=Path, required=True)
    parser.add_argument("--canary-manifest", type=Path, required=True)
    parser.add_argument("--canary-encoded-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit")
    parser.add_argument("--allow-dirty-for-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    binding = build_target_binding_from_files(
        config_path=args.config,
        manifest_path=args.manifest,
        encoded_manifest_path=args.encoded_manifest,
        scan_spec_path=args.scan_spec,
        code_commit=args.code_commit,
        require_clean=not args.allow_dirty_for_tests,
    )
    binding_hash = target_binding_sha256(binding)
    model_source = read_json(args.model_contract)
    model_identity = read_json(args.model_run_identity)
    open_loop_source = read_json(args.open_loop)
    conversation_source = read_json(args.conversation_canary)
    gpu_source = read_json(args.gpu_canary)
    selection_path = args.canary_manifest.with_suffix(args.canary_manifest.suffix + ".selection.json")
    selection = read_json(selection_path)
    source_rows = read_jsonl(args.manifest)
    canary_rows = read_jsonl(args.canary_manifest)
    canary_encoded_rows = read_jsonl(args.canary_encoded_manifest)
    source_by_id = {str(row.get("trial_id", "")): row for row in source_rows}
    canary_ids = [str(row.get("trial_id", "")) for row in canary_rows]
    if (
        not 2 <= len(canary_rows) <= 8
        or any(not trial_id for trial_id in canary_ids)
        or len(set(canary_ids)) != len(canary_ids)
    ):
        raise ContractError("stale readiness evidence: canary subset must contain 2..8 unique named trials")
    for row in canary_rows:
        if source_by_id.get(str(row["trial_id"])) != row:
            raise ContractError(
                f"stale readiness evidence: canary trial is not an exact source-manifest row: {row['trial_id']}")
    if selection.get("trial_ids") != canary_ids or selection.get("trial_count") != len(canary_ids):
        raise ContractError(
            "stale readiness evidence: canary selection sidecar trial coverage differs from the canary manifest")
    bounded_max = selection.get("bounded_max_trials")
    if (
        isinstance(bounded_max, bool) or not isinstance(bounded_max, int)
        or bounded_max < 2 or bounded_max > 8 or len(canary_rows) > bounded_max
    ):
        raise ContractError("stale readiness evidence: invalid bounded canary maximum")
    matched_group = selection.get("matched_group")
    group_fields = ("scenario_id", "direction_id", "speaker_id", "current_value")
    if not isinstance(matched_group, Mapping) or set(matched_group) != set(group_fields):
        raise ContractError("stale readiness evidence: canary matched-group provenance is missing")
    for row in canary_rows:
        observed_group = {
            "scenario_id": str(row.get("scenario_id", "")),
            "direction_id": str(row.get("direction_id", "")),
            "speaker_id": str(row.get("speaker_id", "")),
            "current_value": str(row.get("current_value", row.get("new_value", ""))),
        }
        if any(not value for value in observed_group.values()) or observed_group != dict(matched_group):
            raise ContractError(
                f"stale readiness evidence: canary row escaped its matched group: {row['trial_id']}")
    canary_by_id = {str(row["trial_id"]): row for row in canary_rows}
    clean_row = canary_by_id.get(str(selection.get("clean_trial_id", "")))
    repair_row = canary_by_id.get(str(selection.get("repair_trial_id", "")))
    if (
        clean_row is None or repair_row is None
        or not str(clean_row.get("condition", "")).startswith("clean")
        or str(repair_row.get("condition", "")).startswith("clean")
        or selection.get("clean_condition") != clean_row.get("condition")
        or selection.get("repair_condition") != repair_row.get("condition")
    ):
        raise ContractError("stale readiness evidence: selected clean/repair pairing is invalid")
    canary_data_sha = _data_identity(
        canary_rows, canary_encoded_rows, model_revision=binding["model_revision"])
    expected_config_value_sha = sha256_value(read_json(args.config))
    provenance_mismatches: list[str] = []

    def require_equal(label: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            provenance_mismatches.append(f"{label}: {observed!r} != {expected!r}")

    run_identity_fields = (
        "schema_version", "harness_version", "code_commit", "model_repo", "model_revision",
        "config_sha256", "manifest_sha256", "open_loop_policy_sha256", "data_status",
    )
    identity_body = {name: model_identity.get(name) for name in run_identity_fields}
    if any(model_identity.get(name) is None for name in run_identity_fields):
        provenance_mismatches.append("model run identity is missing canonical identity fields")
    else:
        require_equal(
            "model run identity digest", model_identity.get("run_identity_sha256"),
            sha256_value(identity_body),
        )

    require_equal("model identity code", model_identity.get("code_commit"), binding["code_commit"])
    require_equal("model identity repo", model_identity.get("model_repo"), binding["model_repo"])
    require_equal("model identity revision", model_identity.get("model_revision"), binding["model_revision"])
    require_equal("model identity config", model_identity.get("config_sha256"), expected_config_value_sha)
    require_equal("model identity manifest", model_identity.get("manifest_sha256"), binding["manifest_sha256"])
    require_equal("model contract repo", model_source.get("model_repo"), binding["model_repo"])
    require_equal("model contract revision", model_source.get("model_revision"), binding["model_revision"])
    require_equal("model contract code", model_source.get("code_commit"), binding["code_commit"])
    require_equal("model contract config", model_source.get("config_sha256"), binding["config_sha256"])
    require_equal("model contract manifest", model_source.get("manifest_sha256"), binding["manifest_sha256"])
    require_equal(
        "model contract parent run identity", model_source.get("run_identity_sha256"),
        model_identity.get("run_identity_sha256"),
    )
    require_equal("canary source manifest", selection.get("source_manifest_sha256"), binding["manifest_sha256"])
    require_equal("canary manifest", selection.get("canary_manifest_sha256"), sha256_file(args.canary_manifest))
    require_equal("open-loop code", open_loop_source.get("code_commit"), binding["code_commit"])
    require_equal("open-loop repo", open_loop_source.get("model_repo"), binding["model_repo"])
    require_equal("open-loop revision", open_loop_source.get("model_revision"), binding["model_revision"])
    require_equal("open-loop config", open_loop_source.get("config_sha256"), binding["config_sha256"])
    require_equal(
        "open-loop encoded canary", open_loop_source.get("encoded_manifest_sha256"),
        sha256_file(args.canary_encoded_manifest),
    )
    require_equal("GPU canary code", gpu_source.get("code_commit"), binding["code_commit"])
    require_equal("GPU canary repo", gpu_source.get("model_repo"), binding["model_repo"])
    require_equal("GPU canary revision", gpu_source.get("model_revision"), binding["model_revision"])
    require_equal("GPU canary config", gpu_source.get("config_sha256"), binding["config_sha256"])
    require_equal("GPU canary manifest", gpu_source.get("canary_manifest_sha256"), sha256_file(args.canary_manifest))
    require_equal("GPU canary source manifest", gpu_source.get("source_manifest_sha256"), binding["manifest_sha256"])
    for label, source in (("conversation", conversation_source),):
        require_equal(f"{label} code", source.get("code_commit"), binding["code_commit"])
        require_equal(f"{label} repo", source.get("model_repo"), binding["model_repo"])
        require_equal(f"{label} revision", source.get("model_revision"), binding["model_revision"])
        require_equal(f"{label} config", source.get("config_sha256"), binding["config_sha256"])
        require_equal(f"{label} source manifest", source.get("source_manifest_sha256"), binding["manifest_sha256"])
        require_equal(f"{label} canary manifest", source.get("canary_manifest_sha256"), sha256_file(args.canary_manifest))
    if provenance_mismatches:
        raise ContractError("stale readiness evidence: " + "; ".join(provenance_mismatches[:8]))

    model = _normalize_block(model_source, MODEL_CHECKS)
    open_loop = _normalize_block(
        open_loop_source, OPEN_LOOP_CHECKS,
        {"paired_feedback_identical": "paired_feedback_byte_identical"},
    )
    conversation = _normalize_block(conversation_source, CONVERSATION_CHECKS)
    gpu = _normalize_block(gpu_source, GPU_CANARY_CHECKS)
    evidence = {
        "schema_version": "1.0.0",
        "target_binding_sha256": binding_hash,
        "model_contract": model,
        "open_loop": open_loop,
        "conversation_canary": conversation,
        "gpu_canary": gpu,
        "source_sha256": {
            "model_contract": sha256_file(args.model_contract),
            "model_run_identity": sha256_file(args.model_run_identity),
            "open_loop": sha256_file(args.open_loop),
            "conversation_canary": sha256_file(args.conversation_canary),
            "gpu_canary": sha256_file(args.gpu_canary),
            "canary_manifest": sha256_file(args.canary_manifest),
            "canary_encoded_manifest": sha256_file(args.canary_encoded_manifest),
            "canary_data": canary_data_sha,
        },
    }
    _validate_schema(evidence, "readiness-evidence.schema.json")
    write_json(args.output, evidence)
    print(f"bound readiness evidence -> {args.output}")
    return 0


def assess(argv: Sequence[str]) -> int:
    parser = _parser("Issue a hash-bound GO/NO_GO artifact for one exact paid scan.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoded-manifest", type=Path, required=True)
    parser.add_argument("--scan-spec", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--limits", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit")
    parser.add_argument("--allow-dirty-for-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    binding = build_target_binding_from_files(
        config_path=args.config,
        manifest_path=args.manifest,
        encoded_manifest_path=args.encoded_manifest,
        scan_spec_path=args.scan_spec,
        code_commit=args.code_commit,
        require_clean=not args.allow_dirty_for_tests,
    )
    evidence = read_json(args.evidence)
    report = assess_readiness(
        read_jsonl(args.manifest), read_json(args.config), read_json(args.scan_spec),
        evidence=evidence,
        limits=_load_limits(args.limits),
        target_binding_sha256=target_binding_sha256(binding),
    )
    artifact = build_authorization_artifact(binding, evidence, report)
    _validate_schema(artifact, "paid-scan-authorization.schema.json")
    write_json(args.output, artifact)
    print(f"{artifact['decision']}: {args.output}")
    return 0 if artifact["decision"] == "GO" else 3


def verify(argv: Sequence[str]) -> int:
    parser = _parser("Cryptographically verify a paid-scan GO artifact against current inputs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoded-manifest", type=Path, required=True)
    parser.add_argument("--scan-spec", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--code-commit")
    parser.add_argument("--allow-dirty-for-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    binding = build_target_binding_from_files(
        config_path=args.config,
        manifest_path=args.manifest,
        encoded_manifest_path=args.encoded_manifest,
        scan_spec_path=args.scan_spec,
        code_commit=args.code_commit,
        require_clean=not args.allow_dirty_for_tests,
    )
    artifact = read_json(args.authorization)
    _validate_schema(artifact, "paid-scan-authorization.schema.json")
    verify_authorization_artifact(artifact, binding)
    print(f"verified GO authorization for {target_binding_sha256(binding)}")
    return 0


def _vram(torch: Any) -> tuple[int, int]:
    device_index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device_index)
    return int(free), int(total)


def run_gpu_canary(argv: Sequence[str]) -> int:
    parser = _parser("Run one bounded identity-patch cell and record GPU/activation/runtime measurements.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload-estimate", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args(argv)
    if os.environ.get("NO_TORCH_COMPILE") != "1" or os.environ.get("NO_CUDA_GRAPH") != "1":
        raise ContractError("GPU canary requires NO_TORCH_COMPILE=1 and NO_CUDA_GRAPH=1")
    config = read_json(args.config)
    model = config.get("model", {})
    if model.get("repo") != MODEL_REPO or model.get("revision") != MODEL_REVISION:
        raise ContractError("GPU canary model identity differs from frozen config")
    if args.layer < 0 or args.layer >= int(model.get("layers", 0)):
        raise ContractError("GPU canary layer is outside the model")
    rows = read_jsonl(args.manifest)
    if len(rows) < 2 or len(rows) > 8:
        raise ContractError("GPU canary manifest must contain 2..8 rows")
    clean = next((row for row in rows if str(row.get("condition", "")).startswith("clean")), None)
    repair = next((row for row in rows if not str(row.get("condition", "")).startswith("clean")), None)
    if clean is None or repair is None:
        raise ContractError("GPU canary requires clean and repair rows")
    match_fields = {
        "scenario_id": str(clean.get("scenario_id", "")),
        "direction_id": str(clean.get("direction_id", "")),
        "speaker_id": str(clean.get("speaker_id", "")),
        "current_value": str(clean.get("current_value", clean.get("new_value", ""))),
    }
    repair_match_fields = {
        "scenario_id": str(repair.get("scenario_id", "")),
        "direction_id": str(repair.get("direction_id", "")),
        "speaker_id": str(repair.get("speaker_id", "")),
        "current_value": str(repair.get("current_value", repair.get("new_value", ""))),
    }
    if any(not value for value in match_fields.values()) or match_fields != repair_match_fields:
        raise ContractError(
            "GPU canary clean/repair rows must match scenario, direction, speaker, and current value")
    selection_path = args.manifest.with_suffix(args.manifest.suffix + ".selection.json")
    selection = read_json(selection_path)
    if selection.get("canary_manifest_sha256") != sha256_file(args.manifest):
        raise ContractError("GPU canary selection sidecar does not bind the canary manifest")
    if selection.get("matched_group") != match_fields:
        raise ContractError("GPU canary selection sidecar has stale matched-group provenance")
    if selection.get("clean_trial_id") != clean.get("trial_id"):
        raise ContractError("GPU canary selection sidecar clean trial differs from the manifest")
    if selection.get("repair_trial_id") != repair.get("trial_id"):
        raise ContractError("GPU canary selection sidecar repair trial differs from the manifest")
    source_manifest_sha256 = str(selection.get("source_manifest_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256) is None:
        raise ContractError("GPU canary selection sidecar has no valid source manifest hash")
    contract = repair.get("conversation_contract", {})
    frame = int(contract.get("query_end_frame", int(repair["frame_count"]))) - 1
    if frame < 0:
        raise ContractError("GPU canary query_end_frame has no overlapping frame")
    canary_run_hash = sha256_value({
        "kind": "bounded_gpu_canary",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "code_commit": _git_commit(),
        "config_sha256": sha256_file(args.config),
        "canary_manifest_sha256": sha256_file(args.manifest),
        "source_manifest_sha256": source_manifest_sha256,
    })
    cell = PatchCell(
        canary_run_hash, str(clean["trial_id"]), str(repair["trial_id"]),
        "resid_post", args.layer, None, (frame,), (frame,), sha256_file(args.config),
    )
    canary_root = args.output.parent / "atomic_canary"
    store = AtomicCellStore(canary_root)
    if args.output.exists() or store.contains(cell):
        raise ContractError(
            "GPU canary evidence must be freshly measured; use a new identity-specific canary output root")

    # CUDA availability is checked before importing the checkpoint backend.  A CPU
    # host must leave a NO_GO, not start a potentially large checkpoint download.
    try:
        import torch as canary_torch
    except ImportError as error:
        raise ContractError("PyTorch is unavailable; STOP before checkpoint load") from error
    if not canary_torch.cuda.is_available():
        raise ContractError("CUDA is unavailable; STOP before checkpoint load or paid scan")

    # Importing this module can initialize checkpoint state, so all static/CUDA
    # checks above precede it.
    from experiments.self_repair.mechanistic.runtime import MoshiBackend

    backend = MoshiBackend(model_repo=MODEL_REPO, model_revision=MODEL_REVISION)
    torch = backend.torch
    if not torch.cuda.is_available():
        raise ContractError("CUDA is unavailable; STOP before paid scan")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_vram_at_start, total_vram = _vram(torch)
    started = time.perf_counter()

    def encode(row: Mapping[str, Any]) -> Any:
        path = args.input_artifact_root / require_relative_uri(str(row["audio_uri"]))
        if sha256_file(path) != row["audio_sha256"]:
            raise ContractError(f"GPU canary WAV hash mismatch: {row['trial_id']}")
        contract = row.get("conversation_contract", {})
        target_frames = int(contract.get("target_end_frame_count", row["frame_count"]))
        encoded = backend.encode_conversation_file(path, target_frame_count=target_frames)
        if encoded.target_frame_count != target_frames or encoded.conversation_codes.shape[-1] != target_frames:
            raise ContractError("GPU canary exact output coverage failed")
        return encoded.conversation_codes

    donor_codes = encode(clean)
    recipient_codes = encode(repair)
    if frame >= int(recipient_codes.shape[-1]) or frame >= int(donor_codes.shape[-1]):
        raise ContractError("GPU canary query anchor lies outside an encoded conversation")
    capture = {"sites": ["resid_post"], "capture_layers": [args.layer], "capture_frames": [frame]}
    donor_result = backend.replay_codes(donor_codes, **capture)
    recipient_result = backend.replay_codes(recipient_codes, **capture)
    event_key = ("resid_post", args.layer, frame)
    identity_tensor = recipient_result.event_tensors.get(event_key)
    if identity_tensor is None:
        raise ContractError("GPU canary failed to capture the requested activation")
    identity_result = backend.replay_codes(
        recipient_codes, replacement={event_key: identity_tensor}, **capture)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    finite = bool(np.isfinite(donor_result.logits).all() and np.isfinite(recipient_result.logits).all())
    identity_noop = bool(np.array_equal(recipient_result.logits, identity_result.logits))
    activation_bytes = int(identity_tensor.nbytes)
    model_frames = int(donor_codes.shape[-1] + 2 * recipient_codes.shape[-1])
    if elapsed <= 0 or model_frames <= 0:
        raise ContractError("GPU canary produced invalid runtime measurements")

    # Exercise atomic resume without repeating model work.
    first_write = store.record(cell, {"status": "completed", "synthetic": False})
    resume_skips_existing = store.contains(cell)
    free_vram_after, total_vram_after = _vram(torch)
    if total_vram_after != total_vram:
        raise ContractError("GPU total memory changed during canary")
    measurements: dict[str, Any] = {
        "completed_cells": 1,
        "failed_cells": 0,
        "duplicate_cells": 0,
        "model_frame_count": model_frames,
        "elapsed_seconds": elapsed,
        "mean_cell_seconds": elapsed,
        "seconds_per_model_frame": elapsed / model_frames,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "free_vram_bytes_at_start": free_vram_at_start,
        "free_vram_bytes_after_canary": free_vram_after,
        "device_total_vram_bytes": total_vram,
        "activation_bytes": activation_bytes,
        "donor_frame_count": int(donor_codes.shape[-1]),
        "recipient_frame_count": int(recipient_codes.shape[-1]),
    }
    if args.workload_estimate is not None:
        static = read_json(args.workload_estimate).get("estimate", {})
        measurements["projected_full_gpu_hours_by_cell"] = (
            int(static.get("cell_count", 0)) * elapsed / 3600)
        measurements["projected_full_gpu_hours_by_model_frame"] = (
            int(static.get("total_model_frames", 0)) * elapsed / model_frames / 3600)
        measurements["projected_activation_tensor_bytes"] = int(static.get("activation_tensor_bytes", 0))
        measurements["projected_total_storage_reserved_bytes"] = int(
            static.get("total_storage_reserved_bytes", 0))
    checks = {
        "bounded_grid": len(rows) <= 8,
        "finite_outputs": finite,
        "no_failed_cells": True,
        "resume_no_duplicates": bool(first_write and resume_skips_existing),
        "peak_vram_measured": measurements["peak_vram_bytes"] > 0,
        "activation_bytes_measured": activation_bytes > 0,
        "runtime_measured": elapsed > 0,
    }
    report = {
        "schema_version": "1.0.0",
        "analysis_status": "bounded_real_gpu_canary",
        "passed": all(checks.values()) and identity_noop,
        "checks": checks,
        "identity_patch_noop": identity_noop,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "code_commit": _git_commit(),
        "config_sha256": sha256_file(args.config),
        "canary_manifest_sha256": sha256_file(args.manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "measurements": measurements,
    }
    write_json(args.output, report)
    print(canonical_json(report))
    return 0 if report["passed"] else 3


COMMANDS = {
    "estimate_mechanistic_workload.py": estimate,
    "select_gpu_canary_manifest.py": select_canary_manifest,
    "assemble_readiness_evidence.py": assemble_evidence,
    "assess_mechanistic_readiness.py": assess,
    "verify_paid_scan_authorization.py": verify,
    "run_bounded_gpu_canary.py": run_gpu_canary,
}


def main_for(program: str, argv: Sequence[str] | None = None) -> int:
    try:
        return COMMANDS[Path(program).name](list(sys.argv[1:] if argv is None else argv))
    except (ContractError, ReadinessError, FileNotFoundError, KeyError, ValueError) as error:
        print(f"READINESS ERROR: {error}", file=sys.stderr)
        return 2
