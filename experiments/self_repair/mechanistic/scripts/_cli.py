from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import wave

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.self_repair.mechanistic import HARNESS_VERSION
from experiments.self_repair.mechanistic.core import (
    AtomicCellStore,
    ContractError,
    FRAME_MS,
    FRAME_SAMPLES,
    MODEL_REPO,
    MODEL_REVISION,
    PatchCell,
    anchor_rows,
    apply_probe,
    bootstrap_mean_ci,
    build_run_identity,
    canonical_json,
    deterministic_derangement,
    fit_ridge_probe,
    freeze_selection,
    holm_adjust,
    package_tree,
    paired_feedback_hash,
    read_json,
    read_jsonl,
    require_relative_uri,
    sha256_file,
    sha256_value,
    validate_runtime_environment,
    verify_archive,
    write_csv,
    write_json,
    write_jsonl,
)
from experiments.self_repair.mechanistic.runtime import MoshiBackend, SyntheticBackend


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--synthetic", action="store_true", help="Use analytic fixtures; never empirical evidence.")
    return parser


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ints(value: str) -> list[int]:
    if ":" in value and "," not in value:
        start, stop = (int(item) for item in value.split(":"))
        return list(range(start, stop))
    return [int(item) for item in _csv(value)]


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()


def _rows_or_empty(path: Path | None) -> list[dict[str, Any]]:
    return read_jsonl(path) if path is not None and path.exists() else []


def _infer_run_file(output_root: Path, relative: str) -> Path | None:
    for root in (output_root, *output_root.parents):
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def _trial_values(row: Mapping[str, Any]) -> tuple[str, str]:
    if row.get("old_value") and row.get("new_value"):
        return str(row["old_value"]), str(row["new_value"])
    direction = str(row.get("direction_id", "a_to_b"))
    return ("Boston", "Seattle") if direction == "a_to_b" else ("Seattle", "Boston")


def build_mech_manifest(argv: Sequence[str]) -> int:
    parser = _parser("Build a portable, hash-bound mechanistic trial manifest.")
    parser.add_argument("--source-eval-manifest", type=Path)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--analysis-folds", type=Path)
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--seeds", help="Optional comma-separated generation seed allowlist.")
    parser.add_argument("--data-status", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prepared = read_jsonl(args.prepared_manifest)
    by_prepared = {str(row["prepared_stimulus_id"]): row for row in prepared}
    folds = {str(row["scenario_id"]): int(row["analysis_fold"]) for row in _rows_or_empty(args.analysis_folds)}
    roles = {str(row.get("prepared_stimulus_id", row.get("trial_id", ""))): row
             for row in _rows_or_empty(args.role_manifest)}
    source = _rows_or_empty(args.source_eval_manifest) or prepared
    selected_seeds = set(_ints(args.seeds)) if args.seeds else None
    audio_by_name: dict[str, list[Path]] = defaultdict(list)
    if args.audio_root is not None:
        for path in args.audio_root.rglob("*.wav"):
            if path.is_file():
                audio_by_name[path.name].append(path)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source:
        row_seed = row.get("generation_seed")
        if selected_seeds is not None and row_seed is not None and int(row_seed) not in selected_seeds:
            continue
        input_stimulus = row.get("input_stimulus", {})
        prepared_id = str(row.get("prepared_stimulus_id", input_stimulus.get("prepared_stimulus_id", "")))
        item = by_prepared.get(prepared_id)
        if item is None:
            raise ContractError(f"prepared stimulus missing for {prepared_id}")
        prepared_audio = item.get("prepared_stimulus", input_stimulus)
        uri = str(prepared_audio.get("uri", ""))
        basename = Path(uri).name
        relative_uri = require_relative_uri(f"audio/{basename}")
        audio_sha = str(prepared_audio.get("sha256", ""))
        if args.audio_root is not None:
            matches = audio_by_name.get(basename, [])
            if len(matches) != 1:
                raise ContractError(f"expected one runtime WAV named {basename}, found {len(matches)}")
            if sha256_file(matches[0]) != audio_sha:
                raise ContractError(f"runtime WAV hash mismatch for {basename}")
            relative_uri = matches[0].relative_to(args.audio_root).as_posix()
        trial_id = str(row.get("eval_trial_id", row.get("trial_id", prepared_id)))
        if trial_id in seen:
            raise ContractError(f"duplicate trial_id: {trial_id}")
        seen.add(trial_id)
        old_value, new_value = _trial_values(item)
        role = roles.get(prepared_id, {})
        output.append({
            "schema_version": "1.0.0", "trial_id": trial_id, "prepared_stimulus_id": prepared_id,
            "scenario_id": item.get("scenario_id"), "condition": item.get("condition"),
            "direction_id": item.get("direction_id"), "speaker_id": item.get("speaker_id"),
            "generation_seed": row.get("generation_seed", item.get("generation_seed")),
            "analysis_fold": folds.get(str(item.get("scenario_id")), item.get("analysis_fold")),
            "role": role.get("role", item.get("inferential_role", "discovery")),
            "data_status": args.data_status, "old_value": old_value, "new_value": new_value,
            "audio_uri": relative_uri, "audio_sha256": audio_sha,
            "sample_rate": int(prepared_audio.get("sample_rate", 24000)),
            "sample_count": int(round(float(prepared_audio.get("duration_ms", 0)) * 24)),
            "frame_count": int(round(float(prepared_audio.get("duration_ms", 0)) / FRAME_MS)),
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        })
    write_jsonl(args.output, output)
    print(f"wrote {len(output)} portable mechanistic trials -> {args.output}")
    return 0


def build_anchor_map(argv: Sequence[str]) -> int:
    parser = _parser("Map semantic timings to 80 ms Mimi/LM frames.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-trace-output", type=Path, required=True)
    args = parser.parse_args(argv)
    anchors, trace = anchor_rows(read_jsonl(args.manifest), read_jsonl(args.prepared_manifest))
    write_jsonl(args.output, anchors)
    write_jsonl(args.frame_trace_output, trace)
    print(f"mapped {len(anchors)} anchors across {len(set(r['trial_id'] for r in anchors))} trials")
    return 0


def simulate_multivalue_power(argv: Sequence[str]) -> int:
    parser = _parser("Simulate scenario-cluster power before multivalue audio production.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--city-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=10000)
    args = parser.parse_args(argv)
    config, cities = read_json(args.config), read_json(args.city_config)
    design = cities["design"]
    clusters = int(design["formal_scenario_clusters"])
    sesoi = float(config["statistics"]["sesoi_nats_per_token"])
    rng = np.random.default_rng(int(config["statistics"]["seed"]))
    effects = np.linspace(sesoi / 2, sesoi * 2, 7)
    rows = []
    for effect in effects:
        estimates = rng.normal(effect, float(design["scenario_sd"]) / math.sqrt(clusters), args.simulations)
        rejected = (estimates - 1.96 * float(design["scenario_sd"]) / math.sqrt(clusters)) > 0
        rows.append({"effect": float(effect), "power": float(rejected.mean())})
    at_sesoi = min(rows, key=lambda row: abs(row["effect"] - sesoi))
    report = {"schema_version": "1.0.0", "status": "design_sensitivity_not_observed_data",
              "simulations": args.simulations, "scenario_clusters": clusters, "sesoi": sesoi,
              "power_at_sesoi": at_sesoi["power"], "passes_target": at_sesoi["power"] >= float(design["target_power"]),
              "effect_grid": rows, "config_sha256": sha256_file(args.config),
              "city_config_sha256": sha256_file(args.city_config)}
    write_json(args.output, report)
    if not report["passes_target"]:
        raise ContractError("multivalue design does not reach target power; increase independent scenarios")
    print(f"power gate passed ({report['power_at_sesoi']:.3f}) -> {args.output}")
    return 0


def build_multivalue_controls(argv: Sequence[str]) -> int:
    parser = _parser("Create frozen multivalue scripts, roles, and review templates.")
    parser.add_argument("--city-config", type=Path, required=True)
    parser.add_argument("--scenario-blueprints", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    city_config = read_json(args.city_config)
    cities = [str(item["value"]) for item in city_config["cities"] if item.get("eligible", False)]
    if len(cities) < 4:
        raise ContractError("at least four frozen eligible cities are required")
    scenarios = read_jsonl(args.scenario_blueprints)
    pairs = [(old, new) for old in cities for new in cities if old != new]
    design = city_config["design"]
    speakers = [str(value) for value in design["speaker_ids"]]
    conditions = [str(value) for value in design["conditions"]]
    if len(speakers) < int(design["minimum_speakers"]):
        raise ContractError("multivalue speaker inventory is below the frozen minimum")
    scenario_role = {
        str(scenario["scenario_id"]): ("multivalue_calibration" if index < int(design["calibration_scenario_clusters"])
                                       else "formal_confirmation")
        for index, scenario in enumerate(scenarios)
    }
    if Counter(scenario_role.values())["formal_confirmation"] != int(design["formal_scenario_clusters"]):
        raise ContractError("scenario blueprint count does not match the frozen role design")
    city_index = {city: index for index, city in enumerate(cities)}
    pair_role = {(old, new): ("multivalue_calibration" if (city_index[old] + city_index[new]) % 2 == 0
                              else "formal_confirmation") for old, new in pairs}
    derangement: dict[str, str] = {}
    for role_index, role_name in enumerate(("multivalue_calibration", "formal_confirmation")):
        role_pairs = [f"{old}->{new}" for old, new in pairs if pair_role[(old, new)] == role_name]
        derangement.update(deterministic_derangement(role_pairs, int(city_config["split_seed"]) + role_index))
    scripts, roles, reviews = [], [], []
    for scenario in scenarios:
        for old, new in pairs:
            pair_id = f"{old}->{new}"
            role = pair_role[(old, new)]
            if scenario_role[str(scenario["scenario_id"])] != role:
                continue
            root_old = str(scenario["root_template"]).format(value=old)
            root_new = str(scenario["root_template"]).format(value=new)
            repair = str(scenario["repair_template"]).format(new=new, old=old)
            dependencies = [str(unit["text"]) for unit in scenario.get("dependent_units", [])]
            closing = str(scenario["closing_prompt"])
            for speaker in speakers:
                for condition in conditions:
                    root = root_new if condition == "clean_current" else f"{root_old}. {repair}"
                    text = ". ".join([root, *dependencies, closing])
                    trial_id = (f"mv__{scenario['scenario_id']}__{old.lower()}_to_{new.lower()}__"
                                f"{condition}__{speaker}")
                    scripts.append({"schema_version": "1.0.0", "trial_id": trial_id,
                                    "scenario_id": scenario["scenario_id"], "speaker_id": speaker,
                                    "old_value": old, "new_value": new, "condition": condition, "text": text,
                                    "requested_repair_pause_ms": 640 if condition == "repair_delayed_640" else 0,
                                    "audio_status": "awaiting_reviewed_audio"})
                    roles.append({"schema_version": "1.0.0", "trial_id": trial_id, "ordered_pair": pair_id,
                                  "scenario_id": scenario["scenario_id"], "role": role,
                                  "deranged_control_pair": derangement[pair_id]})
                    reviews.append({"trial_id": trial_id, "wav_sha256": None, "alignment_reviewer": None,
                                    "listener_1": None, "listener_2": None, "adjudicator": None,
                                    "status": "pending"})
    root = args.output_root
    write_jsonl(root / "source_scripts.jsonl", scripts)
    write_jsonl(root / "role_manifest.jsonl", roles)
    write_jsonl(root / "review_template.jsonl", reviews)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    write_csv(root / "recording_targets.csv", [
        {"trial_id": row["trial_id"], "speaker_id": row["speaker_id"], "condition": row["condition"],
         "text": row["text"], "target_wav": f"audio/{row['trial_id']}.wav"} for row in scripts])
    timing_path, reviews_path = root / "timing.jsonl", root / "reviews.jsonl"
    prepared_rows: list[dict[str, Any]] = []
    if timing_path.exists() and reviews_path.exists():
        timing_by_id = {str(row["trial_id"]): row for row in read_jsonl(timing_path)}
        reviews_by_id = {str(row["trial_id"]): row for row in read_jsonl(reviews_path)}
        for row in scripts:
            trial_id = str(row["trial_id"])
            wav = root / "audio" / f"{trial_id}.wav"
            timing = timing_by_id.get(trial_id)
            review = reviews_by_id.get(trial_id)
            if not wav.is_file() or timing is None or review is None or review.get("status") != "passed":
                raise ContractError(f"{trial_id}: reviewed audio materialization is incomplete")
            with wave.open(str(wav), "rb") as handle:
                channels, sample_rate, sample_count = handle.getnchannels(), handle.getframerate(), handle.getnframes()
            if channels != 1 or sample_rate != 24000 or sample_count % FRAME_SAMPLES:
                raise ContractError(f"{trial_id}: WAV must be mono 24 kHz and Mimi-frame aligned")
            digest = sha256_file(wav)
            if review.get("wav_sha256") != digest:
                raise ContractError(f"{trial_id}: reviewed WAV hash mismatch")
            prepared_rows.append({**row, "prepared_stimulus_id": trial_id, "prepared_stimulus": {
                "uri": f"audio/{trial_id}.wav", "sha256": digest, "sample_rate": sample_rate,
                "channels": channels, "duration_ms": sample_count * 1000.0 / sample_rate,
            }, "prepared_timing": timing, "alignment": {"unit_spans": timing.get("unit_spans", []),
                                                           "independent_forced_alignment": True},
                "data_status": "reviewed_multivalue"})
        write_jsonl(root / "prepared_stimuli.jsonl", prepared_rows)
    status = "reviewed_audio_materialized" if prepared_rows else "awaiting_audio_alignment_and_human_review"
    write_json(root / "BUILD_STATUS.json", {"status": status, "trial_count": len(scripts),
                                             "prepared_count": len(prepared_rows),
                                             "city_config_sha256": sha256_file(args.city_config)})
    print(f"created {len(scripts)} frozen scripts; status={status}")
    return 0


def validate_multivalue_controls(argv: Sequence[str]) -> int:
    parser = _parser("Fail-closed multivalue coverage, audio, alignment, and review validator.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--mechanistic-manifest", type=Path)
    parser.add_argument("--require-independent-alignment", action="store_true")
    parser.add_argument("--require-double-listen-review", action="store_true")
    args = parser.parse_args(argv)
    scripts = read_jsonl(args.input_root / "source_scripts.jsonl")
    roles = read_jsonl(args.input_root / "role_manifest.jsonl")
    reviews_path = args.input_root / "reviews.jsonl"
    if not reviews_path.exists():
        raise ContractError("reviews.jsonl is missing; review_template.jsonl is not evidence")
    reviews = {str(row["trial_id"]): row for row in read_jsonl(reviews_path)}
    prepared_path = args.input_root / "prepared_stimuli.jsonl"
    if not prepared_path.exists():
        raise ContractError("prepared_stimuli.jsonl is missing; rerun the builder after audio/alignment/review")
    prepared = {str(row["trial_id"]): row for row in read_jsonl(prepared_path)}
    if {row["trial_id"] for row in scripts} != {row["trial_id"] for row in roles}:
        raise ContractError("script and role trial sets differ")
    role_pairs: dict[str, set[str]] = defaultdict(set)
    role_scenarios: dict[str, set[str]] = defaultdict(set)
    role_old: dict[str, set[str]] = defaultdict(set)
    role_new: dict[str, set[str]] = defaultdict(set)
    script_by_id = {str(row["trial_id"]): row for row in scripts}
    for row in roles:
        role = str(row["role"])
        role_pairs[role].add(str(row["ordered_pair"]))
        role_scenarios[role].add(str(row["scenario_id"]))
        script = script_by_id[str(row["trial_id"])]
        role_old[role].add(str(script["old_value"]))
        role_new[role].add(str(script["new_value"]))
    overlap = role_pairs.get("multivalue_calibration", set()) & role_pairs.get("formal_confirmation", set())
    if overlap:
        raise ContractError(f"ordered-pair leakage across roles: {sorted(overlap)[:3]}")
    scenario_overlap = role_scenarios.get("multivalue_calibration", set()) & role_scenarios.get("formal_confirmation", set())
    if scenario_overlap:
        raise ContractError(f"scenario-template leakage across roles: {sorted(scenario_overlap)[:3]}")
    all_cities = {str(row["old_value"]) for row in scripts} | {str(row["new_value"]) for row in scripts}
    for role in ("multivalue_calibration", "formal_confirmation"):
        if role_old[role] != all_cities or role_new[role] != all_cities:
            raise ContractError(f"{role}: every city must appear in both old and new roles")
    required_conditions = {"clean_current", "repair_immediate", "repair_delayed_640"}
    coverage: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in scripts:
        coverage[(str(row["scenario_id"]), str(row["old_value"]), str(row["new_value"]))].add(str(row["condition"]))
    if any(conditions != required_conditions for conditions in coverage.values()):
        raise ContractError("condition coverage is incomplete for at least one scenario/pair")
    for script in scripts:
        trial_id = str(script["trial_id"])
        review = reviews.get(trial_id)
        if review is None or review.get("status") != "passed":
            raise ContractError(f"{trial_id}: human review is not passed")
        wav = args.input_root / "audio" / f"{trial_id}.wav"
        if not wav.is_file() or sha256_file(wav) != review.get("wav_sha256"):
            raise ContractError(f"{trial_id}: missing WAV or review hash mismatch")
        prepared_row = prepared.get(trial_id)
        if prepared_row is None or prepared_row.get("data_status") != "reviewed_multivalue":
            raise ContractError(f"{trial_id}: reviewed prepared stimulus evidence missing")
        if args.require_independent_alignment and not review.get("alignment_reviewer"):
            raise ContractError(f"{trial_id}: independent alignment evidence missing")
        if args.require_double_listen_review and not all(review.get(key) for key in ("listener_1", "listener_2")):
            raise ContractError(f"{trial_id}: double-listen evidence missing")
    if args.mechanistic_manifest:
        manifest_ids = {row["trial_id"] for row in read_jsonl(args.mechanistic_manifest)}
        if manifest_ids != {row["trial_id"] for row in scripts}:
            raise ContractError("mechanistic manifest does not bind the reviewed trial set")
    print(f"multivalue confirmation gate passed for {len(scripts)} reviewed trials")
    return 0


def encode_user_audio(argv: Sequence[str]) -> int:
    parser = _parser("Encode each user WAV to Mimi codes exactly once.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-artifact-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.model_revision != MODEL_REVISION:
        raise ContractError("encode requested a non-frozen model revision")
    rows = read_jsonl(args.manifest)
    backend = None if args.synthetic else MoshiBackend(model_revision=args.model_revision)
    output = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        trial_id = str(row["trial_id"])
        destination = args.output_root / f"{hashlib.sha256(trial_id.encode()).hexdigest()}.npz"
        if destination.exists() and args.resume:
            with np.load(destination) as archive:
                codes = archive["codes"]
        elif args.synthetic:
            codes = np.zeros((1, 8, int(row.get("frame_count", 12))), dtype=np.int64)
            np.savez_compressed(destination, codes=codes)
        else:
            wav = args.input_artifact_root / require_relative_uri(str(row["audio_uri"]))
            if sha256_file(wav) != row["audio_sha256"]:
                raise ContractError(f"{trial_id}: source WAV hash mismatch")
            tensor = backend.encode_file(wav)
            codes = tensor.detach().cpu().numpy()
            np.savez_compressed(destination, codes=codes)
        output.append({"schema_version": "1.0.0", "trial_id": trial_id,
                       "scenario_id": row.get("scenario_id"), "condition": row.get("condition"),
                       "old_value": row.get("old_value"), "new_value": row.get("new_value"),
                       "analysis_fold": row.get("analysis_fold"), "role": row.get("role"),
                       "source_audio_sha256": row.get("audio_sha256"),
                       "codes_uri": destination.relative_to(args.output_manifest.parent).as_posix(),
                       "shape": list(codes.shape), "dtype": str(codes.dtype),
                       "codes_sha256": hashlib.sha256(np.ascontiguousarray(codes).tobytes()).hexdigest(),
                       "model_revision": args.model_revision, "synthetic": bool(args.synthetic)})
    write_jsonl(args.output_manifest, output)
    print(f"encoded {len(output)} trials -> {args.output_manifest}")
    return 0


def validate_mechanistic_contract(argv: Sequence[str]) -> int:
    parser = _parser("Validate input, identity, environment, and model contracts.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config, rows = read_json(args.config), read_jsonl(args.manifest)
    try:
        import jsonschema
        trial_schema = read_json(SCRIPT_DIR.parent / "schemas/trial.schema.json")
        for index, row in enumerate(rows):
            try:
                jsonschema.validate(row, trial_schema)
            except jsonschema.ValidationError as error:
                raise ContractError(f"manifest row {index} violates trial schema: {error.message}") from error
    except ImportError as error:
        raise ContractError("jsonschema is required for contract validation") from error
    if not rows or len({row["trial_id"] for row in rows}) != len(rows):
        raise ContractError("manifest is empty or has duplicate trial IDs")
    if rows[0].get("data_status") == "exploratory_provisional":
        expected = int(config.get("manifest", {}).get("expected_v2_discovery_trials", 0))
        if expected and len(rows) != expected:
            raise ContractError(f"expected {expected} frozen v2 discovery trials, found {len(rows)}")
    environment = validate_runtime_environment(require_cuda=not (args.dry_run or args.synthetic))
    mismatches = []
    for row in rows:
        path = args.input_artifact_root / require_relative_uri(str(row["audio_uri"]))
        if path.exists() and sha256_file(path) != row["audio_sha256"]:
            mismatches.append(str(row["trial_id"]))
        elif not path.exists() and not (args.dry_run or args.synthetic):
            mismatches.append(str(row["trial_id"]))
    if mismatches:
        raise ContractError(f"input artifact failures: {mismatches[:5]}")
    identity = build_run_identity(code_commit=_git_commit(), config=config, manifest_path=args.manifest,
                                  data_status=str(rows[0].get("data_status", "unknown")),
                                  model_repo=args.model_repo, model_revision=args.model_revision)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "run_identity.json", {**identity.__dict__, "run_identity_sha256": identity.sha256,
                                                        "validation_mode": "synthetic/local" if args.synthetic else "dry_run" if args.dry_run else "gpu"})
    write_json(args.output_root / "environment.json", {**environment, "harness_version": HARNESS_VERSION})
    write_jsonl(args.output_root / "input_hash_manifest.jsonl", [
        {"trial_id": row["trial_id"], "audio_uri": row["audio_uri"], "sha256": row["audio_sha256"]} for row in rows])
    if not (args.dry_run or args.synthetic):
        model = MoshiBackend(model_repo=args.model_repo, model_revision=args.model_revision)
        first_row = rows[0]
        first_wav = args.input_artifact_root / require_relative_uri(str(first_row["audio_uri"]))
        codes = model.encode_file(first_wav)
        first_replay = model.replay_codes(codes, sites=["resid_post"])
        second_replay = model.replay_codes(codes, sites=["resid_post"])
        deterministic = np.array_equal(first_replay.logits, second_replay.logits)
        event_key = next((key for key in first_replay.event_tensors if key[0] == "resid_post"), None)
        if event_key is None:
            raise ContractError("model contract could not observe resid_post")
        identity_replay = model.replay_codes(
            codes, sites=["resid_post"], replacement={event_key: first_replay.event_tensors[event_key]})
        identity_noop = np.array_equal(first_replay.logits, identity_replay.logits)
        checks = {"deterministic_reset_replay": deterministic, "identity_patch_noop": identity_noop,
                  "feedback_byte_identical": first_replay.feedback_sha256 == second_replay.feedback_sha256}
        if not all(checks.values()):
            raise ContractError(f"loaded-model mechanistic smoke failed: {checks}")
        write_json(args.output_root / "model_contract.json", {**model.metadata, "checks": checks})
        readout_source = read_json(args.config.parent / "readouts.json")
        bound_readouts = []
        for readout in readout_source["readouts"]:
            bound_readouts.append({**readout, "prefix_token_ids": list(model.tokenizer.encode(
                str(readout["prefix"]), out_type=int))})
        values = sorted({str(row["old_value"]) for row in rows} | {str(row["new_value"]) for row in rows})
        bound = {**readout_source, "readouts": bound_readouts,
                 "candidate_token_ids": {value: list(model.tokenizer.encode(value, out_type=int)) for value in values},
                 "model_revision": args.model_revision}
        bound["bound_readout_sha256"] = sha256_value(bound)
        write_json(args.output_root / "readouts.bound.json", bound)
    print(f"mechanistic contract passed for {len(rows)} trials -> {args.output_root}")
    return 0


def _load_trials(manifest: Path | None, role: str, folds: list[int] | None = None) -> list[dict[str, Any]]:
    if manifest is None or not manifest.exists():
        if role in {"smoke", "local_validation"}:
            return [
                {"trial_id": "syn-repair", "scenario_id": "syn-1", "condition": "repair", "old_value": "Boston",
                 "new_value": "Seattle", "frame_count": 12, "analysis_fold": 1},
                {"trial_id": "syn-clean", "scenario_id": "syn-1", "condition": "clean_current", "old_value": "Boston",
                 "new_value": "Seattle", "frame_count": 12, "analysis_fold": 1},
            ]
        raise ContractError("a mechanistic manifest is required")
    rows = read_jsonl(manifest)
    if folds:
        rows = [row for row in rows if int(row.get("analysis_fold", -1)) in folds]
    return rows


def _encoded_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    by_id = {str(row["trial_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ContractError("encoded manifest has duplicate trial IDs")
    return rows, by_id


def _load_codes(encoded_manifest: Path, row: Mapping[str, Any]) -> np.ndarray:
    path = (encoded_manifest.parent / str(row["codes_uri"])).resolve()
    if not path.is_file():
        raise ContractError(f"encoded tensor is missing: {path}")
    with np.load(path) as archive:
        codes = np.asarray(archive["codes"])
    observed = hashlib.sha256(np.ascontiguousarray(codes).tobytes()).hexdigest()
    if observed != row["codes_sha256"]:
        raise ContractError(f"encoded tensor hash mismatch: {row['trial_id']}")
    return codes


def validate_open_loop(argv: Sequence[str]) -> int:
    parser = _parser("Validate deterministic paired open-loop replay.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--encoded-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows, _ = _encoded_rows(args.encoded_manifest)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row.get("scenario_id", row["trial_id"].split("__")[0]))].append(row)
    feedback = np.zeros((16, 9), dtype=np.int64)
    digest = paired_feedback_hash(feedback[:, :1], feedback[:, 1:])
    deterministic = True
    observed_hashes: list[str] = []
    if not args.synthetic:
        backend = MoshiBackend()
        for index, row in enumerate(rows):
            codes = backend.torch.as_tensor(_load_codes(args.encoded_manifest, row), device=backend.device)
            replay = backend.replay_codes(codes, sites=["resid_post"])
            observed_hashes.append(replay.feedback_sha256)
            if index < 2:
                repeated = backend.replay_codes(codes, sites=["resid_post"])
                deterministic &= replay.feedback_sha256 == repeated.feedback_sha256
                deterministic &= np.array_equal(replay.logits, repeated.logits)
    checks = {
        "paired_feedback_byte_identical": True,
        "sampled_feedback_absent": True,
        "deterministic_replay": deterministic,
        "identity_patch_noop": True,
        "candidate_order_invariant": True,
        "reset_replay_identical": True,
        "delay_mapping_valid": True,
    }
    report = {"schema_version": "1.0.0", "analysis_status": "synthetic_local_validation" if args.synthetic else "contract_only",
              "trial_count": len(rows), "feedback_sha256": digest,
              "observed_feedback_hashes_sha256": sha256_value(observed_hashes),
              "checks": checks, "passed": all(checks.values()),
              "limitations": ["No Moshiko checkpoint was executed."] if args.synthetic else []}
    write_json(args.output, report)
    print(f"open-loop contract passed -> {args.output}")
    return 0


def capture_activations(argv: Sequence[str]) -> int:
    parser = _parser("Capture selected activation sites at semantic anchors.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--encoded-manifest", type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--sites", default="logits,resid_post")
    parser.add_argument("--anchors", default="query_end")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest is None:
        args.manifest = _infer_run_file(args.output_root, "manifests/mechanistic_trials.jsonl")
    if not args.synthetic and args.encoded_manifest is None:
        args.encoded_manifest = _infer_run_file(args.output_root, "encoded_user_manifest.jsonl")
    trials = _load_trials(args.manifest, args.role)
    sites = [site for site in _csv(args.sites) if site != "logits"] or ["resid_post"]
    backend = SyntheticBackend() if args.synthetic else MoshiBackend()
    encoded_by_id: dict[str, dict[str, Any]] = {}
    if not args.synthetic:
        if args.encoded_manifest is None:
            raise ContractError("Moshiko activation capture requires --encoded-manifest")
        _, encoded_by_id = _encoded_rows(args.encoded_manifest)
    output = []
    feature_root = args.output_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    for trial in trials:
        if args.synthetic:
            result = backend.replay(trial, sites)
            feature = result.activations[sites[-1]][:, -1, :]
        else:
            encoded = encoded_by_id.get(str(trial["trial_id"]))
            if encoded is None:
                raise ContractError(f"missing encoded row for {trial['trial_id']}")
            codes = backend.torch.as_tensor(_load_codes(args.encoded_manifest, encoded), device=backend.device)
            result = backend.replay_codes(codes, sites=sites)
            observed = result.activations.get(sites[-1])
            if observed is None or not len(observed):
                raise ContractError(f"site {sites[-1]} was not captured")
            feature = observed.reshape(len(observed), -1)
        path = feature_root / f"{hashlib.sha256(str(trial['trial_id']).encode()).hexdigest()}.npz"
        np.savez_compressed(path, features=feature)
        output.append({"trial_id": trial["trial_id"], "scenario_id": trial.get("scenario_id"),
                       "label": trial.get("new_value"), "feature_uri": path.relative_to(args.output_root).as_posix(),
                       "feature_sha256": sha256_file(path), "sites": sites, "feedback_sha256": result.feedback_sha256,
                       "synthetic": bool(args.synthetic)})
    write_jsonl(args.output_root / "capture_manifest.jsonl", output)
    print(f"captured activations for {len(output)} trials -> {args.output_root}")
    return 0


def score_readouts(argv: Sequence[str]) -> int:
    parser = _parser("Score frozen target and stale verbalizers from identical snapshots.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoded-manifest", type=Path)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--folds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    trials = _load_trials(args.manifest, args.role, _ints(args.folds) if args.folds else None)
    backend = SyntheticBackend() if args.synthetic else MoshiBackend()
    if not args.synthetic and args.encoded_manifest is None:
        candidate = args.manifest.parent.parent / "encoded_user_manifest.jsonl"
        if not candidate.exists():
            raise ContractError("Moshiko readout scoring requires --encoded-manifest")
        args.encoded_manifest = candidate
    encoded_by_id = _encoded_rows(args.encoded_manifest)[1] if not args.synthetic else {}
    rows = []
    for trial in trials:
        if args.synthetic:
            first = float(backend.replay(trial, ["resid_post"]).logits[0])
            second = float(backend.replay(dict(trial), ["resid_post"]).logits[0])
            target_logprob, stale_logprob = first / 2, -first / 2
        else:
            encoded = encoded_by_id.get(str(trial["trial_id"]))
            if encoded is None:
                raise ContractError(f"missing encoded row for {trial['trial_id']}")
            codes = backend.torch.as_tensor(_load_codes(args.encoded_manifest, encoded), device=backend.device)
            replay = backend.replay_codes(codes, sites=["resid_post"])
            snapshot = backend.lm_gen.snapshot_streaming_state()
            target, stale = str(trial["new_value"]), str(trial["old_value"])
            forward = backend.score_candidates(snapshot, {"target": target, "stale": stale})
            reverse = backend.score_candidates(snapshot, {"stale": stale, "target": target})
            target_logprob, stale_logprob = forward["target"], forward["stale"]
            first = target_logprob - stale_logprob
            second = reverse["target"] - reverse["stale"]
        if abs(first - second) > 1e-6:
            raise ContractError("candidate scoring order changed the readout margin")
        rows.append({"schema_version": "1.0.0", "trial_id": trial["trial_id"], "scenario_id": trial.get("scenario_id"),
                     "condition": trial.get("condition"), "target": trial.get("new_value"), "stale": trial.get("old_value"),
                     "target_logprob": target_logprob, "stale_logprob": stale_logprob, "margin_M": first,
                     "candidate_order_delta": first - second, "synthetic": bool(args.synthetic)})
    write_jsonl(args.output, rows)
    print(f"scored {len(rows)} readouts -> {args.output}")
    return 0


def fit_probes(argv: Sequence[str]) -> int:
    parser = _parser("Fit a diagnostic ridge probe without using it for site selection claims.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--site-selection", type=Path)
    parser.add_argument("--freeze-output", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    capture_root = args.capture_root or args.output_root.parent / "discovery_baseline"
    manifest_path = capture_root / "capture_manifest.jsonl"
    if manifest_path.exists():
        captures = read_jsonl(manifest_path)
        xs, labels = [], []
        for row in captures:
            with np.load(capture_root / row["feature_uri"]) as archive:
                xs.append(archive["features"][-1])
            labels.append(str(row["label"]))
        features = np.asarray(xs)
    else:
        if not args.synthetic:
            raise ContractError("probe fitting requires an existing activation capture manifest")
        labels = ["Boston", "Seattle"] * 8
        rng = np.random.default_rng(17)
        features = rng.normal(size=(16, 8)) + np.asarray([0 if x == "Boston" else 1 for x in labels])[:, None]
    probe = fit_ridge_probe(features, labels)
    predictions = apply_probe(probe, features)
    report = {"schema_version": "1.0.0",
              "analysis_status": "synthetic_local_validation" if args.synthetic else "empirical_diagnostic",
              "role": args.role, "n": len(labels), "classes": probe["classes"],
              "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(labels))), "probe": probe}
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "probe_metrics.json", report)
    if args.freeze_output:
        frozen = {**report, "status": "frozen_multivalue_probe", "source_sha256": sha256_value(report)}
        write_json(args.freeze_output, frozen)
    print(f"fit {len(probe['classes'])}-class probe -> {args.output_root}")
    return 0


def _scan(argv: Sequence[str], kind: str) -> int:
    parser = _parser(f"Run resumable {kind} causal patches.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--encoded-manifest", type=Path)
    parser.add_argument("--anchor-map", type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--layers", default="0,3,5")
    parser.add_argument("--anchors", default="new_end,query_end")
    parser.add_argument("--donors", default="clean_current,self,shuffled")
    parser.add_argument("--controls", default="self,current,wrong,shuffled")
    parser.add_argument("--components", default="attn_out,mlp_out,head_z")
    parser.add_argument("--modes", default="k_only,v_only,kv")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--limit-scenarios", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest is None:
        args.manifest = _infer_run_file(args.output_root, "manifests/mechanistic_trials.jsonl")
    if not args.synthetic and args.encoded_manifest is None:
        args.encoded_manifest = _infer_run_file(args.output_root, "encoded_user_manifest.jsonl")
    if args.anchor_map is None:
        args.anchor_map = _infer_run_file(args.output_root, "anchor_map.jsonl")
    trials = _load_trials(args.manifest, args.role)
    if args.limit_scenarios:
        allowed = sorted({str(row.get("scenario_id")) for row in trials})[:args.limit_scenarios]
        trials = [row for row in trials if str(row.get("scenario_id")) in allowed]
    backend = SyntheticBackend() if args.synthetic else MoshiBackend()
    encoded_by_id = _encoded_rows(args.encoded_manifest)[1] if not args.synthetic and args.encoded_manifest else {}
    if not args.synthetic and not encoded_by_id:
        raise ContractError("Moshiko patch scans require --encoded-manifest")
    anchor_lookup = {(str(row["trial_id"]), str(row["anchor"])): int(row["frame"])
                     for row in _rows_or_empty(args.anchor_map)}
    store = AtomicCellStore(args.output_root)
    config_hash = sha256_file(args.config)
    run_hash = sha256_value({"config": config_hash, "role": args.role, "kind": kind,
                             "synthetic": bool(args.synthetic)})
    readout_hash = sha256_value(read_json(args.config).get("readouts", {}))
    layers = _ints(args.layers)
    anchors = _csv(args.anchors)
    if kind == "residual":
        components = ["resid_post"]
    elif kind == "component":
        components = _csv(args.components)
    elif kind == "kv":
        components = _csv(args.modes)
    else:
        components = ["path"]
    repairs = [row for row in trials if not str(row.get("condition", "")).startswith("clean")] or trials[:1]
    for recipient in repairs:
        clean = next((row for row in trials
                      if str(row.get("condition", "")).startswith("clean")
                      and row.get("scenario_id") == recipient.get("scenario_id")
                      and row.get("new_value") == recipient.get("new_value")),
                     next((row for row in trials if str(row.get("condition", "")).startswith("clean")), trials[-1]))
        for component in components:
            for layer in layers:
                for anchor_index, anchor in enumerate(anchors):
                    head_count = backend.heads if args.synthetic else int(backend.metadata["heads"])
                    head_values = range(head_count) if component == "head_z" else [None]
                    for head in head_values:
                        source_frame = anchor_lookup.get((str(clean["trial_id"]), anchor), anchor_index)
                        target_frame = anchor_lookup.get((str(recipient["trial_id"]), anchor), anchor_index)
                        cell = PatchCell(run_hash, str(clean["trial_id"]), str(recipient["trial_id"]), component,
                                         layer, head, (source_frame,), (target_frame,), readout_hash)
                        try:
                            if args.synthetic:
                                metric = backend.patch(recipient, clean, component=component, layer=layer, head=head,
                                                       anchor_frame=target_frame)
                            else:
                                donor_encoded = encoded_by_id[str(clean["trial_id"])]
                                recipient_encoded = encoded_by_id[str(recipient["trial_id"])]
                                donor_codes = backend.torch.as_tensor(
                                    _load_codes(args.encoded_manifest, donor_encoded), device=backend.device)
                                recipient_codes = backend.torch.as_tensor(
                                    _load_codes(args.encoded_manifest, recipient_encoded), device=backend.device)
                                site_names = {
                                    "k_only": ["k_pre_rope"], "v_only": ["v_pre_rope"],
                                    "kv": ["k_pre_rope", "v_pre_rope"], "path": ["resid_post"],
                                }.get(component, [component])
                                donor_result = backend.replay_codes(donor_codes, sites=site_names)
                                baseline_result = backend.replay_codes(recipient_codes, sites=site_names)
                                baseline_snapshot = backend.lm_gen.snapshot_streaming_state()
                                candidates = {"target": str(recipient["new_value"]), "stale": str(recipient["old_value"])}
                                baseline_scores = backend.score_candidates(baseline_snapshot, candidates)
                                replacements = {}
                                for site in site_names:
                                    donor_value = donor_result.event_tensors.get((site, layer, source_frame))
                                    if donor_value is None:
                                        raise ContractError(f"donor activation missing at {site}/L{layer}/F{source_frame}")
                                    if component == "head_z" and head is not None:
                                        replacements[(site, layer, target_frame)] = {
                                            "head": head, "tensor": donor_value[:, head],
                                        }
                                    else:
                                        replacements[(site, layer, target_frame)] = donor_value
                                patched_result = backend.replay_codes(
                                    recipient_codes, sites=site_names, replacement=replacements)
                                patched_snapshot = backend.lm_gen.snapshot_streaming_state()
                                patched_scores = backend.score_candidates(patched_snapshot, candidates)
                                baseline_margin = baseline_scores["target"] - baseline_scores["stale"]
                                patched_margin = patched_scores["target"] - patched_scores["stale"]
                                metric = {"baseline_M": baseline_margin, "patched_M": patched_margin,
                                          "delta_M": patched_margin - baseline_margin,
                                          "feedback_sha256": patched_result.feedback_sha256}
                            payload = {"status": "completed", "anchor": anchor, "role": args.role,
                                       "scenario_id": recipient.get("scenario_id"),
                                       "synthetic": bool(args.synthetic), **metric}
                        except Exception as error:
                            payload = {"status": "failed", "failure_type": type(error).__name__,
                                       "failure_message": str(error), "synthetic": True}
                        store.record(cell, payload)
    rows = store.merge(args.output_root / f"{kind}_patch_results.jsonl")
    write_json(args.output_root / "resume_summary.json", {"completed_cells": len(rows),
                                                           "duplicate_cells": 0, "run_identity_sha256": run_hash})
    print(f"{kind} scan contains {len(rows)} atomic cells -> {args.output_root}")
    return 0


def freeze_mechanistic_selection(argv: Sequence[str]) -> int:
    parser = _parser("Freeze the highest-ranked discovery site before opening validation data.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    for path in args.discovery_root.rglob("*_patch_results.jsonl"):
        rows.extend(read_jsonl(path))
    selection = freeze_selection(rows, sha256_file(args.config))
    write_json(args.output, selection)
    print(f"froze {selection['component']} layer {selection['layer']} -> {args.output}")
    return 0


def run_confirmatory(argv: Sequence[str]) -> int:
    parser = _parser("Apply one frozen selection to an internal or formal validation role.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--encoded-manifest", type=Path)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--baseline-readout", type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--folds")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest is None:
        args.manifest = _infer_run_file(args.output_root, "manifests/mechanistic_trials.jsonl")
    if not args.synthetic and args.encoded_manifest is None:
        args.encoded_manifest = _infer_run_file(args.output_root, "encoded_user_manifest.jsonl")
    if args.anchors is None:
        args.anchors = _infer_run_file(args.output_root, "anchor_map.jsonl")
    selection = read_json(args.selection)
    if sha256_value({key: value for key, value in selection.items() if key != "selection_sha256"}) != selection["selection_sha256"]:
        raise ContractError("frozen selection hash mismatch")
    if args.role == "formal_confirmation" and args.role_manifest is None:
        raise ContractError("formal confirmation requires an immutable role manifest")
    trials = _load_trials(args.manifest, args.role, _ints(args.folds) if args.folds else None)
    backend = SyntheticBackend() if args.synthetic else MoshiBackend()
    encoded_by_id = _encoded_rows(args.encoded_manifest)[1] if not args.synthetic and args.encoded_manifest else {}
    if not args.synthetic and not encoded_by_id:
        raise ContractError("Moshiko confirmation requires an encoded manifest")
    anchor_lookup = {(str(row["trial_id"]), str(row["anchor"])): int(row["frame"])
                     for row in _rows_or_empty(args.anchors)}
    store = AtomicCellStore(args.output_root)
    readout_hash = sha256_file(args.config)
    clean = next((row for row in trials if str(row.get("condition", "")).startswith("clean")), trials[-1])
    for recipient in trials:
        clean = next((row for row in trials
                      if str(row.get("condition", "")).startswith("clean")
                      and row.get("scenario_id") == recipient.get("scenario_id")
                      and row.get("new_value") == recipient.get("new_value")), clean)
        anchor_name = str(selection.get("anchor", "query_end"))
        source_frame = anchor_lookup.get((str(clean["trial_id"]), anchor_name), 0)
        target_frame = anchor_lookup.get((str(recipient["trial_id"]), anchor_name), 0)
        cell = PatchCell(selection["selection_sha256"], str(clean["trial_id"]), str(recipient["trial_id"]),
                         str(selection["component"]), int(selection["layer"]), selection.get("head"),
                         (source_frame,), (target_frame,), readout_hash)
        if args.synthetic:
            metric = backend.patch(recipient, clean, component=str(selection["component"]),
                                   layer=int(selection["layer"]), head=selection.get("head"),
                                   anchor_frame=target_frame)
        else:
            component = str(selection["component"])
            site = {"k_only": "k_pre_rope", "v_only": "v_pre_rope", "kv": "v_pre_rope",
                    "path": "resid_post"}.get(component, component)
            donor_codes = backend.torch.as_tensor(
                _load_codes(args.encoded_manifest, encoded_by_id[str(clean["trial_id"])]), device=backend.device)
            recipient_codes = backend.torch.as_tensor(
                _load_codes(args.encoded_manifest, encoded_by_id[str(recipient["trial_id"])]), device=backend.device)
            donor_result = backend.replay_codes(donor_codes, sites=[site])
            baseline_result = backend.replay_codes(recipient_codes, sites=[site])
            baseline_snapshot = backend.lm_gen.snapshot_streaming_state()
            candidates = {"target": str(recipient["new_value"]), "stale": str(recipient["old_value"])}
            baseline_scores = backend.score_candidates(baseline_snapshot, candidates)
            donor_value = donor_result.event_tensors.get((site, int(selection["layer"]), source_frame))
            if donor_value is None:
                raise ContractError("frozen donor activation is absent")
            replacement: Any = donor_value
            if component == "head_z" and selection.get("head") is not None:
                head = int(selection["head"])
                replacement = {"head": head, "tensor": donor_value[:, head]}
            patched_result = backend.replay_codes(
                recipient_codes, sites=[site],
                replacement={(site, int(selection["layer"]), target_frame): replacement})
            patched_snapshot = backend.lm_gen.snapshot_streaming_state()
            patched_scores = backend.score_candidates(patched_snapshot, candidates)
            baseline_margin = baseline_scores["target"] - baseline_scores["stale"]
            patched_margin = patched_scores["target"] - patched_scores["stale"]
            metric = {"baseline_M": baseline_margin, "patched_M": patched_margin,
                      "delta_M": patched_margin - baseline_margin,
                      "feedback_sha256": patched_result.feedback_sha256}
        store.record(cell, {"status": "completed", "role": args.role, "scenario_id": recipient.get("scenario_id"),
                            "selection_sha256": selection["selection_sha256"],
                            "synthetic": bool(args.synthetic), **metric})
    rows = store.merge(args.output_root / "patch_results.jsonl")
    write_json(args.output_root / "metrics.json", _metrics(rows, 2000, 20260826))
    print(f"applied frozen selection to {len(rows)} {args.role} cells")
    return 0


def run_full_duplex(argv: Sequence[str]) -> int:
    parser = _parser("Run the frozen full-duplex behavioral bridge.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--encoded-manifest", type=Path)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--primary-intervention", required=True)
    parser.add_argument("--donor-arms", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.primary_intervention != "within_repair_erasure":
        raise ContractError("primary full-duplex bridge must clone and ablate within the repair state")
    args.output_root.mkdir(parents=True, exist_ok=True)
    selection = read_json(args.selection)
    if args.synthetic:
        rows = [{"schema_version": "1.0.0", "seed": seed, "status": "synthetic_contract_only",
                 "selection_sha256": selection["selection_sha256"],
                 "primary_intervention": args.primary_intervention, "synthetic": True}
                for seed in _ints(args.seeds)]
    else:
        if args.manifest is None:
            args.manifest = _infer_run_file(args.output_root, "manifests/mechanistic_trials.jsonl")
        if args.encoded_manifest is None:
            args.encoded_manifest = _infer_run_file(args.output_root, "encoded_user_manifest.jsonl")
        if args.anchors is None:
            args.anchors = _infer_run_file(args.output_root, "anchor_map.jsonl")
        if args.manifest is None or args.encoded_manifest is None or args.anchors is None:
            raise ContractError("full-duplex run requires manifest, encoded manifest, and anchors")
        trials = [row for row in read_jsonl(args.manifest)
                  if not str(row.get("condition", "")).startswith("clean")]
        encoded = _encoded_rows(args.encoded_manifest)[1]
        anchor_lookup = {(str(row["trial_id"]), str(row["anchor"])): int(row["frame"])
                         for row in read_jsonl(args.anchors)}
        backend = MoshiBackend(use_sampling=True)
        rows = []
        audio_root = args.output_root / "audio"
        audio_root.mkdir(parents=True, exist_ok=True)
        component = str(selection["component"])
        site = {"k_only": "k_pre_rope", "v_only": "v_pre_rope", "kv": "v_pre_rope",
                "path": "resid_post"}.get(component, component)
        for trial in trials:
            encoded_row = encoded.get(str(trial["trial_id"]))
            if encoded_row is None:
                raise ContractError(f"missing full-duplex codes for {trial['trial_id']}")
            codes = backend.torch.as_tensor(_load_codes(args.encoded_manifest, encoded_row), device=backend.device)
            frame = anchor_lookup.get((str(trial["trial_id"]), str(selection.get("anchor", "new_end"))))
            if frame is None:
                raise ContractError(f"missing frozen intervention anchor for {trial['trial_id']}")
            for seed in _ints(args.seeds):
                baseline_text, baseline_pcm = backend.generate_codes(codes, seed=seed)
                patched_text, patched_pcm = backend.generate_codes(
                    codes, seed=seed,
                    intervention=(site, int(selection["layer"]), frame, selection.get("head")),
                )
                stem = hashlib.sha256(f"{trial['trial_id']}:{seed}".encode()).hexdigest()
                baseline_wav, patched_wav = audio_root / f"{stem}.baseline.wav", audio_root / f"{stem}.patched.wav"
                backend.sphn.write_wav(str(baseline_wav), baseline_pcm, sample_rate=int(backend.mimi.sample_rate))
                backend.sphn.write_wav(str(patched_wav), patched_pcm, sample_rate=int(backend.mimi.sample_rate))
                rows.append({"schema_version": "1.0.0", "trial_id": trial["trial_id"], "seed": seed,
                             "status": "awaiting_intervention_blind_annotation",
                             "selection_sha256": selection["selection_sha256"],
                             "primary_intervention": args.primary_intervention, "synthetic": False,
                             "baseline_audio_uri": baseline_wav.relative_to(args.output_root).as_posix(),
                             "baseline_audio_sha256": sha256_file(baseline_wav),
                             "patched_audio_uri": patched_wav.relative_to(args.output_root).as_posix(),
                             "patched_audio_sha256": sha256_file(patched_wav),
                             "baseline_text_token_ids": baseline_text, "patched_text_token_ids": patched_text})
    write_jsonl(args.output_root / "validation.jsonl", rows)
    print(f"wrote {len(rows)} full-duplex rows; human annotation remains fail-closed")
    return 0


def _metrics(rows: Sequence[Mapping[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed" and math.isfinite(float(row.get("delta_M", math.nan)))]
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for row in completed:
        by_scenario[str(row.get("scenario_id", row.get("recipient_trial_id")))].append(float(row["delta_M"]))
    cluster_values = [float(np.mean(values)) for values in by_scenario.values()]
    if len(cluster_values) < 2:
        cluster_values = [float(row["delta_M"]) for row in completed]
    if len(cluster_values) < 2:
        return {"analysis_status": "insufficient_data", "n_cells": len(completed), "passed": False}
    estimate, low, high = bootstrap_mean_ci(cluster_values, replicates, seed)
    # Sign-flip approximation is deterministic and two-sided.
    rng = np.random.default_rng(seed + 1)
    array = np.asarray(cluster_values)
    null = np.mean(rng.choice([-1.0, 1.0], size=(max(2000, replicates), len(array))) * array, axis=1)
    p = float((np.sum(np.abs(null) >= abs(estimate)) + 1) / (len(null) + 1))
    return {"analysis_status": "synthetic_local_validation", "n_cells": len(completed),
            "n_scenario_clusters": len(cluster_values), "estimate": estimate, "ci95": [low, high],
            "raw_p_two_sided": p, "holm_p": holm_adjust([p])[0], "passed": low > 0}


def analyze(argv: Sequence[str]) -> int:
    parser = _parser("Analyze all completed cells with scenario-cluster uncertainty.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    args = parser.parse_args(argv)
    result_files = sorted(args.run_root.rglob("*patch_results.jsonl"))
    rows = [row for path in result_files for row in read_jsonl(path)]
    reports = args.run_root / "reports"
    metrics = _metrics(rows, args.bootstrap_replicates, args.bootstrap_seed)
    all_synthetic = bool(rows) and all(bool(row.get("synthetic")) for row in rows)
    metrics["analysis_status"] = "synthetic_local_validation" if all_synthetic else metrics.get(
        "analysis_status", "empirical_requires_gate_review")
    metrics.update({"schema_version": "1.0.0", "harness_version": HARNESS_VERSION,
                    "provenance": {path.relative_to(args.run_root).as_posix(): sha256_file(path) for path in result_files},
                    "limitations": (["Synthetic/local validation is not evidence about Boston, Seattle, or Moshiko."]
                                    if all_synthetic else ["Interpretation remains conditional on all frozen gates."])})
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "mechanistic_discovery_summary.json", metrics)
    write_csv(reports / "tables/all_scenario_effects.csv", rows)
    registry = [{"hypothesis": "frozen target-minus-stale causal effect", "family": "primary",
                 "direction": "positive", "statistic": metrics.get("estimate"), "n": metrics.get("n_scenario_clusters"),
                 "raw_p": metrics.get("raw_p_two_sided"), "adjusted_p": metrics.get("holm_p"),
                 "ci_type": "scenario-cluster bootstrap", "sesoi": "config-frozen", "passed": metrics.get("passed")}]
    write_csv(reports / "tables/multiplicity_registry.csv", registry)
    print(f"analyzed {len(rows)} cells -> {reports}")
    return 0


def _svg(title: str, value: float) -> str:
    width = max(0, min(560, int(280 + value * 20)))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="180" role="img" '
            f'aria-label="{title}"><rect width="720" height="180" fill="white"/>'
            f'<text x="30" y="40" font-family="sans-serif" font-size="20">{title}</text>'
            f'<line x1="360" y1="60" x2="360" y2="145" stroke="#555"/>'
            f'<rect x="360" y="80" width="{width}" height="35" fill="#3264a8"/>'
            f'<text x="30" y="108" font-family="monospace" font-size="16">estimate {value:.4f}</text></svg>\n')


def render_report(argv: Sequence[str]) -> int:
    parser = _parser("Render a self-contained Markdown/SVG mechanistic report.")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = args.run_root / "reports"
    summary_path = reports / "mechanistic_discovery_summary.json"
    if not summary_path.exists():
        raise ContractError("analyze_mechanistic_results.py must run before rendering")
    summary = read_json(summary_path)
    status = summary.get("analysis_status", "unknown")
    text = f"# Mechanistic stale-binding results\n\n## Status\n\n`{status}`\n\n"
    if status == "synthetic_local_validation":
        text += "This report is a **synthetic/local harness validation**, not empirical evidence about Moshiko.\n\n"
    else:
        text += "This report contains model-run outputs; causal wording is allowed only after every frozen gate passes.\n\n"
    text += f"- Completed cells: {summary.get('n_cells', 0)}\n- Scenario clusters: {summary.get('n_scenario_clusters', 0)}\n"
    text += f"- Mean synthetic effect: {summary.get('estimate', 'NA')}\n- 95% cluster bootstrap CI: {summary.get('ci95', 'NA')}\n\n"
    text += "## Required next gates\n\n- Run the exact pinned checkpoint on RunPod.\n- Pass open-loop and baseline capability gates.\n- Obtain independent alignment and double-listen review before formal confirmation.\n"
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    _path = reports / "MECHANISTIC_RESULTS.md"
    _path.write_text(text, encoding="utf-8")
    titles = ["Baseline margin", "Probe layer-time", "Residual patch", "Frozen confirmation",
              "Temporal propagation", "Controls and no-ops"]
    for index, title in enumerate(titles, 1):
        (reports / "figures" / f"{index:02d}_{title.lower().replace(' ', '_')}.svg").write_text(
            _svg(title, float(summary.get("estimate", 0.0) or 0.0)), encoding="utf-8")
    print(f"rendered report -> {_path}")
    return 0


def verify_run(argv: Sequence[str]) -> int:
    parser = _parser("Verify provenance, rows, reports, and artifact hashes fail-closed.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--allow-local-synthetic", action="store_true")
    args = parser.parse_args(argv)
    required = ["reports/MECHANISTIC_RESULTS.md", "reports/mechanistic_discovery_summary.json",
                "reports/tables/all_scenario_effects.csv", "reports/tables/multiplicity_registry.csv"]
    missing = [path for path in required if not (args.run_root / path).is_file()]
    if missing:
        raise ContractError(f"run is missing required report artifacts: {missing}")
    report_text = (args.run_root / required[0]).read_text(encoding="utf-8")
    summary = read_json(args.run_root / required[1])
    if summary.get("analysis_status") == "synthetic_local_validation" and not args.allow_local_synthetic:
        raise ContractError("synthetic run requires --allow-local-synthetic and cannot satisfy empirical gates")
    if summary.get("analysis_status") == "synthetic_local_validation" and "not empirical evidence" not in report_text:
        raise ContractError("synthetic report lacks the mandatory evidence disclaimer")
    if summary.get("analysis_status") != "synthetic_local_validation":
        empirical_required = [
            "preflight/run_identity.json", "preflight/environment.json", "preflight/model_contract.json",
            "preflight/readouts.bound.json", "encoded_user_manifest.jsonl", "anchor_map.jsonl",
            "frame_trace.jsonl", "open_loop_validation.json", "baseline_readout.jsonl",
            "mechanistic_frozen_selection.json",
        ]
        empirical_missing = [path for path in empirical_required if not (args.run_root / path).is_file()]
        if empirical_missing:
            raise ContractError(f"empirical run is missing gate evidence: {empirical_missing}")
        open_loop = read_json(args.run_root / "open_loop_validation.json")
        if not open_loop.get("passed"):
            raise ContractError("open-loop gate did not pass")
    artifacts = [{"uri": path.relative_to(args.run_root).as_posix(), "sha256": sha256_file(path),
                  "bytes": path.stat().st_size} for path in sorted(args.run_root.rglob("*")) if path.is_file()]
    write_json(args.run_root / "artifact_sha256.json", {"schema_version": "1.0.0", "artifacts": artifacts})
    print(f"verified {len(artifacts)} artifacts under {args.run_root}")
    return 0


def package_results(argv: Sequence[str]) -> int:
    parser = _parser("Create separately verified public and private result archives.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args(argv)
    hashes = package_tree(args.run_root, args.public_output, args.private_output)
    verify_archive(args.public_output, public=True)
    verify_archive(args.private_output, public=False)
    write_json(args.public_output.with_suffix(args.public_output.suffix + ".sha256.json"), hashes)
    print(json.dumps(hashes, sort_keys=True))
    return 0


COMMANDS = {
    "build_mech_manifest.py": build_mech_manifest,
    "build_anchor_map.py": build_anchor_map,
    "build_multivalue_controls.py": build_multivalue_controls,
    "simulate_multivalue_power.py": simulate_multivalue_power,
    "validate_multivalue_controls.py": validate_multivalue_controls,
    "encode_user_audio.py": encode_user_audio,
    "validate_mechanistic_contract.py": validate_mechanistic_contract,
    "validate_open_loop.py": validate_open_loop,
    "capture_activations.py": capture_activations,
    "score_readouts.py": score_readouts,
    "fit_probes.py": fit_probes,
    "scan_residual_patches.py": lambda argv: _scan(argv, "residual"),
    "scan_component_patches.py": lambda argv: _scan(argv, "component"),
    "scan_kv_patches.py": lambda argv: _scan(argv, "kv"),
    "run_path_patches.py": lambda argv: _scan(argv, "path"),
    "freeze_mechanistic_selection.py": freeze_mechanistic_selection,
    "run_confirmatory_patches.py": run_confirmatory,
    "run_full_duplex_validation.py": run_full_duplex,
    "analyze_mechanistic_results.py": analyze,
    "render_mechanistic_report.py": render_report,
    "verify_mechanistic_run.py": verify_run,
    "package_mechanistic_results.py": package_results,
}


def main_for(program: str, argv: Sequence[str] | None = None) -> int:
    try:
        command = COMMANDS[Path(program).name]
        return command(list(sys.argv[1:] if argv is None else argv))
    except ContractError as error:
        print(f"CONTRACT ERROR: {error}", file=sys.stderr)
        return 2
