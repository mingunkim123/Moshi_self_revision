#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np

from common import (
    EXPERIMENT_ROOT,
    read_csv,
    read_json,
    resolve_experiment_path,
    sha256_file,
    validate_id,
    write_json,
)


class SilentPrinter:
    def print_header(self) -> None:
        pass

    def print_token(self, token: str) -> None:
        pass

    def log(self, level: str, msg: str) -> None:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible multi-seed Moshi inference over a prepared manifest."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "config/experiment.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXPERIMENT_ROOT / "data/manifest.prepared.csv",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Relative to the experiment directory unless absolute.",
    )
    parser.add_argument("--seeds", help="Comma-separated override")
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--speaker", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-large-model-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def get_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=EXPERIMENT_ROOT.parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def snapshot_revision(path: Path) -> str | None:
    parts = path.resolve().parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def file_metadata(path: Path, hash_file: bool) -> dict[str, Any]:
    stat = path.stat()
    output: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_file:
        output["sha256"] = sha256_file(path)
    return output


def select_trials(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.manifest)
    selected = []
    for row in rows:
        validate_id(row["trial_id"], "trial_id")
        validate_id(row["speaker_id"], "speaker_id")
        validate_id(row["condition_id"], "condition_id")
        if args.condition and row["condition_id"] not in args.condition:
            continue
        if args.speaker and row["speaker_id"] not in args.speaker:
            continue
        audio_path = resolve_experiment_path(row["prepared_audio_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing prepared audio: {audio_path}")
        selected.append(row)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No trials matched the requested filters")
    return selected


def decode_tokens(tokenizer: Any, token_ids: list[int]) -> tuple[str, list[dict[str, Any]]]:
    pieces = []
    timeline = []
    for frame_index, token_id in enumerate(token_ids):
        piece = ""
        if token_id not in (0, 3):
            piece = tokenizer.id_to_piece(token_id).replace("▁", " ")
            pieces.append(piece)
        timeline.append(
            {
                "frame_index": frame_index,
                "time_ms": frame_index * 80,
                "token_id": token_id,
                "piece": piece,
            }
        )
    return "".join(pieces).strip(), timeline


def write_predictions_index(results_root: Path) -> Path:
    records = []
    for result_path in sorted(results_root.glob("raw/*/seed_*/result.json")):
        with result_path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        records.append(
            {
                key: record.get(key)
                for key in (
                    "experiment_name",
                    "trial_id",
                    "speaker_id",
                    "condition_id",
                    "language",
                    "track",
                    "utterance",
                    "target",
                    "stale",
                    "is_repair",
                    "is_clean",
                    "is_long_gap",
                    "clean_match_id",
                    "seed",
                    "input_audio_path",
                    "response_audio_path",
                    "inner_text",
                    "repair_marker_onset_ms",
                    "repair_onset_ms",
                    "repair_end_ms",
                    "user_end_ms",
                    "elapsed_seconds",
                )
            }
        )
    index_path = results_root / "predictions.jsonl"
    temporary = index_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(index_path)
    return index_path


def validate_prepared_audio(path: Path, expected_sha256: str) -> None:
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError(f"Prepared audio checksum changed: {path}")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    trials = select_trials(args)
    seeds = (
        [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else [int(value) for value in config["seeds"]]
    )
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be a non-empty unique list")
    results_root = resolve_experiment_path(args.results_root)

    print(
        f"Validated {len(trials)} trials × {len(seeds)} seeds "
        f"= {len(trials) * len(seeds)} outputs"
    )
    if args.dry_run:
        print("Dry run complete; model was not loaded.")
        return

    os.environ.setdefault("NO_TORCH_COMPILE", "1")
    try:
        import sphn
        import torch
        from moshi.models import loaders
        from moshi.run_inference import InferenceState, seed_all
    except ImportError as error:
        raise RuntimeError(
            "Moshi dependencies are missing. Run experiments/self_repair/runpod/setup.sh first."
        ) from error

    device = str(config["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype_name = str(config["dtype"])
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    revision = config.get("revision") or None
    print(f"Loading checkpoint {config['hf_repo']} revision={revision or 'latest'}")
    checkpoint = loaders.CheckpointInfo.from_hf_repo(
        config["hf_repo"], revision=revision
    )
    mimi = checkpoint.get_mimi(device=device)
    tokenizer = checkpoint.get_text_tokenizer()
    lm = checkpoint.get_moshi(device=device, dtype=dtype_map[dtype_name])
    generation = dict(checkpoint.lm_gen_config)
    generation.update(config["generation"])
    state = InferenceState(
        checkpoint,
        mimi,
        tokenizer,
        lm,
        batch_size=1,
        cfg_coef=float(config["cfg_coef"]),
        device=device,
        **generation,
    )
    state.printer = SilentPrinter()

    hash_large = bool(config.get("hash_large_model_files")) or args.hash_large_model_files
    model_files = {
        "moshi": file_metadata(checkpoint.moshi_weights, hash_large),
        "mimi": file_metadata(checkpoint.mimi_weights, hash_large),
        "tokenizer": file_metadata(checkpoint.tokenizer, True),
    }
    if checkpoint.raw_config is not None:
        model_files["raw_config"] = checkpoint.raw_config
    resolved_revision = snapshot_revision(checkpoint.moshi_weights)
    run_metadata = {
        "experiment_name": config["experiment_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "hf_repo": config["hf_repo"],
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "model_files": model_files,
        "device": device,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "generation": generation,
        "seeds": seeds,
        "token_timestamps": "Approximate 80 ms output frame indices from stream start.",
    }
    results_root.mkdir(parents=True, exist_ok=True)
    write_json(results_root / "run_metadata.json", run_metadata)

    # Warm CUDA graphs without retaining any conversational state.
    seed_all(0)
    silence = torch.zeros(1, 1, state.frame_size, device=device)
    for _ in range(4):
        codes = mimi.encode(silence)
        for code_index in range(codes.shape[-1]):
            tokens = state.lm_gen.step(codes[:, :, code_index : code_index + 1])
            if tokens is not None and lm.dep_q > 0:
                _ = mimi.decode(tokens[:, 1:])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mimi.reset_streaming()
    state.lm_gen.reset_streaming()

    completed = 0
    skipped = 0
    for trial in trials:
        input_path = resolve_experiment_path(trial["prepared_audio_path"])
        validate_prepared_audio(input_path, trial.get("prepared_audio_sha256", ""))
        pcm, _ = sphn.read(
            str(input_path), sample_rate=int(config["audio"]["sample_rate"])
        )
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim == 1:
            pcm = pcm[None, :]
        if pcm.ndim != 2 or pcm.shape[0] < 1:
            raise ValueError(f"Unexpected audio shape {pcm.shape}: {input_path}")
        if pcm.shape[-1] % int(config["audio"]["frame_size"]):
            raise ValueError(f"Audio is not Mimi-frame aligned: {input_path}")
        in_pcm = torch.from_numpy(pcm[None, 0:1]).to(device=device)

        for seed in seeds:
            result_dir = results_root / "raw" / trial["trial_id"] / f"seed_{seed}"
            result_path = result_dir / "result.json"
            if result_path.exists() and not args.overwrite:
                skipped += 1
                continue
            result_dir.mkdir(parents=True, exist_ok=True)
            response_path = result_dir / "response.wav"

            seed_all(seed)
            mimi.reset_streaming()
            state.lm_gen.reset_streaming()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                outputs = state.run(in_pcm)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if len(outputs) != 1:
                raise RuntimeError(f"Expected one output for {trial['trial_id']}")
            text_tokens, response_pcm = outputs[0]
            token_ids = [int(value) for value in text_tokens.reshape(-1).tolist()]
            inner_text, token_timeline = decode_tokens(tokenizer, token_ids)
            response_audio = response_pcm[0].numpy()
            sphn.write_wav(
                str(response_path),
                response_audio,
                sample_rate=int(config["audio"]["sample_rate"]),
            )
            result = {
                "experiment_name": config["experiment_name"],
                "trial_id": trial["trial_id"],
                "speaker_id": trial["speaker_id"],
                "condition_id": trial["condition_id"],
                "language": trial["language"],
                "track": trial["track"],
                "utterance": trial["utterance"],
                "target": trial["target"],
                "stale": trial["stale"],
                "is_repair": trial["is_repair"],
                "is_clean": trial["is_clean"],
                "is_long_gap": trial["is_long_gap"],
                "clean_match_id": trial["clean_match_id"],
                "seed": seed,
                "input_audio_path": str(input_path),
                "input_audio_sha256": trial.get("prepared_audio_sha256"),
                "response_audio_path": str(response_path),
                "response_audio_sha256": sha256_file(response_path),
                "inner_text": inner_text,
                "text_tokens": token_timeline,
                "repair_marker_onset_ms": trial.get("repair_marker_onset_ms"),
                "repair_onset_ms": trial.get("repair_onset_ms"),
                "repair_end_ms": trial.get("repair_end_ms"),
                "user_end_ms": trial.get("user_end_ms"),
                "elapsed_seconds": elapsed,
                "generation": generation,
                "resolved_revision": resolved_revision,
            }
            write_json(result_path, result)
            completed += 1
            print(
                f"[{completed + skipped}/{len(trials) * len(seeds)}] "
                f"{trial['trial_id']} seed={seed}: {inner_text!r}"
            )

    index_path = write_predictions_index(results_root)
    print(f"Completed: {completed}; skipped existing: {skipped}")
    print(f"Prediction index: {index_path}")


if __name__ == "__main__":
    main()
