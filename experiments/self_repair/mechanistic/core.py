from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import tarfile
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import HARNESS_VERSION


MODEL_REPO = "kyutai/moshiko-pytorch-bf16"
MODEL_REVISION = "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"
FRAME_SAMPLES = 1920
SAMPLE_RATE = 24000
FRAME_MS = 80
REQUIRED_SITES = {
    "resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post",
    "q_pre_rope", "k_pre_rope", "v_pre_rope", "q_post_rope",
    "k_post_rope", "v_post_rope", "head_z",
}


class ContractError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ContractError(f"{path}:{number}: row must be an object")
                rows.append(value)
    return rows


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(path, "".join(canonical_json(dict(row)) + "\n" for row in rows))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def require_relative_uri(value: str) -> str:
    if "\\" in value or "\x00" in value:
        raise ContractError(f"unsafe artifact URI: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError(f"artifact URI must be portable and relative: {value!r}")
    return path.as_posix()


def validate_sha256(value: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError(f"{label} is not a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class RunIdentity:
    schema_version: str
    harness_version: str
    code_commit: str
    model_repo: str
    model_revision: str
    config_sha256: str
    manifest_sha256: str
    open_loop_policy_sha256: str
    data_status: str

    @property
    def sha256(self) -> str:
        return sha256_value(asdict(self))


def build_run_identity(
    *, code_commit: str, config: Mapping[str, Any], manifest_path: Path,
    data_status: str, model_repo: str = MODEL_REPO, model_revision: str = MODEL_REVISION,
) -> RunIdentity:
    if re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ContractError("code_commit must be an exact 40-character Git commit")
    if model_revision != MODEL_REVISION:
        raise ContractError("model revision differs from the frozen Moshiko revision")
    policy = config.get("open_loop_policy")
    if not isinstance(policy, Mapping):
        raise ContractError("config.open_loop_policy is required")
    return RunIdentity(
        "1.0.0", HARNESS_VERSION, code_commit, model_repo, model_revision,
        sha256_value(config), sha256_file(manifest_path), sha256_value(policy), data_status,
    )


def validate_runtime_environment(*, require_cuda: bool) -> dict[str, Any]:
    if os.environ.get("NO_TORCH_COMPILE") != "1":
        raise ContractError("NO_TORCH_COMPILE=1 must be set before model construction")
    if os.environ.get("NO_CUDA_GRAPH") != "1":
        raise ContractError("NO_CUDA_GRAPH=1 must be set before model construction")
    report: dict[str, Any] = {
        "no_torch_compile": True, "no_cuda_graph": True, "require_cuda": require_cuda,
    }
    try:
        import torch
        report.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available()})
        if require_cuda and not torch.cuda.is_available():
            raise ContractError("CUDA GPU is required for the Moshiko backend")
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    except ImportError as error:
        raise ContractError("PyTorch is not installed") from error
    return report


def frame_for_ms(milliseconds: int | float, *, frame_ms: int = FRAME_MS) -> int:
    if milliseconds < 0:
        raise ContractError("anchor time cannot be negative")
    return int(round(float(milliseconds) / frame_ms))


def anchor_rows(
    trials: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared_by_id = {
        str(row.get("prepared_stimulus_id", row.get("stimulus_id", ""))): row for row in prepared
    }
    anchors: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for trial in trials:
        trial_id = str(trial.get("trial_id", trial.get("eval_trial_id", "")))
        stimulus_id = str(trial.get("prepared_stimulus_id", trial.get("stimulus_id", "")))
        source = prepared_by_id.get(stimulus_id, trial)
        timing = source.get("prepared_timing", source.get("timing", {}))
        if not isinstance(timing, Mapping):
            raise ContractError(f"{trial_id}: missing timing object")
        frame_count = int(source.get("mimi_frame_count", source.get("frame_count", 0)))
        if frame_count <= 0:
            samples = int(source.get("sample_count", source.get("prepared_sample_count", 0)))
            prepared_audio = source.get("prepared_stimulus", {})
            if samples <= 0 and isinstance(prepared_audio, Mapping):
                samples = int(round(float(prepared_audio.get("duration_ms", 0)) * SAMPLE_RATE / 1000.0))
            frame_count = samples // FRAME_SAMPLES
        if frame_count <= 0:
            raise ContractError(f"{trial_id}: cannot determine frame count")
        alignment = source.get("alignment", {})
        unit_spans = timing.get(
            "unit_spans",
            source.get("unit_spans", alignment.get("unit_spans", []) if isinstance(alignment, Mapping) else []),
        )
        times: dict[str, int | float] = {}
        aliases = {
            "old_end": ("old_value_offset_ms", "old_value_end_ms", "original_value_end_ms"),
            "cue_end": ("repair_cue_offset_ms", "repair_marker_end_ms", "cue_end_ms"),
            "new_end": ("new_value_offset_ms", "repair_end_ms", "new_value_end_ms"),
            "query_end": ("utterance_end_ms", "user_end_ms", "query_end_ms"),
        }
        for anchor, keys in aliases.items():
            for key in keys:
                if key in timing and timing[key] is not None:
                    times[anchor] = timing[key]
                    break
        if isinstance(unit_spans, Sequence):
            for unit in unit_spans:
                if not isinstance(unit, Mapping):
                    continue
                unit_id = str(unit.get("unit_id", ""))
                if unit_id in {"D1", "D2", "D3"}:
                    times[f"{unit_id}_end"] = unit.get("end_ms", unit.get("offset_ms"))
        if "query_end" not in times:
            times["query_end"] = (frame_count - 1) * FRAME_MS
        for name, milliseconds in sorted(times.items()):
            if milliseconds is None:
                continue
            frame = min(frame_count - 1, frame_for_ms(milliseconds))
            if frame < 0 or frame >= frame_count:
                raise ContractError(f"{trial_id}:{name} is outside the encoded sequence")
            anchors.append({
                "trial_id": trial_id, "anchor": name, "frame": frame,
                "time_ms": float(milliseconds), "roundtrip_error_ms": abs(frame * FRAME_MS - float(milliseconds)),
                "minus_one_frame": max(0, frame - 1), "plus_one_frame": min(frame_count - 1, frame + 1),
            })
        for frame in range(frame_count):
            trace.append({"trial_id": trial_id, "frame": frame, "start_ms": frame * FRAME_MS,
                          "end_ms": (frame + 1) * FRAME_MS, "lm_input_offset": frame + 1})
    if not anchors or any(row["roundtrip_error_ms"] > FRAME_MS for row in anchors):
        raise ContractError("anchor map is empty or exceeds the 80 ms roundtrip tolerance")
    return anchors, trace


@dataclass(frozen=True)
class PatchCell:
    run_identity_sha256: str
    donor_trial_id: str
    recipient_trial_id: str
    component: str
    layer: int
    head: int | None
    source_frames: tuple[int, ...]
    target_frames: tuple[int, ...]
    readout_sha256: str

    @property
    def cell_id(self) -> str:
        return sha256_value(asdict(self))


class AtomicCellStore:
    def __init__(self, root: Path):
        self.root = root
        self.cells = root / "cells"
        self.cells.mkdir(parents=True, exist_ok=True)

    def record(self, cell: PatchCell, payload: Mapping[str, Any]) -> bool:
        path = self.cells / f"{cell.cell_id}.json"
        row = json.loads(canonical_json(
            {"schema_version": "1.0.0", "cell_id": cell.cell_id, **asdict(cell), **dict(payload)}
        ))
        if path.exists():
            existing = read_json(path)
            if existing != row:
                raise ContractError(f"conflicting completed patch cell: {cell.cell_id}")
            return False
        write_json(path, row)
        return True

    def rows(self) -> list[dict[str, Any]]:
        rows = [read_json(path) for path in sorted(self.cells.glob("*.json"))]
        ids = [row["cell_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ContractError("duplicate patch cell identity")
        return rows

    def merge(self, output: Path) -> list[dict[str, Any]]:
        rows = self.rows()
        write_jsonl(output, rows)
        return rows


def paired_feedback_hash(text: np.ndarray, audio: np.ndarray | None) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(text).tobytes())
    if audio is not None:
        digest.update(np.ascontiguousarray(audio).tobytes())
    return digest.hexdigest()


def bootstrap_mean_ci(values: Sequence[float], replicates: int, seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ContractError("bootstrap requires at least two finite cluster estimates")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(replicates, array.size), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def fit_ridge_probe(features: np.ndarray, labels: Sequence[str], alpha: float = 1.0) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(labels):
        raise ContractError("probe features and labels have incompatible shapes")
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ContractError("probe requires at least two classes")
    y = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for row, label in enumerate(labels):
        y[row, classes.index(label)] = 1.0
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([z, np.ones(len(z))])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    predictions = np.asarray(classes)[np.argmax(design @ weights, axis=1)]
    return {
        "classes": classes, "mean": mean.tolist(), "scale": scale.tolist(),
        "weights": weights.tolist(), "training_accuracy": float(np.mean(predictions == np.asarray(labels))),
        "alpha": alpha,
    }


def apply_probe(model: Mapping[str, Any], features: np.ndarray) -> list[str]:
    x = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["mean"])
    scale = np.asarray(model["scale"])
    design = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    weights = np.asarray(model["weights"])
    classes = np.asarray(model["classes"])
    return classes[np.argmax(design @ weights, axis=1)].tolist()


def freeze_selection(rows: Sequence[Mapping[str, Any]], config_hash: str) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("status") == "completed" and math.isfinite(float(row.get("delta_M", math.nan)))]
    if not eligible:
        raise ContractError("no completed finite discovery cells can be frozen")
    eligible.sort(key=lambda row: (-abs(float(row["delta_M"])), str(row.get("cell_id", ""))))
    winner = eligible[0]
    selection = {
        "schema_version": "1.0.0", "status": "frozen_discovery_selection",
        "config_sha256": validate_sha256(config_hash, "config hash"),
        "component": winner["component"], "layer": winner["layer"], "head": winner.get("head"),
        "anchor": winner.get("anchor", "query_end"), "direction": "target_minus_stale",
        "selection_source_cell_id": winner["cell_id"],
    }
    selection["selection_sha256"] = sha256_value(selection)
    return selection


def safe_public_member(relative: str) -> bool:
    relative = require_relative_uri(relative)
    suffix = Path(relative).suffix.lower()
    if suffix in {".wav", ".mp3", ".flac", ".pt", ".pth", ".safetensors", ".npy", ".npz"}:
        return False
    lowered = relative.lower()
    return not any(term in lowered for term in ("blind_map", "credential", "private", "token"))


def contains_sensitive_text(path: Path) -> bool:
    if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md", ".txt", ".svg"}:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return True
    patterns = (
        r"\b(?:sk|hf)_[A-Za-z0-9_-]{16,}\b",
        r'(?i)"(?:api[_-]?key|access[_-]?token|secret)"\s*:\s*"(?!null|redacted)[^"]+"',
        r'/(?:Users|home|workspace|root)/[^\s"<]+',
    )
    return any(re.search(pattern, text) for pattern in patterns)


def package_tree(run_root: Path, public_output: Path, private_output: Path) -> dict[str, str]:
    if public_output.resolve() == private_output.resolve():
        raise ContractError("public and private archives must differ")
    public: list[Path] = []
    private: list[Path] = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run_root).as_posix()
        is_public = (safe_public_member(relative) and path.stat().st_size <= 10 * 1024 * 1024
                     and not contains_sensitive_text(path))
        (public if is_public else private).append(path)
    for destination, members in ((public_output, public), (private_output, private)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination, "w:gz") as archive:
            for member in members:
                archive.add(member, arcname=member.relative_to(run_root).as_posix(), recursive=False)
    return {"public_sha256": sha256_file(public_output), "private_sha256": sha256_file(private_output)}


def verify_archive(path: Path, *, public: bool) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names: set[str] = set()
        for member in archive.getmembers():
            name = require_relative_uri(member.name)
            if name in names or not member.isfile():
                raise ContractError(f"archive contains duplicate or non-file member: {name}")
            if public and not safe_public_member(name):
                raise ContractError(f"private artifact leaked into public archive: {name}")
            names.add(name)


def deterministic_derangement(values: Sequence[str], seed: int) -> dict[str, str]:
    if len(set(values)) < 2:
        raise ContractError("derangement requires at least two unique values")
    source = sorted(values)
    rng = random.Random(seed)
    candidate = source[:]
    for _ in range(1000):
        rng.shuffle(candidate)
        if all(left != right for left, right in zip(source, candidate)):
            return dict(zip(source, candidate))
    raise ContractError("could not construct deterministic derangement")
