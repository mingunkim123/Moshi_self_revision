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
    if isinstance(milliseconds, bool) or isinstance(frame_ms, bool):
        raise ContractError("anchor time and frame width must be numeric")
    try:
        time_ms = float(milliseconds)
        width_ms = float(frame_ms)
    except (TypeError, ValueError) as error:
        raise ContractError("anchor time and frame width must be numeric") from error
    if not math.isfinite(time_ms) or not math.isfinite(width_ms):
        raise ContractError("anchor time and frame width must be finite")
    if width_ms <= 0:
        raise ContractError("frame width must be positive")
    if time_ms < 0:
        raise ContractError("anchor time cannot be negative")
    # An event ending exactly on a frame boundary belongs to the preceding
    # frame: [0, 80] -> 0, (80, 160] -> 1, and so on.
    return int(math.ceil(time_ms / width_ms) - 1)


def _user_delay_slots(consumed_audio_frame: int | None) -> list[dict[str, Any]]:
    """Describe the frozen Moshiko user-codebook delays at one LM step."""
    slots: list[dict[str, Any]] = []
    for delay_slot, user_codebooks in ((0, [0]), (1, list(range(1, 8)))):
        source_frame = (
            None
            if consumed_audio_frame is None or consumed_audio_frame < delay_slot
            else consumed_audio_frame - delay_slot
        )
        slots.append({
            "delay_slot": delay_slot,
            "stream": "user_audio",
            "user_codebooks": user_codebooks,
            "source_audio_frame": source_frame,
            "uses_initial_token": source_frame is None,
        })
    return slots


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
        timing = source.get("prepared_timing")
        if not isinstance(timing, Mapping):
            raise ContractError(f"{trial_id}: missing prepared_timing object")
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
        if not isinstance(alignment, Mapping):
            raise ContractError(f"{trial_id}: alignment must be an object")
        unit_spans = alignment.get("unit_spans", [])
        if not isinstance(unit_spans, Sequence) or isinstance(unit_spans, (str, bytes)):
            raise ContractError(f"{trial_id}: alignment.unit_spans must be an array")
        times: dict[str, int | float] = {}
        timebases: dict[str, str] = {}
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
                    timebases[anchor] = "prepared_stream_relative"
                    break
        dependency_units = [
            unit for unit in unit_spans
            if isinstance(unit, Mapping) and str(unit.get("unit_id", "")) in {"D1", "D2", "D3"}
        ]
        if dependency_units:
            preparation = source.get("preparation")
            if not isinstance(preparation, Mapping):
                raise ContractError(
                    f"{trial_id}: preparation.prefix_ms_actual is required for alignment.unit_spans")
            prefix_raw = preparation.get("prefix_ms_actual")
            if isinstance(prefix_raw, bool):
                raise ContractError(f"{trial_id}: invalid preparation.prefix_ms_actual")
            try:
                prefix_ms = float(prefix_raw)
            except (TypeError, ValueError) as error:
                raise ContractError(
                    f"{trial_id}: invalid preparation.prefix_ms_actual") from error
            if not math.isfinite(prefix_ms) or prefix_ms < 0:
                raise ContractError(f"{trial_id}: invalid preparation.prefix_ms_actual")
            for unit in dependency_units:
                unit_id = str(unit["unit_id"])
                offset_raw = unit.get("end_ms", unit.get("offset_ms"))
                if isinstance(offset_raw, bool):
                    raise ContractError(f"{trial_id}:{unit_id} has an invalid end offset")
                try:
                    offset_ms = float(offset_raw)
                except (TypeError, ValueError) as error:
                    raise ContractError(
                        f"{trial_id}:{unit_id} has an invalid end offset") from error
                if not math.isfinite(offset_ms) or offset_ms < 0:
                    raise ContractError(f"{trial_id}:{unit_id} has an invalid end offset")
                anchor = f"{unit_id}_end"
                times[anchor] = offset_ms + prefix_ms
                timebases[anchor] = "alignment_content_relative_plus_prefix"
        if "query_end" not in times:
            times["query_end"] = frame_count * FRAME_MS
            timebases["query_end"] = "encoded_stream_end"
        for name, milliseconds in sorted(times.items()):
            if milliseconds is None:
                continue
            frame = frame_for_ms(milliseconds)
            if frame < 0 or frame >= frame_count:
                raise ContractError(f"{trial_id}:{name} is outside the encoded sequence")
            anchors.append({
                "trial_id": trial_id, "anchor": name, "frame": frame,
                "time_ms": float(milliseconds),
                "timebase": timebases[name],
                "roundtrip_error_ms": abs((frame + 1) * FRAME_MS - float(milliseconds)),
                "minus_one_frame": frame - 1 if frame > 0 else None,
                "plus_one_frame": frame + 1 if frame + 1 < frame_count else None,
            })
        trace.append({
            "trial_id": trial_id,
            "trace_kind": "lm_prime",
            "frame": None,
            "submitted_audio_frame": 0,
            "consumed_audio_frame": None,
            "start_ms": None,
            "end_ms": None,
            "lm_input_offset": 0,
            "lm_step": 0,
            "hidden_absolute_position": 0,
            "max_lm_delay": 1,
            "delay_slots": _user_delay_slots(None),
        })
        for frame in range(frame_count):
            lm_step = frame + 1
            trace.append({
                "trial_id": trial_id,
                "trace_kind": "audio_frame",
                "frame": frame,
                "submitted_audio_frame": frame,
                "consumed_audio_frame": frame,
                "start_ms": frame * FRAME_MS,
                "end_ms": (frame + 1) * FRAME_MS,
                "lm_input_offset": lm_step,
                "lm_step": lm_step,
                "hidden_absolute_position": lm_step,
                "max_lm_delay": 1,
                "delay_slots": _user_delay_slots(frame),
            })
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
        self.failures = root / "failures"
        self.cells.mkdir(parents=True, exist_ok=True)
        self.failures.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _row(cell: PatchCell, payload: Mapping[str, Any]) -> dict[str, Any]:
        status = payload.get("status")
        if status not in {"completed", "failed"}:
            raise ContractError("atomic patch payload status must be completed or failed")
        return json.loads(canonical_json(
            {"schema_version": "1.0.0", "cell_id": cell.cell_id, **asdict(cell), **dict(payload)}
        ))

    def record(self, cell: PatchCell, payload: Mapping[str, Any]) -> bool:
        row = self._row(cell, payload)
        if row["status"] == "failed":
            if self.get(cell) is not None:
                raise ContractError(f"cannot append a failed attempt after completion: {cell.cell_id}")
            failure_id = sha256_value(row)
            path = self.failures / f"{cell.cell_id}.{failure_id}.json"
            row["failure_id"] = failure_id
            if path.exists():
                if read_json(path) != row:
                    raise ContractError(f"conflicting failed patch attempt: {failure_id}")
                return False
            write_json(path, row)
            return True

        path = self.cells / f"{cell.cell_id}.json"
        if path.exists():
            existing = read_json(path)
            if existing != row:
                raise ContractError(f"conflicting completed patch cell: {cell.cell_id}")
            return False
        write_json(path, row)
        return True

    def get(self, cell: PatchCell) -> dict[str, Any] | None:
        """Read and verify a committed cell before doing any replay work.

        The filename alone is not trusted for resume: every canonical identity
        field inside the atomically written row must still match ``cell``.
        """
        path = self.cells / f"{cell.cell_id}.json"
        if not path.exists():
            return None
        row = read_json(path)
        if row.get("cell_id") != cell.cell_id:
            raise ContractError(f"stored patch cell ID mismatch: {path}")
        for key, expected in asdict(cell).items():
            observed = row.get(key)
            if key in {"source_frames", "target_frames"} and isinstance(observed, list):
                observed = tuple(observed)
            if observed != expected:
                raise ContractError(
                    f"stored patch cell identity mismatch for {key}: {cell.cell_id}")
        return row

    def contains(self, cell: PatchCell) -> bool:
        """Return true only for a successfully completed, identity-verified cell.

        Failed attempts live in a separate append-only directory.  They remain
        auditable but do not make ``--resume`` skip the cell, so the same
        immutable identity can be retried after an OOM or transient failure.
        """
        return self.get(cell) is not None

    @staticmethod
    def _verify_row_identity(row: Mapping[str, Any], path: Path) -> None:
        identity_fields = {
            name: row.get(name) for name in PatchCell.__dataclass_fields__
        }
        try:
            reconstructed = PatchCell(**identity_fields)
        except (TypeError, ValueError) as error:
            raise ContractError(f"malformed patch identity in {path}") from error
        if reconstructed.cell_id != row.get("cell_id"):
            raise ContractError(f"stored patch cell ID mismatch: {path}")

    def rows(self) -> list[dict[str, Any]]:
        rows = [read_json(path) for path in sorted(self.cells.glob("*.json"))]
        for path, row in zip(sorted(self.cells.glob("*.json")), rows):
            self._verify_row_identity(row, path)
        ids = [row["cell_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ContractError("duplicate patch cell identity")
        return rows

    def failure_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.failures.glob("*.json")):
            row = read_json(path)
            self._verify_row_identity(row, path)
            expected_failure_id = sha256_value({
                key: value for key, value in row.items() if key != "failure_id"
            })
            if row.get("failure_id") != expected_failure_id:
                raise ContractError(f"stored failure ID mismatch: {path}")
            rows.append(row)
        return rows

    def merge(self, output: Path) -> list[dict[str, Any]]:
        completed = self.rows()
        completed_ids = {row["cell_id"] for row in completed}
        unresolved: dict[str, dict[str, Any]] = {}
        for row in self.failure_rows():
            if row["cell_id"] in completed_ids:
                continue
            existing = unresolved.get(row["cell_id"])
            rank = (int(row.get("attempt_index", 0)), str(row.get("failure_id", "")))
            existing_rank = (
                int(existing.get("attempt_index", 0)), str(existing.get("failure_id", ""))
            ) if existing is not None else (-1, "")
            if rank > existing_rank:
                unresolved[row["cell_id"]] = row
        rows = sorted([*completed, *unresolved.values()], key=lambda row: row["cell_id"])
        write_jsonl(output, rows)
        return rows

    def merge_failures(self, output: Path) -> list[dict[str, Any]]:
        rows = self.failure_rows()
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


def freeze_selection(
    rows: Sequence[Mapping[str, Any]], config_hash: str,
    components: Sequence[str] | None = None,
) -> dict[str, Any]:
    config_hash = validate_sha256(config_hash, "config hash")
    eligible_components: list[str] | None = None
    if components is not None:
        if isinstance(components, (str, bytes)):
            raise ContractError("eligible components must be a sequence")
        eligible_components = []
        for component in components:
            if not isinstance(component, str) or not component or component != component.strip():
                raise ContractError("eligible components must be non-empty, trimmed strings")
            eligible_components.append(component)
        if not eligible_components or len(eligible_components) != len(set(eligible_components)):
            raise ContractError("eligible components must be non-empty and unique")
    eligible: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    for source in rows:
        if source.get("status") != "completed" or source.get("relation") != "clean_current":
            continue
        try:
            effect = float(source.get("delta_M", math.nan))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(effect):
            continue
        row = dict(source)
        cell_id = row.get("cell_id")
        scenario_id = row.get("scenario_id")
        component = row.get("component")
        anchor = row.get("anchor")
        layer = row.get("layer")
        head = row.get("head")
        if not isinstance(cell_id, str) or not cell_id:
            raise ContractError("canonical discovery cell has no cell_id")
        if cell_id in seen_cells:
            raise ContractError(f"duplicate canonical discovery cell: {cell_id}")
        seen_cells.add(cell_id)
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ContractError(f"{cell_id}: canonical discovery cell has no scenario_id")
        if not isinstance(component, str) or not component:
            raise ContractError(f"{cell_id}: canonical discovery cell has no component")
        if eligible_components is not None and component not in eligible_components:
            continue
        if not isinstance(anchor, str) or not anchor:
            raise ContractError(f"{cell_id}: canonical discovery cell has no anchor")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ContractError(f"{cell_id}: canonical discovery cell has an invalid layer")
        if head is not None and (isinstance(head, bool) or not isinstance(head, int) or head < 0):
            raise ContractError(f"{cell_id}: canonical discovery cell has an invalid head")
        readout_sha256 = validate_sha256(
            str(row.get("readout_sha256", "")), f"{cell_id} discovery readout hash")
        provenance = row.get("provenance")
        if provenance is not None:
            if not isinstance(provenance, Mapping):
                raise ContractError(f"{cell_id}: discovery provenance must be an object")
            provenance_config = provenance.get("config_sha256")
            if provenance_config is not None and provenance_config != config_hash:
                raise ContractError(f"{cell_id}: discovery provenance targets a different config")
        path = row.get("path")
        if component == "path" and not isinstance(path, Mapping):
            raise ContractError("selected path cell has no explicit writer/mediator specification")
        row["_selection_effect"] = effect
        row["_selection_readout_sha256"] = readout_sha256
        eligible.append(row)
    if not eligible:
        raise ContractError(
            "no completed finite canonical clean_current discovery cells can be frozen")

    grouped: dict[str, dict[str, Any]] = {}
    for row in eligible:
        identity: dict[str, Any] = {
            "component": row["component"],
            "layer": row["layer"],
            "head": row.get("head"),
            "anchor": row["anchor"],
            "relation": "clean_current",
            "readout_sha256": row["_selection_readout_sha256"],
        }
        if row["component"] == "path":
            identity["path"] = dict(row["path"])
        key = canonical_json(identity)
        group = grouped.setdefault(key, {"identity": identity, "rows": []})
        group["rows"].append(row)

    ranked: list[dict[str, Any]] = []
    for group in grouped.values():
        scenario_effects: dict[str, list[float]] = {}
        for row in group["rows"]:
            scenario_effects.setdefault(str(row["scenario_id"]), []).append(
                float(row["_selection_effect"]))
        scenario_means = [
            {
                "scenario_id": scenario_id,
                "cell_count": len(effects),
                "mean_delta_M": float(sum(effects) / len(effects)),
            }
            for scenario_id, effects in sorted(scenario_effects.items())
        ]
        aggregate = float(
            sum(row["mean_delta_M"] for row in scenario_means) / len(scenario_means))
        identity_sha = sha256_value(group["identity"])
        ranked.append({
            **group,
            "scenario_means": scenario_means,
            "aggregate_delta_M": aggregate,
            "identity_sha256": identity_sha,
        })
    ranked.sort(key=lambda group: (
        -abs(float(group["aggregate_delta_M"])), str(group["identity_sha256"])
    ))
    winner = ranked[0]
    identity = winner["identity"]
    source_rows = winner["rows"]
    source_cell_ids = sorted(str(row["cell_id"]) for row in source_rows)
    source_provenance_by_json: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        provenance = row.get("provenance")
        if isinstance(provenance, Mapping):
            normalized = dict(provenance)
            source_provenance_by_json[canonical_json(normalized)] = normalized
    source_provenance = [
        source_provenance_by_json[key] for key in sorted(source_provenance_by_json)
    ]
    selection = {
        "schema_version": "1.0.0", "status": "frozen_discovery_selection",
        "config_sha256": config_hash,
        "component": identity["component"], "layer": identity["layer"],
        "head": identity["head"], "anchor": identity["anchor"],
        "direction": "target_minus_stale",
        "donor_arm": "clean_current", "relation": "clean_current",
        "readout_sha256": identity["readout_sha256"],
        # Keep the legacy singular trace as a deterministic representative,
        # while the ranking itself is based on every cell listed below.
        "selection_source_cell_id": source_cell_ids[0],
        "selection_source_cell_ids": source_cell_ids,
        "selection_source_cell_count": len(source_cell_ids),
        "selection_source_scenario_count": len(winner["scenario_means"]),
        "selection_scenario_means": winner["scenario_means"],
        "selection_aggregate_delta_M": winner["aggregate_delta_M"],
        "selection_eligibility_policy": {
            "version": "1.0.0",
            "required_relation": "clean_current",
            "eligible_components": eligible_components,
            "aggregation": "mean_within_scenario_then_mean_across_scenarios",
            "ranking": "absolute_aggregate_delta_M",
            "tie_break": "identity_sha256",
        },
        "selection_ranking_policy": (
            "absolute_scenario_balanced_mean_delta_M_then_identity_sha256_v1"),
        "selection_source_aggregate_sha256": sha256_value({
            "identity": identity,
            "cell_ids": source_cell_ids,
            "scenario_means": winner["scenario_means"],
        }),
        "selection_source_provenance": source_provenance,
        "selection_source_provenance_sha256": sha256_value(source_provenance),
    }
    if identity["component"] == "path":
        selection["path"] = dict(identity["path"])
    selection["selection_sha256"] = sha256_value(selection)
    return selection


def safe_public_member(relative: str) -> bool:
    relative = require_relative_uri(relative)
    suffix = Path(relative).suffix.lower()
    public_text_suffixes = {
        ".json", ".jsonl", ".csv", ".md", ".txt", ".svg", ".yaml", ".yml",
        ".toml", ".log", ".xml", ".html", ".sha256",
    }
    # The public package is intentionally an allowlist of inspectable text
    # formats. Unknown/binary formats stay private even if their filename does
    # not reveal that they hold tensors or model state.
    if suffix not in public_text_suffixes:
        return False
    if suffix in {
        ".wav", ".wave", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac",
        ".pcm", ".aiff", ".aif", ".wma",
        ".pt", ".pth", ".safetensors", ".npy", ".npz", ".bin", ".ckpt",
        ".onnx", ".pkl", ".pickle", ".joblib",
    }:
        return False
    lowered = relative.lower()
    parts = tuple(part.lower() for part in PurePosixPath(relative).parts)
    forbidden_parts = {
        "audio", "wav", "activations", "activation_tensors", "tensors",
        "model_cache", "model-cache", "checkpoints", "private", "credentials",
    }
    if any(part in forbidden_parts for part in parts):
        return False
    # The full artifact manifest inventories private filenames and hashes.  It
    # belongs in the private archive; the package checksum sidecar separately
    # authenticates both result archives.
    if relative == "artifact_sha256.json":
        return False
    return not any(term in lowered for term in (
        "blind_map", "credential", "private", "token", "api_key", ".env",
    ))


_SENSITIVE_TEXT_PATTERNS = (
    r"\b(?:sk|hf)_[A-Za-z0-9_-]{16,}\b",
    r'(?i)"(?:api[_-]?key|access[_-]?token|token|secret|password)"\s*:\s*"(?!null|redacted)[^"]+"',
    # URLs are excluded by the negative lookbehind.  A public report should
    # not carry host-specific Unix or Windows filesystem paths.
    r'(?<![A-Za-z0-9:/<])/(?![/<])[^\s"<>]+',
    r'(?i)(?<![A-Za-z0-9])[A-Z]:\\(?:[^\s"<>\\]+\\)+[^\s"<>]+',
)


def contains_sensitive_content(content: str) -> bool:
    return any(re.search(pattern, content) for pattern in _SENSITIVE_TEXT_PATTERNS)


def contains_sensitive_text(path: Path) -> bool:
    if path.suffix.lower() not in {
        ".json", ".jsonl", ".csv", ".md", ".txt", ".svg", ".yaml", ".yml",
        ".toml", ".log", ".xml", ".html", ".sha256",
    }:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return True
    return contains_sensitive_content(text)


def package_tree(run_root: Path, public_output: Path, private_output: Path) -> dict[str, str]:
    if public_output.resolve() == private_output.resolve():
        raise ContractError("public and private archives must differ")
    resolved_root = run_root.resolve()
    for destination in (public_output, private_output):
        try:
            destination.resolve().relative_to(resolved_root)
        except ValueError:
            pass
        else:
            raise ContractError("result archives must be written outside run_root")
    public: list[Path] = []
    private: list[Path] = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run_root).as_posix()
        is_public = (safe_public_member(relative) and path.stat().st_size <= 10 * 1024 * 1024
                     and not contains_sensitive_text(path))
        (public if is_public else private).append(path)
    # Gzip can be incompressible and tar creation briefly needs a complete second
    # copy.  Refuse to start when the destination volume cannot hold the raw
    # member bytes plus a fixed safety reserve.  This is intentionally checked
    # before either archive is opened, so a nearly full RunPod volume is left
    # untouched rather than with a misleading partial package.
    required_by_device: dict[int, int] = {}
    for destination, members in ((public_output, public), (private_output, private)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        device = destination.parent.stat().st_dev
        required_by_device[device] = required_by_device.get(device, 0) + sum(
            member.stat().st_size for member in members
        )
    for destination in (public_output, private_output):
        device = destination.parent.stat().st_dev
        if destination != public_output and public_output.parent.stat().st_dev == device:
            continue
        stats = os.statvfs(destination.parent)
        available = int(stats.f_bavail) * int(stats.f_frsize)
        required = required_by_device[device] + 64 * 1024 * 1024
        if available < required:
            raise ContractError(
                f"insufficient free space for atomic result packaging: {available} < {required} bytes"
            )
    for destination, members in ((public_output, public), (private_output, private)):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with tarfile.open(temporary, "w:gz") as archive:
                for member in members:
                    archive.add(member, arcname=member.relative_to(run_root).as_posix(), recursive=False)
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return {"public_sha256": sha256_file(public_output), "private_sha256": sha256_file(private_output)}


def verify_archive(path: Path, *, public: bool) -> None:
    try:
        with tarfile.open(path, "r:gz") as archive:
            names: set[str] = set()
            for member in archive.getmembers():
                name = require_relative_uri(member.name)
                if name in names or not member.isfile():
                    raise ContractError(f"archive contains duplicate or non-file member: {name}")
                if public and not safe_public_member(name):
                    raise ContractError(f"private artifact leaked into public archive: {name}")
                if public and member.size > 10 * 1024 * 1024:
                    raise ContractError(f"oversized artifact leaked into public archive: {name}")
                if public and Path(name).suffix.lower() in {
                    ".json", ".jsonl", ".csv", ".md", ".txt", ".svg", ".yaml",
                    ".yml", ".toml", ".log", ".xml", ".html", ".sha256",
                }:
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ContractError(f"cannot inspect public archive member: {name}")
                    try:
                        content = handle.read().decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise ContractError(f"public text artifact is not UTF-8: {name}") from error
                    if contains_sensitive_content(content):
                        raise ContractError(f"sensitive content leaked into public archive: {name}")
                names.add(name)
    except ContractError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise ContractError(f"cannot verify result archive {path}: {error}") from error


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
