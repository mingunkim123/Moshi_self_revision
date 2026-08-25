from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Iterator, Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
DATASET_ROOT = EXPERIMENT_ROOT / "dataset_v2"
DEFAULT_CONFIG = DATASET_ROOT / "config/dataset.yaml"
DEFAULT_BLUEPRINTS = DATASET_ROOT / "blueprints/scenarios.jsonl"
DEFAULT_SCRIPTS = DATASET_ROOT / "generated/scripts.jsonl"

CONDITIONS = (
    "clean_final",
    "immediate_repair",
    "delayed_neutral",
    "delayed_one_dependency",
    "delayed_three_dependencies",
)

UNIT_IDS = ("D1", "D2", "D3", "N1", "N2", "N3")
DEPENDENT_IDS = UNIT_IDS[:3]
NEUTRAL_IDS = UNIT_IDS[3:]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the JSON-compatible YAML config without adding a YAML dependency."""
    return json.loads(path.read_text(encoding="utf-8"))


def portable_path(path: Path) -> str:
    """Represent repository paths without embedding a developer's absolute path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(row))
                handle.write("\n")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text))


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", text.casefold()))


def contains_term(text: str, term: str) -> bool:
    normalized = normalized_text(text)
    needle = normalized_text(term)
    return bool(needle and re.search(rf"(?:^|\s){re.escape(needle)}(?:$|\s)", normalized))


def substitute(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for source, target in replacements.items():
            value = value.replace(source, target)
        return value
    if isinstance(value, list):
        return [substitute(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, replacements) for key, item in value.items()}
    return value


def units_by_id(blueprint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    units = [*blueprint["dependent_units"], *blueprint["neutral_units"]]
    return {str(unit["unit_id"]): unit for unit in units}


def dotted_state(patches: Iterable[dict[str, Any]]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for patch in patches:
        for key, value in patch.items():
            if key in state and state[key] != value:
                raise ValueError(f"Conflicting state patch for {key}: {state[key]!r} vs {value!r}")
            state[key] = value
    return state


def iter_duplicates(values: Iterable[str]) -> Iterator[str]:
    seen: set[str] = set()
    emitted: set[str] = set()
    for value in values:
        if value in seen and value not in emitted:
            emitted.add(value)
            yield value
        seen.add(value)
