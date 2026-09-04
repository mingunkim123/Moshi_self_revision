"""Fail-closed verification for mechanistic result trees and packages.

The verifier treats an existing artifact manifest as an immutable claim: it is
checked, never refreshed.  This prevents a second verification pass from
silently blessing files that changed after the first pass.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping

from .core import (
    ContractError,
    AtomicCellStore,
    MODEL_REPO,
    MODEL_REVISION,
    PatchCell,
    read_json,
    read_jsonl,
    require_relative_uri,
    sha256_file,
    sha256_value,
    validate_sha256,
    write_json,
)
from .readiness import AUTHORIZATION_TYPE, ReadinessError, verify_authorization_artifact


ARTIFACT_MANIFEST = "artifact_sha256.json"
PATCH_SCHEMA = Path(__file__).resolve().parent / "schemas/patch-result.schema.json"
AUTHORIZATION_SCHEMA = (
    Path(__file__).resolve().parent / "schemas/paid-scan-authorization.schema.json"
)
ARTIFACT_SCHEMA = Path(__file__).resolve().parent / "schemas/artifact-manifest.schema.json"
PACKAGE_CHECKSUM_SCHEMA = Path(__file__).resolve().parent / "schemas/package-checksum.schema.json"


def _schema_validate(value: Any, schema_path: Path, label: str) -> None:
    try:
        import jsonschema
    except ImportError as error:  # pragma: no cover - pinned runtime dependency
        raise ContractError("jsonschema is required for result verification") from error
    schema = read_json(schema_path)
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise ContractError(f"{label} violates schema at {location}: {errors[0].message}")


def _artifact_paths(run_root: Path) -> list[Path]:
    if not run_root.is_dir() or run_root.is_symlink():
        raise ContractError(f"run root is not a real directory: {run_root}")
    paths: list[Path] = []
    for path in sorted(run_root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"result tree contains a symlink: {path}")
        if path.is_file() and path.relative_to(run_root).as_posix() != ARTIFACT_MANIFEST:
            paths.append(path)
    return paths


def build_artifact_manifest(run_root: Path) -> dict[str, Any]:
    """Build the deterministic manifest body, excluding the manifest itself."""

    return {
        "schema_version": "1.0.0",
        "artifacts": [
            {
                "uri": path.relative_to(run_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in _artifact_paths(run_root)
        ],
    }


def verify_artifact_manifest(run_root: Path, manifest: Mapping[str, Any]) -> int:
    _schema_validate(manifest, ARTIFACT_SCHEMA, ARTIFACT_MANIFEST)
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise ContractError("artifact manifest artifacts must be an array")
    expected_paths = {
        path.relative_to(run_root).as_posix(): path for path in _artifact_paths(run_root)
    }
    observed: dict[str, Mapping[str, Any]] = {}
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise ContractError(f"artifact manifest row {index} is not an object")
        uri = require_relative_uri(str(source.get("uri", "")))
        if uri == ARTIFACT_MANIFEST:
            raise ContractError("artifact manifest must not hash itself")
        if uri in observed:
            raise ContractError(f"artifact manifest contains duplicate URI: {uri}")
        if set(source) != {"uri", "sha256", "bytes"}:
            raise ContractError(f"artifact manifest row fields differ for {uri}")
        validate_sha256(str(source.get("sha256", "")), f"artifact {uri} hash")
        size = source.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ContractError(f"artifact {uri} has an invalid byte count")
        observed[uri] = source
    missing = sorted(set(expected_paths) - set(observed))
    extra = sorted(set(observed) - set(expected_paths))
    if missing or extra:
        raise ContractError(
            f"artifact manifest coverage mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )
    if list(observed) != sorted(observed):
        raise ContractError("artifact manifest rows are not in canonical URI order")
    for uri, row in observed.items():
        path = expected_paths[uri]
        if path.stat().st_size != row["bytes"]:
            raise ContractError(f"artifact byte count mismatch: {uri}")
        if sha256_file(path) != row["sha256"]:
            raise ContractError(f"artifact SHA-256 mismatch: {uri}")
    return len(observed)


def verify_or_create_artifact_manifest(run_root: Path) -> tuple[int, bool]:
    """Verify an existing manifest or atomically establish the first one."""

    path = run_root / ARTIFACT_MANIFEST
    if path.exists():
        return verify_artifact_manifest(run_root, read_json(path)), False
    manifest = build_artifact_manifest(run_root)
    write_json(path, manifest)
    # Re-read the committed bytes and verify the claim we just established.
    return verify_artifact_manifest(run_root, read_json(path)), True


def _non_empty_text(row: Mapping[str, Any], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} requires non-empty {field}")
    return value


def _patch_identity(row: Mapping[str, Any], label: str) -> PatchCell:
    def frame_tuple(field: str) -> tuple[int, ...]:
        values = row.get(field)
        if not isinstance(values, list) or not values:
            raise ContractError(f"{label} requires a non-empty {field} array")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ContractError(f"{label} has invalid {field}")
        if len(values) != len(set(values)):
            raise ContractError(f"{label} has duplicate {field}")
        return tuple(values)

    layer = row.get("layer")
    head = row.get("head")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ContractError(f"{label} has an invalid layer")
    if head is not None and (isinstance(head, bool) or not isinstance(head, int) or head < 0):
        raise ContractError(f"{label} has an invalid head")
    identity = PatchCell(
        run_identity_sha256=validate_sha256(
            _non_empty_text(row, "run_identity_sha256", label), f"{label} run identity"
        ),
        donor_trial_id=_non_empty_text(row, "donor_trial_id", label),
        recipient_trial_id=_non_empty_text(row, "recipient_trial_id", label),
        component=_non_empty_text(row, "component", label),
        layer=layer,
        head=head,
        source_frames=frame_tuple("source_frames"),
        target_frames=frame_tuple("target_frames"),
        readout_sha256=validate_sha256(
            _non_empty_text(row, "readout_sha256", label), f"{label} readout"
        ),
    )
    if identity.cell_id != row.get("cell_id"):
        raise ContractError(f"{label} cell identity SHA-256 mismatch")
    return identity


_REQUIRED_PROVENANCE_FIELDS = (
    "code_commit",
    "harness_version",
    "model_repo",
    "model_revision",
    "config_sha256",
    "manifest_sha256",
    "encoded_manifest_sha256",
    "anchor_map_sha256",
    "readout_sha256",
    "scan_spec_sha256",
    "selection_file_sha256",
    "data_sha256",
    "run_identity_sha256",
)
_OPTIONAL_PROVENANCE_FIELDS = ("role_manifest_sha256", "baseline_readout_sha256")


def validate_patch_result_row(
    source: Mapping[str, Any], *, label: str, require_empirical_provenance: bool,
) -> dict[str, Any]:
    row = dict(source)
    _schema_validate(row, PATCH_SCHEMA, label)
    identity = _patch_identity(row, label)
    status = row.get("status")
    if status == "completed":
        for field in ("baseline_M", "patched_M", "delta_M"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ContractError(f"{label} has invalid completed metric {field}")
        if abs((float(row["patched_M"]) - float(row["baseline_M"])) - float(row["delta_M"])) > 1e-8:
            raise ContractError(f"{label} delta_M disagrees with patched_M - baseline_M")
    elif status == "failed":
        _non_empty_text(row, "failure_type", label)
        _non_empty_text(row, "failure_message", label)
    else:  # The JSON schema also checks this; keep the semantic branch explicit.
        raise ContractError(f"{label} has an invalid status")

    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        if require_empirical_provenance:
            raise ContractError(f"{label} is empirical but has no provenance")
        return row
    missing = [field for field in _REQUIRED_PROVENANCE_FIELDS if field not in provenance]
    unknown = sorted(
        set(provenance) - set(_REQUIRED_PROVENANCE_FIELDS) - set(_OPTIONAL_PROVENANCE_FIELDS)
    )
    if missing or unknown:
        raise ContractError(
            f"{label} provenance fields mismatch; missing={missing}, unknown={unknown}"
        )
    if provenance.get("run_identity_sha256") != identity.run_identity_sha256:
        raise ContractError(f"{label} provenance/run identity mismatch")
    if provenance.get("readout_sha256") != identity.readout_sha256:
        raise ContractError(f"{label} provenance/readout mismatch")
    if provenance.get("model_repo") != MODEL_REPO or provenance.get("model_revision") != MODEL_REVISION:
        raise ContractError(f"{label} provenance has the wrong frozen model")
    commit = provenance.get("code_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ContractError(f"{label} provenance has an invalid code commit")
    for field in ("config_sha256", "manifest_sha256", "anchor_map_sha256", "readout_sha256", "run_identity_sha256"):
        validate_sha256(str(provenance.get(field, "")), f"{label} provenance {field}")
    if require_empirical_provenance:
        for field in ("encoded_manifest_sha256", "scan_spec_sha256", "data_sha256"):
            validate_sha256(str(provenance.get(field, "")), f"{label} provenance {field}")
    for field in ("selection_file_sha256", *_OPTIONAL_PROVENANCE_FIELDS):
        value = provenance.get(field)
        if value is not None:
            validate_sha256(str(value), f"{label} provenance {field}")
    return row


def _authorization_index(run_root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    # Paid runs use a canonical ``*authorization*.json`` name.  Restricting
    # discovery avoids parsing hundreds of thousands of atomic cell JSON files
    # merely to locate a handful of GO artifacts.
    for path in sorted(run_root.rglob("*authorization*.json")):
        try:
            value = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(value, Mapping) or value.get("artifact_type") != AUTHORIZATION_TYPE:
            continue
        _schema_validate(value, AUTHORIZATION_SCHEMA, path.relative_to(run_root).as_posix())
        try:
            binding = verify_authorization_artifact(value, value.get("target_binding", {}))
        except ReadinessError as error:
            raise ContractError(f"invalid paid-scan authorization {path}: {error}") from error
        binding_hash = str(value.get("target_binding_sha256"))
        if binding_hash in result and result[binding_hash] != binding:
            raise ContractError(f"conflicting authorization for target {binding_hash}")
        result[binding_hash] = binding
    return result


def _verify_plan_directory(
    directory: Path,
    *,
    run_root: Path,
    global_ids: set[str],
    authorizations: Mapping[str, Mapping[str, str]],
) -> tuple[int, bool]:
    plan_path = directory / "scan_plan.json"
    plan = read_json(plan_path)
    if not isinstance(plan, Mapping):
        raise ContractError(f"scan plan is not an object: {plan_path}")
    planned_path = directory / "planned_cells.jsonl"
    if not planned_path.is_file():
        raise ContractError(f"scan is missing planned_cells.jsonl: {directory}")
    planned = read_jsonl(planned_path)
    planned_ids: list[str] = []
    for index, row in enumerate(planned):
        label = f"{planned_path.relative_to(run_root)}:{index + 1}"
        identity = _patch_identity(row, label)
        planned_ids.append(identity.cell_id)
    if len(planned_ids) != len(set(planned_ids)):
        raise ContractError(f"scan plan contains duplicate cell IDs: {planned_path}")
    if plan.get("planned_cell_count") != len(planned):
        raise ContractError(f"scan planned-cell count mismatch: {plan_path}")
    if plan.get("planned_cells_sha256") != sha256_value(planned):
        raise ContractError(f"scan planned-cell hash mismatch: {plan_path}")
    _non_empty_text(plan, "kind", str(plan_path))
    result_uri = require_relative_uri(
        _non_empty_text(plan, "result_uri", str(plan_path))
    )
    if Path(result_uri).parent != Path("."):
        raise ContractError(f"scan result_uri must name a file in its scan directory: {plan_path}")
    result_path = directory / result_uri
    if not result_path.is_file():
        raise ContractError(f"scan is missing its canonical result file: {result_path}")
    raw_results = read_jsonl(result_path)
    synthetic_values = {row.get("synthetic") for row in raw_results}
    if len(synthetic_values) != 1 or not synthetic_values <= {True, False}:
        raise ContractError(f"scan mixes or omits synthetic status: {result_path}")
    synthetic = synthetic_values == {True}
    results = [
        validate_patch_result_row(
            row,
            label=f"{result_path.relative_to(run_root)}:{index + 1}",
            require_empirical_provenance=not synthetic,
        )
        for index, row in enumerate(raw_results)
    ]
    result_ids = [str(row["cell_id"]) for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise ContractError(f"result file contains duplicate cell IDs: {result_path}")
    overlap = global_ids.intersection(result_ids)
    if overlap:
        raise ContractError(f"patch cell appears in multiple result files: {sorted(overlap)[:10]}")
    global_ids.update(result_ids)
    missing = sorted(set(planned_ids) - set(result_ids))
    extra = sorted(set(result_ids) - set(planned_ids))
    failed = sorted(str(row["cell_id"]) for row in results if row.get("status") != "completed")
    if missing or extra or failed:
        raise ContractError(
            f"scan cell coverage is incomplete; missing={missing[:10]}, extra={extra[:10]}, "
            f"failed={failed[:10]}"
        )

    by_id = {str(row["cell_id"]): row for row in results}
    plan_by_id = {str(row["cell_id"]): row for row in planned}
    provenance = plan.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ContractError(f"scan plan has no provenance: {plan_path}")
    for cell_id in planned_ids:
        result = by_id[cell_id]
        planned_identity = {
            field: plan_by_id[cell_id].get(field) for field in PatchCell.__dataclass_fields__
        }
        result_identity = {field: result.get(field) for field in PatchCell.__dataclass_fields__}
        if planned_identity != result_identity:
            raise ContractError(f"planned/result identity mismatch for cell {cell_id}")
        if result.get("provenance") != provenance:
            raise ContractError(f"result/scan provenance mismatch for cell {cell_id}")

    cells_dir = directory / "cells"
    if not cells_dir.is_dir():
        raise ContractError(f"scan has no atomic completed-cell directory: {directory}")
    atomic_paths = sorted(cells_dir.glob("*.json"))
    atomic_ids = {path.stem for path in atomic_paths}
    if atomic_ids != set(planned_ids):
        raise ContractError(f"atomic completed-cell coverage mismatch: {directory}")
    for path in atomic_paths:
        if read_json(path) != by_id[path.stem]:
            raise ContractError(f"merged result differs from atomic cell: {path}")

    failures_dir = directory / "failures"
    if not failures_dir.is_dir():
        raise ContractError(f"scan has no append-only failure directory: {directory}")
    store = AtomicCellStore(directory)
    # This also reconstructs every PatchCell/failure ID, so a renamed or
    # hand-edited atomic record cannot hide behind a valid merged JSONL row.
    if store.rows() != sorted(results, key=lambda row: str(row["cell_id"])):
        raise ContractError(f"atomic completed cells differ from merged results: {directory}")
    failure_rows = store.failure_rows()
    failure_log = directory / "failures.jsonl"
    if not failure_log.is_file() or read_jsonl(failure_log) != failure_rows:
        raise ContractError(f"append-only failure log differs from failure atoms: {directory}")
    for index, row in enumerate(failure_rows):
        validate_patch_result_row(
            row,
            label=f"{failure_log.relative_to(run_root)}:{index + 1}",
            require_empirical_provenance=not synthetic,
        )

    resume_path = directory / "resume_summary.json"
    if not resume_path.is_file():
        raise ContractError(f"scan has no resume summary: {directory}")
    resume = read_json(resume_path)
    expected_resume = {
        "planned_cells": len(planned_ids),
        "completed_cells": len(planned_ids),
        "unresolved_failed_cells": 0,
        "duplicate_cells": 0,
        "failure_attempts": len(failure_rows),
        "run_identity_sha256": provenance.get("run_identity_sha256"),
        "planned_cells_sha256": plan.get("planned_cells_sha256"),
    }
    mismatches = {
        field: (resume.get(field), expected) for field, expected in expected_resume.items()
        if resume.get(field) != expected
    }
    if mismatches:
        raise ContractError(f"scan resume summary mismatch: {mismatches}")

    if not synthetic:
        run_hash = str(provenance.get("run_identity_sha256", ""))
        binding = authorizations.get(run_hash)
        if binding is None:
            raise ContractError(f"empirical scan has no matching GO authorization: {run_hash}")
        mapping = {
            "code_commit": "code_commit",
            "model_repo": "model_repo",
            "model_revision": "model_revision",
            "manifest_sha256": "manifest_sha256",
            "encoded_manifest_sha256": "encoded_manifest_sha256",
            "config_sha256": "config_sha256",
            "scan_spec_sha256": "scan_spec_sha256",
            "data_sha256": "data_sha256",
        }
        for provenance_field, binding_field in mapping.items():
            if provenance.get(provenance_field) != binding.get(binding_field):
                raise ContractError(
                    f"empirical scan provenance differs from authorization: {provenance_field}"
                )
    return len(results), synthetic


def verify_patch_artifacts(run_root: Path) -> tuple[int, bool]:
    """Verify exact scan grids, atomic cells, row provenance, and GO bindings."""

    plan_paths = sorted(run_root.rglob("scan_plan.json"))
    if not plan_paths:
        raise ContractError("run contains no exact causal scan plan")
    authorizations = _authorization_index(run_root)
    global_ids: set[str] = set()
    count = 0
    synthetic_modes: set[bool] = set()
    planned_result_paths: set[Path] = set()
    for plan_path in plan_paths:
        plan = read_json(plan_path)
        if not isinstance(plan, Mapping):
            raise ContractError(f"scan plan is not an object: {plan_path}")
        result_uri = require_relative_uri(str(plan.get("result_uri", "")))
        planned_result_paths.add(plan_path.parent / result_uri)
        added, synthetic = _verify_plan_directory(
            plan_path.parent,
            run_root=run_root,
            global_ids=global_ids,
            authorizations=authorizations,
        )
        count += added
        synthetic_modes.add(synthetic)

    # Confirmation artifacts may use a generic filename.  Empirical results
    # must nevertheless have an exact scan plan; an unplanned GPU result is not
    # verifiable evidence.  Synthetic confirmation fixtures remain permitted.
    for result_path in sorted(run_root.rglob("*patch_results.jsonl")):
        if result_path in planned_result_paths:
            continue
        raw = read_jsonl(result_path)
        if not raw or any(row.get("synthetic") is not True for row in raw):
            raise ContractError(f"empirical patch results have no exact scan plan: {result_path}")
        synthetic_modes.add(True)
        for index, row in enumerate(raw):
            validated = validate_patch_result_row(
                row,
                label=f"{result_path.relative_to(run_root)}:{index + 1}",
                require_empirical_provenance=False,
            )
            if validated.get("status") != "completed":
                raise ContractError(f"synthetic confirmation contains a failed cell: {result_path}")
            cell_id = str(validated["cell_id"])
            if cell_id in global_ids:
                raise ContractError(f"patch cell appears in multiple result files: {cell_id}")
            global_ids.add(cell_id)
            count += 1
    if count == 0:
        raise ContractError("run contains no patch result cells")
    if len(synthetic_modes) != 1:
        raise ContractError("run mixes synthetic and empirical patch result sets")
    return count, synthetic_modes == {True}


def verify_analysis_provenance(run_root: Path, summary: Mapping[str, Any]) -> None:
    if summary.get("analysis_status") == "empirical_requires_gate_review":
        from .analysis_protocol import load_frozen_analysis_inputs

        spec_uri = require_relative_uri(str(summary.get("analysis_spec_uri", "")))
        expected_uri = require_relative_uri(str(summary.get("expected_cells_uri", "")))
        spec_path = run_root / spec_uri
        expected_path = run_root / expected_uri
        spec, frozen_cells, result_paths = load_frozen_analysis_inputs(
            run_root=run_root,
            analysis_spec_path=spec_path,
            expected_cells_path=expected_path,
        )
        if summary.get("analysis_spec_sha256") != sha256_file(spec_path):
            raise ContractError("analysis summary has the wrong frozen spec file hash")
        if summary.get("analysis_spec_value_sha256") != spec.get("analysis_spec_sha256"):
            raise ContractError("analysis summary has the wrong frozen spec value hash")
        if summary.get("expected_cells_sha256") != sha256_file(expected_path):
            raise ContractError("analysis summary has the wrong expected-cell file hash")
        if summary.get("expected_cells_value_sha256") != frozen_cells.get(
            "expected_cells_sha256"
        ):
            raise ContractError("analysis summary has the wrong expected-cell value hash")
    else:
        result_paths = sorted(run_root.rglob("*patch_results.jsonl"))
    expected = {
        path.relative_to(run_root).as_posix(): sha256_file(path) for path in result_paths
    }
    if summary.get("provenance") != expected:
        raise ContractError("analysis summary provenance does not match patch result files")
    reported = summary.get("n_cells")
    observed = sum(len(read_jsonl(path)) for path in result_paths)
    if reported != observed:
        raise ContractError(f"analysis summary cell count mismatch: {reported} != {observed}")


def verify_package_checksums(
    manifest: Mapping[str, Any], *, public_path: Path, private_path: Path,
) -> None:
    _schema_validate(manifest, PACKAGE_CHECKSUM_SCHEMA, "package checksum manifest")
    archives = manifest.get("archives")
    if not isinstance(archives, Mapping) or set(archives) != {"public", "private"}:
        raise ContractError("package checksum manifest must describe public and private archives")
    for kind, path in (("public", public_path), ("private", private_path)):
        row = archives[kind]
        if not isinstance(row, Mapping) or set(row) != {"filename", "sha256", "bytes"}:
            raise ContractError(f"package checksum entry is invalid: {kind}")
        if row.get("filename") != path.name:
            raise ContractError(f"package checksum filename mismatch: {kind}")
        validate_sha256(str(row.get("sha256", "")), f"{kind} package hash")
        if row.get("bytes") != path.stat().st_size:
            raise ContractError(f"package byte count mismatch: {kind}")
        if row.get("sha256") != sha256_file(path):
            raise ContractError(f"package SHA-256 mismatch: {kind}")


def package_checksum_manifest(public_path: Path, private_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "archives": {
            "public": {
                "filename": public_path.name,
                "sha256": sha256_file(public_path),
                "bytes": public_path.stat().st_size,
            },
            "private": {
                "filename": private_path.name,
                "sha256": sha256_file(private_path),
                "bytes": private_path.stat().st_size,
            },
        },
    }
