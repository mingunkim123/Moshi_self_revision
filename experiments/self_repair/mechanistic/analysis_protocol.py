"""Frozen, missing-cell-closed causal analysis for patching experiments.

The statistical functions in this module deliberately know nothing about a
checkpoint.  They consume a preregistered set of atomic patch cells and refuse
to change the estimand when a result is missing, duplicated, or relabelled.
The planning helpers additionally bind a human-editable analysis template to
one or more already-materialized ``planned_cells.jsonl`` files *before* any of
their corresponding result files exist.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .core import (
    ContractError,
    bootstrap_mean_ci,
    holm_adjust,
    read_json,
    read_jsonl,
    require_relative_uri,
    sha256_file,
    sha256_value,
    validate_sha256,
    write_json,
)


ANALYSIS_METRICS = frozenset({"baseline_M", "patched_M", "delta_M"})
EXPECTED_IDENTITY_FIELDS = (
    "role",
    "donor_arm",
    "relation",
    "component",
    "layer",
    "head",
    "anchor",
    "donor_trial_id",
    "recipient_trial_id",
)
_MISSING = object()


def _validate_schema(value: Mapping[str, Any], filename: str) -> None:
    try:
        import jsonschema
    except ImportError as error:  # pragma: no cover - pinned runtime dependency
        raise ContractError("jsonschema is required for frozen analysis validation") from error
    schema = read_json(Path(__file__).resolve().parent / "schemas" / filename)
    try:
        jsonschema.validate(dict(value), schema)
    except jsonschema.ValidationError as error:
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise ContractError(
            f"{filename} validation failed at {location}: {error.message}"
        ) from error


def _nested_value(row: Mapping[str, Any], name: str) -> Any:
    value: Any = row
    for part in name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _matches(row: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    """Match exact or explicitly enumerated frozen selector values.

    A selector value is normally a scalar.  ``{"in": [...]}`` is supported
    for a preregistered union of roles or arms, and dotted field names may
    address immutable nested provenance.  No regex/callable expressions are
    accepted, keeping selector behavior portable and schema-auditable.
    """

    for name, expected in selector.items():
        observed = _nested_value(row, str(name))
        if observed is _MISSING:
            return False
        if isinstance(expected, Mapping):
            if set(expected) != {"in"}:
                raise ContractError(
                    f"selector {name!r} only supports the frozen 'in' operator"
                )
            choices = expected["in"]
            if not isinstance(choices, list) or not choices:
                raise ContractError(f"selector {name!r} has an empty/invalid 'in' set")
            if observed not in choices:
                return False
        elif observed != expected:
            return False
    return True


def _validate_hypotheses(hypotheses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not hypotheses:
        raise ContractError("analysis requires at least one frozen hypothesis")
    identifiers: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for hypothesis_index, source in enumerate(hypotheses):
        if not isinstance(source, Mapping):
            raise ContractError(f"hypothesis {hypothesis_index} must be an object")
        hypothesis = dict(source)
        identifier = hypothesis.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ContractError("frozen hypothesis ids must be unique non-empty strings")
        identifiers.add(identifier)
        if hypothesis.get("direction") not in {"positive", "negative"}:
            raise ContractError(f"hypothesis {identifier} has invalid direction")
        sesoi = hypothesis.get("sesoi")
        if (
            isinstance(sesoi, bool)
            or not isinstance(sesoi, (int, float))
            or not math.isfinite(float(sesoi))
            or float(sesoi) < 0
        ):
            raise ContractError(f"hypothesis {identifier} has invalid SESOI")
        terms = hypothesis.get("terms")
        if not isinstance(terms, list) or len(terms) < 2:
            raise ContractError(
                f"hypothesis {identifier} requires at least two frozen contrast terms"
            )
        normalized_terms: list[dict[str, Any]] = []
        for term_index, source_term in enumerate(terms):
            if not isinstance(source_term, Mapping):
                raise ContractError(
                    f"hypothesis {identifier} term {term_index} must be an object"
                )
            selector = source_term.get("selector")
            weight = source_term.get("weight")
            metric = source_term.get("metric", "delta_M")
            if not isinstance(selector, Mapping) or not selector:
                raise ContractError(
                    f"hypothesis {identifier} term {term_index} has no selector"
                )
            for selector_name, selector_value in selector.items():
                if not isinstance(selector_name, str) or not selector_name:
                    raise ContractError(
                        f"hypothesis {identifier} term {term_index} has an invalid selector name"
                    )
                if isinstance(selector_value, Mapping):
                    if set(selector_value) != {"in"}:
                        raise ContractError(
                            f"selector {selector_name!r} only supports the frozen 'in' operator"
                        )
                    choices = selector_value["in"]
                    if not isinstance(choices, list) or not choices:
                        raise ContractError(
                            f"selector {selector_name!r} has an empty/invalid 'in' set"
                        )
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
            ):
                raise ContractError(
                    f"hypothesis {identifier} term {term_index} has an invalid weight"
                )
            if metric not in ANALYSIS_METRICS:
                raise ContractError(
                    f"hypothesis {identifier} term {term_index} has unsupported metric {metric!r}"
                )
            normalized_terms.append({
                "selector": dict(selector),
                "metric": str(metric),
                "weight": float(weight),
            })
        hypothesis["terms"] = normalized_terms
        hypothesis["family"] = str(hypothesis.get("family", "primary"))
        if not hypothesis["family"]:
            raise ContractError(f"hypothesis {identifier} has an empty family")
        hypothesis["sesoi"] = float(sesoi)
        normalized.append(hypothesis)
    return normalized


def _two_sided_sign_flip(values: Sequence[float], *, replicates: int, seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ContractError("sign-flip test requires at least two finite scenario contrasts")
    total_exact = 1 << int(array.size) if array.size <= 20 else None
    observed = abs(float(array.mean()))
    if total_exact is not None and total_exact <= max(2, int(replicates)):
        masks = np.arange(total_exact, dtype=np.uint64)[:, None]
        bits = (masks >> np.arange(array.size, dtype=np.uint64)) & 1
        signs = bits.astype(np.float64) * 2.0 - 1.0
        null = (signs * array).mean(axis=1)
        return float(np.mean(np.abs(null) >= observed - 1e-15))
    rng = np.random.default_rng(seed)
    count = max(2000, int(replicates))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, array.size))
    null = (signs * array).mean(axis=1)
    return float((np.sum(np.abs(null) >= observed - 1e-15) + 1) / (count + 1))


def validate_result_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_cell_ids: Sequence[str] | None = None,
    expected_cells: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate one immutable result per cell and reject incomplete analyses."""

    if not rows:
        raise ContractError("analysis requires at least one patch result")
    observed: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ContractError("every analysis row requires a non-empty cell_id")
        if cell_id in observed:
            raise ContractError(f"duplicate analysis cell: {cell_id}")
        if row.get("status") != "completed":
            raise ContractError(f"analysis cell is not completed: {cell_id}")
        value = row.get("delta_M")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ContractError(f"analysis cell has a non-finite delta_M: {cell_id}")
        scenario = row.get("scenario_id")
        if not isinstance(scenario, str) or not scenario:
            raise ContractError(f"analysis cell has no scenario_id: {cell_id}")
        observed[cell_id] = row
    if expected_cell_ids is not None and expected_cells is not None:
        raise ContractError("provide expected cell IDs or expected cell records, not both")
    expected_by_id: dict[str, Mapping[str, Any]] | None = None
    if expected_cells is not None:
        expected_by_id = {}
        for expected in expected_cells:
            if not isinstance(expected, Mapping):
                raise ContractError("expected analysis cells must be objects")
            cell_id = expected.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id:
                raise ContractError("every expected analysis cell requires a cell_id")
            if cell_id in expected_by_id:
                raise ContractError("expected analysis cell ids contain duplicates")
            for field in EXPECTED_IDENTITY_FIELDS:
                if field not in expected:
                    raise ContractError(
                        f"expected analysis cell {cell_id} has no frozen {field}"
                    )
            expected_by_id[cell_id] = expected
        expected_cell_ids = list(expected_by_id)
    if expected_cell_ids is not None:
        expected = [str(value) for value in expected_cell_ids]
        if len(set(expected)) != len(expected):
            raise ContractError("expected analysis cell ids contain duplicates")
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        if missing or extra:
            raise ContractError(
                f"analysis cell coverage mismatch; missing={missing[:10]}, extra={extra[:10]}"
            )
    if expected_by_id is not None:
        for cell_id, row in observed.items():
            expected = expected_by_id[cell_id]
            for field in EXPECTED_IDENTITY_FIELDS:
                if row.get(field, _MISSING) != expected[field]:
                    raise ContractError(
                        f"analysis cell {cell_id} changed frozen identity field {field}"
                    )
    return [observed[cell_id] for cell_id in sorted(observed)]


def _scenario_contrast(
    rows: Sequence[Mapping[str, Any]], terms: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[float], list[dict[str, Any]]]:
    if len(terms) < 2:
        raise ContractError("a frozen causal contrast requires at least two terms")
    # Only scenarios in the frozen contrast's union are relevant.  This lets a
    # run contain discovery, internal, and formal roles without requiring every
    # role in every unrelated scenario, while still requiring every term for
    # every scenario actually entering this estimand.
    scoped_rows = [
        row for row in rows
        if any(_matches(row, term["selector"]) for term in terms)
    ]
    if not scoped_rows:
        raise ContractError("no result rows match any frozen contrast term")
    scenarios = sorted({str(row["scenario_id"]) for row in scoped_rows})
    values: list[float] = []
    audit: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_rows = [row for row in rows if str(row["scenario_id"]) == scenario]
        contrast = 0.0
        term_audit: list[dict[str, Any]] = []
        for index, term in enumerate(terms):
            selector = term["selector"]
            weight = term["weight"]
            metric = str(term.get("metric", "delta_M"))
            matches = [row for row in scenario_rows if _matches(row, selector)]
            if not matches:
                raise ContractError(
                    f"scenario {scenario} is missing frozen contrast term {index}: {dict(selector)}"
                )
            term_values: list[float] = []
            for row in matches:
                value = row.get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ContractError(
                        f"scenario {scenario} has non-finite {metric} for frozen term {index}"
                    )
                term_values.append(float(value))
            term_mean = float(np.mean(term_values))
            contrast += float(weight) * term_mean
            term_audit.append({
                "selector": dict(selector), "weight": float(weight),
                "metric": metric, "n_cells": len(matches), "term_mean": term_mean,
            })
        values.append(contrast)
        audit.append({"scenario_id": scenario, "contrast": contrast, "terms": term_audit})
    return scenarios, values, audit


def analyze_frozen_contrasts(
    rows: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    *,
    expected_cell_ids: Sequence[str] | None = None,
    expected_cells: Sequence[Mapping[str, Any]] | None = None,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260826,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate preregistered scenario-cluster contrasts with Holm correction.

    A difference-in-differences is expressed as four weighted terms.  Every
    term must be observed in every scenario, so a dropped OOM/error cell can
    never silently change the estimand.
    """

    validated = validate_result_rows(
        rows, expected_cell_ids=expected_cell_ids, expected_cells=expected_cells)
    synthetic_values = {row.get("synthetic") for row in validated}
    if not synthetic_values <= {True, False} or len(synthetic_values) != 1:
        raise ContractError("analysis cannot mix synthetic and empirical result rows")
    hypotheses = _validate_hypotheses(hypotheses)
    if bootstrap_replicates < 100:
        raise ContractError("bootstrap_replicates must be at least 100")
    if not 0 < float(alpha) < 1:
        raise ContractError("analysis alpha must be between zero and one")
    registry: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for index, source in enumerate(hypotheses):
        hypothesis = dict(source)
        identifier = hypothesis.get("id")
        direction = hypothesis["direction"]
        sesoi = hypothesis["sesoi"]
        scenarios, contrasts, audit = _scenario_contrast(validated, hypothesis.get("terms", ()))
        if len(scenarios) < 2:
            raise ContractError(f"hypothesis {identifier} requires at least two scenario clusters")
        estimate, low, high = bootstrap_mean_ci(
            contrasts, int(bootstrap_replicates), int(seed) + index * 1009
        )
        p_value = _two_sided_sign_flip(
            contrasts, replicates=int(bootstrap_replicates), seed=int(seed) + index * 1009 + 1
        )
        raw_p.append(p_value)
        registry.append({
            "hypothesis_id": identifier,
            "family": str(hypothesis.get("family", "primary")),
            "direction": direction,
            "sesoi": float(sesoi),
            "n_scenario_clusters": len(scenarios),
            "estimate": estimate,
            "ci95": [low, high],
            "raw_p_two_sided": p_value,
            "scenario_contrasts": audit,
        })
    adjusted = [1.0] * len(registry)
    families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(registry):
        families[str(row["family"])].append(index)
    for indexes in families.values():
        corrected_family = holm_adjust([raw_p[index] for index in indexes])
        for index, corrected in zip(indexes, corrected_family, strict=True):
            adjusted[index] = corrected
    for row, corrected in zip(registry, adjusted, strict=True):
        row["holm_p"] = corrected
        beyond_sesoi = (
            float(row["ci95"][0]) > float(row["sesoi"])
            if row["direction"] == "positive"
            else float(row["ci95"][1]) < -float(row["sesoi"])
        )
        row["passes_sesoi_ci"] = beyond_sesoi
        row["passed"] = bool(corrected <= alpha and beyond_sesoi)
    all_synthetic = synthetic_values == {True}
    return {
        "schema_version": "1.0.0",
        "analysis_status": (
            "synthetic_local_validation" if all_synthetic else "empirical_requires_gate_review"
        ),
        "test": "two_sided_scenario_cluster_sign_flip",
        "bootstrap": "scenario_cluster_percentile",
        "multiplicity": "holm_within_frozen_family",
        "alpha": float(alpha),
        "n_cells": len(validated),
        "expected_cell_count": (
            len(expected_cells) if expected_cells is not None
            else len(expected_cell_ids) if expected_cell_ids is not None else None
        ),
        "all_cells_complete": True,
        "hypotheses_sha256": sha256_value(list(hypotheses)),
        "registry": registry,
        "passed": all(row["passed"] for row in registry),
    }


def _portable_relative(root: Path, path: Path, label: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ContractError(f"{label} must be inside the run root") from error
    return require_relative_uri(relative)


def _immutable_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if read_json(path) != dict(value):
            raise ContractError(f"refusing to replace a different frozen artifact: {path}")
        return
    write_json(path, value)


def _selection_binding(run_root: Path, path: Path, config_sha256: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ContractError(f"frozen selection must be an object: {path}")
    if value.get("status") != "frozen_discovery_selection":
        raise ContractError(f"selection is not frozen_discovery_selection: {path}")
    if value.get("config_sha256") != config_sha256:
        raise ContractError(f"selection targets a different config: {path}")
    declared = value.get("selection_sha256")
    body = dict(value)
    body.pop("selection_sha256", None)
    if declared != sha256_value(body):
        raise ContractError(f"selection self-hash mismatch: {path}")
    return {
        "uri": _portable_relative(run_root, path, "selection"),
        "sha256": sha256_file(path),
        "selection_sha256": validate_sha256(str(declared), "selection hash"),
    }


def build_expected_cells(
    *, run_root: Path, planned_cell_paths: Sequence[Path], require_pristine: bool = True,
) -> dict[str, Any]:
    """Aggregate exact cells and result URIs from multiple immutable scan plans."""

    if not planned_cell_paths:
        raise ContractError("at least one --planned-cells file is required")
    root = run_root.resolve()
    cells_by_id: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    seen_plan_uris: set[str] = set()
    for planned_path in planned_cell_paths:
        planned_path = planned_path.resolve()
        planned_uri = _portable_relative(root, planned_path, "planned-cells file")
        if planned_uri in seen_plan_uris:
            raise ContractError(f"duplicate planned-cells input: {planned_uri}")
        seen_plan_uris.add(planned_uri)
        if planned_path.name != "planned_cells.jsonl" or not planned_path.is_file():
            raise ContractError(f"expected an existing planned_cells.jsonl: {planned_path}")
        scan_plan_path = planned_path.parent / "scan_plan.json"
        if not scan_plan_path.is_file():
            raise ContractError(f"planned cells have no sibling scan_plan.json: {planned_uri}")
        planned = read_jsonl(planned_path)
        scan_plan = read_json(scan_plan_path)
        if not isinstance(scan_plan, Mapping):
            raise ContractError(f"scan plan must be an object: {scan_plan_path}")
        if scan_plan.get("planned_cell_count") != len(planned):
            raise ContractError(f"scan plan cell count mismatch: {planned_uri}")
        if scan_plan.get("planned_cells_sha256") != sha256_value(planned):
            raise ContractError(f"scan plan does not authenticate planned cells: {planned_uri}")
        role = scan_plan.get("role")
        if not isinstance(role, str) or not role:
            raise ContractError(f"scan plan has no analysis role: {planned_uri}")
        result_name = scan_plan.get("result_uri")
        if not isinstance(result_name, str):
            raise ContractError(f"scan plan has no result_uri: {planned_uri}")
        result_name = require_relative_uri(result_name)
        result_path = (planned_path.parent / PurePosixPath(result_name)).resolve()
        result_uri = _portable_relative(root, result_path, "planned result")
        if require_pristine:
            existing_atomic = [
                *sorted((planned_path.parent / "cells").glob("*.json")),
                *sorted((planned_path.parent / "failures").glob("*.json")),
            ]
            if result_path.exists() or existing_atomic:
                raise ContractError(
                    f"analysis must be frozen before results exist: {planned_uri}"
                )
        source_selection_hash = scan_plan.get("selection_sha256")
        provenance = scan_plan.get("provenance")
        if source_selection_hash is None and isinstance(provenance, Mapping):
            source_selection_hash = provenance.get("selection_file_sha256")
        source = {
            "planned_cells_uri": planned_uri,
            "planned_cells_file_sha256": sha256_file(planned_path),
            "planned_cells_value_sha256": sha256_value(planned),
            "scan_plan_uri": _portable_relative(root, scan_plan_path, "scan plan"),
            "scan_plan_sha256": sha256_file(scan_plan_path),
            "result_uri": result_uri,
            "role": role,
            "kind": scan_plan.get("kind"),
            "cell_count": len(planned),
            "selection_binding": source_selection_hash,
        }
        sources.append(source)
        for row in planned:
            cell_id = row.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id:
                raise ContractError(f"planned row has no cell_id: {planned_uri}")
            if cell_id in cells_by_id:
                raise ContractError(f"planned cell occurs in multiple sources: {cell_id}")
            expected: dict[str, Any] = {"cell_id": cell_id, "role": role}
            for field in EXPECTED_IDENTITY_FIELDS:
                if field == "role":
                    continue
                if field not in row:
                    raise ContractError(
                        f"planned cell {cell_id} has no analysis identity field {field}"
                    )
                expected[field] = row[field]
            cells_by_id[cell_id] = expected
    sources.sort(key=lambda row: str(row["planned_cells_uri"]))
    cells = [cells_by_id[cell_id] for cell_id in sorted(cells_by_id)]
    if not cells:
        raise ContractError("expected-cell aggregation produced no cells")
    body: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "frozen_before_confirmation",
        "cell_count": len(cells),
        "cell_ids": [row["cell_id"] for row in cells],
        "cell_ids_sha256": sha256_value([row["cell_id"] for row in cells]),
        "cells": cells,
        "source_plans": sources,
        "source_plans_sha256": sha256_value(sources),
        "pristine_at_freeze": bool(require_pristine),
    }
    body["expected_cells_sha256"] = sha256_value(body)
    return body


def validate_expected_cells(
    payload: Mapping[str, Any], *, run_root: Path | None = None,
) -> dict[str, Any]:
    """Validate an expected-cell artifact and, optionally, its source plans."""

    value = dict(payload)
    if value.get("status") != "frozen_before_confirmation":
        raise ContractError("expected cells were not frozen before confirmation")
    declared = value.get("expected_cells_sha256")
    body = dict(value)
    body.pop("expected_cells_sha256", None)
    if declared != sha256_value(body):
        raise ContractError("expected-cell artifact self-hash mismatch")
    cells = value.get("cells")
    ids = value.get("cell_ids")
    sources = value.get("source_plans")
    if not isinstance(cells, list) or not isinstance(ids, list) or not isinstance(sources, list):
        raise ContractError("expected-cell artifact has invalid cells/source_plans arrays")
    if any(not isinstance(row, Mapping) for row in cells):
        raise ContractError("expected cell must be an object")
    if value.get("cell_count") != len(cells) or ids != [row.get("cell_id") for row in cells]:
        raise ContractError("expected-cell artifact count/order mismatch")
    if len(set(ids)) != len(ids) or not ids or value.get("cell_ids_sha256") != sha256_value(ids):
        raise ContractError("expected-cell ID set is empty, duplicated, or hash-mismatched")
    if value.get("source_plans_sha256") != sha256_value(sources):
        raise ContractError("expected-cell source-plan hash mismatch")
    for cell in cells:
        for field in ("cell_id", *EXPECTED_IDENTITY_FIELDS):
            if field not in cell:
                raise ContractError(f"expected cell has no {field}")
    if sum(int(source.get("cell_count", -1)) for source in sources if isinstance(source, Mapping)) != len(cells):
        raise ContractError("expected-cell source counts do not sum to the frozen cell count")
    _validate_schema(value, "expected-cells.schema.json")
    if run_root is not None:
        for source in sources:
            if not isinstance(source, Mapping):
                raise ContractError("expected source plan must be an object")
            planned_path = run_root / require_relative_uri(str(source.get("planned_cells_uri", "")))
            scan_plan_path = run_root / require_relative_uri(str(source.get("scan_plan_uri", "")))
            if not planned_path.is_file() or sha256_file(planned_path) != source.get(
                "planned_cells_file_sha256"
            ):
                raise ContractError("frozen planned-cells file is missing or changed")
            if not scan_plan_path.is_file() or sha256_file(scan_plan_path) != source.get(
                "scan_plan_sha256"
            ):
                raise ContractError("frozen scan-plan file is missing or changed")
    return value


def freeze_analysis_artifacts(
    *,
    run_root: Path,
    config_path: Path,
    template_path: Path,
    selection_paths: Sequence[Path],
    planned_cell_paths: Sequence[Path],
    output_spec: Path,
    output_expected_cells: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze a self-hashed empirical analysis plan and exact cell universe."""

    root = run_root.resolve()
    if not config_path.is_file() or not template_path.is_file():
        raise ContractError("analysis freeze requires existing config and template files")
    if not selection_paths:
        raise ContractError("analysis freeze requires at least one frozen selection")
    for output, label in (
        (output_spec, "analysis spec"), (output_expected_cells, "expected cells")
    ):
        _portable_relative(root, output, label)
    config_hash = sha256_file(config_path)
    template = read_json(template_path)
    if not isinstance(template, Mapping):
        raise ContractError("analysis template must be an object")
    if template.get("analysis_kind") != "empirical_preregistered":
        raise ContractError("analysis template must declare empirical_preregistered")
    reserved = {
        "status", "analysis_spec_sha256", "config_sha256", "template_sha256",
        "expected_cells_uri", "expected_cells_file_sha256", "expected_cells_sha256",
        "selection_bindings", "source_plans_sha256", "expected_cell_count",
    }
    if reserved & set(template):
        raise ContractError("analysis template contains fields reserved for freezing")
    hypotheses = template.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ContractError("analysis template requires a hypotheses array")
    normalized_hypotheses = _validate_hypotheses(hypotheses)
    alpha = template.get("alpha")
    replicates = template.get("bootstrap_replicates")
    seed = template.get("bootstrap_seed")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < float(alpha) < 1:
        raise ContractError("analysis template alpha must be between zero and one")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise ContractError("analysis template bootstrap_replicates must be at least 100")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractError("analysis template bootstrap_seed must be an integer")
    expected = build_expected_cells(
        run_root=root, planned_cell_paths=planned_cell_paths, require_pristine=True)
    # Every frozen term must address at least one planned cell.  Result-time
    # validation strengthens this to every term in every included scenario.
    for hypothesis in normalized_hypotheses:
        for index, term in enumerate(hypothesis["terms"]):
            if not any(_matches(cell, term["selector"]) for cell in expected["cells"]):
                raise ContractError(
                    f"hypothesis {hypothesis['id']} term {index} matches no planned cell"
                )
    selection_bindings = sorted(
        (_selection_binding(root, path, config_hash) for path in selection_paths),
        key=lambda row: str(row["uri"]),
    )
    if len({row["uri"] for row in selection_bindings}) != len(selection_bindings):
        raise ContractError("duplicate frozen selection input")
    selection_file_hashes = {row["sha256"] for row in selection_bindings}
    selection_value_hashes = {row["selection_sha256"] for row in selection_bindings}
    for source in expected["source_plans"]:
        binding = source.get("selection_binding")
        if binding is not None and binding not in selection_file_hashes | selection_value_hashes:
            raise ContractError(
                f"source plan {source['planned_cells_uri']} targets an unlisted selection"
            )
    _immutable_write_json(output_expected_cells, expected)
    expected_file_hash = sha256_file(output_expected_cells)
    spec: dict[str, Any] = {
        key: value for key, value in template.items() if key != "hypotheses"
    }
    spec.update({
        "hypotheses": normalized_hypotheses,
        "status": "frozen_before_confirmation",
        "config_sha256": config_hash,
        "template_sha256": sha256_file(template_path),
        "expected_cells_uri": _portable_relative(root, output_expected_cells, "expected cells"),
        "expected_cells_file_sha256": expected_file_hash,
        "expected_cells_sha256": expected["expected_cells_sha256"],
        "expected_cell_count": expected["cell_count"],
        "source_plans_sha256": expected["source_plans_sha256"],
        "selection_bindings": selection_bindings,
    })
    spec["analysis_spec_sha256"] = sha256_value(spec)
    _validate_schema(spec, "analysis-spec.schema.json")
    _immutable_write_json(output_spec, spec)
    return spec, expected


def validate_analysis_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if value.get("analysis_kind") != "empirical_preregistered":
        raise ContractError("analysis spec is not empirical_preregistered")
    if value.get("status") != "frozen_before_confirmation":
        raise ContractError("analysis spec is not frozen_before_confirmation")
    declared = value.get("analysis_spec_sha256")
    body = dict(value)
    body.pop("analysis_spec_sha256", None)
    if declared != sha256_value(body):
        raise ContractError("analysis spec SHA-256 mismatch")
    _validate_schema(value, "analysis-spec.schema.json")
    hypotheses = value.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ContractError("analysis spec requires a hypothesis list")
    value["hypotheses"] = _validate_hypotheses(hypotheses)
    return value


def load_frozen_analysis_inputs(
    *, run_root: Path, analysis_spec_path: Path, expected_cells_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """Load and re-authenticate a frozen spec, expected cells, plans and results."""

    root = run_root.resolve()
    _portable_relative(root, analysis_spec_path.resolve(), "analysis spec")
    spec = validate_analysis_spec(read_json(analysis_spec_path))
    declared_uri = require_relative_uri(str(spec.get("expected_cells_uri", "")))
    declared_path = (root / PurePosixPath(declared_uri)).resolve()
    _portable_relative(root, declared_path, "expected cells")
    if expected_cells_path is not None and expected_cells_path.resolve() != declared_path:
        raise ContractError("--expected-cells differs from the path frozen in analysis spec")
    if not declared_path.is_file() or sha256_file(declared_path) != spec.get(
        "expected_cells_file_sha256"
    ):
        raise ContractError("frozen expected-cell file is missing or changed")
    expected = validate_expected_cells(read_json(declared_path), run_root=root)
    if (
        spec.get("expected_cells_sha256") != expected.get("expected_cells_sha256")
        or spec.get("expected_cell_count") != expected.get("cell_count")
        or spec.get("source_plans_sha256") != expected.get("source_plans_sha256")
    ):
        raise ContractError("analysis spec and expected-cell artifact disagree")
    for binding in spec.get("selection_bindings", []):
        if not isinstance(binding, Mapping):
            raise ContractError("analysis selection binding must be an object")
        path = root / require_relative_uri(str(binding.get("uri", "")))
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise ContractError("frozen selection is missing or changed after preregistration")
    result_files: list[Path] = []
    for source in expected["source_plans"]:
        result = (root / require_relative_uri(str(source["result_uri"]))).resolve()
        _portable_relative(root, result, "planned result")
        if not result.is_file():
            raise ContractError(f"planned result file is missing: {source['result_uri']}")
        ambiguous = [
            path for path in result.parent.glob("*patch_results.jsonl")
            if path.resolve() != result
        ]
        if ambiguous:
            raise ContractError(
                f"planned result directory contains an unregistered patch result: {ambiguous[0].name}"
            )
        result_files.append(result)
    if len({path.resolve() for path in result_files}) != len(result_files):
        raise ContractError("multiple frozen source plans point to the same result file")
    return spec, expected, result_files
