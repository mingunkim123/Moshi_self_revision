"""Leakage-resistant diagnostic probes for mechanistic activation captures.

The probes in this module are deliberately diagnostic.  They are fit only on
the role supplied by the caller, use scenario-grouped cross validation, and
carry the exact training-row digest needed to freeze and later verify them.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .core import (
    ContractError,
    apply_probe,
    fit_ridge_probe,
    sha256_value,
    validate_sha256,
)


FROZEN_PROBE_STATUS = "frozen_diagnostic_probe"
PROBE_TRAINING_ROLES = frozenset({"discovery", "multivalue_calibration"})
PROBE_APPLICATION_ROLES = frozenset({"internal_validation", "formal_confirmation"})


def _training_rows(
    labels: Sequence[str], groups: Sequence[str], row_ids: Sequence[str],
) -> list[dict[str, str]]:
    return [
        {"row_id": row_id, "label": label, "group": group}
        for row_id, label, group in sorted(
            zip(row_ids, labels, groups, strict=True), key=lambda item: item[0]
        )
    ]


def _validated_probe_model(model: Mapping[str, Any]) -> tuple[int, list[str]]:
    """Validate the serialized ridge shapes before NumPy broadcasting can hide errors."""

    try:
        mean = np.asarray(model["mean"], dtype=np.float64)
        scale = np.asarray(model["scale"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        classes = [str(value) for value in model["classes"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("frozen probe model is malformed") from error
    if (
        mean.ndim != 1
        or mean.size < 1
        or scale.shape != mean.shape
        or weights.shape != (mean.size + 1, len(classes))
        or len(classes) < 2
        or len(set(classes)) != len(classes)
        or any(not value for value in classes)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(weights).all()
        or np.any(scale <= 0)
    ):
        raise ContractError("frozen probe model has invalid dimensions or numeric values")
    return int(mean.size), classes


def freeze_probe_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Create the immutable, self-hashed artifact used outside the fit split."""

    if report.get("diagnostic_only") is not True or report.get("causal_use_prohibited") is not True:
        raise ContractError("only explicitly non-causal diagnostic probes may be frozen")
    training_role = str(report.get("training_role", report.get("role", "")))
    if training_role not in PROBE_TRAINING_ROLES:
        raise ContractError(f"probe role {training_role!r} is not an allowed training role")
    if report.get("training_role") != training_role:
        raise ContractError("probe report must explicitly record its training_role")
    frozen = dict(report)
    frozen["status"] = FROZEN_PROBE_STATUS
    frozen["source_sha256"] = sha256_value(report)
    frozen["frozen_probe_sha256"] = sha256_value(frozen)
    validate_frozen_probe_artifact(frozen)
    return frozen


def validate_frozen_probe_artifact(
    frozen: Mapping[str, Any], *, expected_training_rows_sha256: str | None = None,
) -> tuple[int, list[str]]:
    """Fail closed on tampering, forbidden roles, or an inconsistent fit identity."""

    if frozen.get("status") != FROZEN_PROBE_STATUS:
        raise ContractError("probe artifact is not frozen_diagnostic_probe")
    if frozen.get("diagnostic_only") is not True or frozen.get("causal_use_prohibited") is not True:
        raise ContractError("frozen probe is not explicitly diagnostic/non-causal")
    training_role = str(frozen.get("training_role", frozen.get("role", "")))
    if training_role not in PROBE_TRAINING_ROLES:
        raise ContractError(f"frozen probe has forbidden training role: {training_role!r}")
    if frozen.get("role") != training_role:
        raise ContractError("frozen probe role differs from its training_role")
    expected_frozen_hash = validate_sha256(
        str(frozen.get("frozen_probe_sha256", "")), "frozen probe"
    )
    observed_frozen_hash = sha256_value({
        key: value for key, value in frozen.items() if key != "frozen_probe_sha256"
    })
    if observed_frozen_hash != expected_frozen_hash:
        raise ContractError("frozen probe self-hash mismatch")
    expected_source_hash = validate_sha256(
        str(frozen.get("source_sha256", "")), "frozen probe source"
    )
    source_report = {
        key: value for key, value in frozen.items()
        if key not in {"status", "source_sha256", "frozen_probe_sha256"}
    }
    if sha256_value(source_report) != expected_source_hash:
        raise ContractError("frozen probe source hash mismatch")

    training_rows = frozen.get("training_rows")
    if not isinstance(training_rows, list) or not training_rows:
        raise ContractError("frozen probe does not carry its discovery training-row identities")
    normalized_rows: list[dict[str, str]] = []
    for row in training_rows:
        if not isinstance(row, Mapping):
            raise ContractError("frozen probe training-row identity is malformed")
        normalized = {
            "row_id": str(row.get("row_id", "")),
            "label": str(row.get("label", "")),
            "group": str(row.get("group", "")),
        }
        if any(not value for value in normalized.values()):
            raise ContractError("frozen probe training-row identity is incomplete")
        normalized_rows.append(normalized)
    normalized_rows.sort(key=lambda row: row["row_id"])
    if len({row["row_id"] for row in normalized_rows}) != len(normalized_rows):
        raise ContractError("frozen probe training-row identities are not unique")
    training_hash = validate_sha256(
        str(frozen.get("training_rows_sha256", "")), "probe training rows"
    )
    if sha256_value(normalized_rows) != training_hash:
        raise ContractError("frozen probe training-row hash mismatch")
    if expected_training_rows_sha256 is not None:
        expected = validate_sha256(expected_training_rows_sha256, "expected probe training rows")
        if training_hash != expected:
            raise ContractError("frozen probe training-row identity mismatch")
    if int(frozen.get("n_rows", -1)) != len(normalized_rows):
        raise ContractError("frozen probe training-row count mismatch")
    if int(frozen.get("n_groups", -1)) != len({row["group"] for row in normalized_rows}):
        raise ContractError("frozen probe training-group count mismatch")

    probe = frozen.get("probe")
    if not isinstance(probe, Mapping):
        raise ContractError("frozen probe artifact has no model")
    feature_dimension, classes = _validated_probe_model(probe)
    if int(frozen.get("feature_dimension", -1)) != feature_dimension:
        raise ContractError("frozen probe feature dimension metadata mismatch")
    if list(frozen.get("classes", [])) != classes:
        raise ContractError("frozen probe class metadata mismatch")
    if {row["label"] for row in normalized_rows} != set(classes):
        raise ContractError("frozen probe training labels differ from its classes")
    capture_contract = frozen.get("capture_contract")
    if not isinstance(capture_contract, Mapping):
        raise ContractError("frozen probe has no capture provenance contract")
    capture_contract_hash = validate_sha256(
        str(frozen.get("capture_contract_sha256", "")), "probe capture contract"
    )
    if sha256_value(capture_contract) != capture_contract_hash:
        raise ContractError("frozen probe capture provenance hash mismatch")
    if int(capture_contract.get("feature_dimension", -1)) != feature_dimension:
        raise ContractError("frozen probe capture/probe feature dimensions differ")
    coordinate = frozen.get("probe_coordinate")
    if coordinate is not None:
        if (
            not isinstance(coordinate, Mapping)
            or not isinstance(coordinate.get("site"), str)
            or not coordinate.get("site")
            or isinstance(coordinate.get("layer"), bool)
            or not isinstance(coordinate.get("layer"), int)
            or int(coordinate["layer"]) < 0
            or not isinstance(coordinate.get("anchor"), str)
            or not coordinate.get("anchor")
        ):
            raise ContractError("frozen probe has a malformed exact probe coordinate")
        if capture_contract.get("feature_policy") != "flatten_one_exact_captured_tensor":
            raise ContractError("exact frozen probe has a non-exact feature policy")
        if capture_contract.get("probe_coordinate") != coordinate:
            raise ContractError("frozen probe coordinate differs from its capture contract")
        grid_cell_hash = validate_sha256(
            str(frozen.get("probe_grid_cell_sha256", "")), "frozen probe grid cell"
        )
        if frozen.get("probe_grid_cell_identity_policy") != (
            "canonical_report_excluding_capture_manifest_file_sha256"
        ):
            raise ContractError("frozen probe grid-cell identity policy is unsupported")
        grid_cell_body = {
            key: value for key, value in source_report.items()
            if key not in {"probe_grid_cell_sha256", "capture_manifest_sha256"}
        }
        if sha256_value(grid_cell_body) != grid_cell_hash:
            raise ContractError("frozen probe grid-cell identity mismatch")
    for field in ("capture_manifest_sha256", "source_manifest_sha256"):
        value = frozen.get(field)
        if value is not None:
            validate_sha256(str(value), f"frozen probe {field}")
    selection_file_hash = frozen.get("site_selection_sha256")
    selection_identity_hash = frozen.get("site_selection_identity_sha256")
    if (selection_file_hash is None) != (selection_identity_hash is None):
        raise ContractError("frozen probe site-selection provenance is incomplete")
    if selection_file_hash is not None:
        validate_sha256(str(selection_file_hash), "frozen probe site-selection file")
        validate_sha256(str(selection_identity_hash), "frozen probe site-selection identity")
    return feature_dimension, classes


def _validated_inputs(
    features: np.ndarray,
    labels: Sequence[str],
    groups: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    x = np.asarray(features, dtype=np.float64)
    y = [str(label) for label in labels]
    g = [str(group) for group in groups]
    if x.ndim != 2 or x.shape[0] != len(y) or len(y) != len(g):
        raise ContractError("probe features, labels, and groups have incompatible shapes")
    if x.shape[0] < 4 or not np.isfinite(x).all():
        raise ContractError("probe requires at least four finite feature rows")
    if any(not label for label in y) or any(not group for group in g):
        raise ContractError("probe labels and scenario groups must be non-empty")
    if len(set(y)) < 2:
        raise ContractError("probe requires at least two classes")
    if len(set(g)) < 2:
        raise ContractError("grouped probe requires at least two scenario groups")
    return x, y, g


def _balanced_group_folds(
    labels: Sequence[str], groups: Sequence[str], *, folds: int, seed: int,
) -> dict[str, int]:
    """Assign entire groups to deterministic, approximately balanced folds."""

    unique = sorted(set(groups))
    fold_count = min(int(folds), len(unique))
    if fold_count < 2:
        raise ContractError("grouped probe requires at least two folds")
    rng = np.random.default_rng(seed)
    tie_break = {group: float(rng.random()) for group in unique}
    class_names = sorted(set(labels))
    group_counts = {
        group: Counter(label for label, observed_group in zip(labels, groups) if observed_group == group)
        for group in unique
    }
    # Large and class-skewed groups are placed first; seeded values only break
    # genuine ties, which makes the assignment stable under input-row ordering.
    ordered = sorted(
        unique,
        key=lambda group: (
            -sum(group_counts[group].values()),
            -max(group_counts[group].values()),
            tie_break[group],
            group,
        ),
    )
    totals = [0] * fold_count
    class_totals = [Counter() for _ in range(fold_count)]
    assignment: dict[str, int] = {}
    for group in ordered:
        def score(index: int) -> tuple[float, int, int]:
            projected = class_totals[index] + group_counts[group]
            imbalance = sum(projected[name] ** 2 for name in class_names)
            return float(imbalance), totals[index], index

        destination = min(range(fold_count), key=score)
        assignment[group] = destination
        totals[destination] += sum(group_counts[group].values())
        class_totals[destination].update(group_counts[group])
    if set(assignment) != set(unique):
        raise ContractError("not every scenario group was assigned to a probe fold")
    return assignment


def fit_grouped_ridge_probe(
    features: np.ndarray,
    labels: Sequence[str],
    groups: Sequence[str],
    *,
    alpha: float = 1.0,
    folds: int = 5,
    seed: int = 20260826,
    row_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Cross-validate by scenario and fit one frozen K-class ridge probe.

    The returned final probe is fit on all discovery rows only after every
    held-out prediction has been made.  Callers must not use formal rows here.
    """

    x, y, g = _validated_inputs(features, labels, groups)
    if not np.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ContractError("probe alpha must be finite and positive")
    ids = [str(value) for value in (row_ids if row_ids is not None else range(len(y)))]
    if len(ids) != len(y) or len(set(ids)) != len(ids):
        raise ContractError("probe row ids must be unique and match the feature rows")
    assignment = _balanced_group_folds(y, g, folds=folds, seed=seed)
    fold_ids = np.asarray([assignment[group] for group in g], dtype=np.int64)
    predictions: list[str | None] = [None] * len(y)
    fold_reports: list[dict[str, Any]] = []
    all_classes = set(y)
    for fold in sorted(set(fold_ids.tolist())):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        train_labels = [label for label, keep in zip(y, train_mask.tolist()) if keep]
        if set(train_labels) != all_classes:
            raise ContractError(
                f"probe fold {fold} training partition does not contain every class"
            )
        model = fit_ridge_probe(x[train_mask], train_labels, alpha=float(alpha))
        held_out = apply_probe(model, x[test_mask])
        test_indices = np.flatnonzero(test_mask).tolist()
        for index, prediction in zip(test_indices, held_out):
            predictions[index] = prediction
        fold_reports.append({
            "fold": int(fold),
            "train_groups": sorted({group for group, item in assignment.items() if item != fold}),
            "test_groups": sorted({group for group, item in assignment.items() if item == fold}),
            "train_n": int(train_mask.sum()),
            "test_n": int(test_mask.sum()),
            "accuracy": float(np.mean(np.asarray(held_out) == np.asarray(y)[test_mask])),
        })
    if any(value is None for value in predictions):
        raise ContractError("grouped cross-validation did not score every probe row exactly once")
    cv_predictions = [str(value) for value in predictions]
    final_model = fit_ridge_probe(x, y, alpha=float(alpha))
    training_rows = _training_rows(y, g, ids)
    training_rows_sha256 = sha256_value(training_rows)
    return {
        "schema_version": "1.0.0",
        "method": "scenario_grouped_k_class_ridge",
        "diagnostic_only": True,
        "classes": sorted(all_classes),
        "n_rows": len(y),
        "n_groups": len(set(g)),
        "fold_count": len(set(fold_ids.tolist())),
        "folds": fold_reports,
        "cv_predictions": cv_predictions,
        "cv_accuracy": float(np.mean(np.asarray(cv_predictions) == np.asarray(y))),
        "feature_dimension": int(x.shape[1]),
        "training_rows": training_rows,
        "training_rows_sha256": training_rows_sha256,
        "seed": int(seed),
        "probe": final_model,
    }


def apply_frozen_probe(
    frozen: Mapping[str, Any], features: np.ndarray, *, expected_training_rows_sha256: str,
) -> list[str]:
    """Apply a frozen probe only when its discovery-set identity still matches."""

    feature_dimension, _ = validate_frozen_probe_artifact(
        frozen, expected_training_rows_sha256=expected_training_rows_sha256
    )
    x = np.asarray(features, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] != feature_dimension
        or x.shape[0] < 1
        or not np.isfinite(x).all()
    ):
        raise ContractError(
            f"frozen probe feature dimension mismatch: expected (*, {feature_dimension}), "
            f"observed {tuple(x.shape)}"
        )
    return apply_probe(frozen["probe"], x)
