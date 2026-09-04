from __future__ import annotations

import numpy as np
import pytest

from experiments.self_repair.mechanistic.core import ContractError
from experiments.self_repair.mechanistic.core import sha256_value
from experiments.self_repair.mechanistic.probes import (
    apply_frozen_probe,
    fit_grouped_ridge_probe,
    freeze_probe_report,
    validate_frozen_probe_artifact,
)


def _fixture() -> tuple[np.ndarray, list[str], list[str], list[str]]:
    cities = ["Boston", "Seattle", "Denver", "Austin"]
    features, labels, groups, row_ids = [], [], [], []
    for scenario in range(8):
        for city_index, city in enumerate(cities):
            vector = np.zeros(6, dtype=np.float64)
            vector[city_index] = 5.0
            vector[4] = scenario / 10
            vector[5] = (scenario % 2) / 10
            features.append(vector)
            labels.append(city)
            groups.append(f"scenario-{scenario}")
            row_ids.append(f"scenario-{scenario}:{city}")
    return np.asarray(features), labels, groups, row_ids


def _frozen(result: dict) -> dict:
    capture_contract = {
        "schema_version": "1.0.0",
        "feature_dimension": result["feature_dimension"],
        "feature_policy": "unit_fixture",
    }
    return freeze_probe_report({
        **result,
        "role": "discovery",
        "training_role": "discovery",
        "causal_use_prohibited": True,
        "capture_contract": capture_contract,
        "capture_contract_sha256": sha256_value(capture_contract),
        "capture_manifest_sha256": None,
        "source_manifest_sha256": None,
        "site_selection_sha256": None,
        "site_selection_identity_sha256": None,
    })


def test_grouped_probe_supports_four_classes_without_group_leakage() -> None:
    features, labels, groups, row_ids = _fixture()
    result = fit_grouped_ridge_probe(
        features, labels, groups, folds=4, seed=17, row_ids=row_ids
    )
    assert result["classes"] == ["Austin", "Boston", "Denver", "Seattle"]
    assert result["cv_accuracy"] == 1.0
    for fold in result["folds"]:
        assert set(fold["train_groups"]).isdisjoint(fold["test_groups"])
    frozen = _frozen(result)
    predictions = apply_frozen_probe(
        frozen, features[:4], expected_training_rows_sha256=result["training_rows_sha256"]
    )
    assert predictions == labels[:4]


def test_frozen_probe_rejects_wrong_discovery_identity() -> None:
    features, labels, groups, row_ids = _fixture()
    result = _frozen(fit_grouped_ridge_probe(features, labels, groups, row_ids=row_ids))
    with pytest.raises(ContractError, match="identity mismatch"):
        apply_frozen_probe(result, features[:1], expected_training_rows_sha256="0" * 64)


def test_frozen_probe_rejects_tampering_and_feature_dimension_mismatch() -> None:
    features, labels, groups, row_ids = _fixture()
    frozen = _frozen(fit_grouped_ridge_probe(features, labels, groups, row_ids=row_ids))
    tampered = {**frozen, "cv_accuracy": 0.0}
    with pytest.raises(ContractError, match="self-hash mismatch"):
        validate_frozen_probe_artifact(tampered)
    with pytest.raises(ContractError, match="feature dimension mismatch"):
        apply_frozen_probe(
            frozen,
            np.zeros((2, features.shape[1] + 1)),
            expected_training_rows_sha256=frozen["training_rows_sha256"],
        )


def test_grouped_probe_rejects_fold_missing_a_class() -> None:
    features = np.eye(4)
    with pytest.raises(ContractError, match="does not contain every class"):
        fit_grouped_ridge_probe(
            features,
            ["A", "A", "B", "B"],
            ["only-a", "only-a", "only-b", "only-b"],
            folds=2,
        )
