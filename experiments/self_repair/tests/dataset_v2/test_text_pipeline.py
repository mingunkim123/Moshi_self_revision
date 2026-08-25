from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "dataset_v2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    CONDITIONS,
    DATASET_ROOT,
    DEFAULT_BLUEPRINTS,
    DEFAULT_CONFIG,
    read_config,
    read_jsonl,
    sha256_value,
)
from generate_answer_keys import build_answer_keys  # noqa: E402
from generate_scripts import generate_all  # noqa: E402
from validate_blueprints import validate_blueprints  # noqa: E402
from validate_schemas import validate_rows  # noqa: E402
from validate_scripts import validate_scripts  # noqa: E402


class TextPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = read_config(DEFAULT_CONFIG)
        cls.blueprints = read_jsonl(DEFAULT_BLUEPRINTS)
        cls.scripts = generate_all(cls.blueprints, cls.config)

    def test_frozen_blueprints_and_generated_matrix_validate(self) -> None:
        self.assertEqual(validate_blueprints(self.blueprints, self.config), [])
        self.assertEqual(validate_scripts(self.scripts, self.blueprints, self.config), [])
        self.assertEqual(len(self.blueprints), 30)
        self.assertEqual(len(self.scripts), 300)
        self.assertEqual(len({row["text_bundle_id"] for row in self.scripts}), 60)

    def test_generation_is_order_independent_and_deterministic(self) -> None:
        regenerated = generate_all(list(reversed(self.blueprints)), self.config)
        self.assertEqual(regenerated, self.scripts)
        self.assertEqual(sha256_value(regenerated), sha256_value(self.scripts))

    def test_every_bundle_has_one_shared_gold_state_and_five_conditions(self) -> None:
        by_bundle: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in self.scripts:
            by_bundle[str(row["text_bundle_id"])].append(row)
        for rows in by_bundle.values():
            self.assertEqual({row["condition"] for row in rows}, set(CONDITIONS))
            self.assertEqual(len({sha256_value(row["gold_state"]) for row in rows}), 1)

    def test_delayed_one_identity_and_position_are_counterbalanced(self) -> None:
        rows = [
            row for row in self.scripts
            if row["direction_id"] == "a_to_b"
            and row["condition"] == "delayed_one_dependency"
        ]
        cells = Counter(
            (str(row["one_dependency_unit"]), int(row["one_dependency_pre_position"]))
            for row in rows
        )
        self.assertEqual(Counter(unit for unit, _ in cells.elements()), Counter({"D1": 10, "D2": 10, "D3": 10}))
        self.assertEqual(Counter(position for _, position in cells.elements()), Counter({1: 10, 2: 10, 3: 10}))
        self.assertEqual(set(cells.values()), {3, 4})
        for row in rows:
            selected = str(row["one_dependency_unit"])
            position = int(row["one_dependency_pre_position"])
            self.assertEqual(row["pre_repair_units"][position - 1], selected)

    def test_versioned_rows_match_all_three_json_schemas(self) -> None:
        evidence = json.loads(
            (DATASET_ROOT / "config/value_evidence.json").read_text(encoding="utf-8")
        )
        answer_keys = build_answer_keys(self.blueprints, evidence)
        fixtures = {
            "blueprint": self.blueprints,
            "script": self.scripts,
            "answer_key": answer_keys,
        }
        for name, rows in fixtures.items():
            with self.subTest(schema=name):
                schema = json.loads(
                    (DATASET_ROOT / f"schemas/{name}.schema.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(validate_rows(rows, schema), [])
        self.assertEqual(len(answer_keys), 60)


if __name__ == "__main__":
    unittest.main()
