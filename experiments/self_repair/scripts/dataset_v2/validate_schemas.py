#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from common import DATASET_ROOT, portable_path, read_jsonl, write_json


DATASETS = {
    "answer_key": (DATASET_ROOT / "schemas/answer_key.schema.json", DATASET_ROOT / "answer_keys/answer_keys.jsonl"),
    "blueprint": (DATASET_ROOT / "schemas/blueprint.schema.json", DATASET_ROOT / "blueprints/scenarios.jsonl"),
    "script": (DATASET_ROOT / "schemas/script.schema.json", DATASET_ROOT / "generated/scripts.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate versioned JSONL artifacts against JSON Schema.")
    parser.add_argument("kind", choices=sorted(DATASETS))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def validate_rows(rows: list[dict[str, Any]], schema: dict[str, Any]) -> list[str]:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as error:
        raise RuntimeError("Install experiments/self_repair/requirements-v2.txt") from error
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors: list[str] = []
    for row_index, row in enumerate(rows, 1):
        for issue in sorted(validator.iter_errors(row), key=lambda item: list(item.absolute_path)):
            location = ".".join(str(part) for part in issue.absolute_path) or "<root>"
            errors.append(f"row {row_index} {location}: {issue.message}")
    return errors


def main() -> None:
    args = parse_args()
    default_schema, default_input = DATASETS[args.kind]
    schema_path = args.schema or default_schema
    input_path = args.input or default_input
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    rows = read_jsonl(input_path)
    errors = validate_rows(rows, schema)
    report = {
        "schema_version": "2.0.0",
        "kind": args.kind,
        "input": portable_path(input_path),
        "schema": portable_path(schema_path),
        "row_count": len(rows),
        "valid": not errors,
        "errors": errors,
    }
    if args.report:
        write_json(args.report, report)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Schema-valid {args.kind} rows: {len(rows)}")


if __name__ == "__main__":
    main()
