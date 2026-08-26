#!/usr/bin/env python3
"""Build a deterministic MFA dictionary from a frozen base and OOV extension."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from common import DATASET_ROOT, sha256_file, write_json


DEFAULT_EXTENSION = DATASET_ROOT / "config/mfa_english_us_arpa_oov.dict"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extension", type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _dictionary_words(path: Path) -> tuple[set[str], int]:
    words: set[str] = set()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError(f"{path}:{line_number}: expected WORD<TAB>DICTIONARY FIELDS")
            words.add(parts[0].strip())
            rows += 1
    if rows == 0:
        raise ValueError(f"{path}: dictionary is empty")
    return words, rows


def build_dictionary(base: Path, extension: Path, output: Path) -> dict[str, object]:
    base_words, base_rows = _dictionary_words(base)
    extension_words, extension_rows = _dictionary_words(extension)
    overlap = sorted(base_words & extension_words)
    if overlap:
        raise ValueError(f"extension contains words already present in base dictionary: {overlap[:10]}")
    base_bytes = base.read_bytes()
    extension_bytes = extension.read_bytes()
    payload = base_bytes
    if payload and not payload.endswith(b"\n"):
        payload += b"\n"
    payload += extension_bytes
    if payload and not payload.endswith(b"\n"):
        payload += b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {
        "schema_version": "2.0.0",
        "base_sha256": sha256_file(base),
        "extension_sha256": sha256_file(extension),
        "output_sha256": sha256_file(output),
        "base_row_count": base_rows,
        "extension_row_count": extension_rows,
        "extension_word_count": len(extension_words),
        "output_row_count": base_rows + extension_rows,
    }


def main() -> None:
    args = parse_args()
    report = build_dictionary(args.base, args.extension, args.output)
    if args.report:
        write_json(args.report, report)
    print(
        f"Built MFA dictionary with {report['output_row_count']} pronunciations -> {args.output}"
    )


if __name__ == "__main__":
    main()
