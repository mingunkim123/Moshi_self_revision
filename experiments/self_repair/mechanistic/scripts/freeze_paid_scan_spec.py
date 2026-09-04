#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.self_repair.mechanistic.paid_scan_spec import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
