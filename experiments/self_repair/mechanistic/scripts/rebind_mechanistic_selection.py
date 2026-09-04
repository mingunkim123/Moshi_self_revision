#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.self_repair.mechanistic.selection_artifacts import (  # noqa: E402
    rebind_mechanistic_selection_main,
)


if __name__ == "__main__":
    raise SystemExit(rebind_mechanistic_selection_main())
