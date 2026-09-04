from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "moshi"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
