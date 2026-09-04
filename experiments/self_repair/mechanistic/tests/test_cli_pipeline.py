from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from experiments.self_repair.mechanistic.core import read_json


def test_synthetic_end_to_end_shell_pipeline(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "experiments/self_repair/mechanistic/runpod/run_local_validation.sh"
    environment = os.environ.copy()
    environment.update({"PYTHON_BIN": sys.executable, "MECH_LOCAL_RUN_ROOT": str(tmp_path / "run")})
    subprocess.run(["bash", str(script)], cwd=root, env=environment, check=True, capture_output=True, text=True)
    run = tmp_path / "run"
    summary = read_json(run / "reports/mechanistic_discovery_summary.json")
    assert summary["analysis_status"] == "synthetic_local_validation"
    report = (run / "reports/MECHANISTIC_RESULTS.md").read_text(encoding="utf-8")
    assert "not empirical evidence" in report
    artifact_manifest = read_json(run / "artifact_sha256.json")
    assert artifact_manifest["artifacts"]
