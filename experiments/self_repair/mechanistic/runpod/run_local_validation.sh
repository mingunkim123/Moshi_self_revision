#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-mechanistic/bin/python}"
RUN_ROOT="${MECH_LOCAL_RUN_ROOT:-$REPO_ROOT/.cache/mechanistic-local-validation}"
CONFIG="$REPO_ROOT/experiments/self_repair/mechanistic/config/mechanistic.json"

export NO_TORCH_COMPILE=1
export NO_CUDA_GRAPH=1

# A verified artifact manifest seals the run.  Re-invocation must be read-only:
# mutating even a resume counter after sealing would invalidate the archive.
if [[ -f "$RUN_ROOT/artifact_sha256.json" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/verify_mechanistic_run.py" \
    --run-root "$RUN_ROOT" --allow-local-synthetic --synthetic
  echo "Synthetic/local validation already sealed and verified: $RUN_ROOT"
  exit 0
fi

mkdir -p "$RUN_ROOT/discovery/residual"

"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/scan_residual_patches.py" \
  --config "$CONFIG" --role local_validation --output-root "$RUN_ROOT/discovery/residual" --synthetic --resume
# A second pass proves identity-first resume and leaves the final summary in its
# stable all-cells-skipped state before the artifact manifest is sealed.
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/scan_residual_patches.py" \
  --config "$CONFIG" --role local_validation --output-root "$RUN_ROOT/discovery/residual" --synthetic --resume
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/freeze_mechanistic_selection.py" \
  --config "$CONFIG" --discovery-root "$RUN_ROOT/discovery" --output "$RUN_ROOT/mechanistic_frozen_selection.json" --synthetic
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/run_confirmatory_patches.py" \
  --config "$CONFIG" --selection "$RUN_ROOT/mechanistic_frozen_selection.json" --role local_validation \
  --output-root "$RUN_ROOT/internal_validation" --synthetic --resume
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/run_confirmatory_patches.py" \
  --config "$CONFIG" --selection "$RUN_ROOT/mechanistic_frozen_selection.json" --role local_validation \
  --output-root "$RUN_ROOT/internal_validation" --synthetic --resume
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/analyze_mechanistic_results.py" \
  --run-root "$RUN_ROOT" --bootstrap-replicates 2000 --bootstrap-seed 20260826 --synthetic
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/render_mechanistic_report.py" \
  --run-root "$RUN_ROOT" --synthetic
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/verify_mechanistic_run.py" \
  --run-root "$RUN_ROOT" --allow-local-synthetic --synthetic

echo "Synthetic/local validation complete: $RUN_ROOT"
