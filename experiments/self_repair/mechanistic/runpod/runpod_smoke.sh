#!/usr/bin/env bash
set -euo pipefail

: "${MECH_DATA_ROOT:?set MECH_DATA_ROOT to the portable dataset root}"
: "${MECH_RUN_ROOT:?set MECH_RUN_ROOT to an empty identity-specific output directory}"
: "${NO_TORCH_COMPILE:?set NO_TORCH_COMPILE=1 before starting}"
: "${NO_CUDA_GRAPH:?set NO_CUDA_GRAPH=1 before starting}"
test "$NO_TORCH_COMPILE" = 1
test "$NO_CUDA_GRAPH" = 1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CONFIG="$REPO_ROOT/experiments/self_repair/mechanistic/config/mechanistic.json"
MANIFEST="$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl"

"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/validate_mechanistic_contract.py" \
  --config "$CONFIG" --manifest "$MANIFEST" --input-artifact-root "$MECH_DATA_ROOT" \
  --output-root "$MECH_RUN_ROOT/preflight" --model-repo kyutai/moshiko-pytorch-bf16 \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289

"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/encode_user_audio.py" \
  --manifest "$MANIFEST" --input-artifact-root "$MECH_DATA_ROOT" --output-root "$MECH_RUN_ROOT/encoded_user" \
  --output-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289 --resume

echo "Preflight and Mimi encoding passed. Continue with the frozen smoke grid in the runbook."
