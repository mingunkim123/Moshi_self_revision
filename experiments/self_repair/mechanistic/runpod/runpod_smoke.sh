#!/usr/bin/env bash
set -euo pipefail

: "${MECH_DATA_ROOT:?set MECH_DATA_ROOT to the portable dataset root}"
: "${MECH_RUN_ROOT:?set MECH_RUN_ROOT to an empty identity-specific output directory}"
: "${MECH_SCAN_SPEC:?set MECH_SCAN_SPEC to the frozen paid-scan JSON specification}"
: "${NO_TORCH_COMPILE:?set NO_TORCH_COMPILE=1 before starting}"
: "${NO_CUDA_GRAPH:?set NO_CUDA_GRAPH=1 before starting}"
test "$NO_TORCH_COMPILE" = 1
test "$NO_CUDA_GRAPH" = 1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CONFIG="$REPO_ROOT/experiments/self_repair/mechanistic/config/mechanistic.json"
MANIFEST="$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl"
PREFLIGHT="$MECH_RUN_ROOT/preflight"
CANARY_ROOT="$MECH_RUN_ROOT/gpu_canary"
CANARY_MANIFEST="$CANARY_ROOT/canary_trials.jsonl"
CANARY_ENCODED="$CANARY_ROOT/encoded_manifest.jsonl"
STATIC_ESTIMATE="$PREFLIGHT/workload_estimate.json"
MODEL_REVISION="2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"

mkdir -p "$PREFLIGHT" "$CANARY_ROOT"
test -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)"
UNTRACKED_CODE="$(git -C "$REPO_ROOT" ls-files --others --exclude-standard -- \
  experiments/self_repair/mechanistic moshi/moshi)"
if [[ -n "$UNTRACKED_CODE" ]]; then
  echo "NO_GO: executable/source files are not bound to the current Git commit:" >&2
  echo "$UNTRACKED_CODE" >&2
  exit 3
fi

# CPU/static phase: hashes every declared input and computes the exact grid
# without importing the checkpoint backend.
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/estimate_mechanistic_workload.py" \
  --config "$CONFIG" --manifest "$MANIFEST" --scan-spec "$MECH_SCAN_SPEC" \
  --output "$STATIC_ESTIMATE"
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/validate_mechanistic_contract.py" \
  --config "$CONFIG" --manifest "$MANIFEST" --input-artifact-root "$MECH_DATA_ROOT" \
  --output-root "$PREFLIGHT/static_contract" --model-repo kyutai/moshiko-pytorch-bf16 \
  --model-revision "$MODEL_REVISION" --dry-run

# Missing CUDA is a deliberate STOP. No checkpoint download or encoding follows.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NO_GO: nvidia-smi is unavailable. Static estimate: $STATIC_ESTIMATE. STOP before paid scan." >&2
  exit 3
fi
nvidia-smi
if ! "$PYTHON_BIN" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "NO_GO: PyTorch cannot use CUDA. STOP before checkpoint load or paid scan." >&2
  exit 3
fi

# GPU work is bounded to <=4 rows. This script intentionally never encodes the
# full 600-row manifest; expansion is an explicit post-canary action.
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/select_gpu_canary_manifest.py" \
  --manifest "$MANIFEST" --output "$CANARY_MANIFEST" --max-trials 4 --role discovery
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/validate_mechanistic_contract.py" \
  --config "$CONFIG" --manifest "$MANIFEST" --input-artifact-root "$MECH_DATA_ROOT" \
  --output-root "$PREFLIGHT/model_contract" --model-repo kyutai/moshiko-pytorch-bf16 \
  --model-revision "$MODEL_REVISION"
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/encode_user_audio.py" \
  --manifest "$CANARY_MANIFEST" --input-artifact-root "$MECH_DATA_ROOT" \
  --output-root "$CANARY_ROOT/encoded" --output-manifest "$CANARY_ENCODED" \
  --model-revision "$MODEL_REVISION" --resume
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/validate_open_loop.py" \
  --config "$CONFIG" --encoded-manifest "$CANARY_ENCODED" \
  --output "$CANARY_ROOT/open_loop_validation.json"
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/run_bounded_gpu_canary.py" \
  --config "$CONFIG" --manifest "$CANARY_MANIFEST" --input-artifact-root "$MECH_DATA_ROOT" \
  --workload-estimate "$STATIC_ESTIMATE" --output "$CANARY_ROOT/gpu_measurements.json" --layer 0

"$PYTHON_BIN" - "$CANARY_ROOT/gpu_measurements.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    m = json.load(handle)["measurements"]
print("BOUNDED GPU CANARY: "
      f"peak_vram_bytes={m['peak_vram_bytes']} "
      f"mean_cell_seconds={m['mean_cell_seconds']:.6f} "
      f"activation_bytes={m['activation_bytes']} "
      f"eta_cell_hours={m.get('projected_full_gpu_hours_by_cell')} "
      f"eta_frame_hours={m.get('projected_full_gpu_hours_by_model_frame')} "
      f"reserved_storage_bytes={m.get('projected_total_storage_reserved_bytes')}")
PY

CONVERSATION_EVIDENCE="${MECH_CONVERSATION_CANARY:-$CANARY_ROOT/conversation/conversation_canary.json}"
FULL_ENCODED_MANIFEST="${MECH_FULL_ENCODED_MANIFEST:-$MECH_RUN_ROOT/encoded_user_manifest.jsonl}"
if [[ ! -f "$CONVERSATION_EVIDENCE" ]]; then
  echo "NO_GO: missing bounded two-mode conversation evidence at $CONVERSATION_EVIDENCE." >&2
  echo "Require >=4 trials/mode, explicit text+audio tail checks, 0 cap-active/truncated, and exact output coverage." >&2
  exit 3
fi
if [[ ! -f "$FULL_ENCODED_MANIFEST" ]]; then
  echo "NO_GO: full encoded manifest is absent at $FULL_ENCODED_MANIFEST." >&2
  echo "Only the bounded canary was encoded. Review it before explicitly encoding the full set." >&2
  exit 3
fi

"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/assemble_readiness_evidence.py" \
  --config "$CONFIG" --manifest "$MANIFEST" --encoded-manifest "$FULL_ENCODED_MANIFEST" \
  --scan-spec "$MECH_SCAN_SPEC" --model-contract "$PREFLIGHT/model_contract/model_contract.json" \
  --model-run-identity "$PREFLIGHT/model_contract/run_identity.json" \
  --open-loop "$CANARY_ROOT/open_loop_validation.json" \
  --conversation-canary "$CONVERSATION_EVIDENCE" --gpu-canary "$CANARY_ROOT/gpu_measurements.json" \
  --canary-manifest "$CANARY_MANIFEST" --canary-encoded-manifest "$CANARY_ENCODED" \
  --output "$PREFLIGHT/readiness_evidence.json"
"$PYTHON_BIN" "$REPO_ROOT/experiments/self_repair/mechanistic/scripts/assess_mechanistic_readiness.py" \
  --config "$CONFIG" --manifest "$MANIFEST" --encoded-manifest "$FULL_ENCODED_MANIFEST" \
  --scan-spec "$MECH_SCAN_SPEC" --evidence "$PREFLIGHT/readiness_evidence.json" \
  --output "$PREFLIGHT/paid_scan_authorization.json"

echo "GO artifact: $PREFLIGHT/paid_scan_authorization.json"
echo "Pass both --scan-spec and --readiness-go to the exact paid scan command."
