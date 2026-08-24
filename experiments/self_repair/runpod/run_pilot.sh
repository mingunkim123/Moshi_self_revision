#!/usr/bin/env bash
set -euo pipefail

MOSHI_REPO="${MOSHI_REPO:-/workspace/moshi}"
MOSHI_VENV="${MOSHI_VENV:-${MOSHI_REPO}/.venv}"
MOSHI_HF_CACHE="${MOSHI_HF_CACHE:-/workspace/hf-cache}"
EXPERIMENT_DIR="${MOSHI_REPO}/experiments/self_repair"
RECORDINGS_CSV="${1:-${EXPERIMENT_DIR}/data/recordings.csv}"

if [[ ! -x "${MOSHI_VENV}/bin/python" ]]; then
  echo "Virtual environment missing. Run ${EXPERIMENT_DIR}/runpod/setup.sh first." >&2
  exit 1
fi
if [[ ! -f "${RECORDINGS_CSV}" ]]; then
  echo "Recording manifest missing: ${RECORDINGS_CSV}" >&2
  exit 1
fi

export HF_HOME="${MOSHI_HF_CACHE}"
export NO_TORCH_COMPILE=1

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/prepare_stimuli.py" \
  --recordings "${RECORDINGS_CSV}" \
  --output "${EXPERIMENT_DIR}/data/manifest.prepared.csv" \
  --overwrite

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/run_eval.py" \
  --manifest "${EXPERIMENT_DIR}/data/manifest.prepared.csv"

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/make_annotation_sheet.py"

echo
echo "Inference is complete. Fill annotations/annotations.csv, then run:"
echo "  ${MOSHI_VENV}/bin/python ${EXPERIMENT_DIR}/scripts/score_results.py"
