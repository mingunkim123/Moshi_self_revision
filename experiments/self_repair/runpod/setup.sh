#!/usr/bin/env bash
set -euo pipefail

MOSHI_REPO="${MOSHI_REPO:-/workspace/moshi}"
MOSHI_VENV="${MOSHI_VENV:-${MOSHI_REPO}/.venv}"
MOSHI_HF_CACHE="${MOSHI_HF_CACHE:-/workspace/hf-cache}"

if [[ ! -f "${MOSHI_REPO}/moshi/pyproject.toml" ]]; then
  echo "Moshi repository not found at ${MOSHI_REPO}. Set MOSHI_REPO first." >&2
  exit 1
fi

python3 -m venv "${MOSHI_VENV}"
"${MOSHI_VENV}/bin/python" -m pip install --upgrade pip
"${MOSHI_VENV}/bin/python" -m pip install -e "${MOSHI_REPO}/moshi"

mkdir -p \
  "${MOSHI_HF_CACHE}" \
  "${MOSHI_REPO}/experiments/self_repair/data/raw" \
  "${MOSHI_REPO}/experiments/self_repair/data/raw_en" \
  "${MOSHI_REPO}/experiments/self_repair/data/prepared" \
  "${MOSHI_REPO}/experiments/self_repair/data/prepared_en" \
  "${MOSHI_REPO}/experiments/self_repair/results" \
  "${MOSHI_REPO}/experiments/self_repair/results_en" \
  "${MOSHI_REPO}/experiments/self_repair/annotations"

echo
echo "Setup complete. Activate with:"
echo "  source ${MOSHI_VENV}/bin/activate"
echo "  export HF_HOME=${MOSHI_HF_CACHE}"
