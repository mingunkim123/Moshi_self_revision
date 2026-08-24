#!/usr/bin/env bash
set -euo pipefail

MOSHI_REPO="${MOSHI_REPO:-/workspace/moshi}"
MOSHI_VENV="${MOSHI_VENV:-${MOSHI_REPO}/.venv}"
MOSHI_HF_CACHE="${MOSHI_HF_CACHE:-/workspace/hf-cache}"
EXPERIMENT_DIR="${MOSHI_REPO}/experiments/self_repair"
RECORDINGS_CSV="${1:-${EXPERIMENT_DIR}/data/recordings.en.csv}"
CONFIG_JSON="${EXPERIMENT_DIR}/config/experiment.en.json"
PREPARED_MANIFEST="${EXPERIMENT_DIR}/data/manifest.en.prepared.csv"
RESULTS_DIR="${EXPERIMENT_DIR}/results_en"

if [[ ! -x "${MOSHI_VENV}/bin/python" ]]; then
  echo "Virtual environment missing. Run ${EXPERIMENT_DIR}/runpod/setup.sh first." >&2
  exit 1
fi
if [[ ! -f "${RECORDINGS_CSV}" ]]; then
  echo "English recording manifest missing: ${RECORDINGS_CSV}" >&2
  exit 1
fi

export HF_HOME="${MOSHI_HF_CACHE}"
export NO_TORCH_COMPILE=1

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/prepare_stimuli.py" \
  --config "${CONFIG_JSON}" \
  --recordings "${RECORDINGS_CSV}" \
  --output "${PREPARED_MANIFEST}" \
  --prepared-root "${EXPERIMENT_DIR}/data/prepared_en" \
  --overwrite

# Clean, one-city repair, and one-to-two-city expansion; one voice and one seed.
"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/run_eval.py" \
  --config "${CONFIG_JSON}" \
  --manifest "${PREPARED_MANIFEST}" \
  --results-root "${RESULTS_DIR}" \
  --speaker EN01 \
  --condition E1 \
  --condition E3 \
  --condition E8 \
  --seeds 17

echo
echo "Smoke run complete. Inspect these three response WAV files:"
find "${RESULTS_DIR}/raw/EN01__E1/seed_17" \
     "${RESULTS_DIR}/raw/EN01__E3/seed_17" \
     "${RESULTS_DIR}/raw/EN01__E8/seed_17" \
     -name response.wav -print
echo
echo "If they contain audible Moshi responses, run:"
echo "  ${EXPERIMENT_DIR}/runpod/run_english_pilot.sh"
