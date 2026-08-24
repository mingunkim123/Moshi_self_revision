#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOSHI_REPO="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MOSHI_REPO="${MOSHI_REPO:-${DEFAULT_MOSHI_REPO}}"
MOSHI_VENV="${MOSHI_VENV:-${MOSHI_REPO}/.venv}"
EXPERIMENT_DIR="${MOSHI_REPO}/experiments/self_repair"

if [[ ! -x "${MOSHI_VENV}/bin/python" ]]; then
  echo "Virtual environment missing. Run ${EXPERIMENT_DIR}/runpod/setup.sh first." >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to synthesize the TTS stimuli." >&2
  exit 1
fi

"${MOSHI_VENV}/bin/python" -m pip install -r "${EXPERIMENT_DIR}/requirements-tts.txt"
"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/synthesize_english_neural_tts.py" \
  --city-a Boston \
  --city-b Seattle \
  --raw-root "${EXPERIMENT_DIR}/data/raw_us" \
  --recordings "${EXPERIMENT_DIR}/data/recordings.us.csv" \
  --metadata "${EXPERIMENT_DIR}/data/tts_metadata.us.json" \
  --overwrite

echo "Boston/Seattle TTS stimuli are ready."
