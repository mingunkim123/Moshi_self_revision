#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOSHI_REPO="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MOSHI_REPO="${MOSHI_REPO:-${DEFAULT_MOSHI_REPO}}"
MOSHI_VENV="${MOSHI_VENV:-${MOSHI_REPO}/.venv}"
MOSHI_HF_CACHE="${MOSHI_HF_CACHE:-/workspace/hf-cache}"
EXPERIMENT_DIR="${MOSHI_REPO}/experiments/self_repair"
RECORDINGS_CSV="${EXPERIMENT_DIR}/data/recordings.us.csv"
CONDITIONS_CSV="${EXPERIMENT_DIR}/data/conditions.us.csv"
CONFIG_JSON="${EXPERIMENT_DIR}/config/experiment.us.json"
PREPARED_MANIFEST="${EXPERIMENT_DIR}/data/manifest.us.prepared.csv"
RESULTS_DIR="${EXPERIMENT_DIR}/results_us"
ANNOTATIONS_DIR="${EXPERIMENT_DIR}/annotations"

if [[ ! -x "${MOSHI_VENV}/bin/python" ]]; then
  echo "Virtual environment missing. Run ${EXPERIMENT_DIR}/runpod/setup.sh first." >&2
  exit 1
fi
if [[ ! -f "${RECORDINGS_CSV}" ]]; then
  echo "Boston/Seattle TTS is missing. Run ${EXPERIMENT_DIR}/runpod/synthesize_us_tts.sh first." >&2
  exit 1
fi

export HF_HOME="${MOSHI_HF_CACHE}"
export NO_TORCH_COMPILE=1

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/prepare_stimuli.py" \
  --config "${CONFIG_JSON}" \
  --conditions "${CONDITIONS_CSV}" \
  --recordings "${RECORDINGS_CSV}" \
  --output "${PREPARED_MANIFEST}" \
  --prepared-root "${EXPERIMENT_DIR}/data/prepared_us" \
  --overwrite

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/run_eval.py" \
  --config "${CONFIG_JSON}" \
  --manifest "${PREPARED_MANIFEST}" \
  --results-root "${RESULTS_DIR}"

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/auto_label_text.py" \
  --predictions "${RESULTS_DIR}/predictions.jsonl" \
  --output "${ANNOTATIONS_DIR}/annotations.us.auto.csv" \
  --key-output "${ANNOTATIONS_DIR}/annotation_key.us.auto.csv" \
  --overwrite

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/score_results.py" \
  --annotations "${ANNOTATIONS_DIR}/annotations.us.auto.csv" \
  --annotation-key "${ANNOTATIONS_DIR}/annotation_key.us.auto.csv" \
  --manifest "${PREPARED_MANIFEST}" \
  --output-json "${RESULTS_DIR}/metrics.auto.json" \
  --output-md "${RESULTS_DIR}/metrics.auto.md"

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/make_annotation_sheet.py" \
  --predictions "${RESULTS_DIR}/predictions.jsonl" \
  --output "${ANNOTATIONS_DIR}/annotations.us.csv" \
  --key-output "${ANNOTATIONS_DIR}/annotation_key.us.csv" \
  --audio-root "${ANNOTATIONS_DIR}/audio_us" \
  --overwrite

echo
echo "Boston/Seattle inference and preliminary scoring are complete:"
echo "  ${RESULTS_DIR}/metrics.auto.md"
