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

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/run_eval.py" \
  --config "${CONFIG_JSON}" \
  --manifest "${PREPARED_MANIFEST}" \
  --results-root "${RESULTS_DIR}"

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/auto_label_text.py" \
  --predictions "${RESULTS_DIR}/predictions.jsonl" \
  --output "${EXPERIMENT_DIR}/annotations/annotations.en.auto.csv" \
  --key-output "${EXPERIMENT_DIR}/annotations/annotation_key.en.auto.csv" \
  --overwrite

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/score_results.py" \
  --annotations "${EXPERIMENT_DIR}/annotations/annotations.en.auto.csv" \
  --annotation-key "${EXPERIMENT_DIR}/annotations/annotation_key.en.auto.csv" \
  --manifest "${PREPARED_MANIFEST}" \
  --output-json "${RESULTS_DIR}/metrics.auto.json" \
  --output-md "${RESULTS_DIR}/metrics.auto.md"

"${MOSHI_VENV}/bin/python" "${EXPERIMENT_DIR}/scripts/make_annotation_sheet.py" \
  --predictions "${RESULTS_DIR}/predictions.jsonl" \
  --output "${EXPERIMENT_DIR}/annotations/annotations.en.csv" \
  --key-output "${EXPERIMENT_DIR}/annotations/annotation_key.en.csv" \
  --audio-root "${EXPERIMENT_DIR}/annotations/audio_en" \
  --overwrite

echo
echo "English inference and preliminary text scoring are complete:"
echo "  ${RESULTS_DIR}/metrics.auto.md"
echo
echo "Review rows marked AUTO_HEURISTIC_REVIEW or fill annotations.en.csv for human scoring."
echo "Human-score command:"
echo "  ${MOSHI_VENV}/bin/python ${EXPERIMENT_DIR}/scripts/score_results.py"
echo "    --annotations ${EXPERIMENT_DIR}/annotations/annotations.en.csv"
echo "    --annotation-key ${EXPERIMENT_DIR}/annotations/annotation_key.en.csv"
echo "    --manifest ${PREPARED_MANIFEST}"
echo "    --output-json ${RESULTS_DIR}/metrics.json"
echo "    --output-md ${RESULTS_DIR}/metrics.md"
