#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_DIR="${SNAPSHOT_DIR:-$(cat artifacts/relevance_autoresearch/latest_snapshot.txt)}"
MODEL_NAME="${MODEL_NAME:-answerdotai/ModernBERT-base}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TEXT_PREP_MODE="${TEXT_PREP_MODE:-single_blob}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
EPOCHS="${EPOCHS:-2}"
RUN_NAME="${RUN_NAME:-$(printf '%s__%s__%s__bs%s__ga%s' "${MODEL_NAME//\//-}" "${MAX_LENGTH}" "${TEXT_PREP_MODE}" "${BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}")}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/relevance_autoresearch/runs/${RUN_NAME}}"

if [[ ! -d "${SNAPSHOT_DIR}" ]]; then
  echo "Snapshot dir does not exist: ${SNAPSHOT_DIR}" >&2
  exit 1
fi

uv run python -m feedoscope.relevance_experiments.harness \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --max-length "${MAX_LENGTH}" \
  --text-prep-mode "${TEXT_PREP_MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --epochs "${EPOCHS}"
