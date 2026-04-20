#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_DIR="${SNAPSHOT_DIR:-$(cat artifacts/urgency_autoresearch/latest_snapshot.txt)}"
MODEL_NAME="${MODEL_NAME:-answerdotai/ModernBERT-base}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TEXT_PREP_MODE="${TEXT_PREP_MODE:-title_head}"
CLASSIFIER_TYPE="${CLASSIFIER_TYPE:-transformer}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LINEAR_C="${LINEAR_C:-1.0}"
TRAIN_BALANCE_MODE="${TRAIN_BALANCE_MODE:-full}"
EMBED_POOLING="${EMBED_POOLING:-mean}"
EMBED_LAYER_NORM="${EMBED_LAYER_NORM:-0}"
EMBED_TRUNCATE_DIM="${EMBED_TRUNCATE_DIM:-0}"
EPOCHS="${EPOCHS:-2}"
RUN_NAME="${RUN_NAME:-$(printf '%s__%s__%s__%s__bs%s__ga%s__lr%s__epochs%s__%s__c%s__pool%s__ln%s__dim%s' "${MODEL_NAME//\//-}" "${MAX_LENGTH}" "${TEXT_PREP_MODE}" "${CLASSIFIER_TYPE}" "${BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${LEARNING_RATE}" "${EPOCHS}" "${TRAIN_BALANCE_MODE}" "${LINEAR_C}" "${EMBED_POOLING}" "${EMBED_LAYER_NORM}" "${EMBED_TRUNCATE_DIM}")}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/urgency_autoresearch/runs/${RUN_NAME}}"

if [[ ! -d "${SNAPSHOT_DIR}" ]]; then
  echo "Snapshot dir does not exist: ${SNAPSHOT_DIR}" >&2
  exit 1
fi

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM=false

ARGS=(
  --snapshot-dir "${SNAPSHOT_DIR}"
  --model-name "${MODEL_NAME}"
  --max-length "${MAX_LENGTH}"
  --text-prep-mode "${TEXT_PREP_MODE}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --classifier-type "${CLASSIFIER_TYPE}"
  --batch-size "${BATCH_SIZE}"
  --train-balance-mode "${TRAIN_BALANCE_MODE}"
)

if [[ "${CLASSIFIER_TYPE}" == "transformer" ]]; then
  ARGS+=(
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning-rate "${LEARNING_RATE}"
    --epochs "${EPOCHS}"
  )
else
  ARGS+=(
    --linear-c "${LINEAR_C}"
    --embed-pooling "${EMBED_POOLING}"
    --embed-truncate-dim "${EMBED_TRUNCATE_DIM}"
  )
  if [[ "${EMBED_LAYER_NORM}" == "1" ]]; then
    ARGS+=(--embed-layer-norm)
  fi
fi

uv run python -m feedoscope.urgency_experiments.harness "${ARGS[@]}"
