#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_DIR="${SNAPSHOT_DIR:-$(cat artifacts/relevance_autoresearch/latest_snapshot.txt)}"
MODEL_NAME="${MODEL_NAME:-answerdotai/ModernBERT-base}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TEXT_PREP_MODE="${TEXT_PREP_MODE:-single_blob}"
CLASSIFIER_TYPE="${CLASSIFIER_TYPE:-transformer}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LINEAR_C="${LINEAR_C:-1.0}"
TRAIN_BALANCE_MODE="${TRAIN_BALANCE_MODE:-balanced}"
EPOCHS="${EPOCHS:-2}"
RUN_NAME="${RUN_NAME:-$(printf '%s__%s__%s__%s__bs%s__%s' "${MODEL_NAME//\//-}" "${MAX_LENGTH}" "${TEXT_PREP_MODE}" "${CLASSIFIER_TYPE}" "${BATCH_SIZE}" "${TRAIN_BALANCE_MODE}")}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/relevance_autoresearch/runs/${RUN_NAME}}"

if [[ ! -d "${SNAPSHOT_DIR}" ]]; then
  echo "Snapshot dir does not exist: ${SNAPSHOT_DIR}" >&2
  exit 1
fi

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM=false

if [[ "${CLASSIFIER_TYPE}" == "embedding_linear" ]]; then
  SNAPSHOT_DIR="${SNAPSHOT_DIR}" \
  MODEL_NAME="${MODEL_NAME}" \
  MAX_LENGTH="${MAX_LENGTH}" \
  TEXT_PREP_MODE="${TEXT_PREP_MODE}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  SEED="${SEED}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  LINEAR_C="${LINEAR_C}" \
  TRAIN_BALANCE_MODE="${TRAIN_BALANCE_MODE}" \
  uv run python - <<'PY'
import json
import logging
import math
import os
import time
from pathlib import Path

from cleantext import clean
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
import torch
from transformers import AutoModel, AutoTokenizer

from custom_logging import init_logging
from feedoscope import config
from feedoscope.utils import clean_title, strip_html_keep_text

logger = logging.getLogger(__name__)
DEFAULT_EXCELLENT_WEIGHT = config.EXCELLENT_WEIGHT


def _clean_text(text: str) -> str:
    return clean(" ".join(text.split()))


def prepare_single_blob(title: str, content: str) -> str:
    return _clean_text(strip_html_keep_text(f"{clean_title(title)} {content}"))


def prepare_title_head(title: str, content: str, max_length: int, tokenizer) -> str:
    cleaned_title = _clean_text(clean_title(title))
    cleaned_body = _clean_text(strip_html_keep_text(content))
    title_ids = tokenizer.encode(cleaned_title, add_special_tokens=False)
    body_ids = tokenizer.encode(cleaned_body, add_special_tokens=False)
    special_buffer = 4
    if max_length <= special_buffer:
        return tokenizer.decode(title_ids[:1], skip_special_tokens=True).strip()
    title_budget = max(1, min(len(title_ids), max_length - special_buffer))
    kept_title_ids = title_ids[:title_budget]
    remaining_budget = max(0, max_length - special_buffer - len(kept_title_ids))
    head_ids = body_ids[:remaining_budget]
    sep = f" {tokenizer.sep_token} " if tokenizer.sep_token else " "
    parts = [tokenizer.decode(kept_title_ids, skip_special_tokens=True).strip()]
    if head_ids:
        parts.append(tokenizer.decode(head_ids, skip_special_tokens=True).strip())
    return sep.join(part for part in parts if part)


def load_snapshot(snapshot_dir: Path):
    train_df = pd.read_parquet(snapshot_dir / 'train.parquet')
    eval_df = pd.read_parquet(snapshot_dir / 'eval.parquet')
    metadata = json.loads((snapshot_dir / 'metadata.json').read_text())
    return train_df, eval_df, metadata


def apply_training_balance(train_df: pd.DataFrame) -> pd.DataFrame:
    good_df = train_df.loc[train_df['label'] == 1].copy().sort_values('article_id')
    bad_df = train_df.loc[train_df['label'] == 0].copy().sort_values('article_id')
    min_count = min(len(good_df), len(bad_df))
    excellent_mask = (good_df['vote'] == 1) | good_df['starred'].astype(bool)
    excellent_df = good_df.loc[excellent_mask]
    regular_df = good_df.loc[~excellent_mask]
    remaining_slots = max(0, min_count - len(excellent_df))
    regular_df = regular_df.tail(remaining_slots) if remaining_slots > 0 else regular_df.iloc[0:0]
    balanced_good = pd.concat([regular_df, excellent_df], ignore_index=True).sort_values('article_id')
    balanced_bad = bad_df.tail(min_count).copy()
    return pd.concat([balanced_good, balanced_bad], ignore_index=True).sort_values('article_id')


def build_texts(df: pd.DataFrame, tokenizer, max_length: int, text_prep_mode: str) -> list[str]:
    texts = []
    for row in df.to_dict(orient='records'):
        if text_prep_mode == 'single_blob':
            texts.append(prepare_single_blob(row['title'], row['content']))
        elif text_prep_mode == 'title_head':
            texts.append(prepare_title_head(row['title'], row['content'], max_length=max_length, tokenizer=tokenizer))
        else:
            raise ValueError(f'Unsupported text prep mode for embedding harness: {text_prep_mode}')
    return texts


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(model, tokenizer, texts: list[str], max_length: int, batch_size: int, device: torch.device) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                pooled = mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
            elif hasattr(outputs, 'pooler_output'):
                pooled = outputs.pooler_output
            else:
                raise RuntimeError('Model output has neither last_hidden_state nor pooler_output')
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    clipped_probs = np.clip(probs, 1e-7, 1 - 1e-7)
    preds = (clipped_probs >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(labels, clipped_probs)),
        'average_precision': float(average_precision_score(labels, clipped_probs)),
        'log_loss': float(log_loss(labels, clipped_probs)),
    }


def main() -> None:
    init_logging(config.LOGGING_CONFIG)
    snapshot_dir = Path(os.environ['SNAPSHOT_DIR'])
    model_name = os.environ['MODEL_NAME']
    max_length = int(os.environ['MAX_LENGTH'])
    text_prep_mode = os.environ['TEXT_PREP_MODE']
    output_dir = Path(os.environ['OUTPUT_DIR'])
    seed = int(os.environ['SEED'])
    batch_size = int(os.environ['BATCH_SIZE'])
    linear_c = float(os.environ['LINEAR_C'])

    np.random.seed(seed)
    torch.manual_seed(seed)
    train_df, eval_df, metadata = load_snapshot(snapshot_dir)
    train_balance_mode = os.environ['TRAIN_BALANCE_MODE']
    if train_balance_mode == 'balanced':
        balanced_train_df = apply_training_balance(train_df)
    elif train_balance_mode == 'full':
        balanced_train_df = train_df.copy().sort_values('article_id')
    else:
        raise ValueError(f'Unsupported TRAIN_BALANCE_MODE: {train_balance_mode}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not config.ALLOW_TRAINING_WO_GPU:
        raise RuntimeError('GPU not available. Exiting')

    total_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)

    train_texts = build_texts(balanced_train_df, tokenizer=tokenizer, max_length=max_length, text_prep_mode=text_prep_mode)
    eval_texts = build_texts(eval_df, tokenizer=tokenizer, max_length=max_length, text_prep_mode=text_prep_mode)
    train_labels = balanced_train_df['label'].to_numpy()
    eval_labels = eval_df['label'].to_numpy()
    sample_weights = np.where(
        (balanced_train_df['label'].to_numpy() == 1) & ((balanced_train_df['vote'].to_numpy() == 1) | balanced_train_df['starred'].to_numpy()),
        DEFAULT_EXCELLENT_WEIGHT,
        1.0,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_start = time.perf_counter()
    train_embeddings = encode_texts(model, tokenizer, train_texts, max_length, batch_size, device)
    eval_embeddings = encode_texts(model, tokenizer, eval_texts, max_length, batch_size, device)
    clf = LogisticRegression(max_iter=4000, C=linear_c, random_state=seed)
    clf.fit(train_embeddings, train_labels, sample_weight=sample_weights)
    train_seconds = time.perf_counter() - train_start

    probs = clf.predict_proba(eval_embeddings)[:, 1]
    metrics = compute_metrics(eval_labels, probs)
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / math.pow(1024, 3)) if torch.cuda.is_available() else 0.0
    total_seconds = time.perf_counter() - total_start

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        **metrics,
        'peak_vram_gb': peak_vram_gb,
        'train_seconds': float(train_seconds),
        'total_seconds': float(total_seconds),
        'snapshot_id': metadata['snapshot_id'],
        'model_name': model_name,
        'max_length': max_length,
        'text_prep_mode': text_prep_mode,
        'seed': seed,
        'batch_size': batch_size,
        'classifier_type': 'embedding_linear',
        'linear_c': linear_c,
        'train_balance_mode': train_balance_mode,
    }
    (output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    for metric_name in ['average_precision', 'roc_auc', 'log_loss', 'accuracy', 'precision', 'recall', 'f1', 'peak_vram_gb', 'train_seconds', 'total_seconds']:
        print(f"METRIC {metric_name}={results[metric_name]}")


if __name__ == '__main__':
    main()
PY
else
  uv run python -m feedoscope.relevance_experiments.harness \
    --snapshot-dir "${SNAPSHOT_DIR}" \
    --model-name "${MODEL_NAME}" \
    --max-length "${MAX_LENGTH}" \
    --text-prep-mode "${TEXT_PREP_MODE}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --epochs "${EPOCHS}"
fi
