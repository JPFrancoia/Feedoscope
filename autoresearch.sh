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
TRAIN_BALANCE_MODE="${TRAIN_BALANCE_MODE:-full}"
EMBED_POOLING="${EMBED_POOLING:-mean}"
EMBED_PREFIX_MODE="${EMBED_PREFIX_MODE:-none}"
EMBED_PROMPT_MODE="${EMBED_PROMPT_MODE:-none}"
EMBED_FEATURE_MODE="${EMBED_FEATURE_MODE:-dense}"
EMBED_SPARSE_HASH_DIM="${EMBED_SPARSE_HASH_DIM:-8192}"
EMBED_LAYER_NORM="${EMBED_LAYER_NORM:-0}"
EMBED_TRUNCATE_DIM="${EMBED_TRUNCATE_DIM:-0}"
EPOCHS="${EPOCHS:-2}"
RUN_NAME="${RUN_NAME:-$(printf '%s__%s__%s__%s__bs%s__%s__c%s__pool%s__prefix%s__prompt%s__feat%s__hash%s__ln%s__dim%s' "${MODEL_NAME//\//-}" "${MAX_LENGTH}" "${TEXT_PREP_MODE}" "${CLASSIFIER_TYPE}" "${BATCH_SIZE}" "${TRAIN_BALANCE_MODE}" "${LINEAR_C}" "${EMBED_POOLING}" "${EMBED_PREFIX_MODE}" "${EMBED_PROMPT_MODE}" "${EMBED_FEATURE_MODE}" "${EMBED_SPARSE_HASH_DIM}" "${EMBED_LAYER_NORM}" "${EMBED_TRUNCATE_DIM}")}"
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
  EMBED_POOLING="${EMBED_POOLING}" \
  EMBED_PREFIX_MODE="${EMBED_PREFIX_MODE}" \
  EMBED_PROMPT_MODE="${EMBED_PROMPT_MODE}" \
  EMBED_FEATURE_MODE="${EMBED_FEATURE_MODE}" \
  EMBED_SPARSE_HASH_DIM="${EMBED_SPARSE_HASH_DIM}" \
  EMBED_LAYER_NORM="${EMBED_LAYER_NORM}" \
  EMBED_TRUNCATE_DIM="${EMBED_TRUNCATE_DIM}" \
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
from scipy import sparse as sp
from sklearn.feature_extraction import FeatureHasher
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


def prepare_body_head(content: str, max_length: int, tokenizer) -> str:
    cleaned_body = _clean_text(strip_html_keep_text(content))
    body_ids = tokenizer.encode(cleaned_body, add_special_tokens=False)
    special_buffer = 4
    body_budget = max(1, max_length - special_buffer)
    return tokenizer.decode(body_ids[:body_budget], skip_special_tokens=True).strip()


def resolve_text_prefix(prefix_mode: str) -> str:
    if prefix_mode == 'none':
        return ''
    if prefix_mode == 'classification':
        return 'classification: '
    raise ValueError(f'Unsupported EMBED_PREFIX_MODE: {prefix_mode}')


def apply_prompt_template(
    row: dict,
    prepared_text: str,
    tokenizer,
    max_length: int,
    text_prep_mode: str,
    prompt_mode: str,
) -> str:
    if prompt_mode == 'none':
        return prepared_text
    if prompt_mode == 'classification':
        return f'task: classification | query: {prepared_text}'
    if prompt_mode == 'query':
        return f'query: {prepared_text}'
    if prompt_mode == 'document':
        title = _clean_text(clean_title(row['title'])) or 'none'
        if text_prep_mode == 'title_head':
            body = prepare_body_head(row['content'], max_length=max_length, tokenizer=tokenizer)
        elif text_prep_mode == 'single_blob':
            body = _clean_text(strip_html_keep_text(row['content']))
        else:
            body = prepared_text
        body = body or prepared_text
        return f'title: {title} | text: {body}'
    raise ValueError(f'Unsupported EMBED_PROMPT_MODE: {prompt_mode}')


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


def build_texts(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    text_prep_mode: str,
    prefix_mode: str,
    prompt_mode: str,
) -> list[str]:
    texts = []
    prefix = resolve_text_prefix(prefix_mode)
    for row in df.to_dict(orient='records'):
        if text_prep_mode == 'single_blob':
            prepared_text = prepare_single_blob(row['title'], row['content'])
        elif text_prep_mode == 'title_head':
            prepared_text = prepare_title_head(
                row['title'],
                row['content'],
                max_length=max_length,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f'Unsupported text prep mode for embedding harness: {text_prep_mode}')
        prompted_text = apply_prompt_template(
            row,
            prepared_text,
            tokenizer=tokenizer,
            max_length=max_length,
            text_prep_mode=text_prep_mode,
            prompt_mode=prompt_mode,
        )
        texts.append(prefix + prompted_text)
    return texts


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def pool_hidden_states(outputs, attention_mask: torch.Tensor, pooling_mode: str) -> torch.Tensor:
    if pooling_mode == 'mean':
        return mean_pool(outputs.last_hidden_state, attention_mask)
    if pooling_mode == 'cls':
        return outputs.last_hidden_state[:, 0]
    if pooling_mode == 'pooler':
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        raise RuntimeError('pooler_output requested but missing from model output')
    raise ValueError(f'Unsupported EMBED_POOLING: {pooling_mode}')


def l2_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms


def encode_dense_texts(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    pooling_mode: str,
    apply_layer_norm: bool,
    truncate_dim: int,
) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            if not hasattr(outputs, 'last_hidden_state'):
                raise RuntimeError('Embedding harness expects last_hidden_state in model output')
            pooled = pool_hidden_states(outputs, inputs['attention_mask'], pooling_mode)
            if apply_layer_norm:
                pooled = torch.nn.functional.layer_norm(
                    pooled,
                    normalized_shape=(pooled.shape[1],),
                )
            if truncate_dim > 0 and truncate_dim < pooled.shape[1]:
                pooled = pooled[:, :truncate_dim]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def load_bge_m3_model(model_name: str):
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError(
            'BGE-M3 sparse or hybrid features require FlagEmbedding. Install it with `uv add FlagEmbedding`.'
        ) from exc

    return BGEM3FlagModel(model_name, use_fp16=torch.cuda.is_available())


def lexical_weights_to_feature_dicts(lexical_weights) -> list[dict[str, float]]:
    return [{str(key): float(value) for key, value in weights.items()} for weights in lexical_weights]


def encode_bge_m3_features(
    model_name: str,
    texts: list[str],
    batch_size: int,
    max_length: int,
    feature_mode: str,
    sparse_hash_dim: int,
):
    bge_model = load_bge_m3_model(model_name)
    output = bge_model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=feature_mode != 'sparse',
        return_sparse=feature_mode != 'dense',
        return_colbert_vecs=False,
    )

    if feature_mode == 'dense':
        return l2_normalize_rows(np.asarray(output['dense_vecs']))

    hasher = FeatureHasher(
        n_features=sparse_hash_dim,
        input_type='dict',
        alternate_sign=False,
    )
    sparse_features = hasher.transform(
        lexical_weights_to_feature_dicts(output['lexical_weights'])
    ).tocsr()

    if feature_mode == 'sparse':
        return sparse_features

    dense_features = sp.csr_matrix(l2_normalize_rows(np.asarray(output['dense_vecs'])))
    return sp.hstack([dense_features, sparse_features], format='csr')


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
    train_balance_mode = os.environ['TRAIN_BALANCE_MODE']
    embed_pooling = os.environ['EMBED_POOLING']
    embed_prefix_mode = os.environ['EMBED_PREFIX_MODE']
    embed_prompt_mode = os.environ['EMBED_PROMPT_MODE']
    embed_feature_mode = os.environ['EMBED_FEATURE_MODE']
    embed_sparse_hash_dim = int(os.environ['EMBED_SPARSE_HASH_DIM'])
    embed_layer_norm = os.environ['EMBED_LAYER_NORM'] == '1'
    embed_truncate_dim = int(os.environ['EMBED_TRUNCATE_DIM'])

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_df, eval_df, metadata = load_snapshot(snapshot_dir)
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
    logger.info(
        f'Running embedding experiment: model={model_name}, prompt={embed_prompt_mode}, feature_mode={embed_feature_mode}'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    train_texts = build_texts(
        balanced_train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        text_prep_mode=text_prep_mode,
        prefix_mode=embed_prefix_mode,
        prompt_mode=embed_prompt_mode,
    )
    eval_texts = build_texts(
        eval_df,
        tokenizer=tokenizer,
        max_length=max_length,
        text_prep_mode=text_prep_mode,
        prefix_mode=embed_prefix_mode,
        prompt_mode=embed_prompt_mode,
    )

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
    if embed_feature_mode == 'dense':
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(device)
        train_features = encode_dense_texts(
            model,
            tokenizer,
            train_texts,
            max_length,
            batch_size,
            device,
            embed_pooling,
            embed_layer_norm,
            embed_truncate_dim,
        )
        eval_features = encode_dense_texts(
            model,
            tokenizer,
            eval_texts,
            max_length,
            batch_size,
            device,
            embed_pooling,
            embed_layer_norm,
            embed_truncate_dim,
        )
        solver = 'lbfgs'
    elif embed_feature_mode in {'sparse', 'hybrid'}:
        if model_name != 'BAAI/bge-m3':
            raise RuntimeError('Sparse and hybrid feature modes are currently only supported for BAAI/bge-m3')
        train_features = encode_bge_m3_features(
            model_name,
            train_texts,
            batch_size,
            max_length,
            embed_feature_mode,
            embed_sparse_hash_dim,
        )
        eval_features = encode_bge_m3_features(
            model_name,
            eval_texts,
            batch_size,
            max_length,
            embed_feature_mode,
            embed_sparse_hash_dim,
        )
        solver = 'liblinear'
    else:
        raise ValueError(f'Unsupported EMBED_FEATURE_MODE: {embed_feature_mode}')

    clf = LogisticRegression(max_iter=4000, C=linear_c, random_state=seed, solver=solver)
    clf.fit(train_features, train_labels, sample_weight=sample_weights)
    train_seconds = time.perf_counter() - train_start

    probs = clf.predict_proba(eval_features)[:, 1]
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
        'embed_pooling': embed_pooling,
        'embed_prefix_mode': embed_prefix_mode,
        'embed_prompt_mode': embed_prompt_mode,
        'embed_feature_mode': embed_feature_mode,
        'embed_sparse_hash_dim': embed_sparse_hash_dim,
        'embed_layer_norm': embed_layer_norm,
        'embed_truncate_dim': embed_truncate_dim,
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
    --train-balance-mode "${TRAIN_BALANCE_MODE}" \
    --epochs "${EPOCHS}"
fi
