"""Weekly evaluation of relevance and urgency models.

For each model, trains on a subset of the data with a holdout set,
runs inference on the holdout, logs classification metrics, then discards
the eval model. Production models (trained on 100% of data) are NOT affected.

The holdout size is controlled by the VALIDATION_SIZE env var (via config.py).
If VALIDATION_SIZE is 0, the eval is skipped entirely.
"""

import asyncio
import datetime
import json
import logging
import os
import random
import shutil
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import torch
from transformers import PreTrainedTokenizerBase

from custom_logging import init_logging
from feedoscope import (
    config,
    llm_learn,
    llm_learn_urgency,
    relevance_embedding,
    urgency_embedding,
)
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

# Eval model paths use these prefixes. They do NOT match any production model
# prefix (google-embeddinggemma-300m_* or urgency_google-embeddinggemma-300m_*),
# so find_latest_model() will never find or delete them.
EVAL_RELEVANCE_PREFIX = "eval_relevance"
EVAL_URGENCY_PREFIX = "eval_urgency"

MAX_LENGTH = 512
INFERENCE_BATCH_SIZE = 128
EVAL_HISTORY_PATH = "models/eval_history.json"


def _clean_stale_eval_dirs() -> None:
    """Remove leftover eval model directories from a previous interrupted run.

    If the eval script was killed (SIGKILL, OOM, etc.) before the ``finally``
    block could run, the temporary model directories may still exist on disk.
    Calling this at the start of each run ensures a clean slate.
    """
    for prefix in (EVAL_RELEVANCE_PREFIX, EVAL_URGENCY_PREFIX):
        path = f"models/{prefix}"
        if os.path.exists(path):
            logger.warning(
                f"Found stale eval directory {path} from a previous run. Removing."
            )
            shutil.rmtree(path, ignore_errors=True)


def compute_and_log_metrics(
    model_name: str,
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics and log them.

    Args:
        model_name: Name of the model being evaluated (for log prefixing).
        true_labels: Ground truth binary labels.
        predicted_probs: Predicted probabilities for the positive class.

    Returns:
        Dictionary of metric names to their float values.

    """
    pred_labels = (predicted_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "precision": float(precision_score(true_labels, pred_labels, zero_division=0)),
        "recall": float(recall_score(true_labels, pred_labels, zero_division=0)),
        "f1": float(f1_score(true_labels, pred_labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(true_labels, predicted_probs)),
        "average_precision": float(
            average_precision_score(true_labels, predicted_probs)
        ),
        "log_loss": float(log_loss(true_labels, predicted_probs)),
    }

    logger.info(f"[{model_name}] Evaluation results:")
    logger.info(f"[{model_name}]   Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"[{model_name}]   Precision:         {metrics['precision']:.4f}")
    logger.info(f"[{model_name}]   Recall:            {metrics['recall']:.4f}")
    logger.info(f"[{model_name}]   F1:                {metrics['f1']:.4f}")
    logger.info(f"[{model_name}]   ROC AUC:           {metrics['roc_auc']:.4f}")
    logger.info(
        f"[{model_name}]   Average Precision: {metrics['average_precision']:.4f}"
    )
    logger.info(f"[{model_name}]   Log Loss:          {metrics['log_loss']:.4f}")

    return metrics


async def save_eval_results(
    model_name: str,
    training_counts: dict[str, int],
    eval_counts: dict[str, int],
    metrics: dict[str, float],
) -> None:
    """Persist an evaluation record to JSON history and PostgreSQL.

    Creates the file if it does not exist. If the file exists but is
    corrupted, it is overwritten with a fresh list containing only the
    new record. The PostgreSQL insert is intentionally allowed to fail the
    eval job, because Miniflux's ``model_evals`` table is the durable history.

    Args:
        model_name: Name of the model evaluated (e.g. "Relevance", "Urgency").
        training_counts: Article counts used for training, keyed by class.
        eval_counts: Article counts used for evaluation, keyed by class.
        metrics: Metric name to value mapping from ``compute_and_log_metrics``.

    """
    eval_date = datetime.date.today()
    record = {
        "date": eval_date.isoformat(),
        "model": model_name,
        "training": training_counts,
        "eval": eval_counts,
        "metrics": metrics,
    }

    history: list[dict] = []  # type: ignore[type-arg]
    if os.path.exists(EVAL_HISTORY_PATH):
        try:
            with open(EVAL_HISTORY_PATH) as f:
                history = json.load(f)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                f"Could not parse {EVAL_HISTORY_PATH}. Starting fresh history."
            )
            history = []

    history.append(record)

    with open(EVAL_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"[{model_name}] Evaluation results saved to {EVAL_HISTORY_PATH}.")

    await dr.insert_model_eval(
        eval_date=eval_date,
        model_name=model_name,
        training_counts=training_counts,
        eval_counts=eval_counts,
        metrics=metrics,
    )


async def _run_relevance_inference(
    encoder: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    classifier: LogisticRegression,
    articles: list[Article],
    device: torch.device,
) -> np.ndarray:
    """Run held-out relevance inference with the embedding-linear backend."""
    return await relevance_embedding.predict_probabilities(
        articles,
        tokenizer,
        encoder,
        classifier,
        device,
    )


async def _run_urgency_inference(
    encoder: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    classifier: LogisticRegression,
    articles: list[Article],
    device: torch.device,
) -> np.ndarray:
    """Run held-out urgency inference with the embedding-linear backend."""
    return await urgency_embedding.predict_probabilities(
        articles,
        tokenizer,
        encoder,
        classifier,
        device,
    )


async def eval_relevance(device: torch.device) -> None:
    """Evaluate the relevance model accuracy.

    Fetches ALL good/bad articles (no SQL-level holdout), then randomly
    samples VALIDATION_SIZE from each class for the eval set. Trains on
    the remaining articles with the same full-row training data as production,
    then evaluates on the held-out random sample.

    The eval model is discarded after metrics are computed.
    """
    validation_size = config.VALIDATION_SIZE
    logger.info(
        f"[Relevance] Starting evaluation with VALIDATION_SIZE={validation_size}"
    )

    start_time = time.time()

    # Fetch ALL articles (validation_size=0 means no SQL-level holdout).
    all_good = await dr.get_read_articles_training(validation_size=0)
    all_bad = await dr.get_published_articles(validation_size=0)

    logger.info(
        f"[Relevance] Fetched {len(all_good)} good, {len(all_bad)} bad articles total."
    )

    if len(all_good) < validation_size or len(all_bad) < validation_size:
        logger.warning(
            f"[Relevance] Not enough articles to hold out {validation_size} "
            f"per class (good={len(all_good)}, bad={len(all_bad)}). "
            "Skipping eval."
        )
        return

    # Randomly sample VALIDATION_SIZE articles from each class for eval.
    eval_good = random.sample(all_good, validation_size)
    eval_bad = random.sample(all_bad, validation_size)

    eval_good_ids = {a.article_id for a in eval_good}
    eval_bad_ids = {a.article_id for a in eval_bad}

    # Training set: everything NOT in the eval set.
    good_articles = [a for a in all_good if a.article_id not in eval_good_ids]
    bad_articles = [a for a in all_bad if a.article_id not in eval_bad_ids]

    logger.info(
        f"[Relevance] Eval set: {len(eval_good)} good, {len(eval_bad)} bad "
        f"(randomly sampled)."
    )

    logger.info(
        f"[Relevance] Training set: {len(good_articles)} good, {len(bad_articles)} bad."
    )

    # Train on a temp path.
    model_path = f"models/{EVAL_RELEVANCE_PREFIX}"

    try:
        encoder, tokenizer, classifier = await llm_learn.train_model(
            good_articles,
            bad_articles,
            model_path,
            device,
        )
        logger.info("[Relevance] Eval model trained successfully.")

        # Run inference on held-out set.
        good_probs = await _run_relevance_inference(
            encoder,
            tokenizer,
            classifier,
            eval_good,
            device,
        )
        bad_probs = await _run_relevance_inference(
            encoder,
            tokenizer,
            classifier,
            eval_bad,
            device,
        )

        all_probs = np.concatenate([good_probs, bad_probs])
        true_labels = np.concatenate(
            [np.ones(len(good_probs)), np.zeros(len(bad_probs))]
        )

        metrics = compute_and_log_metrics("Relevance", true_labels, all_probs)

        await save_eval_results(
            model_name="Relevance",
            training_counts={"good": len(good_articles), "bad": len(bad_articles)},
            eval_counts={"good": len(eval_good), "bad": len(eval_bad)},
            metrics=metrics,
        )

    finally:
        # Always clean up the eval model.
        shutil.rmtree(model_path, ignore_errors=True)
        logger.info(f"[Relevance] Cleaned up eval model at {model_path}.")

    elapsed_time = time.time() - start_time
    logger.info(f"[Relevance] Evaluation completed in {elapsed_time:.2f} seconds.")


async def eval_urgency(device: torch.device) -> None:
    """Evaluate the urgency model accuracy.

    Trains on read-tagged urgency articles only and holds out a stratified
    subset of those trusted labels for evaluation.

    The eval model is discarded after metrics are computed.
    """
    validation_size = config.VALIDATION_SIZE
    logger.info(f"[Urgency] Starting evaluation with VALIDATION_SIZE={validation_size}")

    start_time = time.time()

    labeled_data = await dr.get_read_articles_with_urgency_tags()

    if not labeled_data:
        logger.warning("[Urgency] No tagged articles found. Skipping eval.")
        return

    logger.info(f"[Urgency] Total read-tagged articles: {len(labeled_data)}.")

    if len(labeled_data) <= validation_size:
        logger.warning(
            f"[Urgency] Not enough read-tagged articles ({len(labeled_data)}) "
            f"to hold out {validation_size}. Skipping eval."
        )
        return

    articles = [article for article, _ in labeled_data]
    labels = np.asarray([label for _, label in labeled_data], dtype=int)
    label_counts = np.bincount(labels, minlength=2)
    if validation_size < 2 or int(label_counts.min()) < 2:
        logger.warning(
            "[Urgency] Not enough examples per class to build a stratified eval split. "
            "Skipping eval."
        )
        return

    train_articles, eval_articles, train_labels, eval_labels = train_test_split(
        articles,
        labels,
        test_size=validation_size,
        random_state=42,
        stratify=labels,
    )

    train_articles = list(train_articles)
    eval_articles = list(eval_articles)
    train_labels = np.asarray(train_labels, dtype=int)
    eval_labels = np.asarray(eval_labels, dtype=int)

    eval_urgent = int(eval_labels.sum())
    eval_evergreen = int(len(eval_labels) - eval_urgent)
    logger.info(
        f"[Urgency] Eval set: {len(eval_articles)} read-tagged articles "
        f"({eval_urgent} urgent, {eval_evergreen} evergreen)."
    )
    logger.info(
        f"[Urgency] Training set: {len(train_articles)} read-tagged articles "
        f"({int(train_labels.sum())} urgent, "
        f"{int(len(train_labels) - train_labels.sum())} evergreen)."
    )

    # Train on a temp path.
    model_path = f"models/{EVAL_URGENCY_PREFIX}"

    try:
        encoder, tokenizer, classifier = await llm_learn_urgency.train_model(
            model_path,
            train_articles,
            train_labels,
            device,
        )
        logger.info("[Urgency] Eval model trained successfully.")

        # Run inference on held-out read articles.
        eval_probs = await _run_urgency_inference(
            encoder,
            tokenizer,
            classifier,
            eval_articles,
            device,
        )

        metrics = compute_and_log_metrics("Urgency", eval_labels, eval_probs)

        await save_eval_results(
            model_name="Urgency",
            training_counts={
                "urgent": int(train_labels.sum()),
                "evergreen": int(len(train_labels) - train_labels.sum()),
            },
            eval_counts={"urgent": eval_urgent, "evergreen": eval_evergreen},
            metrics=metrics,
        )

    finally:
        # Always clean up the eval model.
        shutil.rmtree(model_path, ignore_errors=True)
        logger.info(f"[Urgency] Cleaned up eval model at {model_path}.")

    elapsed_time = time.time() - start_time
    logger.info(f"[Urgency] Evaluation completed in {elapsed_time:.2f} seconds.")


async def main() -> None:
    """Run evaluation for both models sequentially."""
    validation_size = config.VALIDATION_SIZE

    if validation_size == 0:
        logger.info("VALIDATION_SIZE is 0. Skipping evaluation.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type != "cuda" and not config.ALLOW_TRAINING_WO_GPU:
        mes = "GPU not available. Exiting"
        logger.critical(mes)
        raise RuntimeError(mes)

    # Seed for reproducible random sampling within a single run.
    random.seed(42)

    # Remove any leftover eval model directories from a previous crashed run.
    _clean_stale_eval_dirs()

    await dr.global_pool.open(wait=True)

    try:
        await eval_relevance(device)
        await eval_urgency(device)
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
