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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from custom_logging import init_logging
from feedoscope import config, llm_learn, llm_learn_urgency, utils
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

# Eval model paths use these prefixes. They do NOT match any production model
# prefix (jhu-clsp-ettin-encoder-150m_* or urgency-ModernBERT-base_*), so
# find_latest_model() will never find or delete them.
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
        "precision": float(precision_score(true_labels, pred_labels)),
        "recall": float(recall_score(true_labels, pred_labels)),
        "f1": float(f1_score(true_labels, pred_labels)),
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


def save_eval_results(
    model_name: str,
    training_counts: dict[str, int],
    eval_counts: dict[str, int],
    metrics: dict[str, float],
) -> None:
    """Append an evaluation record to the JSON history file on disk.

    Creates the file if it does not exist. If the file exists but is
    corrupted, it is overwritten with a fresh list containing only the
    new record.

    Args:
        model_name: Name of the model evaluated (e.g. "Relevance", "Urgency").
        training_counts: Article counts used for training, keyed by class.
        eval_counts: Article counts used for evaluation, keyed by class.
        metrics: Metric name to value mapping from ``compute_and_log_metrics``.

    """
    record = {
        "date": datetime.date.today().isoformat(),
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


def _run_inference(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    articles: list[Article],
    device: torch.device,
) -> np.ndarray:
    """Run inference on a list of articles and return positive-class probabilities.

    Args:
        model: The trained classification model.
        tokenizer: Tokenizer matching the model.
        articles: Articles to run inference on.
        device: Device to run inference on (CPU or CUDA).

    Returns:
        Array of probabilities for the positive class (shape: [n_articles]).

    """
    texts = utils.prepare_articles_text(articles)

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    all_probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), INFERENCE_BATCH_SIZE):
            end = start + INFERENCE_BATCH_SIZE
            batch_inputs = {k: v[start:end].to(device) for k, v in inputs.items()}
            preds = model(**batch_inputs).logits
            probs = torch.sigmoid(preds[:, 1]).cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs)


async def eval_relevance(device: torch.device) -> None:
    """Evaluate the relevance model accuracy.

    Fetches ALL good/bad articles (no SQL-level holdout), then randomly
    samples VALIDATION_SIZE from each class for the eval set. Trains on
    the remaining articles with the same class-balancing as production,
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

    # Class-balance: same logic as llm_learn.main() lines 237-247.
    # Always keep starred/upvoted ("excellent") articles.
    min_count = min(len(good_articles), len(bad_articles))

    excellent = [a for a in good_articles if a.vote == 1 or a.starred]
    regular = [a for a in good_articles if not (a.vote == 1 or a.starred)]

    remaining_slots = max(0, min_count - len(excellent))
    regular = regular[-remaining_slots:] if remaining_slots > 0 else []
    good_articles = sorted(regular + excellent, key=lambda a: a.article_id)

    bad_articles = bad_articles[-min_count:]

    n_protected = len(excellent)
    logger.info(
        f"[Relevance] Training set: {len(good_articles)} good "
        f"({n_protected} starred/upvoted), {len(bad_articles)} bad."
    )

    # Train on a temp path.
    model_path = f"models/{EVAL_RELEVANCE_PREFIX}"

    tokenizer = AutoTokenizer.from_pretrained(llm_learn.MODEL_NAME)

    try:
        trainer = await llm_learn.train_model(
            model_path, tokenizer, good_articles, bad_articles
        )
        assert trainer.model is not None, "Trainer model is None after training"
        model = trainer.model
        logger.info("[Relevance] Eval model trained successfully.")

        # Run inference on held-out set.
        model.to(device)  # type: ignore[operator]
        good_probs = _run_inference(model, tokenizer, eval_good, device)
        bad_probs = _run_inference(model, tokenizer, eval_bad, device)

        all_probs = np.concatenate([good_probs, bad_probs])
        true_labels = np.concatenate(
            [np.ones(len(good_probs)), np.zeros(len(bad_probs))]
        )

        metrics = compute_and_log_metrics("Relevance", true_labels, all_probs)

        save_eval_results(
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

    Trains on ALL tagged articles (read + unread) with the same smart
    class-balancing as production, but holds out some read tagged articles
    for evaluation. Only read articles have verified labels (the user has
    confirmed or corrected the urgency tag), so they form the eval set.

    The eval model is discarded after metrics are computed.
    """
    validation_size = config.VALIDATION_SIZE
    logger.info(f"[Urgency] Starting evaluation with VALIDATION_SIZE={validation_size}")

    start_time = time.time()

    # Fetch all tagged articles (read + unread).
    labeled_data = await dr.get_articles_with_simplified_time_sensitivity()

    if not labeled_data:
        logger.warning("[Urgency] No tagged articles found. Skipping eval.")
        return

    # Separate read articles (verified labels) from the rest.
    read_data = [(a, l) for a, l in labeled_data if a.status == "read"]
    non_read_data = [(a, l) for a, l in labeled_data if a.status != "read"]

    logger.info(
        f"[Urgency] Total tagged: {len(labeled_data)} "
        f"({len(read_data)} read, {len(non_read_data)} unread)."
    )

    if len(read_data) < validation_size * 2:
        logger.warning(
            f"[Urgency] Not enough read articles ({len(read_data)}) "
            f"to hold out {validation_size}. Skipping eval."
        )
        return

    # Randomly sample VALIDATION_SIZE read articles for evaluation.
    eval_data = random.sample(read_data, validation_size)
    eval_ids = {a.article_id for a, _ in eval_data}
    remaining_read_data = [(a, l) for a, l in read_data if a.article_id not in eval_ids]

    eval_articles = [a for a, _ in eval_data]
    eval_labels = [l for _, l in eval_data]

    eval_urgent = sum(eval_labels)
    eval_evergreen = len(eval_labels) - eval_urgent
    logger.info(
        f"[Urgency] Eval set: {len(eval_data)} read articles "
        f"({eval_urgent} urgent, {eval_evergreen} evergreen)."
    )

    # Training data: remaining read + all unread.
    train_data = remaining_read_data + non_read_data

    # Apply the same smart class-balancing as llm_learn_urgency.py lines 218-276.
    train_articles = [a for a, _ in train_data]
    train_labels = [l for _, l in train_data]

    read_urgent = [
        (a, l)
        for a, l in zip(train_articles, train_labels)
        if l == 1 and a.status == "read"
    ]
    read_evergreen = [
        (a, l)
        for a, l in zip(train_articles, train_labels)
        if l == 0 and a.status == "read"
    ]
    unread_urgent = [
        (a, l)
        for a, l in zip(train_articles, train_labels)
        if l == 1 and a.status != "read"
    ]
    unread_evergreen = [
        (a, l)
        for a, l in zip(train_articles, train_labels)
        if l == 0 and a.status != "read"
    ]

    urgent_count = len(read_urgent) + len(unread_urgent)
    evergreen_count = len(read_evergreen) + len(unread_evergreen)

    logger.info(
        f"[Urgency] Training pool: "
        f"Read: {len(read_urgent)} urgent, {len(read_evergreen)} evergreen. "
        f"Unread: {len(unread_urgent)} urgent, {len(unread_evergreen)} evergreen."
    )

    target = min(urgent_count, evergreen_count)

    def _select_class(
        read_items: list[tuple[Article, int]],
        unread_items: list[tuple[Article, int]],
        target_count: int,
    ) -> list[tuple[Article, int]]:
        """Select articles for one class, always keeping all read articles."""
        selected = list(read_items)
        remaining = target_count - len(selected)
        if remaining > 0:
            sampled = random.sample(unread_items, min(remaining, len(unread_items)))
            selected.extend(sampled)
        return selected

    selected_urgent = _select_class(read_urgent, unread_urgent, target)
    selected_evergreen = _select_class(read_evergreen, unread_evergreen, target)

    balanced_data = selected_urgent + selected_evergreen
    balanced_articles = [a for a, _ in balanced_data]
    balanced_labels = [l for _, l in balanced_data]

    actual_urgent = sum(balanced_labels)
    actual_evergreen = len(balanced_labels) - actual_urgent
    logger.info(
        f"[Urgency] Balanced training set: {actual_urgent} urgent, "
        f"{actual_evergreen} evergreen ({len(balanced_labels)} total)."
    )

    # Train on a temp path.
    model_path = f"models/{EVAL_URGENCY_PREFIX}"

    tokenizer = AutoTokenizer.from_pretrained(llm_learn_urgency.MODEL_NAME)

    try:
        trainer = await llm_learn_urgency.train_model(
            model_path, tokenizer, balanced_articles, balanced_labels
        )
        assert trainer.model is not None, "Trainer model is None after training"
        model = trainer.model
        logger.info("[Urgency] Eval model trained successfully.")

        # Run inference on held-out read articles.
        model.to(device)  # type: ignore[operator]
        eval_probs = _run_inference(model, tokenizer, eval_articles, device)
        true_labels = np.array(eval_labels)

        metrics = compute_and_log_metrics("Urgency", true_labels, eval_probs)

        save_eval_results(
            model_name="Urgency",
            training_counts={
                "urgent": actual_urgent,
                "evergreen": actual_evergreen,
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
