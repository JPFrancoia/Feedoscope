"""Weekly evaluation of relevance and urgency models.

For each model, trains on a subset of the data with a holdout set,
runs inference on the holdout, logs classification metrics, then discards
the eval model. Production models (trained on 100% of data) are NOT affected.

The holdout size is controlled by the VALIDATION_SIZE env var (via config.py).
If VALIDATION_SIZE is 0, the eval is skipped entirely.
"""

import asyncio
import logging
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
# prefix (answerdotai-ModernBERT-base_* or urgency-ModernBERT-base_*), so
# find_latest_model() will never find or delete them.
EVAL_RELEVANCE_PREFIX = "eval_relevance"
EVAL_URGENCY_PREFIX = "eval_urgency"

MAX_LENGTH = 512
INFERENCE_BATCH_SIZE = 128


def compute_and_log_metrics(
    model_name: str,
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
) -> None:
    """Compute classification metrics and log them.

    Args:
        model_name: Name of the model being evaluated (for log prefixing).
        true_labels: Ground truth binary labels.
        predicted_probs: Predicted probabilities for the positive class.

    """
    pred_labels = (predicted_probs >= 0.5).astype(int)

    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    roc_auc = roc_auc_score(true_labels, predicted_probs)
    ap = average_precision_score(true_labels, predicted_probs)
    logloss = log_loss(true_labels, predicted_probs)

    logger.info(f"[{model_name}] Evaluation results:")
    logger.info(f"[{model_name}]   Accuracy:          {acc:.4f}")
    logger.info(f"[{model_name}]   Precision:         {prec:.4f}")
    logger.info(f"[{model_name}]   Recall:            {rec:.4f}")
    logger.info(f"[{model_name}]   F1:                {f1:.4f}")
    logger.info(f"[{model_name}]   ROC AUC:           {roc_auc:.4f}")
    logger.info(f"[{model_name}]   Average Precision: {ap:.4f}")
    logger.info(f"[{model_name}]   Log Loss:          {logloss:.4f}")


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

    Uses the existing VALIDATION_SIZE mechanism: the SQL queries for training
    data (get_read_articles_training, get_published_articles) skip the first
    VALIDATION_SIZE articles by ID order. The companion queries
    (get_sample_good, get_sample_not_good) return exactly those held-out
    articles.

    The eval model is trained on the non-held-out data with the same
    class-balancing as production, then evaluated against the held-out set.
    The model is discarded after metrics are computed.
    """
    validation_size = config.VALIDATION_SIZE
    logger.info(
        f"[Relevance] Starting evaluation with VALIDATION_SIZE={validation_size}"
    )

    start_time = time.time()

    # Fetch training data with holdout excluded (same as llm_learn.main).
    bad_articles = await dr.get_published_articles(validation_size=validation_size)
    good_articles = await dr.get_read_articles_training(validation_size=validation_size)

    logger.info(
        f"[Relevance] Fetched {len(good_articles)} good, "
        f"{len(bad_articles)} bad articles (holdout excluded)."
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

        # Fetch held-out articles.
        eval_good = await dr.get_sample_good(validation_size=validation_size)
        eval_bad = await dr.get_sample_not_good(validation_size=validation_size)

        logger.info(
            f"[Relevance] Eval set: {len(eval_good)} good, {len(eval_bad)} bad."
        )

        if not eval_good or not eval_bad:
            logger.warning("[Relevance] Not enough held-out articles. Skipping eval.")
            return

        # Run inference on held-out set.
        model.to(device)  # type: ignore[operator]
        good_probs = _run_inference(model, tokenizer, eval_good, device)
        bad_probs = _run_inference(model, tokenizer, eval_bad, device)

        all_probs = np.concatenate([good_probs, bad_probs])
        true_labels = np.concatenate(
            [np.ones(len(good_probs)), np.zeros(len(bad_probs))]
        )

        compute_and_log_metrics("Relevance", true_labels, all_probs)

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

    # Sort read articles by ID for deterministic holdout.
    read_data.sort(key=lambda x: x[0].article_id)

    # Hold out the first VALIDATION_SIZE read articles for evaluation.
    eval_data = read_data[:validation_size]
    remaining_read_data = read_data[validation_size:]

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

        compute_and_log_metrics("Urgency", true_labels, eval_probs)

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

    await dr.global_pool.open(wait=True)

    try:
        await eval_relevance(device)
        await eval_urgency(device)
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
