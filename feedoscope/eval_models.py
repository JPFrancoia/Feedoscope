"""Weekly evaluation of the Relevance model."""

import asyncio
from collections.abc import Mapping
import datetime
import json
import logging
import os
import random
import shutil
import time
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from transformers import PreTrainedTokenizerBase

from custom_logging import init_logging
from feedoscope import config, llm_learn, relevance_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

EVAL_RELEVANCE_PREFIX = "eval_relevance"
EVAL_HISTORY_PATH = "models/eval_history.json"
RELEVANCE_PRECISION_BUDGET = 50
EVALUATION_MODEL = "EmbeddingGemma 300M prompted + MLP (AP + Precision@50)"


def _clean_stale_eval_dirs() -> None:
    """Remove the temporary evaluation directory from an interrupted run."""
    path = f"models/{EVAL_RELEVANCE_PREFIX}"
    if os.path.exists(path):
        logger.warning(f"Found stale eval directory {path}. Removing.")
        shutil.rmtree(path, ignore_errors=True)


def compute_relevance_metrics(
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
) -> dict[str, float | None]:
    """Compute ranking metrics for the Relevance holdout."""
    true_labels = np.asarray(true_labels, dtype=int)
    predicted_probs = np.asarray(predicted_probs, dtype=float)
    candidate_count = len(true_labels)
    if not candidate_count:
        return {
            "roc_auc": None,
            "average_precision": None,
            "precision_at_50": None,
        }

    positive_count = int(true_labels.sum())
    effective_budget = min(RELEVANCE_PRECISION_BUDGET, candidate_count)
    cutoff = np.partition(predicted_probs, -effective_budget)[-effective_budget]
    above_cutoff = predicted_probs > cutoff
    at_cutoff = predicted_probs == cutoff
    available_slots = effective_budget - int(above_cutoff.sum())
    credited_positives = true_labels[above_cutoff].sum() + (
        true_labels[at_cutoff].sum() * available_slots / at_cutoff.sum()
    )

    return {
        "roc_auc": (
            float(roc_auc_score(true_labels, predicted_probs))
            if np.unique(true_labels).size == 2
            else None
        ),
        "average_precision": (
            float(average_precision_score(true_labels, predicted_probs))
            if positive_count
            else None
        ),
        "precision_at_50": float(credited_positives / effective_budget),
    }


async def save_eval_results(
    training_counts: dict[str, int],
    eval_counts: dict[str, int],
    metrics: Mapping[str, float | None],
) -> None:
    """Persist one Relevance evaluation record to JSON and PostgreSQL."""
    eval_date = datetime.date.today()
    record = {
        "date": eval_date.isoformat(),
        "model": "Relevance",
        "evaluation_model": EVALUATION_MODEL,
        "training": training_counts,
        "eval": eval_counts,
        "metrics": metrics,
    }

    history: list[dict[str, Any]] = []
    if os.path.exists(EVAL_HISTORY_PATH):
        try:
            with open(EVAL_HISTORY_PATH) as file:
                history = json.load(file)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                f"Could not parse {EVAL_HISTORY_PATH}. Starting fresh history."
            )

    history.append(record)
    with open(EVAL_HISTORY_PATH, "w") as file:
        json.dump(history, file, indent=2)

    await dr.insert_model_eval(
        eval_date=eval_date,
        model_name="Relevance",
        evaluation_model=EVALUATION_MODEL,
        training_counts=training_counts,
        eval_counts=eval_counts,
        metrics=metrics,
    )
    logger.info(f"Saved Relevance evaluation results to {EVAL_HISTORY_PATH}.")


async def _run_relevance_inference(
    encoder: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    classifier: Any,
    articles: list[Article],
    device: torch.device,
) -> np.ndarray:
    """Run held-out Relevance inference."""
    return await relevance_embedding.predict_probabilities(
        articles,
        tokenizer,
        encoder,
        classifier,
        device,
    )


async def eval_relevance(device: torch.device) -> None:
    """Evaluate the Relevance model on a random balanced holdout."""
    validation_size = config.VALIDATION_SIZE
    logger.info(
        f"[Relevance] Starting evaluation with VALIDATION_SIZE={validation_size}"
    )
    start_time = time.time()
    all_good = await dr.get_read_articles_training(validation_size=0)
    all_bad = await dr.get_published_articles(validation_size=0)

    if len(all_good) < validation_size or len(all_bad) < validation_size:
        logger.warning(
            f"[Relevance] Not enough articles to hold out {validation_size} "
            f"per class (good={len(all_good)}, bad={len(all_bad)}). Skipping eval."
        )
        return

    eval_good = random.sample(all_good, validation_size)
    eval_bad = random.sample(all_bad, validation_size)
    eval_good_ids = {article.article_id for article in eval_good}
    eval_bad_ids = {article.article_id for article in eval_bad}
    good_articles = [
        article for article in all_good if article.article_id not in eval_good_ids
    ]
    bad_articles = [
        article for article in all_bad if article.article_id not in eval_bad_ids
    ]
    model_path = f"models/{EVAL_RELEVANCE_PREFIX}"

    try:
        encoder, tokenizer, classifier = await llm_learn.train_model(
            good_articles,
            bad_articles,
            model_path,
            device,
        )
        good_probs = await _run_relevance_inference(
            encoder, tokenizer, classifier, eval_good, device
        )
        bad_probs = await _run_relevance_inference(
            encoder, tokenizer, classifier, eval_bad, device
        )
        metrics = compute_relevance_metrics(
            np.concatenate([np.ones(len(good_probs)), np.zeros(len(bad_probs))]),
            np.concatenate([good_probs, bad_probs]),
        )
        await save_eval_results(
            {"good": len(good_articles), "bad": len(bad_articles)},
            {"good": len(eval_good), "bad": len(eval_bad)},
            metrics,
        )
    finally:
        shutil.rmtree(model_path, ignore_errors=True)

    logger.info(
        f"[Relevance] Evaluation completed in {time.time() - start_time:.2f} seconds."
    )


async def main() -> None:
    """Run the Relevance evaluation."""
    if config.VALIDATION_SIZE == 0:
        logger.info("VALIDATION_SIZE is 0. Skipping evaluation.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not config.ALLOW_TRAINING_WO_GPU:
        raise RuntimeError("GPU not available. Exiting")

    random.seed(42)
    _clean_stale_eval_dirs()
    await dr.global_pool.open(wait=True)
    try:
        await eval_relevance(device)
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
