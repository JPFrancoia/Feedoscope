import asyncio
import datetime
import logging
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from transformers import PreTrainedTokenizerBase

from custom_logging import init_logging
from feedoscope import config, relevance_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

VALIDATION_SIZE = config.VALIDATION_SIZE


def compute_metrics(
    true_labels: np.ndarray, predicted_probs: np.ndarray
) -> dict[str, float]:
    """Compute the relevance validation metrics from predicted probabilities."""
    pred_labels = (predicted_probs >= 0.5).astype(int)
    return {
        "precision": float(precision_score(true_labels, pred_labels)),
        "recall": float(recall_score(true_labels, pred_labels)),
        "f1": float(f1_score(true_labels, pred_labels)),
        "roc_auc": float(roc_auc_score(true_labels, predicted_probs)),
        "average_precision": float(
            average_precision_score(true_labels, predicted_probs)
        ),
        "log_loss": float(log_loss(true_labels, predicted_probs)),
    }


async def predict_combined_probabilities(
    articles: list[Article],
    tokenizer: PreTrainedTokenizerBase,
    encoder: torch.nn.Module,
    relevance_classifier: LogisticRegression,
    super_important_classifier: LogisticRegression,
    device: torch.device,
) -> np.ndarray:
    """Score articles with both heads from one cached embedding lookup."""
    embeddings = await relevance_embedding.encode_articles(
        articles,
        tokenizer,
        encoder,
        device,
    )
    return relevance_embedding.combine_probabilities(
        relevance_embedding.predict_probabilities_from_embeddings(
            embeddings, relevance_classifier
        ),
        relevance_embedding.predict_probabilities_from_embeddings(
            embeddings, super_important_classifier
        ),
    )


def build_model_path(good_articles: list[Article], bad_articles: list[Article]) -> str:
    """Build the on-disk artifact path for a relevance training run."""
    return (
        f"models/{relevance_embedding.get_model_family_prefix()}_"
        f"{datetime.date.today().strftime('%Y_%m_%d')}_"
        f"{len(good_articles)}_good_{len(bad_articles)}_not_good"
    )


async def train_model(
    good_articles: list[Article],
    bad_articles: list[Article],
    model_path: str,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    PreTrainedTokenizerBase,
    LogisticRegression,
    LogisticRegression,
]:
    """Train relevance and super-important heads from one embedding matrix."""
    tokenizer, encoder = relevance_embedding.load_encoder(device)
    training_articles = good_articles + bad_articles
    relevance_labels = np.array([1] * len(good_articles) + [0] * len(bad_articles))
    super_important_labels = np.array(
        [relevance_embedding.is_super_important(article) for article in good_articles],
        dtype=int,
    )
    super_important_count = int(super_important_labels.sum())
    ordinary_read_count = len(good_articles) - super_important_count
    if not super_important_count or not ordinary_read_count:
        raise RuntimeError(
            "Super-important training needs both explicitly preferred and ordinary read articles."
        )
    logger.info(
        f"Training heads with {len(good_articles)} good, {len(bad_articles)} bad, "
        f"{super_important_count} super-important, and {ordinary_read_count} ordinary read articles."
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    embeddings = await relevance_embedding.encode_articles(
        training_articles,
        tokenizer,
        encoder,
        device,
    )
    relevance_classifier = relevance_embedding.fit_classifier(
        embeddings,
        relevance_labels,
        pipeline_label="relevance",
    )
    super_important_classifier = relevance_embedding.fit_classifier(
        embeddings[: len(good_articles)],
        super_important_labels,
        pipeline_label="super-important",
    )
    relevance_embedding.save_two_head_artifact(
        model_path,
        relevance_classifier,
        super_important_classifier,
        train_counts={
            "good": len(good_articles),
            "bad": len(bad_articles),
            "super_important": super_important_count,
            "ordinary_read": ordinary_read_count,
        },
    )
    return encoder, tokenizer, relevance_classifier, super_important_classifier


async def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type != "cuda" and not config.ALLOW_TRAINING_WO_GPU:
        mes = "GPU not available. Exiting"
        logger.critical(mes)
        raise RuntimeError(mes)

    start_time = time.time()

    await dr.global_pool.open(wait=True)

    bad_articles = await dr.get_published_articles(validation_size=VALIDATION_SIZE)
    good_articles = await dr.get_read_articles_training(validation_size=VALIDATION_SIZE)

    logger.info(
        f"Collected {len(good_articles)} good articles "
        f"({sum(1 for article in good_articles if article.vote == 1 or article.starred)} starred/upvoted)."
    )
    logger.info(f"Collected {len(bad_articles)} bad articles.")

    model_path = build_model_path(good_articles, bad_articles)

    if os.path.exists(
        os.path.join(model_path, relevance_embedding.TWO_HEAD_ARTIFACT_FILENAME)
    ):
        logger.info(f"Loading two-head relevance artifact from {model_path}")
        tokenizer, encoder = relevance_embedding.load_encoder(device)
        relevance_classifier, super_important_classifier = (
            relevance_embedding.load_two_head_artifact(model_path)
        )
    else:
        logger.info("Training new two-head relevance backend...")
        (
            encoder,
            tokenizer,
            relevance_classifier,
            super_important_classifier,
        ) = await train_model(
            good_articles,
            bad_articles,
            model_path,
            device,
        )
        logger.info(f"Two-head relevance artifact saved to {model_path}")

    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds.")

    if VALIDATION_SIZE == 0:
        logger.info("No validation size set, skipping validation.")
        await dr.global_pool.close()
        return

    logger.info("Starting relevance validation...")
    good_validation = await dr.get_sample_good(validation_size=VALIDATION_SIZE)
    not_good_validation = await dr.get_sample_not_good(validation_size=VALIDATION_SIZE)
    logger.info(
        f"Loaded {len(good_validation)} good and {len(not_good_validation)} not good validation articles"
    )

    good_probs = await predict_combined_probabilities(
        good_validation,
        tokenizer,
        encoder,
        relevance_classifier,
        super_important_classifier,
        device,
    )
    not_good_probs = await predict_combined_probabilities(
        not_good_validation,
        tokenizer,
        encoder,
        relevance_classifier,
        super_important_classifier,
        device,
    )

    all_probs = np.concatenate([good_probs, not_good_probs])
    true_labels = np.concatenate(
        [np.ones(len(good_probs)), np.zeros(len(not_good_probs))]
    )

    metrics = compute_metrics(true_labels, all_probs)
    logger.info(f"Precision: {metrics['precision']:.2f}")
    logger.info(f"Recall: {metrics['recall']:.2f}")
    logger.info(f"F1 score: {metrics['f1']:.2f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.2f}")
    logger.info(f"Average Precision: {metrics['average_precision']:.2f}")
    logger.info(f"Log Loss: {metrics['log_loss']:.2f}")
    logger.info(f"Peak VRAM: {relevance_embedding.peak_vram_gb():.2f} GB")
    logger.info("Relevance validation completed")

    await dr.global_pool.close()

    elapsed_time = time.time() - start_time
    logger.info(f"Training and validation completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
