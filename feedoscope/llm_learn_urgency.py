import asyncio
import logging
import os
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
from feedoscope import config, relevance_embedding, urgency_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article
from feedoscope.llm_infer import find_latest_model

logger = logging.getLogger(__name__)

VALIDATION_SIZE = config.VALIDATION_SIZE


def compute_metrics(
    true_labels: np.ndarray, predicted_probs: np.ndarray
) -> dict[str, float]:
    """Compute the urgency validation metrics from predicted probabilities."""
    pred_labels = (predicted_probs >= 0.5).astype(int)
    return {
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


def split_training_and_validation(
    labeled_data: list[tuple[Article, int]],
) -> tuple[list[Article], np.ndarray, list[Article], np.ndarray]:
    """Split read-tagged urgency data into train and validation subsets."""
    articles = [article for article, _ in labeled_data]
    labels = np.asarray([label for _, label in labeled_data], dtype=int)

    if VALIDATION_SIZE == 0:
        return articles, labels, [], np.array([], dtype=int)

    if VALIDATION_SIZE < 2:
        raise RuntimeError("VALIDATION_SIZE must be at least 2 for urgency validation.")

    if VALIDATION_SIZE >= len(labeled_data):
        raise RuntimeError(
            "VALIDATION_SIZE must be smaller than the number of labeled urgency articles."
        )

    label_counts = np.bincount(labels, minlength=2)
    if int(label_counts.min()) < 2:
        raise RuntimeError(
            "Need at least two read-tagged urgency articles in each class to "
            "build a stratified validation split."
        )

    train_articles, validation_articles, train_labels, validation_labels = (
        train_test_split(
            articles,
            labels,
            test_size=VALIDATION_SIZE,
            random_state=42,
            stratify=labels,
        )
    )
    return (
        list(train_articles),
        np.asarray(train_labels, dtype=int),
        list(validation_articles),
        np.asarray(validation_labels, dtype=int),
    )


async def train_model(
    model_path: str,
    articles: list[Article],
    labels: np.ndarray,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    PreTrainedTokenizerBase,
    LogisticRegression,
]:
    """Train the embedding-linear urgency backend and save its artifact.

    Args:
        model_path: Output directory for the trained model.
        articles: List of articles to train on.
        labels: List of binary labels (0=evergreen, 1=urgent).
        device: Target device for the shared embedding encoder.

    Returns:
        Loaded encoder, tokenizer, and trained classifier.

    """
    tokenizer, encoder = relevance_embedding.load_encoder(
        device,
        pipeline_label="urgency",
    )
    train_counts = {
        "urgent": int(labels.sum()),
        "evergreen": int(len(labels) - labels.sum()),
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    embeddings = await relevance_embedding.encode_articles(
        articles,
        tokenizer,
        encoder,
        device,
        pipeline_label="urgency",
    )
    classifier = urgency_embedding.fit_classifier(embeddings, labels)
    urgency_embedding.save_artifact(model_path, classifier, train_counts=train_counts)
    return encoder, tokenizer, classifier


async def main() -> None:
    """Train the urgency embedding backend on read-tagged urgency labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type != "cuda" and not config.ALLOW_TRAINING_WO_GPU:
        mes = "GPU not available. Exiting"
        logger.critical(mes)
        raise RuntimeError(mes)

    start_time = time.time()

    await dr.global_pool.open(wait=True)
    try:
        labeled_data = await dr.get_read_articles_with_urgency_tags()

        if not labeled_data:
            logger.error(
                "No read articles with urgency user tags found. "
                "Tag read articles with 0-urgency or 1-urgency first."
            )
            return

        all_labels = np.asarray([label for _, label in labeled_data], dtype=int)
        logger.info(
            f"Fetched {len(labeled_data)} read-tagged urgency articles: "
            f"{int(all_labels.sum())} urgent, "
            f"{int(len(all_labels) - all_labels.sum())} evergreen."
        )

        train_articles, train_labels, validation_articles, validation_labels = (
            split_training_and_validation(labeled_data)
        )
        logger.info(
            f"Training set: {len(train_articles)} articles "
            f"({int(train_labels.sum())} urgent, "
            f"{int(len(train_labels) - train_labels.sum())} evergreen)."
        )
        if validation_articles:
            logger.info(
                f"Validation set: {len(validation_articles)} articles "
                f"({int(validation_labels.sum())} urgent, "
                f"{int(len(validation_labels) - validation_labels.sum())} evergreen)."
            )

        model_path = urgency_embedding.build_model_path(
            urgent_count=int(train_labels.sum()),
            evergreen_count=int(len(train_labels) - train_labels.sum()),
        )

        if os.path.exists(model_path):
            logger.info(f"Loading embedding artifact from {model_path}")
            tokenizer, encoder = relevance_embedding.load_encoder(
                device,
                pipeline_label="urgency",
            )
            classifier = urgency_embedding.load_classifier(model_path)
        else:
            logger.info("Training new embedding-linear urgency backend...")
            encoder, tokenizer, classifier = await train_model(
                model_path,
                train_articles,
                train_labels,
                device,
            )
            logger.info(f"Embedding artifact saved to {model_path}")

        find_latest_model(
            urgency_embedding.get_model_family_prefix(), clean_old_models=True
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds.")

        if not validation_articles:
            logger.info("No validation size set, skipping validation.")
            return

        logger.info("Starting urgency validation...")
        validation_probs = await urgency_embedding.predict_probabilities(
            validation_articles,
            tokenizer,
            encoder,
            classifier,
            device,
        )
        metrics = compute_metrics(validation_labels, validation_probs)
        logger.info(f"Accuracy: {metrics['accuracy']:.2f}")
        logger.info(f"Precision: {metrics['precision']:.2f}")
        logger.info(f"Recall: {metrics['recall']:.2f}")
        logger.info(f"F1 score: {metrics['f1']:.2f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.2f}")
        logger.info(f"Average Precision: {metrics['average_precision']:.2f}")
        logger.info(f"Log Loss: {metrics['log_loss']:.2f}")
        logger.info(f"Peak VRAM: {urgency_embedding.peak_vram_gb():.2f} GB")
        logger.info("Urgency validation completed")

        elapsed_time = time.time() - start_time
        logger.info(f"Training and validation completed in {elapsed_time:.2f} seconds.")
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
