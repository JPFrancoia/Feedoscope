import datetime
import json
import logging
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from transformers import PreTrainedTokenizerBase

from feedoscope import config, relevance_embedding
from feedoscope.entities import Article

logger = logging.getLogger(__name__)


def get_model_family_prefix() -> str:
    """Return the artifact family prefix for urgency embedding models."""
    return (
        f"urgency_{config.RELEVANCE_MODEL_NAME.replace('/', '-')}_"
        f"{config.RELEVANCE_MAX_LENGTH}_{config.RELEVANCE_TEXT_PREP_MODE}_"
        f"embedding_linear_c{config.URGENCY_LINEAR_C}"
    )


def get_model_key() -> str:
    """Return the cache key for the active urgency backend configuration."""
    return (
        "urgency-embedding_linear::"
        f"{config.RELEVANCE_MODEL_NAME}::"
        f"{config.RELEVANCE_MAX_LENGTH}::"
        f"{config.RELEVANCE_TEXT_PREP_MODE}::"
        f"{config.RELEVANCE_PREP_VERSION}::"
        f"c={config.URGENCY_LINEAR_C}"
    )


def build_model_path(urgent_count: int, evergreen_count: int) -> str:
    """Build the on-disk artifact path for an urgency training run."""
    return (
        f"models/{get_model_family_prefix()}_"
        f"{datetime.date.today().strftime('%Y_%m_%d')}_"
        f"{urgent_count}_urgent_{evergreen_count}_evergreen"
    )


def fit_classifier(embeddings: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Fit the urgency logistic-regression head on top of frozen embeddings."""
    logger.info(
        f"Fitting urgency logistic regression on {len(labels)} rows with "
        f"C={config.URGENCY_LINEAR_C}"
    )
    classifier = LogisticRegression(
        max_iter=4000,
        C=config.URGENCY_LINEAR_C,
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(embeddings, labels)
    logger.info("Urgency logistic regression fit completed")
    return classifier


def save_artifact(
    model_path: str,
    classifier: LogisticRegression,
    train_counts: dict[str, int],
) -> None:
    """Persist the urgency classifier and metadata for later inference."""
    logger.info(f"Saving urgency artifact to {model_path}")
    path = Path(model_path)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, path / relevance_embedding.CLASSIFIER_FILENAME)
    metadata = {
        "backend": "embedding_linear",
        "model_name": config.RELEVANCE_MODEL_NAME,
        "encoder_cache_path": str(relevance_embedding.get_encoder_cache_path()),
        "max_length": config.RELEVANCE_MAX_LENGTH,
        "text_prep_mode": config.RELEVANCE_TEXT_PREP_MODE,
        "prep_version": config.RELEVANCE_PREP_VERSION,
        "linear_c": config.URGENCY_LINEAR_C,
        "batch_size": config.RELEVANCE_ENCODER_BATCH_SIZE,
        "model_key": get_model_key(),
        "train_counts": train_counts,
    }
    (path / relevance_embedding.METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    logger.info(f"Saved urgency artifact to {model_path}")


def load_classifier(model_path: str) -> LogisticRegression:
    """Load a previously saved urgency logistic-regression classifier."""
    logger.info(f"Loading urgency classifier from {model_path}")
    return relevance_embedding.load_classifier(model_path)


async def predict_probabilities(
    articles: list[Article],
    tokenizer: PreTrainedTokenizerBase,
    encoder: torch.nn.Module,
    classifier: LogisticRegression,
    device: torch.device,
) -> np.ndarray:
    """Predict urgency probabilities using the shared embedding backend."""
    return await relevance_embedding.predict_probabilities(
        articles,
        tokenizer,
        encoder,
        classifier,
        device,
    )


def peak_vram_gb() -> float:
    """Return peak CUDA memory usage for the current process."""
    return relevance_embedding.peak_vram_gb()
