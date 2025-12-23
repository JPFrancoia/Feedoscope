import asyncio
import logging
import os
import shutil
import time

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from custom_logging import init_logging
from feedoscope import config, utils
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article, RelevanceInferenceResults

logger = logging.getLogger(__name__)

# https://huggingface.co/blog/modernbert

MODEL_NAME = "answerdotai-ModernBERT-base_512_2_epochs_16_batch_size"
MAX_LENGTH = 512  # Maximum length for the tokenizer
INFERENCE_BATCH_SIZE = 128


def find_latest_model(model_name: str, clean_old_models: bool = True) -> str:
    """Find the latest saved model to use for inference.

    This function will find the latest model in the `models` directory, assuming
    the models are sortable by name. The latest model in the sort is considered the
    latest. This should be true if the model names include the training date.

    Args:
        model_name: family of model to use, e.g. "answerdotai-ModernBERT-base"
        clean_old_models: if True, delete all older models starting with model_name
            except the latest one

    Returns:
        The path to the latest model directory.

    Raises:
        FileNotFoundError: if no trained models are found for the given model_name.

    """
    # iterate through the models directory. Look for the model starting with
    # the model_name. sort by name and return the last one.
    models_dir = "models"
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Directory {models_dir} does not exist.")
    model_dirs = [d for d in os.listdir(models_dir) if d.startswith(model_name)]
    if not model_dirs:
        raise FileNotFoundError(
            f"No models found starting with {model_name} in {models_dir}."
        )
    model_dirs.sort()  # Sort by name, assuming the latest model has the highest name
    latest_model = model_dirs[-1]

    if clean_old_models:
        # Delete all older models (keep only the latest)
        for older_model in model_dirs[:-1]:
            older_model_path = os.path.join(models_dir, older_model)
            try:
                shutil.rmtree(older_model_path)
                logger.warning(f"Deleted older model: {older_model_path}")
            except Exception as e:
                logger.error(f"Failed to delete older model {older_model_path}: {e}")

    return os.path.join(models_dir, latest_model)


async def infer(recent_unread_articles: list[Article]) -> RelevanceInferenceResults:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Crash on GPU not available if ALLOW_INFERENCE_WO_GPU is False
    if device.type != "cuda" and not config.ALLOW_INFERENCE_WO_GPU:
        mes = f"GPU not available. Device is '{device}'. Exiting"
        logger.critical(mes)
        raise RuntimeError(mes)

    model_path = find_latest_model(MODEL_NAME.replace("/", "-"))

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)  # Move model to GPU if available

    articles_text = utils.prepare_articles_text(recent_unread_articles)

    logger.debug("Tokenizing articles for inference...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(
        articles_text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    logger.debug("Articles tokenized successfully.")

    logger.debug("Inferencing...")

    all_probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(articles_text), INFERENCE_BATCH_SIZE):
            logger.debug(
                f"Processing batch {start // INFERENCE_BATCH_SIZE + 1} of {len(articles_text) // INFERENCE_BATCH_SIZE + 1}"
            )
            end = start + INFERENCE_BATCH_SIZE
            batch_inputs = {k: v[start:end].to(device) for k, v in inputs.items()}
            preds = model(**batch_inputs).logits
            probs = torch.sigmoid(preds[:, 1]).cpu().numpy()
            all_probs.extend(probs)

    logger.debug("Inference completed successfully.")

    probs = np.round(np.array(all_probs) * 100).astype(int).tolist()

    article_ids = [article.article_id for article in recent_unread_articles]
    article_titles = [
        f"[{score}] {article.title}"
        for score, article in zip(probs, recent_unread_articles)
    ]

    return RelevanceInferenceResults(
        article_ids=article_ids,
        article_titles=article_titles,
        scores=probs,
    )


async def main() -> None:
    await dr.global_pool.open(wait=True)

    recent_unread_articles = await dr.get_previous_days_unread_articles()

    logger.debug(f"Collected {len(recent_unread_articles)} recent unread articles.")

    start_time = time.time()

    results = await infer(recent_unread_articles)

    elapsed_time = time.time() - start_time
    logger.info(
        f"Inference completed in {elapsed_time:.2f} seconds for {len(recent_unread_articles)} articles."
    )

    await dr.update_scores(
        article_ids=results.article_ids,
        article_titles=results.article_titles,
        scores=results.scores,
    )
    logger.debug(
        f"Scores updated in the database for {len(results.article_ids)} articles."
    )

    await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
