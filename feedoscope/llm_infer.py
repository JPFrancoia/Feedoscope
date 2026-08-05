import asyncio
import logging
import os
from pathlib import Path
import shutil
import time

import torch

from custom_logging import init_logging
from feedoscope import config, relevance_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article, RelevanceInferenceResults

logger = logging.getLogger(__name__)

MODEL_NAME = relevance_embedding.get_model_family_prefix()


def find_latest_model(
    model_name: str,
    clean_old_models: bool = True,
    required_filename: str | None = None,
) -> str:
    """Find the latest saved model to use for inference.

    This function will find the latest model in the `models` directory, assuming
    the models are sortable by name. The latest model in the sort is considered the
    latest. This should be true if the model names include the training date.

    Args:
        model_name: family of model to use.
        clean_old_models: if True, delete all older models starting with model_name
            except the latest one.
        required_filename: if set, only select directories containing this file.

    Returns:
        The path to the latest model directory.

    Raises:
        FileNotFoundError: if no trained models are found for the given model_name.

    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Directory {models_dir} does not exist.")
    matching_dirs = [
        entry
        for entry in os.listdir(models_dir)
        if entry.startswith(model_name)
        and os.path.isdir(os.path.join(models_dir, entry))
    ]
    model_dirs = [
        model_dir
        for model_dir in matching_dirs
        if required_filename is None
        or os.path.isfile(os.path.join(models_dir, model_dir, required_filename))
    ]
    if not model_dirs:
        raise FileNotFoundError(
            f"No complete models found starting with {model_name} in {models_dir}."
        )
    model_dirs.sort()
    latest_model = model_dirs[-1]

    if clean_old_models:
        for older_model in matching_dirs:
            if older_model == latest_model:
                continue
            older_model_path = os.path.join(models_dir, older_model)
            try:
                shutil.rmtree(older_model_path)
                logger.warning(f"Deleted older model: {older_model_path}")
            except Exception as e:
                logger.error(f"Failed to delete older model {older_model_path}: {e}")

    return os.path.join(models_dir, latest_model)


def clean_checkpoints(model_path: str) -> None:
    """Delete HuggingFace Trainer checkpoint directories inside a saved model.

    During training, the Trainer creates ``checkpoint-*`` subdirectories for each
    save step/epoch. Once the final model is saved to the same directory, these
    checkpoints are redundant and waste disk space.

    Args:
        model_path: Path to the saved model directory.
    """
    if not os.path.isdir(model_path):
        return

    for entry in os.listdir(model_path):
        entry_path = os.path.join(model_path, entry)
        if os.path.isdir(entry_path) and entry.startswith("checkpoint-"):
            try:
                shutil.rmtree(entry_path)
                logger.info(f"Deleted checkpoint: {entry_path}")
            except Exception as e:
                logger.error(f"Failed to delete checkpoint {entry_path}: {e}")


async def infer(recent_unread_articles: list[Article]) -> RelevanceInferenceResults:
    """Score unread articles with the latest saved relevance artifact."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type != "cuda" and not config.ALLOW_INFERENCE_WO_GPU:
        mes = f"GPU not available. Device is '{device}'. Exiting"
        logger.critical(mes)
        raise RuntimeError(mes)

    model_path = find_latest_model(
        MODEL_NAME,
        clean_old_models=False,
        required_filename=relevance_embedding.TWO_HEAD_ARTIFACT_FILENAME,
    )
    logger.info(f"Loading two-head relevance artifact from {model_path}")

    relevance_classifier, super_important_classifier = (
        relevance_embedding.load_two_head_artifact(model_path)
    )
    # Load before cleanup so an interrupted training run cannot delete the last
    # working artifact just because it created a newer directory.
    find_latest_model(
        MODEL_NAME,
        clean_old_models=True,
        required_filename=relevance_embedding.TWO_HEAD_ARTIFACT_FILENAME,
    )
    tokenizer, encoder = relevance_embedding.load_encoder(device)
    embeddings = await relevance_embedding.encode_articles(
        recent_unread_articles,
        tokenizer,
        encoder,
        device,
    )
    relevance_probs = relevance_embedding.predict_probabilities_from_embeddings(
        embeddings, relevance_classifier
    )
    super_important_probs: list[float] = []
    scores = relevance_probs
    if config.SUPER_IMPORTANT_INFERENCE_ENABLED:
        preference_probs = relevance_embedding.predict_probabilities_from_embeddings(
            embeddings, super_important_classifier
        )
        scores = relevance_embedding.combine_probabilities(
            relevance_probs,
            preference_probs,
            bonus_strength=config.SUPER_IMPORTANT_BONUS,
        )
        super_important_probs = preference_probs.tolist()
    else:
        logger.info(
            "Super-important inference is disabled; using relevance-only scores."
        )

    return RelevanceInferenceResults(
        article_ids=[article.article_id for article in recent_unread_articles],
        article_titles=[article.title for article in recent_unread_articles],
        scores=(scores * 100).tolist(),
        super_important_scores=super_important_probs,
        model_key=Path(model_path).name,
    )


async def main() -> None:
    """Run relevance inference and write scores back to the database."""
    await dr.global_pool.open(wait=True)

    await dr.clear_downvoted_unread_scores()
    recent_unread_articles = await dr.get_previous_days_unread_articles()
    logger.debug(f"Collected {len(recent_unread_articles)} recent unread articles.")

    start_time = time.time()
    results = await infer(recent_unread_articles)
    elapsed_time = time.time() - start_time
    logger.info(
        f"Inference completed in {elapsed_time:.2f} seconds for {len(recent_unread_articles)} articles."
    )

    if config.SUPER_IMPORTANT_INFERENCE_ENABLED:
        await dr.register_super_important_inference(results)
    await dr.update_scores(
        article_ids=results.article_ids,
        article_titles=results.article_titles,
        scores=relevance_embedding.prepare_scores_for_storage(results.scores),
    )
    logger.debug(
        f"Scores updated in the database for {len(results.article_ids)} articles."
    )

    await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
