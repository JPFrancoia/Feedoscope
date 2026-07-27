import asyncio
import logging
import time

import numpy as np
import torch

from custom_logging import init_logging
from feedoscope import (
    config,
    relevance_embedding,
    semantic_freshness_embedding,
)
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article, SemanticFreshnessInferenceResults
from feedoscope.llm_infer import find_latest_model
from feedoscope.llm_infer_urgency import get_articles_for_refresh

logger = logging.getLogger(__name__)


def get_active_model_key() -> str:
    """Return the model key stored in the active artifact."""
    model_path = find_latest_model(
        semantic_freshness_embedding.get_model_family_prefix(), clean_old_models=False
    )
    _, metadata = semantic_freshness_embedding.load_artifact(model_path)
    return str(metadata["model_key"])


async def infer(articles: list[Article]) -> SemanticFreshnessInferenceResults:
    """Predict semantic useful-lifetime distributions for articles."""
    if not articles:
        return SemanticFreshnessInferenceResults(
            article_ids=[], bucket_probabilities=[], expected_lifetime_days=[]
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not config.ALLOW_INFERENCE_WO_GPU:
        raise RuntimeError(f"GPU not available. Device is '{device}'. Exiting")

    model_path = find_latest_model(
        semantic_freshness_embedding.get_model_family_prefix(), clean_old_models=False
    )
    classifiers, _ = semantic_freshness_embedding.load_artifact(model_path)
    tokenizer, encoder = relevance_embedding.load_encoder(
        device,
        pipeline_label="semantic freshness",
    )
    embeddings = await relevance_embedding.encode_articles(
        articles,
        tokenizer,
        encoder,
        device,
        pipeline_label="semantic freshness",
    )
    probabilities = semantic_freshness_embedding.bucket_probabilities(
        embeddings, classifiers
    )
    lifetimes = semantic_freshness_embedding.expected_lifetime_days(probabilities)

    return SemanticFreshnessInferenceResults(
        article_ids=[article.article_id for article in articles],
        bucket_probabilities=probabilities.tolist(),
        expected_lifetime_days=lifetimes.tolist(),
    )


async def main() -> None:
    """Refresh shadow-mode freshness predictions and automatic tags."""
    await dr.global_pool.open(wait=True)
    try:
        model_key = get_active_model_key()
        articles = await get_articles_for_refresh()
        if not articles:
            logger.info("No articles to process. Exiting.")
            return

        start_time = time.time()
        results = await infer(articles)
        await dr.register_semantic_freshness_inference(results, model_key)

        tag_ids = await dr.ensure_semantic_freshness_user_tags()
        horizon_indexes = np.argmax(results.bucket_probabilities, axis=1)
        horizons = [
            semantic_freshness_embedding.HORIZONS[index] for index in horizon_indexes
        ]
        await dr.assign_semantic_freshness_auto_tags(
            results.article_ids,
            horizons,
            tag_ids,
        )
        logger.info(
            f"Freshness inference completed in {time.time() - start_time:.2f} seconds "
            f"for {len(results.article_ids)} articles with model_key={model_key}."
        )
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
