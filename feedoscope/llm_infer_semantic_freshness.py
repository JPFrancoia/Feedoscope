import asyncio
import logging
import time

import torch

from custom_logging import init_logging
from feedoscope import config, relevance_embedding, semantic_freshness_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article, SemanticFreshnessInferenceResults
from feedoscope.llm_infer import find_latest_model
from feedoscope.llm_infer_urgency import get_articles_for_refresh

logger = logging.getLogger(__name__)


async def infer(articles: list[Article]) -> SemanticFreshnessInferenceResults:
    """Predict three-label freshness distributions for articles."""
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
        pipeline_label="freshness",
    )
    embeddings = await relevance_embedding.encode_articles(
        articles,
        tokenizer,
        encoder,
        device,
        pipeline_label="freshness",
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
    """Run a freshness prediction smoke test without changing Miniflux tags."""
    await dr.global_pool.open(wait=True)
    try:
        articles = await get_articles_for_refresh()
        if not articles:
            logger.info("No articles to process. Exiting.")
            return

        start_time = time.time()
        results = await infer(articles)
        logger.info(
            f"Freshness inference completed in {time.time() - start_time:.2f} seconds "
            f"for {len(results.article_ids)} articles."
        )
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
