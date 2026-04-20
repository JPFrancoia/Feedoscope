import asyncio
import logging
import time

import torch

from custom_logging import init_logging
from feedoscope import config, relevance_embedding, urgency_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article, UrgencyInferenceResults
from feedoscope.llm_infer import find_latest_model

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 40
MAX_LOOKBACK_DAYS_SAMPLING = 365
SAMPLING = 1500


def get_active_model_key() -> str:
    """Return the active urgency cache key for the current embedding backend."""
    return urgency_embedding.get_model_key()


async def get_articles_for_refresh(
    number_of_days: int = LOOKBACK_DAYS,
    max_age_in_days: int = MAX_LOOKBACK_DAYS_SAMPLING,
    sampling: int = SAMPLING,
) -> list[Article]:
    """Build the exact unread article set refreshed by both urgency and relevance."""
    recent_articles = await dr.get_previous_days_unread_articles(
        number_of_days=number_of_days
    )
    logger.info(
        f"Fetched {len(recent_articles)} unread articles from the last {number_of_days} days."
    )

    old_articles = await dr.get_old_unread_articles(
        age_in_days=number_of_days,
        max_age_in_days=max_age_in_days,
        sampling=sampling,
    )
    logger.info(
        f"Fetched {len(old_articles)} old unread articles between "
        f"{number_of_days} and {max_age_in_days} days."
    )

    return recent_articles + old_articles


async def infer(articles: list[Article]) -> UrgencyInferenceResults:
    """Run urgency inference using the embedding-linear urgency model.

    Produces a continuous urgency probability (0.0 to 1.0) per article,
    where 0.0 = evergreen and 1.0 = highly urgent/ephemeral.

    Args:
        articles: List of articles to score.

    Returns:
        UrgencyInferenceResults with article IDs and urgency probabilities.

    """
    if not articles:
        return UrgencyInferenceResults(article_ids=[], urgency_scores=[])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type != "cuda" and not config.ALLOW_INFERENCE_WO_GPU:
        mes = f"GPU not available. Device is '{device}'. Exiting"
        logger.critical(mes)
        raise RuntimeError(mes)

    model_path = find_latest_model(urgency_embedding.get_model_family_prefix())

    logger.info(f"Loading embedding-linear urgency artifact from {model_path}")
    classifier = urgency_embedding.load_classifier(model_path)
    tokenizer, encoder = relevance_embedding.load_encoder(device)
    probs = await urgency_embedding.predict_probabilities(
        articles,
        tokenizer,
        encoder,
        classifier,
        device,
    )

    article_ids = [article.article_id for article in articles]

    return UrgencyInferenceResults(
        article_ids=article_ids,
        urgency_scores=probs.tolist(),
    )


async def main(
    number_of_days: int = LOOKBACK_DAYS,
    max_age_in_days: int = MAX_LOOKBACK_DAYS_SAMPLING,
    sampling: int = SAMPLING,
) -> None:
    """Refresh urgency for the same unread article set scored by relevance."""
    await dr.global_pool.open(wait=True)
    try:
        articles = await get_articles_for_refresh(
            number_of_days=number_of_days,
            max_age_in_days=max_age_in_days,
            sampling=sampling,
        )

        if not articles:
            logger.info("No articles to process. Exiting.")
            return

        start_time = time.time()

        results = await infer(articles)

        elapsed_time = time.time() - start_time
        logger.info(
            f"Urgency inference completed in {elapsed_time:.2f} seconds "
            f"for {len(articles)} articles."
        )

        await dr.register_urgency_inference(results, model_key=get_active_model_key())
        logger.info(f"Cached urgency scores for {len(results.article_ids)} articles.")
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
