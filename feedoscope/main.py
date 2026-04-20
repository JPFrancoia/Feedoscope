import asyncio
from datetime import datetime, timezone
import logging
import math
import time

from custom_logging import init_logging
from feedoscope import config, llm_infer, llm_infer_urgency
from feedoscope.data_registry import data_registry as dr
from feedoscope.utils import clean_title

logger = logging.getLogger(__name__)

# Half-life boundaries (in days) for the urgency-based decay.
# The urgency embedding backend produces a probability (0.0 to 1.0):
#   urgency_prob = 0.0 → evergreen content, half-life of 365 days
#   urgency_prob = 1.0 → urgent/ephemeral content, half-life of 10 days
HALF_LIFE_EVERGREEN = 365
HALF_LIFE_URGENT = 10

# All articles that are more recent than this will be rescored at every inference run.
LOOKBACK_DAYS = 40

# We sample SAMPLING articles between LOOKBACK_DAYS and MAX_LOOKBACK_DAYS_SAMPLING
# and we rescore them. This is to make sure these old-ish articles get rescored from
# time to time, but we save some computing time.
MAX_LOOKBACK_DAYS_SAMPLING = 365
SAMPLING = 1500


def compute_decay_rate(urgency_prob: float) -> float:
    """Interpolate decay rate between evergreen and urgent half-lives.

    We interpolate the half-life linearly between HALF_LIFE_EVERGREEN and
    HALF_LIFE_URGENT based on the urgency probability, then compute the
    exponential decay rate from that half-life.

    At urgency_prob=0.5, the half-life is ~187 days (similar to the old
    score=2 half-life of 183 days).

    Args:
        urgency_prob: Probability of urgency from the current urgency backend (0.0 to 1.0).

    Returns:
        The exponential decay rate constant.

    """
    half_life = HALF_LIFE_EVERGREEN + urgency_prob * (
        HALF_LIFE_URGENT - HALF_LIFE_EVERGREEN
    )
    return math.log(2) / half_life


def decay_relevance_score(
    original_score: int,
    date_entered: datetime,
    urgency_prob: float,
) -> int:
    """Apply time-decay to a relevance score based on urgency probability.

    Args:
        original_score: The raw relevance score (0-100).
        date_entered: When the article was published.
        urgency_prob: Probability of urgency (0.0 to 1.0).

    Returns:
        The decayed relevance score.

    """
    days_passed = (
        (datetime.now(timezone.utc) - date_entered).total_seconds() / 3600 / 24
    )
    decay_rate = compute_decay_rate(urgency_prob)
    decayed_score = original_score * math.exp(-decay_rate * days_passed)

    return int(round(decayed_score))


async def main() -> None:
    await dr.global_pool.open(wait=True)
    try:
        # Step 1: Build the active article set once. Urgency refresh must mirror
        # relevance refresh exactly, so both backends operate on this same list.
        articles = await llm_infer_urgency.get_articles_for_refresh(
            number_of_days=LOOKBACK_DAYS,
            max_age_in_days=MAX_LOOKBACK_DAYS_SAMPLING,
            sampling=SAMPLING,
        )
        logger.info(f"Total articles to be scored: {len(articles)}")

        if not articles:
            logger.info("No articles to score. Exiting.")
            return

        # Remove past scores and time sensitivity from titles.
        # This should be done in llm_infer.infer as well, but better safe than sorry.
        for art in articles:
            art.title = clean_title(art.title)

        start_time = time.time()

        # Step 2: Refresh urgency scores for the same active article set.
        logger.info("Starting urgency inference for the active article set...")
        urgency_results = await llm_infer_urgency.infer(articles)
        await dr.register_urgency_inference(
            urgency_results,
            model_key=llm_infer_urgency.get_active_model_key(),
        )
        logger.info(
            f"Cached urgency scores for {len(urgency_results.article_ids)} articles."
        )

        # Step 3: Run relevance inference.
        logger.info("Starting inference for relevance scores...")
        relevance_scores = await llm_infer.infer(articles)

        # Step 4: Fetch the refreshed urgency scores for decay calculation.
        article_ids = [article.article_id for article in articles]
        urgency_scores = await dr.get_urgency_scores_for_articles(
            article_ids,
            model_key=llm_infer_urgency.get_active_model_key(),
        )
        logger.info(
            f"Found refreshed urgency scores for {len(urgency_scores)}/{len(articles)} articles."
        )

        # Step 5: Apply time-decay using urgency probabilities.
        for idx in range(len(articles)):
            assert articles[idx].article_id == relevance_scores.article_ids[idx]

            urgency_prob = urgency_scores.get(articles[idx].article_id)
            if urgency_prob is None:
                decayed_score = relevance_scores.scores[idx]
                logger.warning(
                    f"Article {articles[idx].article_id} has no refreshed urgency score. "
                    "Skipping decay."
                )
            else:
                decayed_score = decay_relevance_score(
                    original_score=relevance_scores.scores[idx],
                    date_entered=articles[idx].date_entered,
                    urgency_prob=urgency_prob,
                )

            relevance_scores.scores[idx] = decayed_score

        inference_time = time.time() - start_time
        logger.info(
            f"Inference completed in {inference_time:.2f} seconds "
            f"for {len(articles)} articles."
        )

        # Step 6: Write final decayed scores to DB.
        await dr.update_scores(
            article_ids=relevance_scores.article_ids,
            article_titles=relevance_scores.article_titles,
            scores=relevance_scores.scores,
        )

        db_write_time = time.time() - inference_time - start_time
        logger.debug(
            f"Scores updated in the database for {len(relevance_scores.article_ids)} "
            f"articles in {db_write_time:.2f} seconds."
        )
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
