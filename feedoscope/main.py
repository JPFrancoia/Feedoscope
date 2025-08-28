import asyncio
from datetime import datetime
import logging
import math
from typing import Literal

from custom_logging import init_logging
from feedoscope import config, infer_time_sensitivity, llm_infer
from feedoscope.data_registry import data_registry as dr
from feedoscope.utils import clean_title

logger = logging.getLogger(__name__)

# Decay Constant (k) = ln(2) / (Desired Half-Life)
# -> Decay Constant (k) = 0.693 / (Desired Half-Life)
# Example:
# The relevance should be cut in half in just 1 day.
# Calculation: k=0.693/1=0.693

DECAY_RATES = {
    1: 0,  # No decay
    2: 0.0154,  # Half-life of 45 days
    3: 0.0347,  # Half-life of 20 days
    4: 0.0693,  # Half-life of 10 days
    5: 0.1386,  # Half-life of 5 days
}

LOOKBACK_DAYS = 45


def decay_relevance_score(
    original_score: int,
    date_entered: datetime,
    time_sensitivity: Literal[1, 2, 3, 4, 5],
) -> int:

    days_passed = (datetime.now() - date_entered).total_seconds() / 3600 / 24
    decayed_score = original_score * math.exp(
        -DECAY_RATES[time_sensitivity] * days_passed
    )

    return int(round(decayed_score))


async def main() -> None:
    await dr.global_pool.open(wait=True)

    # Different behaviour than for relevance scoring (we use the module's main
    # function directly instead of infer). This function will also write the time
    # sensitivity into the database. This is because the time sensitivity a) takes
    # time to infer and b) doesn't change once determined.
    logger.info("Starting inference for time sensitivity...")
    await infer_time_sensitivity.main(LOOKBACK_DAYS)

    # Get articles that are unread from the last N days.
    articles = await dr.get_previous_days_unread_articles(number_of_days=LOOKBACK_DAYS)

    # Remove past scores and time sensitivity from titles.
    # This should be done in llm_infer.infer as well, but better safe than sorry
    for art in articles:
        art.title = clean_title(art.title)

    logger.info("Starting inference for relevance scores...")
    relevance_scores = await llm_infer.infer(articles)

    for idx in range(len(articles)):
        # Make sure the order of the articles is the same.
        assert articles[idx].article_id == relevance_scores.article_ids[idx]

        # If we don't have a time sensitivity score, just use the raw relevance score.
        # It might be because we haven't computed the time sensitivity yet, or
        # because there was a problem. Relevance alone is better than nothing.
        if articles[idx].time_sensitivity_score is None:
            decayed_score = relevance_scores.scores[idx]
            logger.warning(
                f"Article {articles[idx].article_id} has no time sensitivity score. Skipping decay."
            )
            relevance_scores.article_titles[idx] = (
                f"[{decayed_score}] "
                f"{articles[idx].title} "
            )
        else:
            decayed_score = decay_relevance_score(
                original_score=relevance_scores.scores[idx],
                date_entered=articles[idx].date_entered,
                time_sensitivity=articles[idx].time_sensitivity_score,  # type: ignore
            )
            relevance_scores.article_titles[idx] = (
                f"[{decayed_score}] "
                f"{articles[idx].title} "
                f"(TS: {articles[idx].time_sensitivity_score})"
            )

        relevance_scores.scores[idx] = decayed_score

    await dr.update_scores(
        article_ids=relevance_scores.article_ids,
        article_titles=relevance_scores.article_titles,
        scores=relevance_scores.scores,
    )
    logger.debug(
        f"Scores updated in the database for {len(relevance_scores.article_ids)} articles."
    )


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
