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
    2: 0.0019,  # Half-life of 365 days
    3: 0.0154,  # Half-life of 45 days
    4: 0.0693,  # Half-life of 10 days
    5: 0.1386,  # Half-life of 5 days
}

LOOKBACK_DAYS = 14


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

    # Get articles that DO NOT have time sensitivity yet.
    # If this script is run regularly, this should be a small number.
    articles = await dr.get_previous_days_articles_wo_time_sensitivity(
        number_of_days=LOOKBACK_DAYS
    )

    logger.info("Starting inference for time sensitivity...")
    time_sensitivities = await infer_time_sensitivity.infer(articles)

    # Get articles that are unread from the last N days.
    articles = await dr.get_previous_days_unread_articles(number_of_days=LOOKBACK_DAYS)

    logger.info("Starting inference for relevance scores...")
    relevance_scores = await llm_infer.infer(articles)

    for idx in range(len(articles)):
        # Make sure the order of the articles is the same.
        # If not, one of the infer functions returned articles in a different order.
        assert (
            articles[idx].article_id
            == relevance_scores.article_ids[idx]
            == time_sensitivities[idx].article_id
        )

        decayed_score = decay_relevance_score(
            original_score=relevance_scores.scores[idx],
            date_entered=articles[idx].date_entered,
            time_sensitivity=time_sensitivities[idx].score,
        )

        relevance_scores.article_titles[idx] = (
            f"[{decayed_score}] "
            f"{articles[idx].title} "
            f"(TS: {time_sensitivities[idx].score})"
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
