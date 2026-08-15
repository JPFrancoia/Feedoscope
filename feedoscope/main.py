import argparse
import asyncio
from datetime import datetime, timezone
import logging
import math
import time

from custom_logging import init_logging
from feedoscope import config, llm_infer, relevance_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article
from feedoscope.utils import clean_title

logger = logging.getLogger(__name__)


def validate_age_range(
    min_age_days: int | None,
    max_age_days: int | None,
) -> tuple[int, int] | None:
    """Validate an optional non-overlapping inference age range."""
    if min_age_days is None and max_age_days is None:
        return None
    if min_age_days is None or max_age_days is None:
        raise ValueError("min-age-days and max-age-days must be provided together")
    if min_age_days < 0 or max_age_days <= min_age_days:
        raise ValueError("age range must satisfy 0 <= min-age-days < max-age-days")
    return min_age_days, max_age_days


def parse_args() -> argparse.Namespace:
    """Parse optional age-block arguments for controlled inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-age-days", type=int)
    parser.add_argument("--max-age-days", type=int)
    args = parser.parse_args()
    try:
        validate_age_range(args.min_age_days, args.max_age_days)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def is_valid_half_life(half_life_days: float | None) -> bool:
    """Return whether a half-life can safely drive score decay."""
    return (
        half_life_days is not None
        and math.isfinite(half_life_days)
        and half_life_days > 0
    )


def calculate_score_horizon_days(half_life_days: float) -> float:
    """Return the maximum age that can produce a positive stored score."""
    if not is_valid_half_life(half_life_days):
        raise ValueError("half_life_days must be finite and positive")

    minimum_positive_raw_score = 100 * (1 - (1 - 0.5 / 100) ** 3)
    return half_life_days * math.log2(100 / minimum_positive_raw_score)


def decay_relevance_score(
    original_score: float,
    date_entered: datetime,
    half_life_days: float,
) -> float:
    """Apply exponential time decay using a concrete half-life in days."""
    if not is_valid_half_life(half_life_days):
        raise ValueError("half_life_days must be finite and positive")

    days_passed = max(
        0.0,
        (datetime.now(timezone.utc) - date_entered).total_seconds() / 3600 / 24,
    )
    return original_score * math.exp(-math.log(2) * days_passed / half_life_days)


async def get_articles_for_scoring(score_horizon_days: float) -> list[Article]:
    """Return unread articles that can still produce a positive stored score."""
    return await dr.get_previous_days_unread_articles(score_horizon_days)


async def main(
    min_age_days: int | None = None,
    max_age_days: int | None = None,
) -> None:
    age_range = validate_age_range(min_age_days, max_age_days)
    score_horizon_days = calculate_score_horizon_days(config.AGE_DECAY_HALF_LIFE_DAYS)
    init_logging(config.LOGGING_CONFIG)
    if age_range is None:
        logger.info(
            f"Starting inference: score horizon={score_horizon_days:.4f}d, "
            f"half-life={config.AGE_DECAY_HALF_LIFE_DAYS}d"
        )
    else:
        logger.info(
            f"Starting inference for article ages [{age_range[0]}, {age_range[1]}) "
            f"days with half-life={config.AGE_DECAY_HALF_LIFE_DAYS}d"
        )
    logger.info("Opening database pool...")
    await dr.global_pool.open(wait=True)
    logger.info("Database pool opened.")
    try:
        await dr.clear_downvoted_unread_scores()
        if age_range is None:
            articles = await get_articles_for_scoring(score_horizon_days)
        else:
            articles = await dr.get_unread_articles_by_age(*age_range)
            logger.info(
                f"Fetched {len(articles)} unread articles aged "
                f"[{age_range[0]}, {age_range[1]}) days."
            )
        await dr.clear_expired_unread_scores(score_horizon_days)
        logger.info(f"Total articles to be scored: {len(articles)}")

        if not articles:
            logger.info("No articles to score. Exiting.")
            return

        for article in articles:
            article.title = clean_title(article.title)

        start_time = time.time()
        logger.info("Starting inference for relevance scores...")
        relevance_scores = await llm_infer.infer(articles)
        logger.info(
            f"Relevance inference completed in {time.time() - start_time:.2f} seconds "
            f"for {len(relevance_scores.article_ids)} articles."
        )

        for idx, article in enumerate(articles):
            assert article.article_id == relevance_scores.article_ids[idx]
            relevance_scores.scores[idx] = decay_relevance_score(
                original_score=relevance_scores.scores[idx],
                date_entered=article.date_entered,
                half_life_days=config.AGE_DECAY_HALF_LIFE_DAYS,
            )

        await dr.update_scores(
            article_ids=relevance_scores.article_ids,
            article_titles=relevance_scores.article_titles,
            scores=relevance_embedding.prepare_scores_for_storage(
                relevance_scores.scores
            ),
        )
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(
        main(
            min_age_days=cli_args.min_age_days,
            max_age_days=cli_args.max_age_days,
        )
    )
