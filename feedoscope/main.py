import argparse
import asyncio
from datetime import datetime, timezone
import logging
import math
import time

from custom_logging import init_logging
from feedoscope import (
    config,
    llm_infer,
    llm_infer_semantic_freshness,
    llm_infer_urgency,
)
from feedoscope.data_registry import data_registry as dr
from feedoscope.utils import clean_title

logger = logging.getLogger(__name__)

# All articles that are more recent than this will be rescored at every inference run.
LOOKBACK_DAYS = 40

# We sample SAMPLING articles between LOOKBACK_DAYS and MAX_LOOKBACK_DAYS_SAMPLING
# and we rescore them. This is to make sure these old-ish articles get rescored from
# time to time, but we save some computing time.
MAX_LOOKBACK_DAYS_SAMPLING = 365
SAMPLING = 1500


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


def compute_urgency_half_life(urgency_prob: float) -> float:
    """Interpolate the legacy urgency-based relevance half-life in days."""
    return config.HALF_LIFE_EVERGREEN + urgency_prob * (
        config.HALF_LIFE_URGENT - config.HALF_LIFE_EVERGREEN
    )


def is_valid_half_life(half_life_days: float | None) -> bool:
    """Return whether a predicted lifetime can safely drive score decay."""
    return (
        half_life_days is not None
        and math.isfinite(half_life_days)
        and half_life_days > 0
    )


def get_decay_half_life(
    urgency_prob: float | None,
    expected_lifetime_days: float | None,
) -> float | None:
    """Return the half-life selected by the active decay backend."""
    if config.RELEVANCE_DECAY_BACKEND == "semantic_freshness":
        return (
            expected_lifetime_days
            if is_valid_half_life(expected_lifetime_days)
            else None
        )
    if urgency_prob is None:
        return None
    return compute_urgency_half_life(urgency_prob)


def decay_relevance_score(
    original_score: float,
    date_entered: datetime,
    half_life_days: float,
) -> int:
    """Apply exponential time decay using a concrete half-life in days.

    Args:
        original_score: The unrounded raw relevance score (0-100).
        date_entered: When the article was published.
        half_life_days: Days until the score is halved.

    Returns:
        The decayed relevance score.

    Raises:
        ValueError: If the half-life is not finite and positive.
    """
    if not is_valid_half_life(half_life_days):
        raise ValueError("half_life_days must be finite and positive")

    days_passed = max(
        0.0,
        (datetime.now(timezone.utc) - date_entered).total_seconds() / 3600 / 24,
    )
    decayed_score = original_score * math.exp(
        -math.log(2) * days_passed / half_life_days
    )

    return int(round(decayed_score))


async def main(
    min_age_days: int | None = None,
    max_age_days: int | None = None,
) -> None:
    age_range = validate_age_range(min_age_days, max_age_days)
    init_logging(config.LOGGING_CONFIG)
    if age_range is None:
        logger.info(
            f"Starting inference: lookback={LOOKBACK_DAYS}d, sampling={SAMPLING}, "
            f"decay={config.RELEVANCE_DECAY_BACKEND}"
        )
    else:
        logger.info(
            f"Starting inference for article ages [{age_range[0]}, {age_range[1]}) "
            f"days with decay={config.RELEVANCE_DECAY_BACKEND}"
        )
    logger.info("Opening database pool...")
    await dr.global_pool.open(wait=True)
    logger.info("Database pool opened.")
    try:
        await dr.clear_downvoted_unread_scores()
        urgency_model_key = llm_infer_urgency.get_active_model_key()
        logger.info(f"Active urgency model key: {urgency_model_key}")

        # Step 1: Build the active article set once. Urgency refresh must mirror
        # relevance refresh exactly, so both backends operate on this same list.
        if age_range is None:
            articles = await llm_infer_urgency.get_articles_for_refresh(
                number_of_days=LOOKBACK_DAYS,
                max_age_in_days=MAX_LOOKBACK_DAYS_SAMPLING,
                sampling=SAMPLING,
            )
        else:
            articles = await dr.get_unread_articles_by_age(*age_range)
            logger.info(
                f"Fetched {len(articles)} unread articles aged "
                f"[{age_range[0]}, {age_range[1]}) days."
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
        urgency_start = time.time()
        urgency_results = await llm_infer_urgency.infer(articles)
        await dr.register_urgency_inference(
            urgency_results,
            model_key=urgency_model_key,
        )
        urgency_elapsed = time.time() - urgency_start
        logger.info(
            f"Urgency refresh completed in {urgency_elapsed:.2f} seconds for "
            f"{len(urgency_results.article_ids)} articles with "
            f"model_key={urgency_model_key}."
        )

        # Step 3: Predict freshness for score decay without changing article tags.
        logger.info("Starting freshness inference...")
        freshness_start = time.time()
        freshness_half_lives: dict[int, float] = {}
        try:
            freshness_results = await llm_infer_semantic_freshness.infer(articles)
            freshness_half_lives = dict(
                zip(
                    freshness_results.article_ids,
                    freshness_results.expected_lifetime_days,
                    strict=True,
                )
            )
        except Exception:
            logger.exception(
                "Freshness inference failed; semantic decay will keep raw scores."
            )
        else:
            logger.info(
                f"Freshness inference completed in "
                f"{time.time() - freshness_start:.2f} seconds for "
                f"{len(freshness_results.article_ids)} articles."
            )

        # Step 4: Run relevance inference.
        logger.info("Starting inference for relevance scores...")
        relevance_start = time.time()
        relevance_scores = await llm_infer.infer(articles)
        relevance_elapsed = time.time() - relevance_start
        logger.info(
            f"Relevance inference completed in {relevance_elapsed:.2f} seconds "
            f"for {len(relevance_scores.article_ids)} articles."
        )
        await dr.register_super_important_inference(relevance_scores)

        # Step 5: Fetch the legacy urgency scores only when the rollback backend is active.
        urgency_scores: dict[int, float] = {}
        if config.RELEVANCE_DECAY_BACKEND == "urgency":
            article_ids = [article.article_id for article in articles]
            urgency_scores = await dr.get_urgency_scores_for_articles(
                article_ids,
                model_key=urgency_model_key,
            )
            logger.info(
                f"Found refreshed urgency scores for {len(urgency_scores)}/{len(articles)} "
                "articles."
            )

        # Step 6: Apply time decay from the selected backend.
        for idx in range(len(articles)):
            article = articles[idx]
            assert article.article_id == relevance_scores.article_ids[idx]

            urgency_prob = urgency_scores.get(article.article_id)
            half_life_days = get_decay_half_life(
                urgency_prob=urgency_prob,
                expected_lifetime_days=freshness_half_lives.get(article.article_id),
            )
            if half_life_days is None:
                if config.RELEVANCE_DECAY_BACKEND == "semantic_freshness":
                    logger.warning(
                        f"Article {article.article_id} has no valid semantic freshness "
                        "lifetime. Skipping decay."
                    )
                else:
                    logger.warning(
                        f"Article {article.article_id} has no refreshed urgency score. "
                        "Skipping decay."
                    )
                continue

            relevance_scores.scores[idx] = decay_relevance_score(
                original_score=relevance_scores.scores[idx],
                date_entered=article.date_entered,
                half_life_days=half_life_days,
            )

        inference_time = time.time() - start_time
        logger.info(
            f"Inference completed in {inference_time:.2f} seconds "
            f"for {len(articles)} articles."
        )

        # Step 7: Write final decayed scores to DB.
        await dr.update_scores(
            article_ids=relevance_scores.article_ids,
            article_titles=relevance_scores.article_titles,
            scores=[round(score) for score in relevance_scores.scores],
        )

        db_write_time = time.time() - inference_time - start_time
        logger.debug(
            f"Scores updated in the database for {len(relevance_scores.article_ids)} "
            f"articles in {db_write_time:.2f} seconds."
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
