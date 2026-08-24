import argparse
import asyncio
import logging

from custom_logging import init_logging
from feedoscope import article_summaries, config
from feedoscope.data_registry import data_registry as dr

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the bounded summary-generation command arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--by-score", action="store_true")
    selection.add_argument("--days", type=int)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("limit must be positive")
    if args.days is not None and args.days <= 0:
        parser.error("days must be positive")
    return args


async def main(
    limit: int = 100, by_score: bool = False, days: int | None = None
) -> int:
    """Generate and store one-line summaries for the selected entries."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if days is not None and days <= 0:
        raise ValueError("days must be positive")
    if by_score and days is not None:
        raise ValueError("by_score and days cannot both be set")

    await dr.global_pool.open(wait=True)
    try:
        if days is not None:
            articles = await dr.get_recent_articles_for_summary(days, limit)
        elif by_score:
            articles = await dr.get_highest_scored_articles_for_summary(limit)
        else:
            articles = await dr.get_latest_articles_for_summary(limit)
        if not articles:
            logger.info("No articles selected for summary generation.")
            return 0

        summary_model = article_summaries.load_summary_model()
        saved = 0
        for article in articles:
            try:
                summary = article_summaries.summarize_article(article, summary_model)
            except Exception:
                logger.exception(f"Failed to summarize article {article.article_id}.")
                continue
            if summary is None:
                continue
            await dr.update_one_line_summary(article.article_id, summary)
            saved += 1
            logger.info(
                f"Stored summary {saved}/{len(articles)} for {article.article_id}."
            )

        return saved
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    cli_args = parse_args()
    asyncio.run(main(cli_args.limit, cli_args.by_score, cli_args.days))
