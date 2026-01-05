from functools import lru_cache
from importlib.resources import files
import logging
from typing import Any, LiteralString, cast

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from feedoscope import config
from feedoscope.entities import Article, TimeSensitivity

logger = logging.getLogger(__name__)


# For explanation about type hinting, see:
# https://www.psycopg.org/psycopg3/docs/advanced/typing.html#generic-pool-types
global_pool = AsyncConnectionPool(
    config.DATABASE_URL,
    open=False,
    connection_class=AsyncConnection[DictRow],  # provides type hints
    kwargs={
        "row_factory": dict_row,
    },
    max_size=10,
    max_lifetime=10 * 60,
    max_idle=5 * 60,
)


# We limit the maxsize to prevent any foot gun
@lru_cache(maxsize=100)
def _get_query_from_file(filename: str) -> LiteralString:
    query = files("feedoscope.data_registry.sql").joinpath(filename).read_text().strip()

    query = cast(LiteralString, query)

    return query


async def get_read_articles_training(
    validation_size: int = 100,
) -> list[Article]:
    """Get read articles for training.

    These articles are consdered "good", aka "interesting" by the user.
    A small portion of these articles will be used for validation (100 for now).
    The articles are returned ordered by article id descending, so the order is
    deterministic.

    Args:
        validation_size: Number of articles to leave for validation

    Returns:
        List of good articles for training.

    """
    query = _get_query_from_file("get_read_articles_training.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            {"validation_size": validation_size},
        )
        data = await cur.fetchall()

    return [Article(**article) for article in data]


async def get_unread_articles_training() -> list[Article]:
    """Get unread articles for training.

    These articles are considered unlabelled, they could be good or bad.
    We fetch a large number of these articles to train the model.
    7000 articles are fetched for now, and they are ordered by article id descending.

    Returns:
        List of unread articles for training.

    """
    query = _get_query_from_file("get_unread_articles_training.sql")

    # TODO: parametrize how many articles to fetch for training, for now it's hardcoded to 7000

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            # {"param": param_value},
        )
        data = await cur.fetchall()

    return [Article(**article) for article in data]


async def get_published_articles(validation_size: int = 0) -> list[Article]:
    """Fetch published articles.

    Published articles are considered "bad", aka "not interesting" by the user.
    This is because there is no buttn to mark an article as "not interesting" in
    ttrss' UI, and I don't use the published articles feature.
    All published articles are fetched, ordered by article id descending.

    Args:
        validation_size: Number of articles to leave for validation. Default to 0
            for PU learning, because not using published (aka bad) articles for
            training with PU learning.

    Returns:
        A list of published articles, aka "not interesting" articles.

    """
    query = _get_query_from_file("get_published_articles.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            {"validation_size": validation_size},
        )
        data = await cur.fetchall()

    return [Article(**article) for article in data]


async def get_sample_good(validation_size: int) -> list[Article]:
    """Get a sample of good articles for validation.

    Args:
        validation_size: Number of articles left for validation

    Returns:
        A list of good articles, aka "interesting" articles.

    """
    query = _get_query_from_file("get_sample_good.sql")

    # TODO: parametrize how many articles to fetch for validation, for now it's hardcoded to 100
    # NOTE: the 100 is directly linked to the number of articles we DON'T select in get_read_articles_training

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            {"validation_size": validation_size},
        )
        data = await cur.fetchall()

    return [Article(**article) for article in data]


async def get_sample_not_good(validation_size: int) -> list[Article]:
    """Get a sample of not good articles for validation.

    Args:
        validation_size: Number of articles left for validation

    Returns:
        A list of not good articles, aka "not interesting" articles.

    """
    query = _get_query_from_file("get_sample_not_good.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            {"validation_size": validation_size},
        )
        data = await cur.fetchall()

    return [Article(**article) for article in data]


async def get_previous_days_unread_articles(number_of_days: int = 14) -> list[Article]:
    """Get unread articles from the previous X days.

    This is used to fetch articles that are not read yet, but are still
    within the last X days.
    Only articles that are unread AND with a score of 0 are considered.

    Args:
        number_of_days: Number of days to look back for unread articles.

    Returns:
        A list of unread articles from the previous X days.

    """
    query = _get_query_from_file("get_previous_days_unread_articles.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            {
                "number_of_days": number_of_days,
            },
        )
        data = await cur.fetchall()

    return [Article(**article) for article in data]


async def get_old_unread_articles(
    age_in_days: int = 30, max_age_in_days: int = 365, sampling: int = 1500
) -> list[Article]:

    query = _get_query_from_file("get_old_unread_articles.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            {
                "age_in_days": age_in_days,
                "max_age_in_days": max_age_in_days,
                "sampling": sampling,
            },
        )
        data = await cur.fetchall()

    return [Article(**article) for article in data]


async def update_scores(
    article_ids: list[int], article_titles: list[str], scores: list[int]
) -> None:
    """Update the scores of articles in the database.

    Args:
        article_titles: List of article titles (unused, kept for backward compatibility).
        article_ids: List of article IDs to update.
        scores: List of scores to set for the articles.

    """
    scores_query = _get_query_from_file("update_scores.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.executemany(
            scores_query,
            [
                {"score": score, "int_id": int_id}
                for score, int_id in zip(scores, article_ids)
            ],
        )


# WARNING: this returns a different article format than the other functions
async def get_previous_days_articles_wo_time_sensitivity(
    number_of_days: int = 14,
) -> list[Article]:
    query = _get_query_from_file("get_previous_days_wo_time_sensitivity_articles.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            {"number_of_days": number_of_days},
        )
        data = await cur.fetchall()

    return [Article(**article) for article in data]


async def register_time_sensitivity_for_articles(
    time_sensitivities: list[TimeSensitivity],
) -> None:
    query = _get_query_from_file("register_time_sensitivity_for_articles.sql")

    async with (
        global_pool.connection() as conn,
        conn.cursor() as cur,
        cur.copy(query) as copy,
    ):
        for sensitivity in time_sensitivities:
            row = (
                sensitivity.article_id,
                sensitivity.score,
                sensitivity.confidence,
                sensitivity.explanation,
            )
            await copy.write_row(row)
