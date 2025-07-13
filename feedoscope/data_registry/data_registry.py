from functools import lru_cache
from importlib.resources import files
import logging
from typing import Any, LiteralString, cast

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from feedoscope import config

logger = logging.getLogger(__name__)


global_pool = AsyncConnectionPool(
    config.DATABASE_URL,
    open=False,
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


async def get_read_articles_training() -> list[dict[str, Any]]:
    """Get read articles for training.

    These articles are consdered "good", aka "interesting" by the user.
    A small portion of these articles will be used for validation (100 for now).
    The articles are returned ordered by article id descending, so the order is
    deterministic.

    Returns:
        List of good articles for training.

    """
    query = _get_query_from_file("get_read_articles_training.sql")

    # TODO: parametrize how many articles to leave for validation, for now it's hardcoded to 100

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            # {"param": param_value},
        )
        data = await cur.fetchall()

    return data


async def get_unread_articles_training() -> list[dict[str, Any]]:
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

    return data


async def get_published_articles() -> list[dict[str, Any]]:
    """Fetch published articles.

    Published articles are considered "bad", aka "not interesting" by the user.
    This is because there is no buttn to mark an article as "not interesting" in
    ttrss' UI, and I don't use the published articles feature.
    All published articles are fetched, ordered by article id descending.

    Returns:
        A list of published articles, aka "not interesting" articles.

    """
    query = _get_query_from_file("get_published_articles.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            # {"param": param_value},
        )
        data = await cur.fetchall()

    return data


async def get_sample_good() -> list[dict[str, Any]]:
    """Get a sample of good articles for validation.

    Returns:
        A list of good articles, aka "interesting" articles.

    """
    query = _get_query_from_file("get_sample_good.sql")

    # TODO: parametrize how many articles to fetch for validation, for now it's hardcoded to 100
    # NOTE: the 100 is directly linked to the number of articles we DON'T select in get_read_articles_training

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            # {"param": param_value},
        )
        data = await cur.fetchall()

    return data


async def get_sample_not_good() -> list[dict[str, Any]]:
    """Get a sample of not good articles for validation.

    Returns:
        A list of not good articles, aka "not interesting" articles.

    """
    query = _get_query_from_file("get_sample_not_good.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            # {"param": param_value},
        )
        data = await cur.fetchall()

    return data
