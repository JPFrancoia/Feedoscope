import logging
from functools import lru_cache
from importlib.resources import files
from typing import Any, LiteralString, cast

from feedoscope import config
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

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


async def get_articles() -> list[tuple[Any, ...]]:
    query = _get_query_from_file("get_articles.sql")

    async with global_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                query,
                # {"param": param_value},
            )
            data = await cur.fetchall()

    return data


async def get_unread_articles() -> list[tuple[Any, ...]]:
    query = _get_query_from_file("get_unread_articles.sql")

    async with global_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                query,
                # {"param": param_value},
            )
            data = await cur.fetchall()

    return data


async def get_published_articles() -> list[tuple[Any, ...]]:
    query = _get_query_from_file("get_published_articles.sql")

    async with global_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                query,
                # {"param": param_value},
            )
            data = await cur.fetchall()

    return data


async def get_sample_good() -> list[tuple[Any, ...]]:
    query = _get_query_from_file("get_sample_good.sql")

    async with global_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                query,
                # {"param": param_value},
            )
            data = await cur.fetchall()

    return data


async def get_sample_not_good() -> list[tuple[Any, ...]]:
    query = _get_query_from_file("get_sample_not_good.sql")

    async with global_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                query,
                # {"param": param_value},
            )
            data = await cur.fetchall()

    return data
