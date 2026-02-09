from functools import lru_cache
from importlib.resources import files
import logging
from typing import Any, LiteralString, cast

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from feedoscope import config
from feedoscope.entities import (
    Article,
    SimplifiedTimeSensitivity,
    TimeSensitivity,
    UrgencyInferenceResults,
)

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


# --- Simplified time sensitivity (Phase 2: decoder model labeling) ---


async def get_articles_wo_simplified_time_sensitivity() -> list[Article]:
    """Get articles from the last 6 months without a simplified time sensitivity score.

    Filters to articles published within the last 6 months. Re-runnable: only
    returns articles missing from time_sensitivity_simplified.

    Returns:
        Articles from the last 6 months without a simplified time sensitivity score.

    """
    query = _get_query_from_file("get_articles_wo_simplified_time_sensitivity.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(query)
        data = await cur.fetchall()

    return [Article(**article) for article in data]


async def register_simplified_time_sensitivity(
    time_sensitivities: list[SimplifiedTimeSensitivity],
) -> None:
    """Bulk insert simplified time sensitivity scores via COPY.

    Args:
        time_sensitivities: List of simplified time sensitivity results to insert.

    """
    query = _get_query_from_file("register_simplified_time_sensitivity.sql")

    async with (
        global_pool.connection() as conn,
        conn.cursor() as cur,
        cur.copy(query) as copy,
    ):
        for sensitivity in time_sensitivities:
            row = (
                sensitivity.article_id,
                sensitivity.score,
                sensitivity.explanation,
            )
            await copy.write_row(row)


# --- Training data for distilled urgency model (Phase 3) ---


async def get_articles_with_simplified_time_sensitivity() -> list[tuple[Article, int]]:
    """Get all articles with their simplified time sensitivity labels.

    Used for training the distilled urgency ModernBERT model. Returns article
    data joined with the binary urgency label (0 or 1).

    Returns:
        List of (Article, urgency_label) tuples.

    """
    query = _get_query_from_file("get_articles_with_simplified_time_sensitivity.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(query)
        data = await cur.fetchall()

    results: list[tuple[Article, int]] = []
    for row in data:
        urgency_label = row.pop("urgency_label")
        article = Article(**row)
        results.append((article, urgency_label))

    return results


# --- Urgency inference caching (Phase 4: distilled model) ---


async def get_articles_wo_urgency_inference(
    number_of_days: int = 14,
) -> list[Article]:
    """Get recent unread articles without a cached urgency inference score.

    Only processes articles that haven't been scored yet by the distilled
    urgency model. Used by llm_infer_urgency.py.

    Args:
        number_of_days: Number of days to look back for unread articles.

    Returns:
        List of unread articles without cached urgency scores.

    """
    query = _get_query_from_file("get_articles_wo_urgency_inference.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(query, {"number_of_days": number_of_days})
        data = await cur.fetchall()

    return [Article(**article) for article in data]


async def register_urgency_inference(
    results: UrgencyInferenceResults,
) -> None:
    """Bulk insert urgency inference scores via COPY.

    Args:
        results: Urgency inference results containing article IDs and scores.

    """
    query = _get_query_from_file("register_urgency_inference.sql")

    async with (
        global_pool.connection() as conn,
        conn.cursor() as cur,
        cur.copy(query) as copy,
    ):
        for article_id, score in zip(results.article_ids, results.urgency_scores):
            await copy.write_row((article_id, score))


async def get_urgency_scores_for_articles(
    article_ids: list[int],
) -> dict[int, float]:
    """Fetch cached urgency scores for a set of articles.

    Used by main.py to look up urgency probabilities for time-decay calculation.

    Args:
        article_ids: List of article IDs to look up.

    Returns:
        Dict mapping article_id to urgency_score (0.0 to 1.0).
        Articles without a cached score are not included.

    """
    query = _get_query_from_file("get_urgency_scores_for_articles.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(query, {"article_ids": article_ids})
        data = await cur.fetchall()

    return {row["article_id"]: row["urgency_score"] for row in data}


# --- Urgency user tags ---


async def ensure_urgency_user_tags() -> dict[str, int]:
    """Ensure urgency user tags exist and return their IDs.

    Creates '0-urgency' and '1-urgency' tags for user_id=1
    if they don't already exist, then fetches their IDs.

    Returns:
        Dict mapping tag title to tag ID, e.g.
        {'0-urgency': 42, '1-urgency': 43}.

    """
    upsert_query = _get_query_from_file("upsert_urgency_user_tags.sql")
    select_query = _get_query_from_file("ensure_urgency_user_tags.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(upsert_query)
        await cur.execute(select_query)
        data = await cur.fetchall()

    return {row["title"]: row["id"] for row in data}


async def assign_urgency_tags_for_articles(
    article_ids: list[int],
    scores: list[int],
    tag_ids: dict[str, int],
) -> None:
    """Assign urgency user tags based on simplified time sensitivity scores.

    For each article, removes any existing urgency tag and assigns the
    correct one based on the score (0 or 1).

    Args:
        article_ids: List of article IDs to tag.
        scores: Corresponding urgency scores (0 or 1) for each article.
        tag_ids: Dict mapping tag title to tag ID (from ensure_urgency_user_tags).

    """
    query = _get_query_from_file("set_urgency_user_tag_for_entry.sql")

    tag_id_by_score = {
        0: tag_ids["0-urgency"],
        1: tag_ids["1-urgency"],
    }

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.executemany(
            query,
            [
                {"entry_id": article_id, "user_tag_id": tag_id_by_score[score]}
                for article_id, score in zip(article_ids, scores)
            ],
        )
