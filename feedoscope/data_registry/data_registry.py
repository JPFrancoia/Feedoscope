from collections.abc import Mapping
import datetime
from functools import lru_cache
from importlib.resources import files
import logging
from typing import LiteralString, cast

import numpy as np
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from feedoscope import config
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

SCORE_UPDATE_BATCH_SIZE = 1000


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


def _parse_embedding_bytes(raw_bytes: bytes | memoryview) -> np.ndarray:
    """Convert stored raw float32 bytes into a dense NumPy embedding."""
    buffer = raw_bytes.tobytes() if isinstance(raw_bytes, memoryview) else raw_bytes
    return np.frombuffer(buffer, dtype=np.float32).copy()


def _format_embedding_bytes(embedding: np.ndarray) -> bytes:
    """Serialize a dense embedding as raw float32 bytes for storage."""
    return np.asarray(embedding, dtype=np.float32).tobytes()


async def get_articles_for_embedding_warm(
    after_article_id: int,
    batch_size: int,
) -> list[Article]:
    """Return one ascending entry batch for the resumable embedding warmer."""
    query = _get_query_from_file("get_articles_for_embedding_warm.sql")
    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            {"after_article_id": after_article_id, "batch_size": batch_size},
        )
        data = await cur.fetchall()
    return [Article(**article) for article in data]


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


async def clear_downvoted_unread_scores() -> int:
    """Clear stale scores from unread articles excluded after a downvote."""
    query = _get_query_from_file("clear_downvoted_unread_scores.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(query)
        cleared = cur.rowcount

    logger.info(f"Cleared scores from {cleared} downvoted unread articles.")
    return cleared


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


async def get_unread_articles_by_age(
    min_age_days: int, max_age_days: int
) -> list[Article]:
    """Get every unread article in the requested age range.

    Args:
        min_age_days: Youngest included article age in days.
        max_age_days: Oldest excluded article age in days.

    Returns:
        Unread, non-downvoted, unstarred articles in the age range.
    """
    query = _get_query_from_file("get_unread_articles_by_age.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            {
                "min_age_days": min_age_days,
                "max_age_days": max_age_days,
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
    if len(article_ids) != len(scores):
        raise ValueError("article_ids and scores must align")

    scores_query = _get_query_from_file("update_scores.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        for start in range(0, len(article_ids), SCORE_UPDATE_BATCH_SIZE):
            end = start + SCORE_UPDATE_BATCH_SIZE
            rows = [
                {"score": score, "int_id": int_id}
                for score, int_id in zip(
                    scores[start:end], article_ids[start:end], strict=True
                )
            ]
            await cur.executemany(scores_query, rows)
            await conn.commit()
            logger.info(
                f"Updated article scores {start + 1}-{min(end, len(article_ids))}/"
                f"{len(article_ids)}."
            )


async def insert_model_eval(
    eval_date: datetime.date,
    model_name: str,
    evaluation_model: str,
    training_counts: dict[str, int],
    eval_counts: dict[str, int],
    metrics: Mapping[str, float | None],
) -> None:
    """Insert a model evaluation result.

    Args:
        eval_date: Date the evaluation ran.
        model_name: Name of the evaluated model section.
        evaluation_model: Encoder and prediction algorithm used for evaluation.
        training_counts: Training sample counts by class.
        eval_counts: Evaluation sample counts by class.
        metrics: Classification metrics from the evaluation run.

    """
    query = _get_query_from_file("insert_model_eval.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            {
                "eval_date": eval_date,
                "model_name": model_name,
                "evaluation_model": evaluation_model,
                "training": Jsonb(training_counts),
                "eval_counts": Jsonb(eval_counts),
                "metrics_accuracy": metrics.get("accuracy"),
                "metrics_precision": metrics.get(
                    "precision", metrics.get("precision_at_50")
                ),
                "metrics_recall": metrics.get("recall"),
                "metrics_f1": metrics.get("f1", metrics.get("macro_f1")),
                "metrics_roc_auc": metrics.get(
                    "roc_auc", metrics.get("long_lived_auc")
                ),
                "metrics_average_precision": metrics.get("average_precision"),
                "metrics_log_loss": metrics.get("log_loss"),
                "metrics_rps": metrics.get("rps"),
                "metrics_weighted_kappa": metrics.get("weighted_kappa"),
                "metrics_log_duration_mae": metrics.get("log_duration_mae"),
                "metrics_super_important_average_precision": metrics.get(
                    "super_important_average_precision"
                ),
                "metrics_relevance_average_precision": metrics.get(
                    "relevance_average_precision"
                ),
                "metrics_recall_at_10": metrics.get("recall_at_10"),
                "metrics_recall_at_25": metrics.get("recall_at_25"),
                "metrics_recall_at_50": metrics.get("recall_at_50"),
                "metrics_super_important_bonus": metrics.get("super_important_bonus"),
            },
        )

    logger.info(f"Inserted {model_name} evaluation result for {eval_date}.")


async def get_relevance_embeddings(
    article_ids: list[int],
    model_name: str,
    max_length: int,
    text_prep_mode: str,
    prep_version: int,
) -> dict[int, tuple[str, np.ndarray]]:
    """Fetch cached relevance embeddings for one embedding configuration.

    Args:
        article_ids: Article IDs to look up.
        model_name: Encoder repository or identifier.
        max_length: Token budget used to prepare the text.
        text_prep_mode: Relevance text preparation mode.
        prep_version: Explicit cache-busting version for prep logic.

    Returns:
        Mapping of article ID to ``(text_hash, embedding)``.

    """
    if not article_ids:
        return {}

    query = _get_query_from_file("get_relevance_embeddings.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            query,
            {
                "article_ids": article_ids,
                "model_name": model_name,
                "max_length": max_length,
                "text_prep_mode": text_prep_mode,
                "prep_version": prep_version,
            },
        )
        data = await cur.fetchall()

    return {
        row["article_id"]: (
            row["text_hash"],
            _parse_embedding_bytes(row["embedding"]),
        )
        for row in data
    }


async def upsert_relevance_embeddings(
    rows: list[tuple[int, str, np.ndarray]],
    model_name: str,
    max_length: int,
    text_prep_mode: str,
    prep_version: int,
) -> None:
    """Insert or replace cached relevance embeddings for one configuration.

    Args:
        rows: ``(article_id, text_hash, embedding)`` rows to upsert.
        model_name: Encoder repository or identifier.
        max_length: Token budget used to prepare the text.
        text_prep_mode: Relevance text preparation mode.
        prep_version: Explicit cache-busting version for prep logic.

    """
    if not rows:
        return

    query = _get_query_from_file("upsert_relevance_embeddings.sql")

    async with global_pool.connection() as conn, conn.cursor() as cur:
        await cur.executemany(
            query,
            [
                {
                    "article_id": article_id,
                    "model_name": model_name,
                    "max_length": max_length,
                    "text_prep_mode": text_prep_mode,
                    "prep_version": prep_version,
                    "text_hash": text_hash,
                    "embedding": _format_embedding_bytes(embedding),
                }
                for article_id, text_hash, embedding in rows
            ],
        )
