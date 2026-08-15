import asyncio
from contextlib import AbstractAsyncContextManager
import datetime
from typing import Any

import numpy as np
from psycopg.types.json import Jsonb
import pytest

from feedoscope import eval_models
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article


def article(article_id: int, date_entered: datetime.datetime, vote: int) -> Article:
    return Article(
        article_id=article_id,
        title=f"Article {article_id}",
        starred=False,
        feed_name="Feed",
        content="Content",
        link=f"https://example.com/{article_id}",
        author="Author",
        date_entered=date_entered,
        last_read=date_entered,
        tags=[],
        vote=vote,
        status="read" if vote >= 0 else "unread",
    )


def test_forward_holdout_uses_a_strict_time_boundary() -> None:
    start = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    all_good = [
        article(index + 1, start + datetime.timedelta(days=index), 0)
        for index in range(6)
    ]
    all_bad = [
        article(index + 101, start + datetime.timedelta(days=index), -1)
        for index in range(4)
    ]

    good_fit, bad_fit, eval_good, eval_bad, cutoff = eval_models.build_forward_holdout(
        all_good, all_bad, validation_size=2
    )

    assert [item.article_id for item in good_fit] == [1, 2]
    assert [item.article_id for item in bad_fit] == [101, 102]
    assert [item.article_id for item in eval_good] == [5, 6]
    assert [item.article_id for item in eval_bad] == [103, 104]
    assert cutoff == start + datetime.timedelta(days=2)
    cutoff_ids = {
        item.article_id for item in all_good + all_bad if item.date_entered == cutoff
    }
    assert cutoff_ids == {3, 103}
    assert cutoff_ids.isdisjoint(item.article_id for item in good_fit + bad_fit)
    assert max(item.date_entered for item in good_fit + bad_fit) < min(
        item.date_entered for item in eval_good + eval_bad
    )
    assert {item.article_id for item in good_fit + bad_fit}.isdisjoint(
        item.article_id for item in eval_good + eval_bad
    )


def test_perfect_relevance_ranking_metrics() -> None:
    labels = np.array([1] * 50 + [0] * 50)
    probabilities = np.arange(100, 0, -1, dtype=float)

    metrics = eval_models.compute_relevance_metrics(labels, probabilities)

    assert metrics == {
        "roc_auc": 1.0,
        "average_precision": pytest.approx(1.0),
        "precision_at_50": 1.0,
    }


def test_relevance_precision_counts_top_results() -> None:
    labels = np.array([1] * 47 + [0] * 3 + [1] * 53 + [0] * 97)
    metrics = eval_models.compute_relevance_metrics(
        labels, np.arange(200, 0, -1, dtype=float)
    )

    assert metrics["precision_at_50"] == 0.94


def test_relevance_precision_is_invariant_to_cutoff_tie_order() -> None:
    probabilities = np.array([2.0] * 45 + [1.0] * 10 + [0.0] * 5)
    labels = np.array([1] * 40 + [0] * 5 + [1] * 4 + [0] * 6 + [1] * 5)
    reordered_labels = np.array([1] * 40 + [0] * 5 + [0] * 6 + [1] * 4 + [1] * 5)

    metrics = eval_models.compute_relevance_metrics(labels, probabilities)
    reordered_metrics = eval_models.compute_relevance_metrics(
        reordered_labels, probabilities
    )

    assert metrics["precision_at_50"] == pytest.approx(0.84)
    assert reordered_metrics["precision_at_50"] == metrics["precision_at_50"]


def test_relevance_metrics_are_nullable_without_candidates() -> None:
    assert eval_models.compute_relevance_metrics(np.array([]), np.array([])) == {
        "roc_auc": None,
        "average_precision": None,
        "precision_at_50": None,
    }


def test_save_eval_results_persists_relevance_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    history_path = tmp_path / "eval_history.json"
    captured: dict[str, object] = {}

    async def insert_model_eval(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(eval_models, "EVAL_HISTORY_PATH", str(history_path))
    monkeypatch.setattr(eval_models.dr, "insert_model_eval", insert_model_eval)

    asyncio.run(eval_models.save_eval_results({"good": 1}, {"bad": 1}, {"f1": 0.5}))

    assert '"model": "Relevance"' in history_path.read_text()
    assert captured["model_name"] == "Relevance"
    assert captured["evaluation_model"] == (
        "EmbeddingGemma 300M prompted + weighted logistic regression "
        "(forward AP + Precision@50)"
    )


def test_model_eval_metrics_preserve_canonical_keys() -> None:
    assert dr.normalize_model_eval_metrics(
        {
            "precision": 0.8,
            "precision_at_50": 0.94,
            "future_metric": 0.7,
            "missing_metric": None,
        }
    ) == {
        "precision": 0.8,
        "precision_at_50": 0.94,
        "future_metric": 0.7,
    }


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_model_eval_metrics_reject_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        dr.normalize_model_eval_metrics({"metric": value})


@pytest.mark.parametrize("value", (True, "0.5"))
def test_model_eval_metrics_reject_non_numeric_values(value: object) -> None:
    with pytest.raises(TypeError, match="must be numeric"):
        dr.normalize_model_eval_metrics({"metric": value})  # type: ignore[dict-item]


def test_model_eval_metrics_reject_invalid_keys() -> None:
    with pytest.raises(ValueError, match="Invalid model metric key"):
        dr.normalize_model_eval_metrics({"Precision@50": 0.94})


def test_insert_model_eval_persists_canonical_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Cursor(AbstractAsyncContextManager["Cursor"]):
        async def __aexit__(self, *args: object) -> None:
            return None

        async def execute(self, query: str, parameters: dict[str, object]) -> None:
            captured.update(parameters)

    class Connection(AbstractAsyncContextManager["Connection"]):
        async def __aexit__(self, *args: object) -> None:
            return None

        def cursor(self) -> Cursor:
            return Cursor()

    class Pool:
        def connection(self) -> Connection:
            return Connection()

    monkeypatch.setattr(dr, "global_pool", Pool())
    monkeypatch.setattr(dr, "_get_query_from_file", lambda filename: filename)

    asyncio.run(
        dr.insert_model_eval(
            datetime.date(2026, 8, 1),
            "Relevance",
            eval_models.EVALUATION_MODEL,
            {"good": 1},
            {"bad": 1},
            {"roc_auc": 0.5, "average_precision": 0.6, "precision_at_50": 0.94},
        )
    )

    metrics = captured["metrics"]
    assert isinstance(metrics, Jsonb)
    assert metrics.obj == {
        "roc_auc": 0.5,
        "average_precision": 0.6,
        "precision_at_50": 0.94,
    }
    assert not any(key.startswith("metrics_") for key in captured)
