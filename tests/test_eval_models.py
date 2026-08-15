import asyncio
from contextlib import AbstractAsyncContextManager
import datetime
from typing import Any

import numpy as np
import pytest

from feedoscope import eval_models
from feedoscope.data_registry import data_registry as dr


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
    assert captured["evaluation_model"] == eval_models.EVALUATION_MODEL


def test_insert_model_eval_maps_relevance_metrics(
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

    assert captured["metrics_precision"] == 0.94
    assert captured["metrics_roc_auc"] == 0.5
    assert captured["metrics_average_precision"] == 0.6
    assert captured["metrics_rps"] is None
