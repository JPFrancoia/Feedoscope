import asyncio
from contextlib import AbstractAsyncContextManager
import datetime
from typing import Any

import numpy as np
import pytest

from feedoscope import eval_models
from feedoscope.data_registry import data_registry as dr


def test_perfect_freshness_metrics() -> None:
    metrics = eval_models.compute_freshness_metrics(np.arange(3), np.eye(3))

    assert metrics == {
        "rps": 0.0,
        "macro_f1": 1.0,
        "weighted_kappa": 1.0,
        "log_duration_mae": 0.0,
        "long_lived_auc": 1.0,
    }


def test_freshness_rps_is_normalized_across_two_boundaries() -> None:
    probabilities = np.eye(3)[[2, 1, 0]]

    metrics = eval_models.compute_freshness_metrics(np.arange(3), probabilities)

    assert metrics["rps"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["weighted_kappa"] == pytest.approx(-1.0)
    assert metrics["log_duration_mae"] == pytest.approx(
        2 * abs(np.log(365) - np.log(7)) / 3
    )
    assert metrics["long_lived_auc"] == 0.25


def test_freshness_long_lived_auc_is_nullable() -> None:
    metrics = eval_models.compute_freshness_metrics(
        np.array([0, 1]), np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    )

    assert metrics["long_lived_auc"] is None


def test_freshness_weighted_kappa_is_nullable() -> None:
    metrics = eval_models.compute_freshness_metrics(
        np.array([0, 0]), np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )

    assert metrics["weighted_kappa"] is None


def test_freshness_eval_uses_newest_rows_for_holdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        (article_id, label, "bootstrap")
        for article_id, label in enumerate([0, 1, 2, 1, 2])
    ]
    captured: dict[str, Any] = {}

    async def get_data() -> list[tuple[int, int, str]]:
        return rows

    async def encode_articles(
        articles: list[int], *args: object, **kwargs: object
    ) -> np.ndarray:
        captured["articles"] = articles
        return np.arange(10, dtype=float).reshape(5, 2)

    def fit_classifiers(embeddings: np.ndarray, labels: np.ndarray) -> list[object]:
        captured["train_embeddings"] = embeddings
        captured["train_labels"] = labels
        return [object(), object()]

    def bucket_probabilities(
        embeddings: np.ndarray, classifiers: list[object]
    ) -> np.ndarray:
        captured["eval_embeddings"] = embeddings
        return np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

    async def save_eval_results(**kwargs: object) -> None:
        captured["saved"] = kwargs

    monkeypatch.setattr(eval_models.config, "VALIDATION_SIZE", 2)
    monkeypatch.setattr(dr, "get_semantic_freshness_training_data", get_data)
    monkeypatch.setattr(
        eval_models.relevance_embedding,
        "load_encoder",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        eval_models.relevance_embedding, "encode_articles", encode_articles
    )
    monkeypatch.setattr(
        eval_models.semantic_freshness_embedding, "fit_classifiers", fit_classifiers
    )
    monkeypatch.setattr(
        eval_models.semantic_freshness_embedding,
        "bucket_probabilities",
        bucket_probabilities,
    )
    monkeypatch.setattr(eval_models, "save_eval_results", save_eval_results)

    asyncio.run(eval_models.eval_freshness(eval_models.torch.device("cpu")))

    assert captured["articles"] == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(captured["train_labels"], [0, 1, 2])
    np.testing.assert_array_equal(
        captured["train_embeddings"], np.arange(6).reshape(3, 2)
    )
    np.testing.assert_array_equal(
        captured["eval_embeddings"], np.arange(6, 10).reshape(2, 2)
    )
    assert captured["saved"]["training_counts"] == {
        "fresh_d": 1,
        "fresh_m": 1,
        "fresh_y": 1,
    }
    assert captured["saved"]["eval_counts"] == {
        "fresh_d": 0,
        "fresh_m": 1,
        "fresh_y": 1,
    }


def test_evaluation_model_label_identifies_each_prompted_head() -> None:
    assert (
        eval_models.evaluation_model_label("Relevance")
        == "EmbeddingGemma 300M prompted + MLP"
    )
    assert (
        eval_models.evaluation_model_label("Urgency")
        == "EmbeddingGemma 300M prompted + logistic regression"
    )


def test_save_eval_results_persists_evaluation_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    history_path = tmp_path / "eval_history.json"
    captured: dict[str, object] = {}

    async def insert_model_eval(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(eval_models, "EVAL_HISTORY_PATH", str(history_path))
    monkeypatch.setattr(eval_models.dr, "insert_model_eval", insert_model_eval)

    asyncio.run(
        eval_models.save_eval_results(
            "Relevance",
            {"good": 1},
            {"bad": 1},
            {"f1": 0.5},
        )
    )

    record = history_path.read_text()
    assert '"evaluation_model": "EmbeddingGemma 300M prompted + MLP"' in record
    assert captured["evaluation_model"] == "EmbeddingGemma 300M prompted + MLP"


def test_insert_model_eval_maps_nullable_model_specific_metrics(
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
            "Freshness",
            "EmbeddingGemma 300M + logistic regression",
            {"fresh_d": 1},
            {"fresh_y": 1},
            {
                "macro_f1": 0.75,
                "long_lived_auc": None,
                "rps": 0.2,
                "weighted_kappa": 0.5,
                "log_duration_mae": 0.3,
            },
        )
    )

    assert captured["evaluation_model"] == "EmbeddingGemma 300M + logistic regression"
    assert captured["metrics_accuracy"] is None
    assert captured["metrics_f1"] == 0.75
    assert captured["metrics_roc_auc"] is None
    assert captured["metrics_rps"] == 0.2
    assert captured["metrics_weighted_kappa"] == 0.5
    assert captured["metrics_log_duration_mae"] == 0.3

    captured.clear()
    asyncio.run(
        dr.insert_model_eval(
            datetime.date(2026, 8, 1),
            "Relevance",
            "EmbeddingGemma 300M + logistic regression",
            {"good": 1},
            {"bad": 1},
            {
                "accuracy": 0.1,
                "precision": 0.2,
                "recall": 0.3,
                "f1": 0.4,
                "roc_auc": 0.5,
                "average_precision": 0.6,
                "log_loss": 0.7,
            },
        )
    )

    assert captured["metrics_accuracy"] == 0.1
    assert captured["metrics_precision"] == 0.2
    assert captured["metrics_recall"] == 0.3
    assert captured["metrics_f1"] == 0.4
    assert captured["metrics_roc_auc"] == 0.5
    assert captured["metrics_average_precision"] == 0.6
    assert captured["metrics_log_loss"] == 0.7
    assert captured["metrics_rps"] is None
    assert captured["metrics_weighted_kappa"] is None
    assert captured["metrics_log_duration_mae"] is None
    assert captured["metrics_super_important_average_precision"] is None

    captured.clear()
    asyncio.run(
        dr.insert_model_eval(
            datetime.date(2026, 8, 1),
            "Super-important",
            "EmbeddingGemma 300M + logistic regression",
            {"super_important": 10, "ordinary_read": 20, "bad": 30},
            {"super_important": 5, "ordinary_read": 10, "bad": 15},
            {
                "super_important_average_precision": 0.8,
                "relevance_average_precision": 0.9,
                "recall_at_10": 0.2,
                "recall_at_25": 0.4,
                "recall_at_50": 0.6,
                "super_important_bonus": 0.5,
            },
        )
    )

    assert captured["metrics_super_important_average_precision"] == 0.8
    assert captured["metrics_relevance_average_precision"] == 0.9
    assert captured["metrics_recall_at_10"] == 0.2
    assert captured["metrics_recall_at_25"] == 0.4
    assert captured["metrics_recall_at_50"] == 0.6
    assert captured["metrics_super_important_bonus"] == 0.5
    assert captured["metrics_accuracy"] is None
