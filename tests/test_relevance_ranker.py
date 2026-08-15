import asyncio
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from transformers import PreTrainedTokenizerBase

from feedoscope import llm_infer, relevance_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article


def test_prompted_embedding_key_and_text_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = cast(
        PreTrainedTokenizerBase,
        SimpleNamespace(encode=lambda text, **_: text.split()),
    )
    monkeypatch.setattr(
        relevance_embedding.relevance_text,
        "prepare_articles_text",
        lambda *_, **kwargs: [f"article-budget-{kwargs['max_length']}"],
    )

    cache = relevance_embedding.get_cache_config()
    texts = relevance_embedding.prepare_articles_text(
        [cast(Article, SimpleNamespace())],
        tokenizer,
    )

    assert cache["model_name"] == "google/embeddinggemma-300m-classification-v1"
    assert cache["prompt"] == "task: classification | query: "
    assert texts == ["task: classification | query: article-budget-2044"]


def test_explicit_preference_label_uses_star_or_upvote() -> None:
    assert relevance_embedding.is_important(
        cast(Article, SimpleNamespace(status="read", vote=1, starred=False))
    )
    assert relevance_embedding.is_important(
        cast(Article, SimpleNamespace(status="read", vote=0, starred=True))
    )
    assert not relevance_embedding.is_important(
        cast(Article, SimpleNamespace(status="read", vote=0, starred=False))
    )
    assert not relevance_embedding.is_important(
        cast(Article, SimpleNamespace(status="read", vote=-1, starred=True))
    )
    assert not relevance_embedding.is_important(
        cast(Article, SimpleNamespace(status="unread", vote=1, starred=False))
    )


def test_relevance_training_weights_important_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(relevance_embedding.config, "IMPORTANT_ARTICLE_WEIGHT", 20.0)
    articles = [
        cast(Article, SimpleNamespace(status="read", vote=1, starred=False)),
        cast(Article, SimpleNamespace(status="read", vote=0, starred=False)),
        cast(Article, SimpleNamespace(status="unread", vote=-1, starred=False)),
    ]

    sample_weights = relevance_embedding.build_relevance_sample_weights(articles)

    np.testing.assert_array_equal(sample_weights, [20.0, 1.0, 1.0])


def test_relevance_mlp_receives_sample_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, np.ndarray | None] = {}

    class RecordingMLP:
        def __init__(self, **_: object) -> None:
            pass

        def fit(
            self,
            _: np.ndarray,
            __: np.ndarray,
            sample_weight: np.ndarray | None = None,
        ) -> None:
            captured["sample_weight"] = sample_weight

    sample_weights = np.array([20.0, 1.0])
    monkeypatch.setattr(relevance_embedding, "MLPClassifier", RecordingMLP)

    relevance_embedding.fit_classifier(
        np.array([[1.0], [2.0]]),
        np.array([1, 0]),
        sample_weights=sample_weights,
    )

    assert captured["sample_weight"] is not None
    np.testing.assert_array_equal(captured["sample_weight"], sample_weights)


def test_model_family_changes_with_important_article_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = relevance_embedding.get_model_family_prefix()

    monkeypatch.setattr(relevance_embedding.config, "IMPORTANT_ARTICLE_WEIGHT", 7)

    assert relevance_embedding.get_model_family_prefix() != original


def test_artifact_round_trip_and_rejects_old_shape(tmp_path: Path) -> None:
    embeddings = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])
    relevance_classifier = MLPClassifier(random_state=42).fit(embeddings, labels)

    relevance_embedding.save_relevance_artifact(
        str(tmp_path),
        relevance_classifier,
        {"good": 2, "bad": 2},
    )

    loaded = relevance_embedding.load_relevance_artifact(str(tmp_path))
    np.testing.assert_allclose(
        loaded.predict_proba(embeddings),
        relevance_classifier.predict_proba(embeddings),
    )

    joblib.dump(
        relevance_classifier,
        tmp_path / relevance_embedding.ARTIFACT_FILENAME,
    )
    with pytest.raises(RuntimeError, match="not compatible"):
        relevance_embedding.load_relevance_artifact(str(tmp_path))


def test_latest_model_skips_and_cleans_incomplete_training_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    complete = tmp_path / "models" / "ranker_2026_07_01"
    incomplete = tmp_path / "models" / "ranker_2026_08_01"
    complete.mkdir(parents=True)
    incomplete.mkdir()
    (complete / relevance_embedding.ARTIFACT_FILENAME).write_bytes(b"model")

    selected = llm_infer.find_latest_model(
        "ranker_",
        clean_old_models=False,
        required_filename=relevance_embedding.ARTIFACT_FILENAME,
    )

    assert selected == "models/ranker_2026_07_01"
    assert incomplete.exists()

    llm_infer.find_latest_model(
        "ranker_",
        clean_old_models=True,
        required_filename=relevance_embedding.ARTIFACT_FILENAME,
    )
    assert complete.exists()
    assert not incomplete.exists()


def test_inference_scores_come_from_the_relevance_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles = [
        cast(Article, SimpleNamespace(article_id=1, title="First")),
        cast(Article, SimpleNamespace(article_id=2, title="Second")),
    ]
    predicted_with: list[str] = []

    async def encode(*_: object, **__: object) -> np.ndarray:
        return np.array([[1.0], [2.0]])

    def predict(_: np.ndarray, classifier: str) -> np.ndarray:
        predicted_with.append(classifier)
        return np.array([0.8, 0.3])

    monkeypatch.setattr(llm_infer.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(llm_infer.config, "ALLOW_INFERENCE_WO_GPU", True)
    monkeypatch.setattr(
        llm_infer,
        "find_latest_model",
        lambda *_, **__: "models/relevance-test",
    )
    monkeypatch.setattr(
        relevance_embedding,
        "load_relevance_artifact",
        lambda _: "relevance",
    )
    monkeypatch.setattr(
        relevance_embedding,
        "load_encoder",
        lambda _: ("tokenizer", "encoder"),
    )
    monkeypatch.setattr(relevance_embedding, "encode_articles", encode)
    monkeypatch.setattr(
        relevance_embedding,
        "predict_probabilities_from_embeddings",
        predict,
    )

    results = asyncio.run(llm_infer.infer(articles))

    assert results.scores == [80.0, 30.0]
    assert predicted_with == ["relevance"]


def test_score_updates_commit_bounded_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches: list[list[dict[str, object]]] = []
    commits: list[None] = []

    class Cursor(AbstractAsyncContextManager["Cursor"]):
        async def __aexit__(self, *args: object) -> None:
            return None

        async def executemany(
            self,
            query: str,
            rows: list[dict[str, object]],
        ) -> None:
            assert query == "update_scores.sql"
            batches.append(rows)

    class Connection(AbstractAsyncContextManager["Connection"]):
        async def __aexit__(self, *args: object) -> None:
            return None

        def cursor(self) -> Cursor:
            return Cursor()

        async def commit(self) -> None:
            commits.append(None)

    class Pool:
        def connection(self) -> Connection:
            return Connection()

    monkeypatch.setattr(dr, "global_pool", Pool())
    monkeypatch.setattr(dr, "SCORE_UPDATE_BATCH_SIZE", 2)
    monkeypatch.setattr(dr, "_get_query_from_file", lambda filename: filename)

    asyncio.run(
        dr.update_scores(
            article_ids=[1, 2, 3, 4, 5],
            article_titles=["1", "2", "3", "4", "5"],
            scores=[10, 20, 30, 40, 50],
        )
    )

    assert batches == [
        [{"score": 10, "int_id": 1}, {"score": 20, "int_id": 2}],
        [{"score": 30, "int_id": 3}, {"score": 40, "int_id": 4}],
        [{"score": 50, "int_id": 5}],
    ]
    assert len(commits) == 3

    with pytest.raises(ValueError, match="must align"):
        asyncio.run(
            dr.update_scores(
                article_ids=[1],
                article_titles=["1"],
                scores=[10, 20],
            )
        )


def test_downvoted_unread_scores_are_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []

    class Cursor(AbstractAsyncContextManager["Cursor"]):
        rowcount = 7

        async def __aexit__(self, *args: object) -> None:
            return None

        async def execute(self, query: str) -> None:
            executed.append(query)

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

    assert asyncio.run(dr.clear_downvoted_unread_scores()) == 7
    assert executed == ["clear_downvoted_unread_scores.sql"]
