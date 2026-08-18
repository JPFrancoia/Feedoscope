import asyncio
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pytest

from feedoscope import llm_infer, relevance_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article


def test_title_and_body_embedding_contracts_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article = cast(Article, SimpleNamespace(title="Title", content="Body"))
    monkeypatch.setattr(
        relevance_embedding.relevance_text,
        "prepare_single_blob",
        lambda title, body: f"{title}|{body}",
    )

    title_cache = relevance_embedding.get_cache_config("title")
    body_cache = relevance_embedding.get_cache_config("body")

    assert title_cache == {
        "model_name": "google/embeddinggemma-300m-classification-v1",
        "max_length": 512,
        "text_prep_mode": "title",
        "prep_version": 2,
        "prompt": "task: classification | query: ",
    }
    assert body_cache == {
        **title_cache,
        "max_length": 2048,
        "text_prep_mode": "body",
    }
    assert relevance_embedding.prepare_articles_text([article], "title") == [
        "task: classification | query: Title|"
    ]
    assert relevance_embedding.prepare_articles_text([article], "body") == [
        "task: classification | query: |Body"
    ]


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


def test_field_encoder_requests_title_then_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields: list[str] = []

    async def encode(*_: object, **kwargs: object) -> np.ndarray:
        fields.append(cast(str, kwargs["field"]))
        return np.array([[1.0, 0.0]])

    monkeypatch.setattr(relevance_embedding, "encode_articles", encode)

    title, body = asyncio.run(
        relevance_embedding.encode_article_fields(
            [cast(Article, SimpleNamespace())],
            cast(Any, SimpleNamespace()),
            cast(Any, SimpleNamespace()),
            cast(Any, SimpleNamespace()),
        )
    )

    assert fields == ["title", "body"]
    np.testing.assert_array_equal(title, body)


def test_word_vocabularies_fit_only_training_text() -> None:
    fit_articles = [
        cast(
            Article,
            SimpleNamespace(
                title=f"shared {'bad' if index < 2 else 'good'}",
                content=f"common {'reject' if index < 2 else 'accept'}",
            ),
        )
        for index in range(4)
    ]
    title_vectorizer = relevance_embedding.build_tfidf_vectorizer()
    body_vectorizer = relevance_embedding.build_tfidf_vectorizer()
    embeddings = np.ones((4, 2), dtype=np.float32)
    relevance_embedding.build_features(
        fit_articles,
        embeddings,
        embeddings,
        title_vectorizer,
        body_vectorizer,
        fit_vectorizers=True,
    )
    title_vocabulary = dict(title_vectorizer.vocabulary_)
    body_vocabulary = dict(body_vectorizer.vocabulary_)
    evaluation = [
        cast(
            Article,
            SimpleNamespace(title="futuretitle", content="futurebody"),
        )
    ]

    features = relevance_embedding.build_features(
        evaluation,
        np.ones((1, 2), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        title_vectorizer,
        body_vectorizer,
    )

    assert title_vectorizer.vocabulary_ == title_vocabulary
    assert body_vectorizer.vocabulary_ == body_vocabulary
    assert features.shape[1] == 4 + len(title_vocabulary) + len(body_vocabulary)


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


def test_relevance_logistic_regression_receives_sample_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, np.ndarray | None] = {}

    class RecordingLogisticRegression:
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
    monkeypatch.setattr(
        relevance_embedding,
        "LogisticRegression",
        RecordingLogisticRegression,
    )

    relevance_embedding.fit_classifier(
        np.array([[1.0], [2.0]]),
        np.array([1, 0]),
        sample_weights=sample_weights,
    )

    assert captured["sample_weight"] is not None
    np.testing.assert_array_equal(captured["sample_weight"], sample_weights)


@pytest.mark.parametrize(
    ("config_name", "value"),
    (("IMPORTANT_ARTICLE_WEIGHT", 7), ("RELEVANCE_LINEAR_C", 1)),
)
def test_model_family_changes_with_classifier_config(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    value: float,
) -> None:
    original = relevance_embedding.get_model_family_prefix()

    monkeypatch.setattr(relevance_embedding.config, config_name, value)

    assert relevance_embedding.get_model_family_prefix() != original


def test_artifact_round_trip_and_rejects_incompatible_models(tmp_path: Path) -> None:
    articles = [
        cast(
            Article,
            SimpleNamespace(
                title=f"{'bad' if index < 2 else 'good'} common",
                content=f"{'reject' if index < 2 else 'accept'} shared",
            ),
        )
        for index in range(4)
    ]
    title_embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]])
    body_embeddings = title_embeddings[:, ::-1]
    labels = np.array([0, 0, 1, 1])
    relevance_model = relevance_embedding.fit_relevance_model(
        articles,
        title_embeddings,
        body_embeddings,
        labels,
        np.ones(4),
    )
    relevance_embedding.save_relevance_artifact(
        str(tmp_path),
        relevance_model,
        {"good": 2, "bad": 2},
    )

    loaded = relevance_embedding.load_relevance_artifact(str(tmp_path))
    expected = relevance_embedding.build_features(
        articles,
        title_embeddings,
        body_embeddings,
        relevance_model.title_vectorizer,
        relevance_model.body_vectorizer,
    )
    actual = relevance_embedding.build_features(
        articles,
        title_embeddings,
        body_embeddings,
        loaded.title_vectorizer,
        loaded.body_vectorizer,
    )
    np.testing.assert_allclose(
        loaded.classifier.predict_proba(actual),
        relevance_model.classifier.predict_proba(expected),
    )

    artifact_path = tmp_path / relevance_embedding.ARTIFACT_FILENAME
    artifact = joblib.load(artifact_path)
    artifact["relevance_classifier"].set_params(C=1)
    joblib.dump(artifact, artifact_path)
    with pytest.raises(RuntimeError, match="not compatible"):
        relevance_embedding.load_relevance_artifact(str(tmp_path))

    joblib.dump(
        {
            "relevance_classifier": relevance_model.classifier,
            "metadata": relevance_embedding.build_metadata({"good": 2, "bad": 2}, 2),
        },
        artifact_path,
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


def test_inference_scores_come_from_the_complete_relevance_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles = [
        cast(Article, SimpleNamespace(article_id=1, title="First")),
        cast(Article, SimpleNamespace(article_id=2, title="Second")),
    ]
    predicted_with: list[str] = []

    async def predict(*args: object, **__: object) -> np.ndarray:
        predicted_with.append(cast(str, args[3]))
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
    monkeypatch.setattr(relevance_embedding, "predict_probabilities", predict)

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


def test_unread_score_cleanup_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[tuple[str, dict[str, float] | None]] = []

    class Cursor(AbstractAsyncContextManager["Cursor"]):
        rowcount = 7

        async def __aexit__(self, *args: object) -> None:
            return None

        async def execute(
            self,
            query: str,
            params: dict[str, float] | None = None,
        ) -> None:
            executed.append((query, params))

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
    assert asyncio.run(dr.clear_expired_unread_scores(42.5)) == 7
    assert executed == [
        ("clear_downvoted_unread_scores.sql", None),
        ("clear_expired_unread_scores.sql", {"score_horizon_days": 42.5}),
    ]
