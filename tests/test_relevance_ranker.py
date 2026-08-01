import asyncio
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from feedoscope import eval_models, llm_infer, relevance_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article, RelevanceInferenceResults


def test_explicit_preference_label_uses_star_or_upvote() -> None:
    assert relevance_embedding.is_super_important(
        cast(Article, SimpleNamespace(status="read", vote=1, starred=False))
    )
    assert relevance_embedding.is_super_important(
        cast(Article, SimpleNamespace(status="read", vote=0, starred=True))
    )
    assert not relevance_embedding.is_super_important(
        cast(Article, SimpleNamespace(status="read", vote=0, starred=False))
    )
    assert not relevance_embedding.is_super_important(
        cast(Article, SimpleNamespace(status="read", vote=-1, starred=True))
    )
    assert not relevance_embedding.is_super_important(
        cast(Article, SimpleNamespace(status="unread", vote=1, starred=False))
    )


def test_combined_score_requires_relevance_and_preference() -> None:
    combined = relevance_embedding.combine_probabilities(
        np.array([0.9, 0.7]), np.array([0.1, 0.8])
    )

    np.testing.assert_allclose(combined, [0.09, 0.56])
    with pytest.raises(ValueError, match="must align"):
        relevance_embedding.combine_probabilities(np.array([0.9]), np.array([0.1, 0.8]))


def test_two_head_artifact_round_trip_and_rejects_old_shape(tmp_path: Path) -> None:
    embeddings = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])
    relevance_classifier = LogisticRegression().fit(embeddings, labels)
    super_important_classifier = LogisticRegression().fit(embeddings, labels)

    relevance_embedding.save_two_head_artifact(
        str(tmp_path),
        relevance_classifier,
        super_important_classifier,
        {"good": 2, "bad": 2, "super_important": 1, "ordinary_read": 1},
    )

    loaded_relevance, loaded_super_important = (
        relevance_embedding.load_two_head_artifact(str(tmp_path))
    )
    np.testing.assert_allclose(
        loaded_relevance.predict_proba(embeddings),
        relevance_classifier.predict_proba(embeddings),
    )
    np.testing.assert_allclose(
        loaded_super_important.predict_proba(embeddings),
        super_important_classifier.predict_proba(embeddings),
    )

    joblib.dump(
        relevance_classifier,
        tmp_path / relevance_embedding.TWO_HEAD_ARTIFACT_FILENAME,
    )
    with pytest.raises(RuntimeError, match="not compatible"):
        relevance_embedding.load_two_head_artifact(str(tmp_path))


def test_latest_model_skips_and_cleans_incomplete_training_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    complete = tmp_path / "models" / "ranker_2026_07_01"
    incomplete = tmp_path / "models" / "ranker_2026_08_01"
    complete.mkdir(parents=True)
    incomplete.mkdir()
    (complete / relevance_embedding.TWO_HEAD_ARTIFACT_FILENAME).write_bytes(b"model")

    selected = llm_infer.find_latest_model(
        "ranker_",
        clean_old_models=False,
        required_filename=relevance_embedding.TWO_HEAD_ARTIFACT_FILENAME,
    )

    assert selected == "models/ranker_2026_07_01"
    assert incomplete.exists()

    llm_infer.find_latest_model(
        "ranker_",
        clean_old_models=True,
        required_filename=relevance_embedding.TWO_HEAD_ARTIFACT_FILENAME,
    )
    assert complete.exists()
    assert not incomplete.exists()


def test_chronological_split_excludes_unsettled_labels() -> None:
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    articles = [
        cast(
            Article,
            SimpleNamespace(
                article_id=index,
                last_read=now - timedelta(days=100 - index),
            ),
        )
        for index in range(10)
    ]
    articles.append(
        cast(Article, SimpleNamespace(article_id=10, last_read=now - timedelta(days=1)))
    )

    training, holdout = eval_models.split_super_important_eval_articles(
        articles,
        now=now,
    )

    assert [article.article_id for article in training] == list(range(8))
    assert [article.article_id for article in holdout] == [8, 9]


def test_ranking_metrics_reward_super_important_first() -> None:
    articles = [
        cast(Article, SimpleNamespace(status="read", vote=1, starred=False)),
        cast(Article, SimpleNamespace(status="read", vote=0, starred=False)),
        cast(Article, SimpleNamespace(status="unread", vote=-1, starred=False)),
    ]

    good_ranking = eval_models.compute_super_important_ranking_metrics(
        articles,
        np.array([0.9, 0.8, 0.1]),
        budgets=(1, 2),
    )
    reversed_ranking = eval_models.compute_super_important_ranking_metrics(
        articles,
        np.array([0.1, 0.8, 0.9]),
        budgets=(1, 2),
    )

    assert good_ranking["super_important_average_precision"] == 1.0
    assert good_ranking["recall_at_1"] == 1.0
    assert good_ranking["ndcg_at_2"] > reversed_ranking["ndcg_at_2"]


def test_ranking_metrics_reject_malformed_scores() -> None:
    article = cast(
        Article,
        SimpleNamespace(status="read", vote=1, starred=False),
    )

    with pytest.raises(ValueError, match="align"):
        eval_models.compute_super_important_ranking_metrics(
            [article],
            np.array([0.1, 0.2]),
        )
    with pytest.raises(ValueError, match="finite"):
        eval_models.compute_super_important_ranking_metrics(
            [article],
            np.array([np.nan]),
        )


def test_super_important_upsert_uses_model_key_and_raw_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class Cursor(AbstractAsyncContextManager["Cursor"]):
        async def __aexit__(self, *args: object) -> None:
            return None

        async def executemany(
            self,
            query: str,
            rows: list[dict[str, object]],
        ) -> None:
            assert query == "upsert_super_important_inference.sql"
            captured.extend(rows)

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
    results = RelevanceInferenceResults(
        article_ids=[7],
        article_titles=["Article"],
        scores=[42.125],
        super_important_scores=[0.625],
        model_key="artifact-v1",
    )

    asyncio.run(dr.register_super_important_inference(results))

    assert captured == [
        {
            "article_id": 7,
            "model_key": "artifact-v1",
            "super_important_score": 0.625,
        }
    ]
