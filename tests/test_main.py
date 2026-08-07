import asyncio
from datetime import datetime, timedelta, timezone
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://test")

from feedoscope import main
from feedoscope.entities import Article, RelevanceInferenceResults


def test_inference_age_range_validation() -> None:
    assert main.validate_age_range(None, None) is None
    assert main.validate_age_range(0, 30) == (0, 30)
    assert main.validate_age_range(30, 90) == (30, 90)

    with pytest.raises(ValueError, match="provided together"):
        main.validate_age_range(30, None)
    with pytest.raises(ValueError, match="must satisfy"):
        main.validate_age_range(90, 30)


def test_semantic_lifetime_is_score_half_life() -> None:
    score = main.decay_relevance_score(
        original_score=100,
        date_entered=datetime.now(timezone.utc) - timedelta(days=10),
        half_life_days=10,
    )

    assert score == pytest.approx(50, abs=0.001)


@pytest.mark.parametrize(
    ("score", "expected"),
    (
        (0, 0),
        (10, 3.4511),
        (50, 20.6299),
        (90, 53.5841),
        (99, 78.4557),
        (100, 100),
    ),
)
def test_relevance_score_spreading(score: float, expected: float) -> None:
    assert main.relevance_embedding.spread_relevance_score(score) == pytest.approx(
        expected, abs=0.001
    )


def test_relevance_score_spreading_preserves_order() -> None:
    scores = range(101)
    spread_scores = [
        main.relevance_embedding.spread_relevance_score(score) for score in scores
    ]

    assert all(left < right for left, right in zip(spread_scores, spread_scores[1:]))


def test_decay_precedes_score_spreading_and_rounding() -> None:
    decayed_score = main.decay_relevance_score(
        original_score=99,
        date_entered=datetime.now(timezone.utc) - timedelta(days=10),
        half_life_days=10,
    )

    assert decayed_score == pytest.approx(49.5, abs=0.001)
    assert main.relevance_embedding.prepare_scores_for_storage([decayed_score]) == [20]


@pytest.mark.parametrize("score", (-1, 101, float("nan"), float("inf")))
def test_invalid_relevance_score_is_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        main.relevance_embedding.spread_relevance_score(score)


@pytest.mark.parametrize(
    "half_life_days", (None, 0.0, -1.0, float("nan"), float("inf"))
)
def test_invalid_half_life_is_rejected(half_life_days: float | None) -> None:
    assert not main.is_valid_half_life(half_life_days)
    with pytest.raises(ValueError, match="finite and positive"):
        main.decay_relevance_score(
            original_score=100,
            date_entered=datetime.now(timezone.utc),
            half_life_days=half_life_days,  # type: ignore[arg-type]
        )


def test_age_backend_uses_configured_half_life(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main.config, "RELEVANCE_DECAY_BACKEND", "age")
    monkeypatch.setattr(main.config, "AGE_DECAY_HALF_LIFE_DAYS", 7.0)

    assert main.get_decay_half_life(urgency_prob=None, expected_lifetime_days=None) == 7
    assert main.get_decay_half_life(urgency_prob=1.0, expected_lifetime_days=365) == 7


def test_age_decay_config_defaults_to_seven_days() -> None:
    env = os.environ.copy()
    env.pop("RELEVANCE_DECAY_BACKEND", None)
    env.pop("AGE_DECAY_HALF_LIFE_DAYS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from feedoscope import config; "
            "print(config.RELEVANCE_DECAY_BACKEND, config.AGE_DECAY_HALF_LIFE_DAYS)",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "age 7.0"


@pytest.mark.parametrize("value", ("0", "nan"))
def test_age_decay_config_rejects_invalid_half_life(value: str) -> None:
    env = os.environ | {"AGE_DECAY_HALF_LIFE_DAYS": value}
    result = subprocess.run(
        [sys.executable, "-c", "from feedoscope import config"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "AGE_DECAY_HALF_LIFE_DAYS must be finite and positive" in result.stderr


def test_age_backend_skips_model_decay_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article = Article(
        article_id=1,
        title="Article",
        starred=False,
        feed_name="Feed",
        content="Content",
        link="https://example.com/article",
        author="Author",
        date_entered=datetime.now(timezone.utc),
        last_read=None,
        time_sensitivity_score=None,
        tags=[],
        vote=0,
        status="unread",
    )
    relevance_results = RelevanceInferenceResults(
        article_ids=[1],
        article_titles=["Article"],
        scores=[100.0],
        model_key="test",
    )
    urgency_model_key = Mock(return_value="urgency-test")
    urgency_infer = AsyncMock()
    freshness_infer = AsyncMock()
    update_scores = AsyncMock()

    monkeypatch.setattr(main.config, "RELEVANCE_DECAY_BACKEND", "age")
    monkeypatch.setattr(main.config, "AGE_DECAY_HALF_LIFE_DAYS", 7.0)
    monkeypatch.setattr(
        main.dr,
        "global_pool",
        SimpleNamespace(open=AsyncMock(), close=AsyncMock()),
    )
    monkeypatch.setattr(main.dr, "clear_downvoted_unread_scores", AsyncMock())
    monkeypatch.setattr(main.dr, "update_scores", update_scores)
    monkeypatch.setattr(
        main.llm_infer_urgency,
        "get_articles_for_refresh",
        AsyncMock(return_value=[article]),
    )
    monkeypatch.setattr(
        main.llm_infer_urgency, "get_active_model_key", urgency_model_key
    )
    monkeypatch.setattr(main.llm_infer_urgency, "infer", urgency_infer)
    monkeypatch.setattr(main.llm_infer_semantic_freshness, "infer", freshness_infer)
    monkeypatch.setattr(
        main.llm_infer,
        "infer",
        AsyncMock(return_value=relevance_results),
    )

    asyncio.run(main.main())

    urgency_model_key.assert_not_called()
    urgency_infer.assert_not_awaited()
    freshness_infer.assert_not_awaited()
    update_scores.assert_awaited_once()


def test_semantic_backend_uses_lifetime_or_skips_decay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main.config, "RELEVANCE_DECAY_BACKEND", "semantic_freshness")

    assert main.get_decay_half_life(urgency_prob=1.0, expected_lifetime_days=20) == 20
    assert (
        main.get_decay_half_life(urgency_prob=1.0, expected_lifetime_days=None) is None
    )


def test_urgency_backend_retains_legacy_half_lives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main.config, "RELEVANCE_DECAY_BACKEND", "urgency")

    assert (
        main.get_decay_half_life(urgency_prob=0.0, expected_lifetime_days=None) == 120
    )
    assert main.get_decay_half_life(urgency_prob=1.0, expected_lifetime_days=None) == 10
    assert (
        main.get_decay_half_life(urgency_prob=None, expected_lifetime_days=20) is None
    )
