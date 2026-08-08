import asyncio
from datetime import datetime, timedelta, timezone
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://test")

from feedoscope import main
from feedoscope.entities import Article, RelevanceInferenceResults


def article(date_entered: datetime | None = None) -> Article:
    return Article(
        article_id=1,
        title="Article",
        starred=False,
        feed_name="Feed",
        content="Content",
        link="https://example.com/article",
        author="Author",
        date_entered=date_entered or datetime.now(timezone.utc),
        last_read=None,
        tags=[],
        vote=0,
        status="unread",
    )


def test_inference_age_range_validation() -> None:
    assert main.validate_age_range(None, None) is None
    assert main.validate_age_range(0, 30) == (0, 30)

    with pytest.raises(ValueError, match="provided together"):
        main.validate_age_range(30, None)
    with pytest.raises(ValueError, match="must satisfy"):
        main.validate_age_range(90, 30)


def test_age_decay_uses_configured_half_life() -> None:
    score = main.decay_relevance_score(
        original_score=100,
        date_entered=datetime.now(timezone.utc) - timedelta(days=10),
        half_life_days=10,
    )

    assert score == pytest.approx(50, abs=0.001)


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


def test_age_decay_config_defaults_to_seven_days() -> None:
    env = os.environ.copy()
    env.pop("AGE_DECAY_HALF_LIFE_DAYS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from feedoscope import config; print(config.AGE_DECAY_HALF_LIFE_DAYS)",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "7.0"


@pytest.mark.parametrize("value", ("0", "nan"))
def test_age_decay_config_rejects_invalid_half_life(value: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from feedoscope import config"],
        env=os.environ | {"AGE_DECAY_HALF_LIFE_DAYS": value},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "AGE_DECAY_HALF_LIFE_DAYS must be finite and positive" in result.stderr


def test_main_scores_articles_with_fixed_age_decay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relevance_results = RelevanceInferenceResults(
        article_ids=[1],
        article_titles=["Article"],
        scores=[100.0],
        model_key="test",
    )
    update_scores = AsyncMock()
    monkeypatch.setattr(main.config, "AGE_DECAY_HALF_LIFE_DAYS", 7.0)
    monkeypatch.setattr(
        main.dr,
        "global_pool",
        SimpleNamespace(open=AsyncMock(), close=AsyncMock()),
    )
    monkeypatch.setattr(main.dr, "clear_downvoted_unread_scores", AsyncMock())
    monkeypatch.setattr(
        main,
        "get_articles_for_scoring",
        AsyncMock(
            return_value=[article(datetime.now(timezone.utc) - timedelta(days=7))]
        ),
    )
    monkeypatch.setattr(main.dr, "update_scores", update_scores)
    monkeypatch.setattr(
        main.llm_infer, "infer", AsyncMock(return_value=relevance_results)
    )

    asyncio.run(main.main())

    update_scores.assert_awaited_once()
    assert update_scores.await_args is not None
    assert update_scores.await_args.kwargs["scores"] == [21]
