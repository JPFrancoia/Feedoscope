from datetime import datetime, timedelta, timezone
import os

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://test")

from feedoscope import main


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
