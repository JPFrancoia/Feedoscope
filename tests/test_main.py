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

    assert score == 50


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
