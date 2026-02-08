from enum import StrEnum, unique
from typing import Literal, Optional

from pydantic import AwareDatetime, BaseModel, Field


@unique
class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Article(BaseModel):
    """
    Article entity for Miniflux database.

    Main differences from TTRSS:
    - article_id is now the 'id' column from entries table (Miniflux)
    - feed_name comes from feeds.title
    - starred replaces marked (same meaning)
    - vote column indicates user preference (-1=bad, 0=neutral, 1=good)
    - tags are stored directly in entries table
    """

    article_id: int  # entries.id in Miniflux
    title: str
    starred: bool  # Replaces 'marked' from TTRSS
    feed_name: str  # From feeds.title
    content: str
    link: str  # URL in Miniflux
    author: str
    date_entered: AwareDatetime  # published_at in Miniflux
    last_read: Optional[AwareDatetime] = Field(...)  # changed_at when status='read'
    time_sensitivity_score: Optional[Literal[1, 2, 3, 4, 5]] = Field(...)
    tags: list[str]  # Directly from entries.tags array in Miniflux
    vote: int  # -1, 0, or 1 in Miniflux


class RelevanceInferenceResults(BaseModel):
    article_ids: list[int]
    article_titles: list[str]
    scores: list[int]


class TimeSensitivity(BaseModel):
    article_id: int

    # 1 is not time-sensitive at all, 5 is extremely time-sensitive
    score: Literal[1, 2, 3, 4, 5]

    confidence: ConfidenceLevel
    explanation: str


class SimplifiedTimeSensitivity(BaseModel):
    """Simplified binary time sensitivity from decoder model (Ministral-8B).

    0 = not urgent (evergreen, still relevant in one year)
    1 = urgent (ephemeral, loses relevance within a year)
    """

    article_id: int
    score: Literal[0, 1]
    explanation: str


class UrgencyInferenceResults(BaseModel):
    """Results from the distilled ModernBERT urgency model.

    urgency_scores are probabilities of the "urgent" class (0.0 to 1.0),
    used for continuous time-decay calculation.
    """

    article_ids: list[int]
    urgency_scores: list[float]
