from enum import StrEnum, unique
from typing import Optional

from pydantic import BaseModel, Field, NaiveDatetime


@unique
class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Article(BaseModel):
    article_id: int
    title: str
    marked: bool
    feed_name: str
    content: str
    link: str
    author: str
    date_entered: NaiveDatetime
    last_read: Optional[NaiveDatetime] = Field(...)
    labels: list[str]
    tags: list[str]


class RelevanceInferenceResults(BaseModel):
    article_ids: list[int]
    article_titles: list[str]
    scores: list[int]


class TimeSensitivity(BaseModel):
    article_id: int
    score: int = Field(ge=1, le=5)
    confidence: ConfidenceLevel
    explanation: str
