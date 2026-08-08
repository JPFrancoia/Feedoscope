from typing import Optional

from pydantic import AwareDatetime, BaseModel, Field


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
    tags: list[str]  # Directly from entries.tags array in Miniflux
    vote: int  # -1, 0, or 1 in Miniflux
    status: str  # 'read', 'unread', or 'removed' in Miniflux


class RelevanceInferenceResults(BaseModel):
    article_ids: list[int]
    article_titles: list[str]
    scores: list[float]
    model_key: str
