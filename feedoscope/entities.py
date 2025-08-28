from pydantic import BaseModel, NaiveDatetime


class Article(BaseModel):
    article_id: int
    title: str
    marked: bool
    feed_name: str
    content: str
    link: str
    author: str
    date_entered: NaiveDatetime
    last_read: NaiveDatetime | None = ...
    labels: list[str]
    tags: list[str]


class RelevanceInferenceResults(BaseModel):
    article_ids: list[int]
    article_titles: list[str]
    scores: list[int]
