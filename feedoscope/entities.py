from pydantic import BaseModel, AwareDatetime


class Article(BaseModel):
    article_id: int
    title: str
    marked: bool
    feed_title: str
    content: str
    link: str
    author: str
    date_entered: AwareDatetime
    last_read: AwareDatetime

    # FIXME: we can do better and get a list of str from the database
    labels: str
    tags: str


class RelevanceInferenceResults(BaseModel):
    article_ids: list[int]
    article_titles: list[str]
    scores: list[int]
