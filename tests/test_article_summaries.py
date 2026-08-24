import asyncio
from datetime import datetime, timezone
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("DATABASE_URL", "postgresql://test")

from feedoscope import article_summaries, summarize_articles
from feedoscope.entities import Article


class WordTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
    ) -> list[str]:
        del add_special_tokens, truncation
        return text.split()[:max_length]

    def decode(self, token_ids: list[str], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return " ".join(token_ids)


def article() -> Article:
    return Article(
        article_id=1,
        title="[85] Summary title",
        starred=False,
        feed_name="Feed",
        content="<p>First paragraph.</p><p>Second paragraph.</p>",
        link="https://example.com/article",
        author="Author",
        date_entered=datetime.now(timezone.utc),
        last_read=None,
        tags=[],
        vote=0,
        status="unread",
    )


def test_prepare_article_source_strips_html_and_limits_tokens() -> None:
    source = article_summaries.prepare_article_source(article(), WordTokenizer(), 5)

    assert source == "Title: Summary title Article source:"


def test_normalize_summary_rejects_multiple_sentences() -> None:
    assert (
        article_summaries.normalize_summary("One factual sentence.")
        == "One factual sentence."
    )
    assert (
        article_summaries.normalize_summary("First sentence. Second sentence.") is None
    )
    assert (
        article_summaries.normalize_summary("First sentence. second sentence.") is None
    )
    assert (
        article_summaries.normalize_summary("The U.S. agency published a report.")
        == "The U.S. agency published a report."
    )
    assert article_summaries.normalize_summary("Missing final punctuation") is None


def test_summarize_article_uses_tokenized_chat_input() -> None:
    class Inputs(dict):
        def to(self, device: str) -> "Inputs":
            assert device == "cuda"
            return self

    class Processor:
        tokenizer = WordTokenizer()

        def apply_chat_template(self, *args, **kwargs) -> Inputs:
            assert args
            assert kwargs["tokenize"]
            return Inputs({"input_ids": SimpleNamespace(shape=(1, 3))})

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert token_ids == [4]
            assert skip_special_tokens
            return "A factual summary."

    class Model:
        device = "cuda"

        def generate(self, **kwargs) -> list[list[int]]:
            assert "input_ids" in kwargs
            return [[1, 2, 3, 4]]

    summary = article_summaries.summarize_article(article(), (Processor(), Model()))

    assert summary == "A factual summary."


def test_summarize_article_retries_invalid_output() -> None:
    class Inputs(dict):
        def to(self, device: str) -> "Inputs":
            assert device == "cuda"
            return self

    class Processor:
        tokenizer = WordTokenizer()

        def __init__(self) -> None:
            self.messages: list[list[dict[str, object]]] = []

        def apply_chat_template(self, messages, *args, **kwargs) -> Inputs:
            del args
            assert kwargs["tokenize"]
            self.messages.append(messages)
            return Inputs({"input_ids": SimpleNamespace(shape=(1, 3))})

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return (
                "Two sentences. Invalid." if token_ids == [4] else "Corrected summary."
            )

    class Model:
        device = "cuda"
        calls = 0

        def generate(self, **kwargs) -> list[list[int]]:
            assert "input_ids" in kwargs
            self.calls += 1
            return [[1, 2, 3, 4 if self.calls == 1 else 5]]

    processor = Processor()
    summary = article_summaries.summarize_article(article(), (processor, Model()))

    assert summary == "Corrected summary."
    assert len(processor.messages) == 2
    assert processor.messages[1][-2]["role"] == "assistant"


def test_main_selects_highest_scored_articles(monkeypatch) -> None:
    get_articles = AsyncMock(return_value=[])
    monkeypatch.setattr(
        summarize_articles.dr,
        "global_pool",
        SimpleNamespace(open=AsyncMock(), close=AsyncMock()),
    )
    monkeypatch.setattr(
        summarize_articles.dr,
        "get_highest_scored_articles_for_summary",
        get_articles,
    )

    assert asyncio.run(summarize_articles.main(limit=200, by_score=True)) == 0
    get_articles.assert_awaited_once_with(200)


def test_main_saves_only_valid_summaries(monkeypatch) -> None:
    update_summary = AsyncMock()
    summarize = iter(["A factual summary.", None])
    monkeypatch.setattr(
        summarize_articles.dr,
        "global_pool",
        SimpleNamespace(open=AsyncMock(), close=AsyncMock()),
    )
    monkeypatch.setattr(
        summarize_articles.dr,
        "get_latest_articles_for_summary",
        AsyncMock(
            return_value=[article(), article().model_copy(update={"article_id": 2})]
        ),
    )
    monkeypatch.setattr(
        summarize_articles.dr, "update_one_line_summary", update_summary
    )
    monkeypatch.setattr(
        summarize_articles.article_summaries,
        "load_summary_model",
        lambda: object(),
    )
    monkeypatch.setattr(
        summarize_articles.article_summaries,
        "summarize_article",
        lambda _article, _pipeline: next(summarize),
    )

    assert asyncio.run(summarize_articles.main(limit=2)) == 1
    update_summary.assert_awaited_once_with(1, "A factual summary.")
