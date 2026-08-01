import logging
from types import SimpleNamespace
from typing import cast

import pytest
from transformers import PreTrainedTokenizerBase

from feedoscope import relevance_text
from feedoscope.entities import Article


def test_article_text_preparation_logs_progress(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    times = iter((0.0, 31.0, 62.0))
    monkeypatch.setattr(relevance_text.time, "monotonic", lambda: next(times))
    articles = [
        cast(Article, SimpleNamespace(title="Title", content="Body")),
        cast(Article, SimpleNamespace(title="Title", content="Body")),
    ]

    with caplog.at_level(logging.INFO, logger=relevance_text.__name__):
        relevance_text.prepare_articles_text(
            articles,
            tokenizer=cast(PreTrainedTokenizerBase, SimpleNamespace()),
            max_length=10,
            mode="single_blob",
        )

    assert "Prepared article text 1/2." in caplog.messages
    assert "Prepared article text 2/2." in caplog.messages
