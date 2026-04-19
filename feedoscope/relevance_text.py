from typing import Literal

from cleantext import clean  # type: ignore[import]
from transformers import PreTrainedTokenizerBase

from feedoscope.entities import Article
from feedoscope.utils import clean_title, strip_html_keep_text

TextPrepMode = Literal["single_blob", "title_head"]


def _clean_text(text: str) -> str:
    """Normalize whitespace and run the shared text cleaner."""
    return clean(" ".join(text.split()))


def _encode_truncated(
    tokenizer: PreTrainedTokenizerBase, text: str, max_length: int
) -> list[int]:
    """Tokenize text while enforcing a hard upper bound on token count."""
    if max_length <= 0 or not text:
        return []

    return tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )


def prepare_single_blob(title: str, content: str) -> str:
    """Build a single cleaned text blob from title and article body."""
    return _clean_text(strip_html_keep_text(f"{clean_title(title)} {content}"))


def prepare_title_head(
    tokenizer: PreTrainedTokenizerBase,
    title: str,
    content: str,
    max_length: int,
) -> str:
    """Keep the full title plus the first body tokens that fit the budget."""
    cleaned_title = _clean_text(clean_title(title))
    cleaned_body = _clean_text(strip_html_keep_text(content))
    special_buffer = 4

    if max_length <= special_buffer:
        title_ids = _encode_truncated(tokenizer, cleaned_title, 1)
        return tokenizer.decode(title_ids, skip_special_tokens=True).strip()

    title_budget = max(1, max_length - special_buffer)
    kept_title_ids = _encode_truncated(tokenizer, cleaned_title, title_budget)
    remaining_budget = max(0, max_length - special_buffer - len(kept_title_ids))

    # Only tokenize the body up to the remaining budget so long articles do not
    # trigger tokenizer length warnings before we slice them down.
    head_ids = _encode_truncated(tokenizer, cleaned_body, remaining_budget)

    sep = f" {tokenizer.sep_token} " if tokenizer.sep_token else " "
    parts = [tokenizer.decode(kept_title_ids, skip_special_tokens=True).strip()]
    if head_ids:
        parts.append(tokenizer.decode(head_ids, skip_special_tokens=True).strip())

    return sep.join(part for part in parts if part)


def prepare_articles_text(
    articles: list[Article],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    mode: TextPrepMode,
) -> list[str]:
    """Prepare model-ready text for each article using the configured mode."""
    texts: list[str] = []
    for article in articles:
        if mode == "single_blob":
            text = prepare_single_blob(article.title, article.content)
        elif mode == "title_head":
            text = prepare_title_head(
                tokenizer=tokenizer,
                title=article.title,
                content=article.content,
                max_length=max_length,
            )
        else:
            raise ValueError(f"Unsupported relevance text prep mode: {mode}")
        texts.append(text)

    return texts
