import asyncio
import json
import logging
from typing import Any, AsyncGenerator

from bs4 import BeautifulSoup
from cleantext import clean  # type: ignore[import]
from llama_cpp import Llama  # type: ignore[import]
from pydantic import ValidationError

from custom_logging import init_logging
from feedoscope import config, utils
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article, SimplifiedTimeSensitivity

logger = logging.getLogger(__name__)


PROMPT = """
[INST]
You are a news analysis AI. Your sole function is to classify an article's urgency and return a valid JSON object.

**Objective:**
Determine if this article is time-sensitive. Ask yourself: "Would this article still be relevant and informative one year from now?"

- If YES (the article would still be relevant in one year) → score = 0
- If NO (the article would lose most of its relevance within a year) → score = 1

**JSON Output Schema:**
{
  "score": <0 or 1>,
  "explanation": <string, one sentence justifying your choice>
}

**Instructions:**
Respond with a single valid JSON object. No additional text, labels, or markdown.

**Article Data:**
Title: {{headline}}
Summary: {{article_summary_or_first_paragraph}}
[/INST]
"""


MODEL_PATH = "models/Ministral-8B-Instruct-2410-Q6_K_L.gguf"


def strip_html_keep_text(html: str) -> str:
    """Strip HTML tags and return clean text."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def prepare_content(content: str, llm: Llama, title: str) -> str:
    """Prepare article content, truncating to fit within the context window."""
    content = clean(strip_html_keep_text(content))

    # Calculate token budget: context window - prompt template - title - response buffer
    prompt_template_tokens = len(llm.tokenize(PROMPT.encode("utf-8")))
    title_tokens = len(llm.tokenize(title.encode("utf-8")))
    response_tokens = 150  # Reserve tokens for the JSON response (shorter than 1-5)
    available_tokens = 1024 - prompt_template_tokens - title_tokens - response_tokens

    # Tokenize and truncate to available budget
    content_tokens = llm.tokenize(content.encode("utf-8"))
    if len(content_tokens) > available_tokens:
        content_tokens = content_tokens[:available_tokens]

    return llm.detokenize(content_tokens).decode("utf-8", errors="ignore")


def best_effort_json_parse(result: str) -> dict[str, Any]:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    result = result.replace("```json\n", "").replace("\n```", "").strip()
    return json.loads(result)


async def infer(
    articles: list[Article],
) -> AsyncGenerator[SimplifiedTimeSensitivity, None]:
    """Infer simplified time sensitivity for articles using Ministral-8B.

    Yields:
        SimplifiedTimeSensitivity for each successfully scored article.

    """
    logger.debug("Loading Llama model for simplified time sensitivity inference...")
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, verbose=False, n_ctx=1024)
    logger.debug("Llama model loaded.")

    for idx, article in enumerate(articles, start=1):
        clean_title = utils.clean_title(article.title)
        final_prompt = PROMPT.replace("{{headline}}", clean_title)
        final_prompt = final_prompt.replace(
            "{{article_summary_or_first_paragraph}}",
            prepare_content(article.content, llm, clean_title),
        )

        output = llm(
            prompt=final_prompt,
            max_tokens=256,
            stop=["[/INST]"],
            echo=False,
        )
        result = output.get("choices")[0]["text"]

        try:
            parsed_result = best_effort_json_parse(result)
            parsed_result["article_id"] = article.article_id
            sensitivity = SimplifiedTimeSensitivity(**parsed_result)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM output: {result}")
            continue
        except ValidationError:
            logger.exception(f"Validation error for article {article.title}")
            continue

        logger.debug(
            f"{idx}/{len(articles)} Article: {article.title}, data: {sensitivity}"
        )

        yield sensitivity


async def _register_and_tag_batch(
    batch: list[SimplifiedTimeSensitivity],
    tag_ids: dict[str, int],
) -> None:
    """Register a batch of simplified time sensitivities and assign urgency tags.

    Two things happen for each article in the batch:

    1. The LLM-assigned score (0 or 1) is stored in the time_sensitivity_simplified
       table. This is the raw LLM output and is always written.

    2. A "0-urgency" or "1-urgency" user tag is assigned to the article in Miniflux.
       This tag is ONLY set if the article doesn't already have an urgency tag.
       If the user has already manually tagged the article (e.g. to correct a
       misclassification), that manual tag is preserved and the LLM's tag
       assignment is silently skipped.

    The user tags serve as the ground-truth labels for training the distilled
    urgency model (make train_urgency). By never overwriting existing tags,
    manual corrections are always reflected in the next training run.
    """
    await dr.register_simplified_time_sensitivity(batch)
    await dr.assign_urgency_tags_for_articles(
        article_ids=[s.article_id for s in batch],
        scores=[s.score for s in batch],
        tag_ids=tag_ids,
    )


async def main() -> None:
    """Score all unscored articles with simplified binary time sensitivity.

    This is the LLM labeling step of the urgency pipeline. It uses Ministral-8B
    to classify articles as evergreen (0) or time-sensitive (1), then:

    1. Stores the score in the time_sensitivity_simplified table.
    2. Assigns a "0-urgency" or "1-urgency" user tag in Miniflux — but ONLY
       if the article doesn't already have an urgency tag. Existing tags
       (including manual corrections) are never overwritten.

    Only articles from the last 6 months that are NOT already in the
    time_sensitivity_simplified table are processed (idempotent).
    """
    await dr.global_pool.open(wait=True)

    # Ensure the "0-urgency" and "1-urgency" tag definitions exist in the
    # Miniflux user_tags table, and fetch their database IDs for later use.
    tag_ids = await dr.ensure_urgency_user_tags()
    logger.info(f"Urgency user tag IDs: {tag_ids}")

    articles = await dr.get_articles_wo_simplified_time_sensitivity()

    logger.info(
        f"Fetched {len(articles)} articles without simplified time sensitivity."
    )

    if not articles:
        logger.info("No articles to process. Exiting.")
        return

    batch: list[SimplifiedTimeSensitivity] = []
    total_processed = 0

    async for sensitivity in infer(articles):
        batch.append(sensitivity)

        if len(batch) >= 10:
            await _register_and_tag_batch(batch, tag_ids)
            total_processed += len(batch)
            logger.info(
                f"Registered batch of {len(batch)} simplified time sensitivities. "
                f"Total: {total_processed}"
            )
            batch = []

    # Write remaining items that didn't reach batch size
    if batch:
        await _register_and_tag_batch(batch, tag_ids)
        total_processed += len(batch)
        logger.info(
            f"Registered final batch of {len(batch)} simplified time sensitivities. "
            f"Total: {total_processed}"
        )

    logger.info(f"Completed processing. Total registered: {total_processed} articles.")


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
