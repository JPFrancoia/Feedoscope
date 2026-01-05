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
from feedoscope.entities import Article, TimeSensitivity

logger = logging.getLogger(__name__)


PROMPT = """
[INST]
You are a news analysis AI that evaluates the time-sensitivity of news articles. Your sole function is to return a valid JSON object based on the provided data.

**Objective:**
Analyze the provided article information and determine its time-sensitivity rating on a scale of 1 to 5. Time-sensitivity refers to how quickly the information becomes outdated, not its overall importance.

**JSON Output Schema:**
{
  "score": <integer between 1 and 5>,
  "confidence": <string, "high", "medium", or "low">,
  "explanation": <string, a concise explanation for the rating>,
}

**Rating Scale Definitions:**
- **1 (Evergreen):** Content is historical, biographical, or a foundational explainer. Keywords: "history of", "profile", "explainer", "deep dive".
- **2 (Low):** A feature, trend analysis, or opinion piece relevant for months. Keywords: "analysis", "opinion", "trend", "culture".
- **3 (Medium):** Story tied to an ongoing but not breaking event, relevant for days/weeks. Keywords: "debate", "upcoming", "policy", "investigation".
- **4 (High):** Reports on a specific, recent event; loses relevance in 24-48 hours. Keywords: "announces", "reports", "wins", "results", "verdict".
- **5 (Critical):** Live coverage of a rapidly unfolding event; loses relevance in hours. Keywords: "live", "breaking", "unfolding", "evacuation", "alert".

**Instructions:**
Analyze the following article. Provide your response as a single, valid JSON object and nothing else. Do not include any additional text, labels, or markdown formatting. Ensure your JSON is well-formed and valid.

**Article Data:**
Title: {{headline}}
Summary: {{article_summary_or_first_paragraph}}
[/INST]
"""


MODEL_PATH = "models/Ministral-8B-Instruct-2410-Q6_K_L.gguf"
MAX_CONTENT_LENGTH = 1024


def strip_html_keep_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def prepare_content(content: str, llm: Llama, title: str) -> str:
    content = clean(strip_html_keep_text(content))

    # Calculate token budget: context window - prompt template - title - response buffer
    prompt_template_tokens = len(llm.tokenize(PROMPT.encode("utf-8")))
    title_tokens = len(llm.tokenize(title.encode("utf-8")))
    response_tokens = 200  # Reserve tokens for the JSON response
    available_tokens = 1024 - prompt_template_tokens - title_tokens - response_tokens

    # Tokenize and truncate to available budget
    content_tokens = llm.tokenize(content.encode("utf-8"))
    if len(content_tokens) > available_tokens:
        content_tokens = content_tokens[:available_tokens]

    return llm.detokenize(content_tokens).decode("utf-8", errors="ignore")


def best_effort_json_parse(result: str) -> dict[str, Any]:

    result = result.replace("```json\n", "").replace("\n```", "").strip()

    # sometimes the output contains ```json at the beginning, or ``` at the end
    # We should try to clean that up before parsing the JSON
    return json.loads(result)


async def infer(
    recent_unread_articles: list[Article],
) -> AsyncGenerator[TimeSensitivity, None]:
    logger.debug("Loading Llama model for time sensitivity inference...")
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, verbose=False, n_ctx=1024)
    logger.debug("Llama model loaded.")

    for idx, article in enumerate(recent_unread_articles, start=1):
        clean_title = utils.clean_title(article.title)
        final_prompt = PROMPT.replace("{{headline}}", clean_title)
        final_prompt = final_prompt.replace(
            "{{article_summary_or_first_paragraph}}",
            prepare_content(article.content, llm, clean_title),
        )

        output = llm(
            prompt=final_prompt,
            max_tokens=4096,
            stop=["[/INST]"],
            echo=False,
        )
        result = output.get("choices")[0]["text"]

        try:
            parsed_result = best_effort_json_parse(result)
            parsed_result["article_id"] = article.article_id
            sensitivity = TimeSensitivity(**parsed_result)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM output: {result}")
            continue
        except ValidationError:
            logger.exception(f"Validation error for article {article.title}")
            continue

        logger.debug(
            f"{idx}/{len(recent_unread_articles)} Article: {article.title}, data: {sensitivity}"
        )

        yield sensitivity


async def main(number_of_days: int = 14) -> None:

    await dr.global_pool.open(wait=True)

    recent_unread_articles = await dr.get_previous_days_articles_wo_time_sensitivity(
        number_of_days=number_of_days
    )

    logger.info(
        f"Fetched {len(recent_unread_articles)} unread articles from the last {number_of_days} days without time sensitivity."
    )

    batch: list[TimeSensitivity] = []
    total_processed = 0

    async for sensitivity in infer(recent_unread_articles):
        batch.append(sensitivity)

        if len(batch) >= 10:
            await dr.register_time_sensitivity_for_articles(batch)
            total_processed += len(batch)
            logger.info(
                f"Registered batch of {len(batch)} time sensitivities. Total: {total_processed}"
            )
            batch = []

    # Write remaining items that didn't reach batch size
    if batch:
        await dr.register_time_sensitivity_for_articles(batch)
        total_processed += len(batch)
        logger.info(
            f"Registered final batch of {len(batch)} time sensitivities. Total: {total_processed}"
        )

    logger.info(f"Completed processing. Total registered: {total_processed} articles.")


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)

    asyncio.run(main())
