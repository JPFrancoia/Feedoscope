import asyncio
import json
import logging
from typing import Any

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


MODEL_PATH = "./mistral/Ministral-8B-Instruct-2410-Q6_K_L.gguf"
MAX_CONTENT_LENGTH = 1024


def strip_html_keep_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def prepare_content(content: str, llm: Llama) -> str:
    content = clean(strip_html_keep_text(content))

    # Once sanitized, we truncate the content to fit within the model's token limit.
    # We cut by chars so if we cut to 1024 chars and the model's window is 1024
    # tokens, we are sure to fit.
    tokens = llm.tokenize(content.encode("utf-8")[:MAX_CONTENT_LENGTH])

    return llm.detokenize(tokens).decode("utf-8", errors="ignore")


def best_effort_json_parse(result: str) -> dict[str, Any]:

    result = result.replace("```json\n", "").replace("\n```", "").strip()

    # sometimes the output contains ```json at the beginning, or ``` at the end
    # We should try to clean that up before parsing the JSON
    return json.loads(result)



async def infer(recent_unread_articles: list[Article]) -> list[TimeSensitivity]:
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, verbose=False, n_ctx=1024)

    time_sensitivities: list[TimeSensitivity] = []

    for article in recent_unread_articles:
        final_prompt = PROMPT.replace("{{headline}}", utils.clean_title(article.title))
        final_prompt = final_prompt.replace(
            "{{article_summary_or_first_paragraph}}",
            prepare_content(article.content, llm),
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

        time_sensitivities.append(sensitivity)

        logger.debug(f"Article: {article.title}, data: {sensitivity}")

    return time_sensitivities


async def main(number_of_days: int = 14) -> None:

    await dr.global_pool.open(wait=True)

    recent_unread_articles = await dr.get_previous_days_articles_wo_time_sensitivity(
        number_of_days=number_of_days
    )

    time_sensitivities = await infer(recent_unread_articles)

    # TODO: assign labels to the article, into the database

    if time_sensitivities:
        await dr.register_time_sensitivity_for_articles(time_sensitivities)
        logger.info(
            f"Registered time sensitivities for {len(time_sensitivities)} articles."
        )


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)

    asyncio.run(main())
