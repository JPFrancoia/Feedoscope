import asyncio
import json
import logging
import re

from bs4 import BeautifulSoup
from cleantext import clean
from llama_cpp import Llama

from custom_logging import init_logging
from feedoscope import config
from feedoscope.data_registry import data_registry as dr

logger = logging.getLogger(__name__)


PROMPT = """
INST]
You are a news analyst AI. Your task is to rate the time-sensitivity of a news article on a scale of 1 to 5.

Time-sensitivity is defined as how quickly information loses its primary relevance, NOT its overall importance.

**Rating Scale Definitions:**
- **1 (Evergreen):** Content is historical, biographical, or a foundational explainer. Keywords: "history of", "profile", "explainer", "deep dive".
- **2 (Low):** A feature, trend analysis, or opinion piece relevant for months. Keywords: "analysis", "opinion", "trend", "culture".
- **3 (Medium):** Story tied to an ongoing but not breaking event, relevant for days/weeks. Keywords: "debate", "upcoming", "policy", "investigation".
- **4 (High):** Reports on a specific, recent event; loses relevance in 24-48 hours. Keywords: "announces", "reports", "wins", "results", "verdict".
- **5 (Critical):** Live coverage of a rapidly unfolding event; loses relevance in hours. Keywords: "live", "breaking", "unfolding", "evacuation", "alert".

**Instructions:**
Analyze the article above. Your response MUST be a single integer from 1 to 5. Do not provide any other text, labels, or markdown formatting. Your entire response should be only the number.

**Article Information:**
Title: {{headline}}
Summary: {{article_summary_or_first_paragraph}}

[/INST]
"""

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


def strip_html_keep_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def prepare_content(content: str, llm) -> str:
    content = clean(strip_html_keep_text(content))
    tokens = llm.tokenize(content.encode("utf-8")[:1024])
    truncated_tokens = tokens[:512]
    return llm.detokenize(truncated_tokens).decode("utf-8", errors="ignore")


async def main() -> None:

    model_path = "./mistral/Ministral-8B-Instruct-2410-Q6_K_L.gguf"
    llm = Llama(model_path=model_path, n_gpu_layers=-1, verbose=False, n_ctx=1024)

    await dr.global_pool.open(wait=True)

    recent_unread_articles = await dr.get_previous_days_articles_wo_time_sensitivity(
        number_of_days=14
    )

    time_sensitivities: list[dict] = []

    for article in recent_unread_articles:
        title = re.sub(r"^\[[^]]*\]\s*", "", article["title"])
        final_prompt = PROMPT.replace("{{headline}}", title)
        final_prompt = final_prompt.replace(
            "{{article_summary_or_first_paragraph}}",
            prepare_content(article["content"], llm),
        )

        output = llm(
            prompt=final_prompt,
            max_tokens=4096,
            stop=["[/INST]"],
            echo=False,
        )
        result = output.get("choices")[0]["text"]

        try:
            parsed_result = json.loads(result)
            parsed_result["article_id"] = article["article_id"]
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM output: {result}")
            continue

        time_sensitivities.append(parsed_result)

        logger.debug(f"Article: {title}, data: {parsed_result}")

        # TODO: assign labels to the article, into the database

    if time_sensitivities:
        await dr.register_time_sensitivity_for_articles(time_sensitivities)
        logger.info(
            f"Registered time sensitivities for {len(time_sensitivities)} articles."
        )


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)

    asyncio.run(main())
