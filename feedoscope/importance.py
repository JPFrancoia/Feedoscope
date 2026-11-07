"""Run explicit Jev importance inference without changing reader feedback."""

import argparse
import asyncio
import datetime as dt
import json
import logging
import math
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from custom_logging import init_logging
from feedoscope import config
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import ImportanceArticle, ImportanceResult

logger = logging.getLogger(__name__)
REQUEST_BYTES = 30_000
CLASSES = {"important", "good_only"}


def validate_prompt(prompt: object, model: str) -> None:
    """Check the complete binary prompt and a fixed Jev model before use."""
    if not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
        raise ValueError("Use a fixed Jev model version, such as jev-1.14.0")
    if not isinstance(prompt, dict) or set(prompt) != {"reader_profile", "question"}:
        raise ValueError("Prompt must contain reader_profile and question")
    if (
        not isinstance(prompt["reader_profile"], str)
        or not prompt["reader_profile"].strip()
    ):
        raise ValueError("reader_profile must be nonempty text")
    question = prompt["question"]
    if not isinstance(question, dict) or set(question) != {
        "type",
        "instructions",
        "criteria",
    }:
        raise ValueError("Question must contain type, instructions, and criteria")
    if (
        question["type"] != "choice"
        or not isinstance(question["instructions"], str)
        or not question["instructions"].strip()
    ):
        raise ValueError("Question must be a choice with nonempty instructions")
    criteria = question["criteria"]
    if (
        not isinstance(criteria, dict)
        or set(criteria) != CLASSES
        or any(not isinstance(v, str) or not v.strip() for v in criteria.values())
    ):
        raise ValueError("Criteria must describe important and good_only")
    request_payload(ImportanceArticle(entry_id=1, title="", content=""), prompt, model)


def request_payload(
    article: ImportanceArticle, prompt: dict[str, Any], model: str
) -> dict[str, Any]:
    """Prepare the benchmark input and truncate middle text within its budget."""
    soup = BeautifulSoup(article.content, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    payload: dict[str, Any] = {
        "model": model,
        "state": {
            "reader_profile": prompt["reader_profile"],
            "article": {"title": article.title, "body": body_text},
        },
        "questions": {"importance": prompt["question"]},
    }
    source = body_text.encode()
    size = len(source)
    while len(json.dumps(payload, ensure_ascii=False).encode()) > REQUEST_BYTES:
        if size < 256:
            raise ValueError("Prompt and title exceed request budget")
        size = size * 9 // 10
        head = size * 3 // 4
        tail = size - head
        payload["state"]["article"]["body"] = (
            source[:head].decode(errors="ignore")
            + "\n[Middle omitted for context limit]\n"
            + source[-tail:].decode(errors="ignore")
        )
    return payload


def validate_response(response: Any, model: str) -> ImportanceResult:
    """Reject malformed responses before storing the returned probability and confidence."""
    try:
        if response["model"] != model:
            raise ValueError("Unexpected response model")
        answer = response["answers"]["importance"]
        probabilities = answer["probabilities"]
        if answer["type"] != "choice" or set(probabilities) != CLASSES:
            raise ValueError("Unexpected answer type or classes")
        values = [*probabilities.values(), answer["confidence"]]
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in values
        ):
            raise ValueError("Invalid probability or confidence")
        if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-4):
            raise ValueError("Probabilities do not sum to one")
        choice = answer["choice"]
        if choice not in CLASSES or probabilities[choice] < max(probabilities.values()):
            raise ValueError("Choice disagrees with probabilities")
        if not math.isclose(
            answer["confidence"], 2 * max(probabilities.values()) - 1, abs_tol=0.002
        ):
            raise ValueError("Confidence disagrees with binary Choice formula")
        return ImportanceResult(
            is_important=choice == "important",
            importance_probability=probabilities["important"],
            confidence=answer["confidence"],
            inferred_at=dt.datetime.now(dt.UTC),
        )
    except (KeyError, TypeError, AttributeError):
        raise ValueError("Malformed Jev response") from None


def call_jev(payload: dict[str, Any], api_key: str) -> ImportanceResult:
    """Call Jev over verified HTTPS with bounded transient-error retries."""
    request = Request(
        "https://api.typesafe.ai/v1/systemone",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=120) as response:
                data = json.load(response)
            return validate_response(data, payload["model"])
        except HTTPError as error:
            status = error.code
            error.close()
            if status not in (429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f"TypeSafe HTTP {status}") from None
        except (URLError, TimeoutError):
            if attempt == 3:
                raise RuntimeError(
                    "TypeSafe connection failed after four attempts"
                ) from None
        except json.JSONDecodeError:
            raise ValueError("TypeSafe returned invalid JSON") from None
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError("TypeSafe retry limit reached")


async def infer(entry_ids: list[int]) -> list[dict[str, Any]]:
    """Classify explicit articles with one prompt version and save each success."""
    if not entry_ids or any(type(i) is not int or i <= 0 for i in entry_ids):
        raise ValueError("Supply positive article IDs")
    if not config.JEV_KEY:
        raise ValueError("Set JEV_KEY before importance inference")
    version = await dr.get_latest_importance_prompt()
    validate_prompt(version.prompt, version.model)
    articles = await dr.get_importance_articles(sorted(set(entry_ids)))
    missing = set(entry_ids) - {article.entry_id for article in articles}
    if missing:
        raise ValueError(f"Articles are missing or removed: {sorted(missing)}")
    # Prepare all inputs before paid calls so invalid input cannot cause a partial run.
    requests = [
        (article.entry_id, request_payload(article, version.prompt, version.model))
        for article in articles
    ]
    results = []
    for entry_id, payload in requests:
        result = await asyncio.to_thread(call_jev, payload, config.JEV_KEY)
        stored = await dr.upsert_entry_importance(entry_id, version.id, result)
        row = {
            "entry_id": entry_id,
            "importance_prompt_id": version.id,
            **result.model_dump(mode="json"),
            "stored": stored,
        }
        results.append(row)
        logger.info(
            f"Importance entry {entry_id}: {result.is_important}, prompt {version.id}, stored={stored}"
        )
    return results


async def run(args: argparse.Namespace) -> None:
    """Run one explicit command and close the database pool on every exit."""
    prompt: dict[str, Any] = {}
    if args.command == "add-prompt":
        prompt = json.loads(args.file.read_text())
        validate_prompt(prompt, args.model)
    await dr.global_pool.open(wait=True)
    try:
        if args.command == "add-prompt":
            version = await dr.insert_importance_prompt(prompt, args.model)
            print(
                json.dumps({"importance_prompt_id": version.id, "model": version.model})
            )
        else:
            results = await infer(args.entry_id)
            for result in results:
                print(json.dumps(result, allow_nan=False))
    finally:
        await dr.global_pool.close()


def main() -> None:
    """Parse prompt import or explicit article inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add-prompt", help="Insert an immutable prompt version")
    add.add_argument("--file", type=Path, required=True)
    add.add_argument("--model", required=True)
    inference = commands.add_parser(
        "infer", help="Classify only the supplied article IDs"
    )
    inference.add_argument("--entry-id", type=int, action="append", required=True)
    args = parser.parse_args()
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
