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
RECENT_RUN_LOCK = 730114
INPUT_USD_PER_MILLION = 0.042


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


def call_jev(
    payload: dict[str, Any], api_key: str, usage: dict[str, int] | None = None
) -> ImportanceResult:
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
        if usage is not None:
            usage["requests"] += 1
        try:
            with urlopen(request, timeout=120) as response:
                data = json.load(response)
            if usage is not None:
                usage["responses"] += 1
                reported = data.get("usage", {}) if isinstance(data, dict) else {}
                for field in ("input_tokens", "output_tokens"):
                    count = reported.get(field) if isinstance(reported, dict) else None
                    if type(count) is int and count >= 0:
                        usage[field] += count
                    else:
                        usage["missing_" + field] += 1
            return validate_response(data, payload["model"])
        except HTTPError as error:
            if usage is not None:
                usage["failed_requests"] += 1
            status = error.code
            error.close()
            if status not in (429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f"TypeSafe HTTP {status}") from None
        except (URLError, TimeoutError):
            if usage is not None:
                usage["failed_requests"] += 1
            if attempt == 3:
                raise RuntimeError(
                    "TypeSafe connection failed after four attempts"
                ) from None
        except json.JSONDecodeError:
            if usage is not None:
                usage["failed_requests"] += 1
            raise ValueError("TypeSafe returned invalid JSON") from None
        except ValueError:
            if usage is not None:
                usage["failed_requests"] += 1
            raise
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


async def infer_recent(days: int) -> None:
    """Classify recent missing articles once and print usage even after failure."""
    if type(days) is not int or days <= 0:
        raise ValueError("Days must be a positive integer")
    if not config.JEV_KEY:
        raise ValueError("Set JEV_KEY before importance inference")
    started = time.monotonic()
    until = dt.datetime.now(dt.UTC)
    usage = dict.fromkeys(
        [
            "requests",
            "responses",
            "input_tokens",
            "output_tokens",
            "missing_input_tokens",
            "missing_output_tokens",
            "failed_requests",
        ],
        0,
    )
    summary: dict[str, Any] = {
        "type": "importance_summary",
        "days": days,
        "until": until.isoformat(),
        "status": "failed",
        "selected": 0,
        "stored": 0,
        "skipped": 0,
    }
    try:
        # The transaction releases this lock on success, failure, or connection loss.
        async with dr.global_pool.connection() as conn:
            cursor = await conn.execute(
                "select pg_try_advisory_xact_lock(%s) as acquired", (RECENT_RUN_LOCK,)
            )
            row = await cursor.fetchone()
            if not row or not row["acquired"]:
                raise RuntimeError("Another recent importance run is active")
            version = await dr.get_latest_importance_prompt()
            validate_prompt(version.prompt, version.model)
            articles = await dr.get_recent_missing_importance_articles(days, until)
            requests = [
                (a.entry_id, request_payload(a, version.prompt, version.model))
                for a in articles
            ]
            summary.update(
                selected=len(requests),
                importance_prompt_id=version.id,
                model=version.model,
            )
            for entry_id, payload in requests:
                if await dr.has_entry_importance(entry_id):
                    summary["skipped"] += 1
                    continue
                result = await asyncio.to_thread(
                    call_jev, payload, config.JEV_KEY, usage
                )
                stored = await dr.insert_entry_importance(entry_id, version.id, result)
                summary["stored" if stored else "skipped"] += 1
                print(
                    json.dumps(
                        {
                            "entry_id": entry_id,
                            "importance_prompt_id": version.id,
                            **result.model_dump(mode="json"),
                            "stored": stored,
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )
            summary["status"] = "complete"
    finally:
        summary.update(usage)
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["estimated_cost_usd"] = (
            usage["input_tokens"] * INPUT_USD_PER_MILLION / 1_000_000
        )
        summary["input_usd_per_million"] = INPUT_USD_PER_MILLION
        summary["usage_complete"] = (
            usage["requests"] == usage["responses"]
            and usage["missing_input_tokens"] == 0
            and usage["missing_output_tokens"] == 0
        )
        print(json.dumps(summary, allow_nan=False), flush=True)


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
        elif args.command == "infer-recent":
            await infer_recent(args.days)
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
    recent = commands.add_parser(
        "infer-recent", help="Classify recent articles without any importance record"
    )
    recent.add_argument("--days", type=int, required=True)
    args = parser.parse_args()
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
