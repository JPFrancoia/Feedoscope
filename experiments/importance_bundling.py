"""Compare original and article-local Jev requests without database access."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from feedoscope.importance import INPUT_USD_PER_MILLION, validate_response

MODEL = "jev-1.14.0"
QUESTION_BYTES = 32_000
BUNDLE_BYTES = 60_000


def encoded(value: Any) -> bytes:
    """Encode the same JSON representation for hashing and transmission."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode()


def sha(value: bytes) -> str:
    """Return a reproducible input fingerprint."""
    return hashlib.sha256(value).hexdigest()


def original_payload(article: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    """Preserve the original benchmark preparation, including its truncation."""
    payload: dict[str, Any] = {
        "model": MODEL,
        "state": {
            "reader_profile": prompt["reader_profile"],
            "article": {"title": article["title"], "body": article["body"]},
        },
        "questions": {"importance": prompt["question"]},
    }
    source = article["body"].encode()
    size = len(source)
    while len(encoded(payload)) > 30_000:
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


def bundle_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep each article inside only its own independent question."""
    if not payloads:
        raise ValueError("A bundle needs at least one article")
    state = {"reader_profile": payloads[0]["state"]["reader_profile"]}
    questions = {}
    for index, payload in enumerate(payloads):
        if (
            payload["model"] != MODEL
            or payload["state"]["reader_profile"] != state["reader_profile"]
        ):
            raise ValueError("Bundle model and profile must match")
        question = payload["questions"]["importance"]
        questions[f"article_{index}"] = {
            **question,
            "instructions": {
                "instructions": question["instructions"],
                "article": payload["state"]["article"],
            },
        }
    result = {"model": MODEL, "state": state, "questions": questions}
    if (
        any(
            len(encoded({"state": state, "question": question})) > QUESTION_BYTES
            for question in questions.values()
        )
        or len(encoded(result)) > BUNDLE_BYTES
    ):
        raise ValueError("Bundle exceeds context budget")
    return result


def pack(
    articles: list[dict[str, Any]],
    prompt: dict[str, Any],
    size: int,
    original: bool = False,
) -> list[dict[str, Any]]:
    """Pack articles without changing their individual prepared text."""
    if type(size) is not int or size < 1:
        raise ValueError("Bundle size must be positive")
    batches = []
    pending: list[dict[str, Any]] = []
    ids: list[int] = []
    for article in articles:
        payload = original_payload(article, prompt)
        if original:
            batches.append({"ids": [article["article_id"]], "payload": payload})
            continue
        bundle_payload([payload])
        try:
            bundle_payload([*pending, payload])
            fits = len(pending) < size
        except ValueError:
            fits = False
        if pending and not fits:
            batches.append({"ids": ids, "payload": bundle_payload(pending)})
            pending, ids = [], []
        pending.append(payload)
        ids.append(article["article_id"])
    if pending:
        batches.append({"ids": ids, "payload": bundle_payload(pending)})
    return batches


def validate(response: Any, payload: dict[str, Any]) -> dict[str, float]:
    """Reject missing, extra, or malformed answers before mapping article IDs."""
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise ValueError("Malformed bundled response")
    if set(response["answers"]) != set(payload["questions"]):
        raise ValueError("Response question keys differ")
    return {
        key: validate_response(
            {"model": response.get("model"), "answers": {"importance": answer}}, MODEL
        ).importance_probability
        for key, answer in response["answers"].items()
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read complete private cache records without modifying corrupt files."""
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def run_mode(
    path: Path, batches: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    """Run sequential requests and resume only matching validated cache rows."""
    records = read_records(path)
    if len(records) > len(batches):
        raise ValueError("Cache contains extra requests")
    for record, batch in zip(records, batches):
        if (
            record["ids"] != batch["ids"]
            or record["question_keys"] != list(batch["payload"]["questions"])
            or record["request_sha256"] != sha(encoded(batch["payload"]))
        ):
            raise ValueError("Cached request differs")
        validate(record["response"], batch["payload"])
    for batch in batches[len(records) :]:
        payload = batch["payload"]
        request = Request(
            "https://api.typesafe.ai/v1/systemone",
            data=encoded(payload),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        attempts = 0
        for attempt in range(4):
            attempts += 1
            try:
                with urlopen(request, timeout=120) as response:
                    data = json.load(response)
                break
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
            time.sleep(2 ** (attempt + 1))
        record = {
            "ids": batch["ids"],
            "question_keys": list(payload["questions"]),
            "request_sha256": sha(encoded(payload)),
            "request_bytes": len(encoded(payload)),
            "seconds": time.monotonic() - started,
            "attempts": attempts,
            "response": data,
        }
        # Keep billed responses, including invalid ones, before validation fails.
        path.touch(mode=0o600, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        validate(data, payload)
        records.append(record)
    return records


def report(
    articles: list[dict[str, Any]], modes: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Compare matched predictions and reported usage from each request mode."""
    ids = [row["article_id"] for row in articles]
    labels = np.array([row["label"] for row in articles])
    output: dict[str, Any] = {}
    vectors = {}
    choices = {}
    for name, records in modes.items():
        predictions = {}
        decisions = {}
        for record in records:
            answers = record["response"]["answers"]
            for article_id, question_key in zip(
                record["ids"], record["question_keys"], strict=True
            ):
                answer = answers[question_key]
                if article_id in predictions:
                    raise ValueError("Duplicate prediction")
                predictions[article_id] = answer["probabilities"]["important"]
                decisions[article_id] = answer["choice"] == "important"
        if set(predictions) != set(ids):
            raise ValueError("Incomplete mode")
        p = np.array([predictions[i] for i in ids])
        chosen = np.array([decisions[i] for i in ids])
        vectors[name], choices[name] = p, chosen
        totals = {"input_tokens": 0, "output_tokens": 0}
        complete = all(row["attempts"] == 1 for row in records)
        for row in records:
            usage = row["response"].get("usage")
            for field in totals:
                count = usage.get(field) if isinstance(usage, dict) else None
                if type(count) is int and count >= 0:
                    totals[field] += count
                else:
                    complete = False
        input_tokens = totals["input_tokens"]
        output[name] = {
            "articles": len(ids),
            "requests": len(records),
            "attempts": sum(r["attempts"] for r in records),
            "seconds": sum(r["seconds"] for r in records),
            "input_tokens": input_tokens,
            "output_tokens": totals["output_tokens"],
            "usage_complete": complete,
            "estimated_cost_usd": (
                input_tokens * INPUT_USD_PER_MILLION / 1_000_000 if complete else None
            ),
            "accuracy": float(accuracy_score(labels, chosen)),
            "average_precision": float(average_precision_score(labels, p)),
            "roc_auc": float(roc_auc_score(labels, p)),
            "brier": float(brier_score_loss(labels, p)),
            "log_loss": float(
                log_loss(labels, np.column_stack([1 - p, p]), labels=[0, 1])
            ),
            "bundle_sizes": [len(r["ids"]) for r in records],
        }
    reference = vectors["original"]
    for name, p in vectors.items():
        output[name]["mean_absolute_probability_change"] = float(
            np.mean(np.abs(p - reference))
        )
        output[name]["max_absolute_probability_change"] = float(
            np.max(np.abs(p - reference))
        )
        output[name]["decision_changes"] = int(
            np.sum(choices[name] != choices["original"])
        )
    rng = np.random.default_rng(42)
    strata = [np.flatnonzero(labels == value) for value in (0, 1)]
    differences: dict[str, list[float]] = {name: [] for name in vectors}
    for _ in range(2000):
        sample = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        baseline = average_precision_score(labels[sample], reference[sample])
        for name, p in vectors.items():
            differences[name].append(
                float(average_precision_score(labels[sample], p[sample]) - baseline)
            )
    for name, values in differences.items():
        output[name]["paired_ap_difference_95"] = np.quantile(
            values, [0.025, 0.975]
        ).tolist()
    return output


def main() -> None:
    """Run an isolated, hashed experiment from existing private input files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import shlex

    key = ""
    for line in args.key_file.read_text().splitlines():
        assignment = line.strip().removeprefix("export ")
        if assignment.startswith("JEV_KEY="):
            parts = shlex.split(assignment.split("=", 1)[1], comments=True)
            if len(parts) == 1:
                key = parts[0]
    if not key:
        raise ValueError("No JEV_KEY found")
    prompt_path = args.source / "prompt.json"
    article_path = args.source / ("profile.jsonl" if args.smoke else "eval.jsonl")
    prompt = json.loads(prompt_path.read_text())
    articles = read_records(article_path)
    if args.smoke:
        articles = articles[:4]
    else:
        if len(articles) != 150 or len({a["article_id"] for a in articles}) != 150:
            raise ValueError("Expected 150 distinct evaluation articles")
        cached = {
            r["article_id"]: r for r in read_records(args.source / "predictions.jsonl")
        }
        for article in articles:
            if (
                sha(
                    json.dumps(
                        original_payload(article, prompt),
                        sort_keys=True,
                        ensure_ascii=False,
                    ).encode()
                )
                != cached[article["article_id"]]["request_sha256"]
            ):
                raise ValueError("Original benchmark input differs")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest = {
        "prompt_sha256": sha(prompt_path.read_bytes()),
        "articles_sha256": sha(article_path.read_bytes()),
        "protocol_sha256": sha(Path(__file__).read_bytes()),
        "model": MODEL,
        "smoke": args.smoke,
        "mode_order": ["original", "local_single", "bundle4", "bundle8", "bundle16"],
        "question_bytes": QUESTION_BYTES,
        "request_bytes": BUNDLE_BYTES,
    }
    lock = args.output / "manifest.json"
    if lock.exists() and json.loads(lock.read_text()) != manifest:
        raise ValueError("Experiment inputs or code changed")
    lock.write_text(json.dumps(manifest, indent=2) + "\n")
    lock.chmod(0o600)
    modes = {}
    for name, size in [
        ("original", 1),
        ("local_single", 1),
        ("bundle4", 4),
        ("bundle8", 8),
        ("bundle16", 16),
    ]:
        batches = pack(articles, prompt, size, original=name == "original")
        modes[name] = run_mode(args.output / (name + ".jsonl"), batches, key)
        print(
            json.dumps(
                {"mode": name, "articles": len(articles), "requests": len(batches)}
            ),
            flush=True,
        )
    if not args.smoke:
        result = report(articles, modes)
        path = args.output / "results.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        path.chmod(0o600)
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
