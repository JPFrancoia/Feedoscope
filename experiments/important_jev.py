"""Run an isolated, text-only Jev importance experiment without database access."""

import argparse
from collections import Counter
from collections.abc import Callable
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL = "jev-1.14.0"
SEED = 42
REQUEST_BYTES = 30_000
CLASSES = {"important", "good_only"}


def digest(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any, replace: bool = False) -> None:
    """Write a private JSON file, or replace it when requested."""
    with path.open("w" if replace else "x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete records from a JSONL file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def clean_body(content: str) -> str:
    """Remove markup and non-article script content without shortening the text."""
    soup = BeautifulSoup(content or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def text_key(article: dict[str, Any]) -> str:
    """Group identical bodies, or identical titles when the body is empty."""
    text = article["body"] or article["title"]
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def assign_split(
    important: list[dict[str, Any]],
    good: list[dict[str, Any]],
    profile_count: int = 225,
    eval_count: int = 75,
    good_profile_count: int = 75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Select disjoint classes and text groups before any profiling."""
    rng = random.Random(SEED)
    important_keys = {text_key(row) for row in important}
    good_keys = {text_key(row) for row in good}
    conflicts = important_keys & good_keys
    pools = []
    for rows in (important, good):
        ordered = sorted(rows, key=lambda row: row["article_id"])
        rng.shuffle(ordered)
        seen = set(conflicts)
        unique = []
        for row in ordered:
            key = text_key(row)
            if key not in seen:
                unique.append(row)
                seen.add(key)
        pools.append(unique)
    positives, negatives = pools
    if len(positives) < profile_count + eval_count:
        raise ValueError("Not enough distinct important articles")
    if len(negatives) < good_profile_count + eval_count:
        raise ValueError("Not enough distinct good-only articles")
    evaluation = positives[:eval_count] + negatives[:eval_count]
    profile = (
        positives[eval_count : eval_count + profile_count]
        + negatives[eval_count : eval_count + good_profile_count]
    )
    rng.shuffle(evaluation)
    assert {r["article_id"] for r in profile}.isdisjoint(
        r["article_id"] for r in evaluation
    )
    assert {text_key(r) for r in profile}.isdisjoint(text_key(r) for r in evaluation)
    return (
        profile,
        evaluation,
        {
            "conflicting_text_groups": len(conflicts),
            "distinct_important": len(positives),
            "distinct_good_only": len(negatives),
            "unused_important": len(important) - profile_count - eval_count,
        },
    )


def prepare(snapshot: Path, output: Path) -> None:
    """Verify source files and create private profiling and evaluation sets."""
    for line in (snapshot / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if digest(snapshot / name.strip()) != expected:
            raise ValueError(f"Snapshot checksum mismatch: {name}")
    important_rows = pq.read_table(
        snapshot / "important.parquet", columns=["article_id", "is_important"]
    ).to_pylist()
    if not all(row["is_important"] for row in important_rows):
        raise ValueError("important.parquet contains a non-important row")
    good_rows = pq.read_table(
        snapshot / "good.parquet", columns=["article_id", "is_important"]
    ).to_pylist()
    important_ids = {row["article_id"] for row in important_rows}
    good_important_ids = {row["article_id"] for row in good_rows if row["is_important"]}
    if important_ids != good_important_ids:
        raise ValueError("important.parquet IDs differ from good.parquet labels")
    important: list[dict[str, Any]] = []
    good: list[dict[str, Any]] = []
    for row in pq.read_table(snapshot / "good.parquet").to_pylist():
        article = {
            "article_id": row["article_id"],
            "label": int(row["is_important"]),
            "title": row["title"] or "",
            "body": clean_body(row["content"]),
        }
        (important if row["is_important"] else good).append(article)
    profile, evaluation, counts = assign_split(important, good)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, rows in (("profile", profile), ("eval", evaluation)):
        path = output / f"{name}.jsonl"
        with path.open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        path.chmod(0o600)
    for shard in range(5):
        rows = (
            profile[shard * 45 : (shard + 1) * 45]
            + profile[225 + shard * 15 : 225 + (shard + 1) * 15]
        )
        write_json(output / f"profile_batch_{shard + 1}.json", rows)
    write_json(
        output / "split_manifest.json",
        {
            "seed": SEED,
            "snapshot": str(snapshot.resolve()),
            "source_sha256": {
                name: digest(snapshot / name)
                for name in ("manifest.json", "good.parquet", "important.parquet")
            },
            "split_sha256": {
                name: digest(output / name) for name in ("profile.jsonl", "eval.jsonl")
            },
            "profile_counts": dict(Counter(str(r["label"]) for r in profile)),
            "eval_counts": dict(Counter(str(r["label"]) for r in evaluation)),
            "selection": counts,
            "text_preparation": "BeautifulSoup text, whitespace collapsed, scripts removed",
        },
    )
    print(json.dumps({"output": str(output), "selection": counts}))


def request_payload(article: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    """Build text-only input within a conservative per-question byte budget."""
    payload: dict[str, Any] = {
        "model": MODEL,
        "state": {
            "reader_profile": prompt["reader_profile"],
            "article": {"title": article["title"], "body": article["body"]},
        },
        "questions": {"importance": prompt["question"]},
    }
    body = article["body"].encode()
    original = len(body)
    while len(json.dumps(payload, ensure_ascii=False).encode()) > REQUEST_BYTES:
        if len(body) < 256:
            raise ValueError("Prompt and title exceed request budget")
        body = body[: len(body) * 9 // 10]
        head_size = len(body) * 3 // 4
        tail_size = len(body) - head_size
        source = article["body"].encode()
        payload["state"]["article"]["body"] = (
            source[:head_size].decode(errors="ignore")
            + "\n[Middle omitted for context limit]\n"
            + source[-tail_size:].decode(errors="ignore")
        )
    assert original >= len(body)
    return payload


def validate_answer(response: dict[str, Any]) -> dict[str, Any]:
    """Reject malformed probabilities, model changes, and inconsistent choices."""
    if response.get("model") != MODEL:
        raise ValueError("Unexpected response model")
    answer = response["answers"]["importance"]
    probabilities = answer["probabilities"]
    if answer.get("type") != "choice" or set(probabilities) != CLASSES:
        raise ValueError("Unexpected answer type or classes")
    values = list(probabilities.values()) + [answer["confidence"]]
    if any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in values):
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
    return answer


def load_key(env_file: Path | None) -> str:
    """Read JEV_KEY without loading or displaying unrelated credentials."""
    if os.environ.get("JEV_KEY"):
        return os.environ["JEV_KEY"]
    if env_file:
        for line in env_file.read_text().splitlines():
            match = re.fullmatch(r"\s*(?:export\s+)?JEV_KEY=(.*)", line)
            if match:
                parts = shlex.split(match[1], comments=True)
                if len(parts) == 1 and parts[0]:
                    return parts[0]
    raise ValueError("Set JEV_KEY or supply its env file")


def call_jev(
    payload: dict[str, Any],
    key: str,
    validator: Callable[[dict[str, Any]], Any] = validate_answer,
) -> dict[str, Any]:
    """Call TypeSafe with TLS verification and bounded transient-error retries."""
    request = Request(
        "https://api.typesafe.ai/v1/systemone",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
            validator(result)
            return result
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f"TypeSafe HTTP {error.code}") from error
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Retry limit reached")


def evaluate(output: Path, env_file: Path | None, smoke: bool = False) -> None:
    """Freeze the prompt and append validated responses, with safe resume."""
    manifest = json.loads((output / "split_manifest.json").read_text())
    for name, expected in manifest["split_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Split changed: {name}")
    prompt_path = output / "prompt.json"
    frozen = {
        "prompt_sha256": digest(prompt_path),
        "model": MODEL,
        "request_bytes": REQUEST_BYTES,
    }
    lock = output / "frozen_prompt.json"
    if lock.exists():
        if json.loads(lock.read_text()) != frozen:
            raise ValueError("Frozen prompt changed")
    else:
        write_json(lock, frozen)
    prompt = json.loads(prompt_path.read_text())
    rows = read_jsonl(output / ("profile.jsonl" if smoke else "eval.jsonl"))
    if smoke:
        rows = rows[:1]
    predictions = output / ("smoke.jsonl" if smoke else "predictions.jsonl")
    done = read_jsonl(predictions) if predictions.exists() else []
    ids = [row["article_id"] for row in done]
    if len(ids) != len(set(ids)) or not set(ids) <= {r["article_id"] for r in rows}:
        raise ValueError("Unexpected or duplicate cached prediction IDs")
    by_id = {row["article_id"]: row for row in rows}
    for record in done:
        validate_answer(record["response"])
        payload = request_payload(by_id[record["article_id"]], prompt)
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if record["request_sha256"] != expected:
            raise ValueError("Cached response belongs to a different request")
    key = load_key(env_file)
    with predictions.open("a") as handle:
        predictions.chmod(0o600)
        for article in rows:
            if article["article_id"] in ids:
                continue
            payload = request_payload(article, prompt)
            started = time.monotonic()
            response = call_jev(payload, key)
            record = {
                "article_id": article["article_id"],
                "response": response,
                "request_sha256": hashlib.sha256(
                    json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
                "request_bytes": len(json.dumps(payload, ensure_ascii=False).encode()),
                "truncated": payload["state"]["article"]["body"] != article["body"],
                "seconds": time.monotonic() - started,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            ids.append(article["article_id"])
            print(f"Completed {len(ids)}/{len(rows)}", flush=True)


def metrics(
    labels: list[int], probabilities: list[float], confidences: list[float]
) -> dict[str, Any]:
    """Calculate classification, ranking, calibration, and selective metrics."""
    y, p, confidence = map(np.asarray, (labels, probabilities, confidences))
    predicted = p >= 0.5
    correct = predicted == y
    result = {
        "count": len(y),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix(
            y, predicted, labels=[0, 1]
        ).tolist(),
        "mean_confidence_correct": (
            float(confidence[correct].mean()) if correct.any() else None
        ),
        "mean_confidence_wrong": (
            float(confidence[~correct].mean()) if (~correct).any() else None
        ),
        "confidently_wrong_0.8": int(((confidence >= 0.8) & ~correct).sum()),
        "confidently_wrong_0.9": int(((confidence >= 0.9) & ~correct).sum()),
    }
    bins = []
    for index in range(10):
        mask = (p >= index / 10) & ((p < (index + 1) / 10) if index < 9 else (p <= 1))
        if mask.any():
            bins.append(
                {
                    "lower": index / 10,
                    "upper": (index + 1) / 10,
                    "count": int(mask.sum()),
                    "mean_probability": float(p[mask].mean()),
                    "important_fraction": float(y[mask].mean()),
                }
            )
    result["calibration_bins"] = bins
    result["ece_10_bins"] = sum(
        b["count"] * abs(b["mean_probability"] - b["important_fraction"]) for b in bins
    ) / len(y)
    selective = []
    for threshold in (0, 0.5, 0.8, 0.9):
        mask = confidence >= threshold
        selective.append(
            {
                "confidence_threshold": threshold,
                "count": int(mask.sum()),
                "coverage": float(mask.mean()),
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
                "errors": int((~correct[mask]).sum()),
                "important_count": int(y[mask].sum()),
            }
        )
    result["selective"] = selective
    return result


def report(output: Path) -> None:
    """Require complete evaluation and save metrics with private error details."""
    manifest = json.loads((output / "split_manifest.json").read_text())
    for name, expected in manifest["split_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Split changed: {name}")
    frozen = json.loads((output / "frozen_prompt.json").read_text())
    if frozen["prompt_sha256"] != digest(output / "prompt.json"):
        raise ValueError("Frozen prompt changed")
    prompt = json.loads((output / "prompt.json").read_text())
    rows = read_jsonl(output / "eval.jsonl")
    if Counter(row["label"] for row in rows) != {0: 75, 1: 75}:
        raise ValueError("Require 75 important and 75 good-only labels")
    predictions = read_jsonl(output / "predictions.jsonl")
    by_id = {r["article_id"]: r for r in predictions}
    if (
        len(predictions) != 150
        or len(by_id) != 150
        or set(by_id) != {r["article_id"] for r in rows}
    ):
        raise ValueError("Require exactly 150 distinct evaluation predictions")
    labels, probabilities, confidences, errors = [], [], [], []
    for row in rows:
        record = by_id[row["article_id"]]
        payload = request_payload(row, prompt)
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if record["request_sha256"] != expected:
            raise ValueError("Response belongs to a different request")
        answer = validate_answer(record["response"])
        probability = answer["probabilities"]["important"]
        labels.append(row["label"])
        probabilities.append(probability)
        confidences.append(answer["confidence"])
        if int(probability >= 0.5) != row["label"]:
            errors.append(
                {
                    "article_id": row["article_id"],
                    "title": row["title"],
                    "label": row["label"],
                    "p_important": probability,
                    "confidence": answer["confidence"],
                }
            )
    result = metrics(labels, probabilities, confidences)
    result["model"] = MODEL
    result["prompt_sha256"] = frozen["prompt_sha256"]
    result["split_sha256"] = manifest["split_sha256"]
    result["prediction_sha256"] = digest(output / "predictions.jsonl")
    result["log_loss_probability_clip"] = float(np.finfo(float).eps)
    result["baseline_constant_0.5"] = {
        "brier": 0.25,
        "log_loss": math.log(2),
        "average_precision": 0.5,
        "roc_auc": 0.5,
    }
    result["truncated_articles"] = sum(r["truncated"] for r in predictions)
    result["input_tokens"] = sum(
        r["response"]["usage"]["input_tokens"] for r in predictions
    )
    result["estimated_usd_at_0.042_per_million_input_tokens"] = (
        result["input_tokens"] * 0.042 / 1_000_000
    )
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(labels), np.asarray(probabilities)
    samples: dict[str, list[float]] = {
        "accuracy": [],
        "roc_auc": [],
        "average_precision": [],
        "brier": [],
        "log_loss": [],
    }
    for _ in range(2000):
        indices = np.concatenate(
            [
                rng.choice(np.flatnonzero(y == label), 75, replace=True)
                for label in (0, 1)
            ]
        )
        yy, pp = y[indices], p[indices]
        samples["accuracy"].append(float(accuracy_score(yy, pp >= 0.5)))
        samples["roc_auc"].append(float(roc_auc_score(yy, pp)))
        samples["average_precision"].append(float(average_precision_score(yy, pp)))
        samples["brier"].append(float(brier_score_loss(yy, pp)))
        samples["log_loss"].append(float(log_loss(yy, pp, labels=[0, 1])))
    result["stratified_bootstrap_95_percent_intervals_2000_samples"] = {
        name: np.quantile(values, [0.025, 0.975]).tolist()
        for name, values in samples.items()
    }
    result["errors_descending_confidence"] = sorted(
        errors, key=lambda row: row["confidence"], reverse=True
    )
    write_json(output / "metrics.json", result, replace=True)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "errors_descending_confidence"},
            indent=2,
        )
    )


def main() -> None:
    """Run a requested experiment stage."""
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "smoke", "evaluate", "report"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        if not args.snapshot:
            parser.error("prepare requires --snapshot")
        prepare(args.snapshot, args.output)
    elif args.stage in ("evaluate", "smoke"):
        evaluate(args.output, args.env_file, smoke=args.stage == "smoke")
    else:
        report(args.output)


if __name__ == "__main__":
    main()
