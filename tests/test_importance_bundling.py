import copy
import io
import json
import math
from pathlib import Path
from typing import Any

import pytest

from experiments import importance_bundling as b


def prompt() -> dict[str, Any]:
    return {
        "reader_profile": "Useful engineering methods matter.",
        "question": {
            "type": "choice",
            "instructions": "Using reader_profile, classify article.",
            "criteria": {"important": "Useful", "good_only": "Ordinary"},
        },
    }


def article(i: int) -> dict[str, Any]:
    return {
        "article_id": i,
        "label": i % 2,
        "title": f"Article {i}",
        "body": f"Body {i}",
    }


def response(payload: dict[str, Any], p: float = 0.9) -> dict[str, Any]:
    return {
        "model": b.MODEL,
        "answers": {
            key: {
                "type": "choice",
                "choice": "important" if p >= 0.5 else "good_only",
                "confidence": abs(2 * p - 1),
                "probabilities": {"important": p, "good_only": 1 - p},
            }
            for key in reversed(payload["questions"])
        },
        "usage": {"input_tokens": 50, "output_tokens": 32},
    }


def test_articles_are_local_and_original_prompt_is_unchanged() -> None:
    original = [b.original_payload(article(i), prompt()) for i in range(3)]
    before = copy.deepcopy(original)
    bundle = b.bundle_payload(original)
    assert original == before
    assert bundle["state"] == {"reader_profile": prompt()["reader_profile"]}
    for i, question in enumerate(bundle["questions"].values()):
        assert question["criteria"] == prompt()["question"]["criteria"]
        assert question["instructions"] == {
            "instructions": prompt()["question"]["instructions"],
            "article": original[i]["state"]["article"],
        }
        assert "article_id" not in json.dumps(question)
        assert "label" not in json.dumps(question)


def test_packing_respects_bytes_count_and_preserves_long_unicode_text() -> None:
    rows = [article(i) for i in range(20)]
    rows[2]["body"] = "é" * 70_000
    rows[3]["body"] = "漢" * 70_000
    rows[4]["body"] = "A" * 70_000
    batches = b.pack(rows, prompt(), 8)
    assert [i for batch in batches for i in batch["ids"]] == list(range(20))
    assert len(batches[0]["ids"]) < 8
    by_id = {r["article_id"]: b.original_payload(r, prompt()) for r in rows}
    for batch in batches:
        assert len(batch["ids"]) <= 8
        assert len(b.encoded(batch["payload"])) <= b.BUNDLE_BYTES
        for i, question in zip(
            batch["ids"], batch["payload"]["questions"].values(), strict=True
        ):
            assert question["instructions"]["article"] == by_id[i]["state"]["article"]
            assert (
                len(
                    b.encoded(
                        {"state": batch["payload"]["state"], "question": question}
                    )
                )
                <= b.QUESTION_BYTES
            )
    with pytest.raises(ValueError, match="positive"):
        b.pack(rows, prompt(), 0)


def test_bad_profiles_empty_and_oversize_bundles_fail() -> None:
    one = b.original_payload(article(1), prompt())
    two = copy.deepcopy(one)
    two["state"]["reader_profile"] = "Different"
    with pytest.raises(ValueError, match="match"):
        b.bundle_payload([one, two])
    with pytest.raises(ValueError, match="at least"):
        b.bundle_payload([])
    with pytest.raises(ValueError, match="budget"):
        b.bundle_payload([one] * 1000)


@pytest.mark.parametrize("bad", ["missing", "extra", "model", "nan", "confidence"])
def test_bundle_rejects_invalid_answers(bad: str) -> None:
    payload = b.bundle_payload([b.original_payload(article(1), prompt())])
    result = response(payload)
    if bad == "missing":
        result["answers"] = {}
    elif bad == "extra":
        result["answers"]["other"] = result["answers"]["article_0"]
    elif bad == "model":
        result["model"] = "jev-latest"
    elif bad == "nan":
        result["answers"]["article_0"]["probabilities"]["important"] = float("nan")
    else:
        result["answers"]["article_0"]["confidence"] = 0.1
    with pytest.raises(ValueError):
        b.validate(result, payload)


def test_resume_checks_cache_and_does_not_repeat_paid_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches = b.pack([article(1), article(2)], prompt(), 4)
    payload = batches[0]["payload"]
    calls = []

    def fake_open(*args: Any, **kwargs: Any) -> io.BytesIO:
        calls.append(args)
        return io.BytesIO(json.dumps(response(payload)).encode())

    monkeypatch.setattr(b, "urlopen", fake_open)
    path = tmp_path / "responses.jsonl"
    first = b.run_mode(path, batches, "test")
    assert b.run_mode(path, batches, "test") == first
    assert len(calls) == 1
    changed = copy.deepcopy(first[0])
    changed["ids"].reverse()
    path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="differs"):
        b.run_mode(path, batches, "test")
    first[0]["question_keys"].reverse()
    path.write_text(json.dumps(first[0]) + "\n")
    with pytest.raises(ValueError, match="differs"):
        b.run_mode(path, batches, "test")
    path.write_text('{"incomplete"')
    before = path.read_bytes()
    with pytest.raises(json.JSONDecodeError):
        b.run_mode(path, batches, "test")
    assert path.read_bytes() == before


def test_report_uses_named_mapping_not_response_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [article(i) for i in range(4)]
    batches = b.pack(rows, prompt(), 4)

    def fake_open(*args: Any, **kwargs: Any) -> io.BytesIO:
        value = response(batches[0]["payload"])
        for i, answer in enumerate(reversed(list(value["answers"].values()))):
            p = 0.9 if i % 2 else 0.1
            answer["probabilities"] = {"important": p, "good_only": 1 - p}
            answer["choice"] = "important" if p > 0.5 else "good_only"
            answer["confidence"] = 0.8
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(b, "urlopen", fake_open)
    records = b.run_mode(tmp_path / "run.jsonl", batches, "test")
    result = b.report(rows, {"original": records, "bundle4": records})
    assert (
        result["original"]["accuracy"] == result["original"]["average_precision"] == 1.0
    )
    assert result["bundle4"]["decision_changes"] == 0
    assert result["bundle4"]["usage_complete"]
    assert result["bundle4"]["paired_ap_difference_95"] == [0.0, 0.0]


@pytest.mark.parametrize(
    "usage",
    [
        {},
        None,
        {"input_tokens": None},
        {"input_tokens": True},
        {"input_tokens": -1},
        {"input_tokens": 50},
        {"input_tokens": 50, "output_tokens": 32},
    ],
)
def test_report_usage_and_known_metrics(usage: Any) -> None:
    rows = [article(0), article(1)]
    records = []
    for row in rows:
        payload = b.original_payload(row, prompt())
        value = response(payload, 0.8 if row["label"] else 0.2)
        value["usage"] = usage
        records.append(
            {
                "ids": [row["article_id"]],
                "question_keys": ["importance"],
                "response": value,
                "seconds": 0.25,
                "attempts": 1,
            }
        )
    result = b.report(rows, {"original": records})["original"]
    assert result["seconds"] == 0.5
    assert result["accuracy"] == result["average_precision"] == result["roc_auc"] == 1
    assert result["brier"] == pytest.approx(0.04)
    assert result["log_loss"] == pytest.approx(-math.log(0.8))
    complete = usage == {"input_tokens": 50, "output_tokens": 32}
    assert result["usage_complete"] == complete
    if complete:
        assert result["estimated_cost_usd"] == pytest.approx(100 * 0.042 / 1_000_000)
    else:
        assert result["estimated_cost_usd"] is None
