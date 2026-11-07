import argparse
import asyncio
import datetime as dt
from email.message import Message
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.error import HTTPError, URLError

import pytest

from feedoscope import importance as imp
from feedoscope.entities import ImportanceArticle, ImportancePrompt

MODEL = "jev-1.14.0"


def prompt() -> dict[str, Any]:
    return {
        "reader_profile": "Useful engineering methods matter.",
        "question": {
            "type": "choice",
            "instructions": "Is this article important?",
            "criteria": {"important": "Useful", "good_only": "Ordinary"},
        },
    }


def response(p: float = 0.9) -> dict[str, Any]:
    return {
        "model": MODEL,
        "answers": {
            "importance": {
                "type": "choice",
                "choice": "important" if p >= 0.5 else "good_only",
                "probabilities": {"important": p, "good_only": 1 - p},
                "confidence": abs(2 * p - 1),
            }
        },
    }


def bundle_response(payload: dict[str, Any], p: float = 0.9) -> dict[str, Any]:
    return {
        "model": MODEL,
        "answers": {
            key: response(p)["answers"]["importance"] for key in payload["questions"]
        },
    }


def version(id: int = 1) -> ImportancePrompt:
    return ImportancePrompt(
        id=id, prompt=prompt(), model=MODEL, created_at=dt.datetime.now(dt.UTC)
    )


def test_request_preserves_prompt_and_sends_only_cleaned_text() -> None:
    article = ImportanceArticle(
        entry_id=123,
        title="A title",
        content="<script>secret</script><p>Body &amp; text</p><style>hidden</style>",
    )
    payload = imp.request_payload(article, prompt(), MODEL)
    assert payload == {
        "model": MODEL,
        "state": {
            "reader_profile": prompt()["reader_profile"],
            "article": {"title": "A title", "body": "Body & text"},
        },
        "questions": {"importance": prompt()["question"]},
    }
    assert "123" not in json.dumps(payload)


def test_context_limit_preserves_unicode_head_and_tail() -> None:
    article = ImportanceArticle(
        entry_id=1, title="Title", content="HEAD " + "é界 " * 50_000 + " TAIL"
    )
    payload = imp.request_payload(article, prompt(), MODEL)
    body = payload["state"]["article"]["body"]
    assert len(json.dumps(payload, ensure_ascii=False).encode()) <= imp.REQUEST_BYTES
    assert body.startswith("HEAD") and body.endswith("TAIL")
    assert "[Middle omitted for context limit]" in body
    assert "\ufffd" not in body


@pytest.mark.parametrize(
    "change", ["profile", "question", "classes", "instructions", "oversized", "extra"]
)
def test_invalid_prompts_fail_before_use(change: str) -> None:
    p = prompt()
    if change == "profile":
        p["reader_profile"] = ""
    elif change == "question":
        p["question"]["type"] = "noul"
    elif change == "classes":
        p["question"]["criteria"] = {"yes": "yes", "no": "no"}
    elif change == "instructions":
        p["question"]["instructions"] = None
    elif change == "oversized":
        p["reader_profile"] = "x" * 40_000
    else:
        p["unexpected"] = "unused input"
    with pytest.raises(ValueError):
        imp.validate_prompt(p, MODEL)


def test_only_fixed_model_versions_are_accepted() -> None:
    imp.validate_prompt(prompt(), MODEL)
    for model in ["jev-latest", "", "gpt-6-luna"]:
        with pytest.raises(ValueError, match="fixed Jev"):
            imp.validate_prompt(prompt(), model)


@pytest.mark.parametrize("p", [0, 0.1, 0.5, 0.9, 1])
def test_raw_probability_and_confidence_are_both_preserved(p: float) -> None:
    result = imp.validate_response(response(p), MODEL)
    assert result.importance_probability == p
    assert result.confidence == abs(2 * p - 1)
    assert result.is_important == (p >= 0.5)
    assert result.inferred_at.tzinfo is not None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.9"])
def test_invalid_numbers_are_rejected(value: Any) -> None:
    for field in ["probability", "confidence"]:
        data = response()
        answer = data["answers"]["importance"]
        if field == "probability":
            answer["probabilities"]["important"] = value
        else:
            answer["confidence"] = value
        with pytest.raises(ValueError):
            imp.validate_response(data, MODEL)


@pytest.mark.parametrize("change", ["model", "choice", "sum", "confidence", "shape"])
def test_malformed_or_inconsistent_responses_are_rejected(change: str) -> None:
    data = response()
    answer = data["answers"]["importance"]
    if change == "model":
        data["model"] = "jev-1.13.0"
    elif change == "choice":
        answer["choice"] = "good_only"
    elif change == "sum":
        answer["probabilities"]["good_only"] = 0.5
    elif change == "confidence":
        answer["confidence"] = 0.1
    else:
        data = {}
    with pytest.raises(ValueError):
        imp.validate_response(data, MODEL)


def test_tied_choices_are_valid_but_have_zero_confidence() -> None:
    data = response(0.5)
    data["answers"]["importance"]["choice"] = "good_only"
    assert not imp.validate_response(data, MODEL).is_important


def test_http_retries_and_validates_the_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = HTTPError(
        "https://api.typesafe.ai", 429, "limited", Message(), io.BytesIO(b"private")
    )
    stream = MagicMock()
    stream.__enter__.return_value = io.StringIO(json.dumps(response()))
    calls = MagicMock(side_effect=[error, stream])
    monkeypatch.setattr(imp, "urlopen", calls)
    sleep = MagicMock()
    monkeypatch.setattr(imp.time, "sleep", sleep)
    result = imp.call_jev({"model": MODEL, "questions": {"importance": {}}}, "fake-key")
    assert result["importance"].importance_probability == 0.9
    assert calls.call_count == 2
    request = calls.call_args.args[0]
    assert request.full_url == "https://api.typesafe.ai/v1/systemone"
    assert request.get_header("Authorization") == "Bearer fake-key"
    assert calls.call_args.kwargs == {"timeout": 120}
    sleep.assert_called_once_with(2)


def test_http_failure_does_not_disclose_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = HTTPError(
        "https://api.typesafe.ai", 401, "failed", Message(), io.BytesIO(b"private body")
    )
    monkeypatch.setattr(imp, "urlopen", MagicMock(side_effect=error))
    with pytest.raises(RuntimeError, match="TypeSafe HTTP 401") as raised:
        imp.call_jev({"model": MODEL, "questions": {"importance": {}}}, "fake-key")
    assert "private" not in str(raised.value)
    assert raised.value.__suppress_context__


def test_connection_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = MagicMock(side_effect=URLError("connection failed"))
    monkeypatch.setattr(imp, "urlopen", calls)
    monkeypatch.setattr(imp.time, "sleep", MagicMock())
    with pytest.raises(RuntimeError, match="four attempts"):
        imp.call_jev({"model": MODEL, "questions": {"importance": {}}}, "fake-key")
    assert calls.call_count == 4


def setup_infer(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[AsyncMock, AsyncMock, MagicMock]:
    monkeypatch.setattr(imp.config, "JEV_KEY", "fake-key")
    latest = AsyncMock(side_effect=[version(1), version(2)])
    monkeypatch.setattr(imp.dr, "get_latest_importance_prompt", latest)
    articles = [
        ImportanceArticle(entry_id=i, title=f"Title {i}", content="Text")
        for i in [1, 2]
    ]
    monkeypatch.setattr(
        imp.dr, "get_importance_articles", AsyncMock(return_value=articles)
    )
    write = AsyncMock(return_value=True)
    monkeypatch.setattr(imp.dr, "upsert_entry_importance", write)
    call = MagicMock(
        side_effect=lambda payload, key: imp.validate_answers(
            bundle_response(payload), payload
        )
    )
    monkeypatch.setattr(imp, "call_jev", call)
    return latest, write, call


def test_run_keeps_one_prompt_version_and_deduplicates_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest, write, call = setup_infer(monkeypatch)
    rows = asyncio.run(imp.infer([1, 2, 1]))
    latest.assert_awaited_once()
    get_articles = imp.dr.get_importance_articles
    assert isinstance(get_articles, AsyncMock)
    get_articles.assert_awaited_once_with([1, 2])
    assert [args.args[1] for args in write.await_args_list] == [1, 1]
    assert len(rows) == 2 and call.call_count == 1


def test_failure_preserves_successes_without_writing_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, write, call = setup_infer(monkeypatch)
    monkeypatch.setattr(imp, "BUNDLE_SIZE", 1)
    call.side_effect = [
        {"article_0": imp.validate_response(response(), MODEL)},
        RuntimeError("failed"),
    ]
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(imp.infer([1, 2]))
    write.assert_awaited_once()
    assert write.await_args is not None
    assert write.await_args.args[0] == 1


def test_missing_entries_fail_before_paid_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, write, call = setup_infer(monkeypatch)
    with pytest.raises(ValueError, match="missing or removed"):
        asyncio.run(imp.infer([1, 2, 3]))
    write.assert_not_awaited()
    call.assert_not_called()


def test_missing_prompt_or_key_prevents_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    latest, _, call = setup_infer(monkeypatch)
    latest.side_effect = RuntimeError("No importance prompt exists")
    with pytest.raises(RuntimeError, match="No importance prompt"):
        asyncio.run(imp.infer([1]))
    call.assert_not_called()
    monkeypatch.setattr(imp.config, "JEV_KEY", "")
    with pytest.raises(ValueError, match="Set JEV_KEY"):
        asyncio.run(imp.infer([1]))


def test_command_imports_private_prompt_without_printing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "prompt.json"
    path.write_text(json.dumps(prompt()))
    pool = MagicMock(open=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(imp.dr, "global_pool", pool)
    insert = AsyncMock(return_value=version())
    monkeypatch.setattr(imp.dr, "insert_importance_prompt", insert)
    asyncio.run(
        imp.run(argparse.Namespace(command="add-prompt", file=path, model=MODEL))
    )
    insert.assert_awaited_once_with(prompt(), MODEL)
    assert json.loads(capsys.readouterr().out) == {
        "importance_prompt_id": 1,
        "model": MODEL,
    }
    pool.close.assert_awaited_once()


def test_command_closes_pool_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = MagicMock(open=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(imp.dr, "global_pool", pool)
    monkeypatch.setattr(imp, "infer", AsyncMock(side_effect=RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(imp.run(argparse.Namespace(command="infer", entry_id=[1])))
    pool.close.assert_awaited_once()
