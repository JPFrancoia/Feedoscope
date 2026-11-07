"""Check normal inference bundles without sending private data or paid requests."""

import asyncio
import copy
import io
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_importance import MODEL, bundle_response, prompt, setup_infer
from test_importance_recent import recent_setup

from feedoscope import importance as imp
from feedoscope.entities import ImportanceArticle

REAL_CALL_JEV = imp.call_jev


def prepared(count: int, content: str = "Body") -> list[tuple[int, dict[str, Any]]]:
    return [
        (
            i,
            imp.request_payload(
                ImportanceArticle(entry_id=i, title=f"Title {i}", content=content),
                prompt(),
                MODEL,
            ),
        )
        for i in range(1, count + 1)
    ]


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode()


def test_bundles_keep_articles_local_and_limit_count() -> None:
    items = prepared(17)
    batches = imp.pack_requests(items)
    assert [len(b) for b in batches] == [8, 8, 1]
    for batch in batches:
        payload = imp.bundle_payload([p for _, p in batch])
        assert payload["state"] == {"reader_profile": prompt()["reader_profile"]}
        assert len(encoded(payload)) <= imp.BUNDLE_BYTES
        for index, (_, original) in enumerate(batch):
            question = payload["questions"][f"article_{index}"]
            assert question["criteria"] == prompt()["question"]["criteria"]
            assert (
                question["instructions"]["instructions"]
                == prompt()["question"]["instructions"]
            )
            assert question["instructions"]["article"] == original["state"]["article"]
            assert (
                len(encoded({"state": payload["state"], "question": question}))
                <= imp.QUESTION_BYTES
            )
    assert imp.pack_requests([]) == []


def test_packing_splits_on_bytes_without_further_truncation() -> None:
    items = prepared(7, "HEAD " + "é界 " * 50_000 + " TAIL")
    before = copy.deepcopy(items)
    batches = imp.pack_requests(items)
    assert len(batches) > 1
    assert items == before
    assert [item for batch in batches for item in batch] == items
    for batch in batches:
        assert len(batch) < imp.BUNDLE_SIZE
        payload = imp.bundle_payload([p for _, p in batch])
        assert len(encoded(payload)) <= imp.BUNDLE_BYTES
        for question in payload["questions"].values():
            assert (
                len(encoded({"state": payload["state"], "question": question}))
                <= imp.QUESTION_BYTES
            )
            body = question["instructions"]["article"]["body"]
            assert (
                body.startswith("HEAD")
                and body.endswith("TAIL")
                and "\ufffd" not in body
            )


@pytest.mark.parametrize("kind", ["request", "question", "count", "model", "profile"])
def test_bundle_rejects_bad_limits_and_mixed_inputs(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = [p for _, p in prepared(2)]
    if kind == "request":
        monkeypatch.setattr(
            imp, "BUNDLE_BYTES", len(encoded(imp.bundle_payload(payloads))) - 1
        )
    elif kind == "question":
        monkeypatch.setattr(imp, "QUESTION_BYTES", 10)
    elif kind == "count":
        payloads *= 5
    elif kind == "model":
        payloads[-1]["model"] = "jev-1.13.0"
    else:
        payloads[-1]["state"]["reader_profile"] = "Different reader"
    with pytest.raises(ValueError):
        imp.bundle_payload(payloads)


def test_exact_byte_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [p for _, p in prepared(2)]
    payload = imp.bundle_payload(items)
    monkeypatch.setattr(imp, "BUNDLE_BYTES", len(encoded(payload)))
    monkeypatch.setattr(
        imp,
        "QUESTION_BYTES",
        max(
            len(encoded({"state": payload["state"], "question": q}))
            for q in payload["questions"].values()
        ),
    )
    assert imp.bundle_payload(items) == payload


@pytest.mark.parametrize("limit", ["QUESTION_BYTES", "BUNDLE_BYTES"])
def test_invalid_later_input_prevents_all_paid_calls(
    monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    _, write, call = setup_infer(monkeypatch)
    monkeypatch.setattr(
        imp.dr,
        "get_importance_articles",
        AsyncMock(
            return_value=[
                ImportanceArticle(
                    entry_id=i, title="Title", content="x" * (2000 if i == 10 else 10)
                )
                for i in range(1, 11)
            ]
        ),
    )
    monkeypatch.setattr(imp, limit, 1000)
    with pytest.raises(ValueError, match="context budget"):
        asyncio.run(imp.infer(list(range(1, 11))))
    call.assert_not_called()
    write.assert_not_awaited()


@pytest.mark.parametrize(
    "change", ["missing", "extra", "invalid", "model", "not_object"]
)
def test_invalid_bundle_saves_nothing(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, write, _ = setup_infer(monkeypatch)

    def send(request: Any, timeout: int) -> Any:
        payload = json.loads(request.data)
        data: Any = bundle_response(payload)
        if change == "missing":
            del data["answers"]["article_1"]
        elif change == "extra":
            data["answers"]["extra"] = data["answers"]["article_0"]
        elif change == "invalid":
            data["answers"]["article_1"]["confidence"] = -1
        elif change == "model":
            data["model"] = "jev-1.13.0"
        else:
            data = []
        stream = MagicMock()
        stream.__enter__.return_value = io.StringIO(json.dumps(data))
        return stream

    monkeypatch.setattr(imp, "call_jev", REAL_CALL_JEV)
    send_mock = MagicMock(side_effect=send)
    monkeypatch.setattr(imp, "urlopen", send_mock)
    with pytest.raises(ValueError):
        asyncio.run(imp.infer([1, 2]))
    write.assert_not_awaited()
    send_mock.assert_called_once()


def test_reordered_answers_save_to_correct_articles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, write, _ = setup_infer(monkeypatch)

    def send(request: Any, timeout: int) -> Any:
        payload = json.loads(request.data)
        positive = bundle_response(payload, 0.9)["answers"]["article_0"]
        negative = bundle_response(payload, 0.1)["answers"]["article_1"]
        data = {
            "model": MODEL,
            "answers": {"article_1": negative, "article_0": positive},
        }
        stream = MagicMock()
        stream.__enter__.return_value = io.StringIO(json.dumps(data))
        return stream

    monkeypatch.setattr(imp, "call_jev", REAL_CALL_JEV)
    send_mock = MagicMock(side_effect=send)
    monkeypatch.setattr(imp, "urlopen", send_mock)
    rows = asyncio.run(imp.infer([1, 2]))
    assert [row["is_important"] for row in rows] == [True, False]
    assert [
        (a.args[0], a.args[2].importance_probability) for a in write.await_args_list
    ] == [(1, 0.9), (2, 0.1)]
    send_mock.assert_called_once()


def test_recent_usage_counts_requests_not_articles(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, insert = recent_setup(monkeypatch)

    def send(request: Any, timeout: int) -> Any:
        payload = json.loads(request.data)
        data = bundle_response(payload)
        data["usage"] = {"input_tokens": 123, "output_tokens": 64}
        stream = MagicMock()
        stream.__enter__.return_value = io.StringIO(json.dumps(data))
        return stream

    monkeypatch.setattr(imp, "call_jev", REAL_CALL_JEV)
    send_mock = MagicMock(side_effect=send)
    monkeypatch.setattr(imp, "urlopen", send_mock)
    asyncio.run(imp.infer_recent(7))
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["stored"] == 2 and insert.await_count == 2
    assert summary["requests"] == summary["responses"] == 1
    assert summary["input_tokens"] == 123 and summary["output_tokens"] == 64
    assert summary["usage_complete"]
    assert summary["estimated_cost_usd"] == pytest.approx(123 * 0.042 / 1_000_000)
    send_mock.assert_called_once()


def test_recent_empty_bundles_make_no_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    call, insert = recent_setup(monkeypatch)
    monkeypatch.setattr(imp.dr, "has_entry_importance", AsyncMock(return_value=True))
    asyncio.run(imp.infer_recent(7))
    call.assert_not_called()
    insert.assert_not_awaited()
    summary = json.loads(capsys.readouterr().out)
    assert summary["skipped"] == 2 and summary["requests"] == 0


def test_recent_later_bundle_rechecks_and_keeps_concurrent_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    call, insert = recent_setup(monkeypatch)
    monkeypatch.setattr(
        imp.dr,
        "get_recent_missing_importance_articles",
        AsyncMock(
            return_value=[
                ImportanceArticle(entry_id=i, title="Title", content="Body")
                for i in range(1, 11)
            ]
        ),
    )
    has = AsyncMock(side_effect=[False] * 8 + [True, False])
    monkeypatch.setattr(imp.dr, "has_entry_importance", has)
    # Another writer can also insert after the pre-request check.
    insert.side_effect = [True] * 8 + [False]
    asyncio.run(imp.infer_recent(7))
    assert call.call_count == 2
    assert len(call.call_args.args[0]["questions"]) == 1
    assert [a.args[0] for a in insert.await_args_list] == [*range(1, 9), 10]
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["stored"] == 8 and summary["skipped"] == 2
