"""Check missing-only inference and API usage without live requests."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import datetime as dt
import io
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_importance import MODEL, bundle_response, prompt, response

from feedoscope import importance as imp
from feedoscope.entities import ImportanceArticle, ImportancePrompt, ImportanceResult


def recent_setup(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, AsyncMock]:
    cursor = MagicMock(fetchone=AsyncMock(return_value={"acquired": True}))
    conn = MagicMock(execute=AsyncMock(return_value=cursor))

    @asynccontextmanager
    async def connection() -> AsyncIterator[MagicMock]:
        yield conn

    monkeypatch.setattr(imp.dr, "global_pool", MagicMock(connection=connection))
    monkeypatch.setattr(imp.config, "JEV_KEY", "synthetic-key")
    monkeypatch.setattr(
        imp.dr,
        "get_latest_importance_prompt",
        AsyncMock(
            return_value=ImportancePrompt(
                id=2, model=MODEL, prompt=prompt(), created_at=dt.datetime.now(dt.UTC)
            )
        ),
    )
    monkeypatch.setattr(
        imp.dr,
        "get_recent_missing_importance_articles",
        AsyncMock(
            return_value=[
                ImportanceArticle(entry_id=i, title="Synthetic", content="Body")
                for i in (1, 2)
            ]
        ),
    )
    monkeypatch.setattr(imp.dr, "has_entry_importance", AsyncMock(return_value=False))
    insert = AsyncMock(return_value=True)
    monkeypatch.setattr(imp.dr, "insert_entry_importance", insert)

    def call(
        payload: dict[str, Any], key: str, usage: dict[str, int]
    ) -> dict[str, ImportanceResult]:
        usage["requests"] += 1
        usage["responses"] += 1
        usage["input_tokens"] += 100
        usage["output_tokens"] += 32
        return imp.validate_answers(bundle_response(payload), payload)

    mocked = MagicMock(side_effect=call)
    monkeypatch.setattr(imp, "call_jev", mocked)
    return mocked, insert


def test_recent_skips_records_added_after_selection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    call, insert = recent_setup(monkeypatch)
    monkeypatch.setattr(
        imp.dr, "has_entry_importance", AsyncMock(side_effect=[True, False])
    )
    asyncio.run(imp.infer_recent(7))
    call.assert_called_once()
    insert.assert_awaited_once()
    assert insert.await_args is not None
    assert insert.await_args.args[:2] == (2, 2)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summary = rows[-1]
    assert (
        summary["selected"] == 2 and summary["skipped"] == 1 and summary["stored"] == 1
    )
    assert summary["status"] == "complete" and summary["usage_complete"]
    assert summary["input_tokens"] == 100 and summary["output_tokens"] == 32
    assert summary["estimated_cost_usd"] == pytest.approx(0.0000042)
    assert summary["elapsed_seconds"] >= 0
    query = imp.dr.get_recent_missing_importance_articles
    assert isinstance(query, AsyncMock)
    assert query.await_args is not None
    assert query.await_args.args[0] == 7
    assert query.await_args.args[1].tzinfo == dt.UTC
    latest = imp.dr.get_latest_importance_prompt
    assert isinstance(latest, AsyncMock)
    latest.assert_awaited_once()


def test_recent_failure_keeps_success_and_prints_incomplete_cost(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    call, insert = recent_setup(monkeypatch)
    monkeypatch.setattr(imp, "BUNDLE_SIZE", 1)
    success = call.side_effect

    def fail(
        payload: dict[str, Any], key: str, usage: dict[str, int]
    ) -> dict[str, ImportanceResult]:
        if call.call_count == 1:
            assert callable(success)
            return success(payload, key, usage)
        usage["requests"] += 1
        usage["failed_requests"] += 1
        raise RuntimeError("synthetic failure")

    call.side_effect = fail
    with pytest.raises(RuntimeError, match="synthetic failure"):
        asyncio.run(imp.infer_recent(7))
    insert.assert_awaited_once()
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["status"] == "failed" and summary["stored"] == 1
    assert summary["requests"] == 2 and not summary["usage_complete"]
    assert summary["failed_requests"] == 1


def test_recent_excludes_concurrent_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    call, insert = recent_setup(monkeypatch)

    # Replace the lock result without opening another database connection.
    @asynccontextmanager
    async def locked() -> AsyncIterator[MagicMock]:
        yield MagicMock(
            execute=AsyncMock(
                return_value=MagicMock(
                    fetchone=AsyncMock(return_value={"acquired": False})
                )
            )
        )

    monkeypatch.setattr(imp.dr, "global_pool", MagicMock(connection=locked))
    with pytest.raises(RuntimeError, match="Another recent"):
        asyncio.run(imp.infer_recent(7))
    call.assert_not_called()
    insert.assert_not_awaited()
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


@pytest.mark.parametrize("days", [0, -1, True, 1.5])
def test_recent_invalid_days_prevent_calls(
    monkeypatch: pytest.MonkeyPatch, days: Any
) -> None:
    call, insert = recent_setup(monkeypatch)
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(imp.infer_recent(days))
    call.assert_not_called()
    insert.assert_not_awaited()


@pytest.mark.parametrize(
    "usage", [{"input_tokens": 123, "output_tokens": 32}, {}, {"input_tokens": True}]
)
def test_call_counts_reported_tokens(
    monkeypatch: pytest.MonkeyPatch, usage: dict[str, Any]
) -> None:
    data = response()
    data["usage"] = usage
    stream = MagicMock()
    stream.__enter__.return_value = io.StringIO(json.dumps(data))
    monkeypatch.setattr(imp, "urlopen", MagicMock(return_value=stream))
    totals = dict.fromkeys(
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
    imp.call_jev(
        {"model": MODEL, "questions": {"importance": {}}}, "synthetic-key", totals
    )
    assert totals["requests"] == totals["responses"] == 1
    expected = usage.get("input_tokens")
    valid = type(expected) is int
    assert totals["input_tokens"] == (expected if valid else 0)
    assert totals["missing_input_tokens"] == (0 if valid else 1)


@pytest.mark.parametrize("data", [None, []])
def test_non_object_response_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, data: Any
) -> None:
    stream = MagicMock()
    stream.__enter__.return_value = io.StringIO(json.dumps(data))
    monkeypatch.setattr(imp, "urlopen", MagicMock(return_value=stream))
    totals = dict.fromkeys(
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
    with pytest.raises(ValueError, match="Malformed Jev response"):
        imp.call_jev({"model": MODEL}, "synthetic", totals)
    assert totals["failed_requests"] == 1
    assert totals["missing_input_tokens"] == 1


def test_missing_output_usage_is_not_complete(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    call, _ = recent_setup(monkeypatch)
    original = call.side_effect

    def missing(
        payload: dict[str, Any], key: str, usage: dict[str, int]
    ) -> dict[str, ImportanceResult]:
        assert callable(original)
        result = original(payload, key, usage)
        usage["missing_output_tokens"] += 1
        return result

    call.side_effect = missing
    asyncio.run(imp.infer_recent(7))
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert not summary["usage_complete"]
    assert summary["missing_output_tokens"] == 1
