import asyncio
import datetime as dt
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from feedoscope import export_snapshot as snapshot


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status in ("read", "unread"):
        for vote in (-1, 0, 1):
            for starred in (False, True):
                row: dict[str, Any] = dict.fromkeys(snapshot.SCHEMA.names)
                row.update(
                    article_id=len(rows) + 1,
                    title="A title with café",
                    content="<p>First line.</p>\n<p>Second line.</p>",
                    published_at=dt.datetime(2026, 11, 6, tzinfo=dt.UTC),
                    status=status,
                    vote=vote,
                    starred=starred,
                    tags=None,
                    user_tags=["A tag"],
                    is_good=status == "read" and vote >= 0,
                    is_bad=vote == -1,
                    is_important=vote >= 0 and (starred or vote == 1),
                )
                rows.append(row)
    return rows


def _connection(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, Any]],
    **source_changes: Any,
) -> tuple[AsyncMock, MagicMock, MagicMock]:
    source = {
        "articles": len(rows),
        "good": sum(row["is_good"] for row in rows),
        "bad": sum(row["is_bad"] for row in rows),
        "important": sum(row["is_important"] for row in rows),
        "important_read": sum(row["is_good"] and row["is_important"] for row in rows),
        "article_id_sum": str(sum(row["article_id"] for row in rows)),
        "read_only": "on",
        "default_read_only": "on",
        "isolation": "repeatable read",
        "snapshot_time": dt.datetime(2026, 11, 6, tzinfo=dt.UTC),
        "snapshot": "1:1:",
    }
    source.update(source_changes)
    cursor = MagicMock()
    cursor.__aenter__.return_value = cursor
    cursor.description = [SimpleNamespace(name=name) for name in snapshot.SCHEMA.names]
    cursor.execute = AsyncMock()
    cursor.fetchmany = AsyncMock(
        side_effect=[rows[i : i + 2] for i in range(0, len(rows), 2)] + [[]]
    )
    conn = MagicMock()
    conn.__aenter__.return_value = conn
    conn.cursor.return_value = cursor
    conn.execute = AsyncMock(
        return_value=SimpleNamespace(fetchone=AsyncMock(return_value=source))
    )
    connect = AsyncMock(return_value=conn)
    monkeypatch.setattr(snapshot, "AsyncConnection", SimpleNamespace(connect=connect))
    monkeypatch.setattr(snapshot, "BATCH_SIZE", 2)
    return connect, conn, cursor


def _run(tmp_path: Path) -> Path:
    return asyncio.run(
        snapshot.export_snapshot("postgresql://private-source", tmp_path)
    )


def test_export_snapshot_preserves_rows_and_uses_read_only_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _rows()
    connect, conn, cursor = _connection(monkeypatch, rows)
    output = _run(tmp_path)

    assert connect.await_args is not None
    assert connect.await_args.kwargs["autocommit"] is True
    assert "default_transaction_read_only=on" in connect.await_args.kwargs["options"]
    commands = [call.args[0] for call in conn.execute.await_args_list]
    assert commands == [
        "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY",
        snapshot._query("get_article_snapshot_counts.sql"),
        "ROLLBACK",
    ]
    cursor.execute.assert_awaited_once_with(
        snapshot._query("export_article_snapshot.sql")
    )
    conn.cursor.assert_called_once_with(name="private_article_export")
    assert pq.read_table(output / "articles.parquet").equals(
        pa.Table.from_pylist(rows, schema=snapshot.SCHEMA)
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_counts"] == {
        "articles": 12,
        "good": 4,
        "bad": 4,
        "important": 6,
    }
    assert manifest["important_read_count"] == 3
    assert "private-source" not in (output / "manifest.json").read_text()
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("rows", [[], [_rows()[2]]])
def test_empty_datasets_and_subsets_are_valid_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]
) -> None:
    _connection(monkeypatch, rows)
    output = _run(tmp_path)

    for name in ("articles", "good", "bad", "important"):
        table = pq.read_table(output / f"{name}.parquet")
        assert table.schema == snapshot.SCHEMA
    assert pq.read_table(output / "important.parquet").num_rows == 0
    assert (output / "manifest.json").is_file()


def test_repeated_exports_preserve_previous_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _connection(monkeypatch, _rows())
    first = _run(tmp_path)
    original = {path.name: path.read_bytes() for path in first.iterdir()}
    _connection(monkeypatch, _rows())
    second = _run(tmp_path)

    assert second != first
    assert original == {path.name: path.read_bytes() for path in first.iterdir()}


@pytest.mark.parametrize(
    "source_changes",
    [
        {"read_only": "off"},
        {"default_read_only": "off"},
        {"isolation": "read committed"},
    ],
)
def test_unsafe_transactions_fail_before_article_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_changes: dict[str, str]
) -> None:
    _, conn, cursor = _connection(monkeypatch, _rows(), **source_changes)
    with pytest.raises(RuntimeError, match="transaction is not read-only"):
        _run(tmp_path)

    cursor.execute.assert_not_awaited()
    assert conn.execute.await_args.args == ("ROLLBACK",)
    assert not list(tmp_path.glob("*/manifest.json"))


def test_fetch_failure_rolls_back_without_a_completion_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, conn, cursor = _connection(monkeypatch, _rows())
    cursor.fetchmany.side_effect = RuntimeError("Source connection failed")
    with pytest.raises(RuntimeError, match="Source connection failed"):
        _run(tmp_path)

    assert conn.execute.await_args.args == ("ROLLBACK",)
    assert not list(tmp_path.glob("*/manifest.json"))


@pytest.mark.parametrize("failure", ["count", "duplicate", "label", "schema"])
def test_invalid_exports_have_no_completion_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    rows = _rows()
    source_changes = {}
    if failure == "count":
        source_changes["articles"] = 99
    elif failure == "duplicate":
        rows[1]["article_id"] = rows[0]["article_id"]
    elif failure == "label":
        rows[0]["is_good"] = True
    _, conn, cursor = _connection(monkeypatch, rows, **source_changes)
    if failure == "schema":
        cursor.description = []
    with pytest.raises(RuntimeError):
        _run(tmp_path)

    assert conn.execute.await_args.args == ("ROLLBACK",)
    assert not list(tmp_path.glob("*/manifest.json"))


def test_changed_subset_content_fails_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _connection(monkeypatch, _rows())
    output = _run(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    path = output / "good.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["content"] = "Changed content"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=snapshot.SCHEMA), path, row_group_size=2
    )

    with pytest.raises(RuntimeError, match="Subset rows differ for good"):
        snapshot._verify_tables(
            output, manifest["source_counts"], manifest["article_id_sum"]
        )


def test_missing_database_url_does_not_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect, _, _ = _connection(monkeypatch, [])
    with pytest.raises(ValueError, match="DATABASE_URL is not set"):
        asyncio.run(snapshot.export_snapshot("", tmp_path))

    connect.assert_not_awaited()
    assert not list(tmp_path.iterdir())
