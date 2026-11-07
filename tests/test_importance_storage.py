"""Check importance storage against disposable local PostgreSQL."""

import asyncio
from collections.abc import Iterator
import datetime as dt
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
import pytest

from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import ImportanceResult

MIGRATIONS = Path(__file__).resolve().parents[1] / "db/migrations"
UP = (MIGRATIONS / "000010_importance.up.sql").read_text()
DOWN = (MIGRATIONS / "000010_importance.down.sql").read_text()
MODEL = "jev-1.14.0"
PROMPT: dict[str, Any] = {
    "reader_profile": "Synthetic profile",
    "question": {
        "type": "choice",
        "instructions": "Important?",
        "criteria": {"important": "Useful", "good_only": "Ordinary"},
    },
}


@pytest.fixture
def database() -> Iterator[str]:
    """Create only an isolated schema in an explicitly selected loopback database."""
    url = os.getenv("TEST_STORAGE_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_STORAGE_DATABASE_URL to isolated local PostgreSQL")
    parsed = urlparse(url)
    if (
        parsed.hostname not in ("127.0.0.1", "localhost", "::1")
        or parsed.path != "/importance_test"
    ):
        pytest.fail(
            "Importance storage tests require a loopback importance_test database"
        )
    schema = "importance_" + uuid4().hex
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(sql.SQL("create schema {}").format(sql.Identifier(schema)))
        scoped = make_conninfo(url, options=f"-csearch_path={schema}")
        try:
            with psycopg.connect(scoped) as setup:
                setup.execute(
                    "create table entries (id bigint primary key, title text not null, content text, status text not null, vote integer, starred boolean, score integer)"
                )
                setup.execute(
                    "insert into entries values (1,'Title','Body','unread',0,false,42),(2,'Second',null,'read',1,true,70),(3,'Removed','Private','removed',0,false,0)"
                )
                setup.execute(UP)
            yield scoped
        finally:
            conn.execute(
                sql.SQL("drop schema {} cascade").format(sql.Identifier(schema))
            )


def test_migration_constraints_immutability_and_deletion(database: str) -> None:
    """Reject invalid stored values and keep article and prompt relationships intact."""
    with psycopg.connect(database, autocommit=True) as conn:
        entry_before = conn.execute("select * from entries order by id").fetchall()
        first = conn.execute(
            "insert into importance_prompt (prompt,model) values (%s,%s) returning id",
            (Jsonb(PROMPT), MODEL),
        ).fetchone()
        assert first is not None
        prompt_id = first[0]
        for field in ["prompt", "model", "created_at", "id"]:
            with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
                conn.execute(
                    sql.SQL("update importance_prompt set {} = {}").format(
                        sql.Identifier(field), sql.Identifier(field)
                    )
                )
        for model in ["jev-latest", "", "other"]:
            with pytest.raises(psycopg.errors.CheckViolation):
                conn.execute(
                    "insert into importance_prompt (prompt,model) values (%s,%s)",
                    (Jsonb(PROMPT), model),
                )
        for value in [-0.1, 1.1, float("nan"), float("inf"), float("-inf")]:
            for column in ["importance_probability", "confidence"]:
                row = {
                    "entry_id": 1,
                    "importance_prompt_id": prompt_id,
                    "is_important": True,
                    "importance_probability": 0.9,
                    "confidence": 0.8,
                    "inferred_at": dt.datetime.now(dt.UTC),
                }
                row[column] = value
                with pytest.raises(psycopg.errors.CheckViolation):
                    conn.execute(
                        dr._get_query_from_file("upsert_entry_importance.sql"), row
                    )
        for entry_id, prompt_ref in [(999, prompt_id), (1, 999)]:
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                conn.execute(
                    "insert into entries_importance values (%s,%s,true,0.9,0.8,now())",
                    (entry_id, prompt_ref),
                )
        for column in [
            "entry_id",
            "importance_prompt_id",
            "is_important",
            "importance_probability",
            "confidence",
            "inferred_at",
        ]:
            row = {
                "entry_id": 1,
                "importance_prompt_id": prompt_id,
                "is_important": True,
                "importance_probability": 0.9,
                "confidence": 0.8,
                "inferred_at": dt.datetime.now(dt.UTC),
            }
            row[column] = None
            with pytest.raises(psycopg.errors.NotNullViolation):
                conn.execute(
                    dr._get_query_from_file("upsert_entry_importance.sql"), row
                )
        conn.execute(
            "insert into entries_importance values (1,%s,true,0.9,0.8,now())",
            (prompt_id,),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute(
                "insert into entries_importance values (1,%s,true,0.9,0.8,now())",
                (prompt_id,),
            )
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            conn.execute("delete from importance_prompt where id=%s", (prompt_id,))
        assert (
            conn.execute("select * from entries order by id").fetchall() == entry_before
        )
        conn.execute("delete from entries where id=1")
        assert conn.execute("select count(*) from entries_importance").fetchone() == (
            0,
        )
        assert conn.execute("select count(*) from importance_prompt").fetchone() == (1,)


def test_rollback_and_repeat_upgrade_preserve_entries(database: str) -> None:
    """Remove only this migration's objects and allow a repeat upgrade."""
    with psycopg.connect(database, autocommit=True) as conn:
        before = conn.execute("select * from entries order by id").fetchall()
        for _ in range(2):
            conn.execute(DOWN)
            assert conn.execute(
                "select to_regclass('importance_prompt'),to_regclass('entries_importance'),to_regprocedure('prevent_importance_prompt_update()')"
            ).fetchone() == (None, None, None)
            assert (
                conn.execute("select * from entries order by id").fetchall() == before
            )
            conn.execute(UP)
            assert conn.execute(
                "select count(*) from importance_prompt"
            ).fetchone() == (0,)


def test_registry_latest_prompt_and_prediction_order(
    database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use real registry queries and reject late older predictions."""

    async def check() -> None:
        async with AsyncConnectionPool(
            database, open=False, kwargs={"row_factory": dict_row}
        ) as pool:
            monkeypatch.setattr(dr, "global_pool", pool)
            with pytest.raises(RuntimeError, match="No importance prompt"):
                await dr.get_latest_importance_prompt()
            first = await dr.insert_importance_prompt(PROMPT, MODEL)
            second = await dr.insert_importance_prompt(PROMPT, MODEL)
            assert second.id > first.id
            assert (await dr.get_latest_importance_prompt()).id == second.id
            articles = await dr.get_importance_articles([1, 2, 3, 999])
            assert [a.entry_id for a in articles] == [1, 2]
            assert articles[1].content == ""
            now = dt.datetime.now(dt.UTC)
            positive = ImportanceResult(
                is_important=True,
                importance_probability=0.9,
                confidence=0.8,
                inferred_at=now,
            )
            negative = ImportanceResult(
                is_important=False,
                importance_probability=0.1,
                confidence=0.8,
                inferred_at=now + dt.timedelta(seconds=1),
            )
            assert await dr.upsert_entry_importance(1, first.id, positive)
            assert await dr.upsert_entry_importance(1, second.id, negative)
            assert not await dr.upsert_entry_importance(1, first.id, negative)
            assert not await dr.upsert_entry_importance(1, second.id, positive)
            async with pool.connection() as conn:
                cursor = await conn.execute("select * from entries_importance")
                rows = await cursor.fetchall()
                assert len(rows) == 1
                assert rows[0] == {
                    "entry_id": 1,
                    "importance_prompt_id": second.id,
                    **negative.model_dump(),
                }
                cursor = await conn.execute(
                    "select vote,starred,score from entries where id=1"
                )
                assert await cursor.fetchone() == {
                    "vote": 0,
                    "starred": False,
                    "score": 42,
                }

    asyncio.run(check())
