"""Check retired-table removal in an isolated, rolled-back PostgreSQL schema."""

import os
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest

MIGRATIONS = Path(__file__).resolve().parents[1] / "db/migrations"
TABLES = (
    "urgency_inference",
    "time_sensitivity",
    "freshness_bootstrap_labels",
    "time_sensitivity_simplified",
)


def test_retired_tables_upgrade_and_rollback() -> None:
    database_url = os.environ.get("TEST_STORAGE_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_STORAGE_DATABASE_URL to isolated local PostgreSQL")
    if urlparse(database_url).hostname not in ("localhost", "127.0.0.1", "::1"):
        pytest.fail("Migration checks require a loopback database")

    with psycopg.connect(database_url) as conn:
        try:
            schema = "retired_tables_" + uuid4().hex
            conn.execute(sql.SQL("create schema {}").format(sql.Identifier(schema)))
            conn.execute(
                sql.SQL("set local search_path to {}").format(sql.Identifier(schema))
            )
            conn.execute("create table entries (id bigint primary key)")
            conn.execute("insert into entries values (1)")
            conn.execute("create table user_tags (user_id bigint, title text)")
            conn.execute("create table relevance_embeddings (article_id bigint)")
            conn.execute("insert into relevance_embeddings values (1)")
            for version in (1, 2, 3, 5, 6):
                path = next(MIGRATIONS.glob(f"{version:06d}_*.up.sql"))
                conn.execute(path.read_text())
            conn.execute(
                "insert into urgency_inference (article_id, model_key, urgency_score) "
                "values (1, 'fixture', 0.5)"
            )
            conn.execute("insert into time_sensitivity (article_id) values (1)")
            conn.execute(
                "insert into freshness_bootstrap_labels (article_id, label, source) "
                "values (1, 'fresh_d', 'fixture')"
            )
            conn.execute(
                "insert into time_sensitivity_simplified (article_id, score) "
                "values (1, 1)"
            )
            columns = """
                select table_name, column_name, data_type, udt_name, is_nullable,
                    column_default from information_schema.columns
                where table_schema = %s order by table_name, ordinal_position
            """
            constraints = """
                select r.relname, c.conname, pg_get_constraintdef(c.oid)
                from pg_constraint c join pg_class r on r.oid = c.conrelid
                join pg_namespace n on n.oid = r.relnamespace
                where n.nspname = %s order by r.relname, c.conname
            """
            before_columns = conn.execute(columns, (schema,)).fetchall()
            before_constraints = conn.execute(constraints, (schema,)).fetchall()
            up = (MIGRATIONS / "000009_drop_retired_tables.up.sql").read_text()
            down = (MIGRATIONS / "000009_drop_retired_tables.down.sql").read_text()

            # A dependency must block removal without deleting earlier tables.
            conn.execute(
                "create view retained_dependency as select * from time_sensitivity"
            )
            with pytest.raises(psycopg.errors.DependentObjectsStillExist):
                with conn.transaction():
                    conn.execute(up)
            for table in TABLES:
                assert conn.execute(
                    sql.SQL("select count(*) from {}").format(sql.Identifier(table))
                ).fetchone() == (1,)
            conn.execute("drop view retained_dependency")

            for _ in range(2):
                conn.execute(up)
                for table in TABLES:
                    assert conn.execute(
                        "select to_regclass(%s)", (table,)
                    ).fetchone() == (None,)
                assert conn.execute("select * from entries").fetchall() == [(1,)]
                assert conn.execute(
                    "select * from relevance_embeddings"
                ).fetchall() == [(1,)]
                assert conn.execute(
                    "select title from user_tags order by title"
                ).fetchall() == [
                    ("fresh_d",),
                    ("fresh_m",),
                    ("fresh_y",),
                ]
                conn.execute(down)
                assert conn.execute(columns, (schema,)).fetchall() == before_columns
                assert (
                    conn.execute(constraints, (schema,)).fetchall()
                    == before_constraints
                )
                for table in TABLES:
                    assert conn.execute(
                        sql.SQL("select count(*) from {}").format(sql.Identifier(table))
                    ).fetchone() == (0,)
        finally:
            conn.rollback()
