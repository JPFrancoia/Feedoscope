"""Export private article snapshots without database writes."""

import asyncio
from contextlib import ExitStack
import datetime as dt
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
from typing import Any, LiteralString, cast

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

OUTPUT_ROOT = Path("artifacts/important_classifier")
BATCH_SIZE = 500
READ_ONLY_OPTIONS = (
    "-c default_transaction_read_only=on -c statement_timeout=120000 "
    "-c lock_timeout=3000 -c timezone=UTC"
)
LABEL_RULES = {
    "good": "status = 'read' AND vote >= 0",
    "bad": "vote = -1",
    "important": "vote >= 0 AND (starred OR vote = 1)",
    "important_training": "is_good AND is_important",
}
SCHEMA = pa.schema(
    [
        ("article_id", pa.int64()),
        ("user_id", pa.int64()),
        ("feed_id", pa.int64()),
        ("hash", pa.string()),
        ("published_at", pa.timestamp("us", tz="UTC")),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("changed_at", pa.timestamp("us", tz="UTC")),
        ("title", pa.string()),
        ("url", pa.string()),
        ("author", pa.string()),
        ("content", pa.string()),
        ("status", pa.string()),
        ("vote", pa.int64()),
        ("starred", pa.bool_()),
        ("saved_for_later", pa.bool_()),
        ("tags", pa.list_(pa.string())),
        ("user_tags", pa.list_(pa.string())),
        ("comments_url", pa.string()),
        ("reading_time", pa.int64()),
        ("one_line_summary", pa.string()),
        ("score", pa.int64()),
        ("raw_relevance_score", pa.float64()),
        ("feed_name", pa.string()),
        ("feed_site_url", pa.string()),
        ("feed_description", pa.string()),
        ("category_id", pa.int64()),
        ("category_name", pa.string()),
        ("is_good", pa.bool_()),
        ("is_bad", pa.bool_()),
        ("is_important", pa.bool_()),
    ]
)


def _query(filename: str) -> LiteralString:
    return cast(
        LiteralString,
        files("feedoscope.data_registry.sql").joinpath(filename).read_text(),
    )


async def _write_tables(conn: AsyncConnection[DictRow], output: Path) -> None:
    query = _query("export_article_snapshot.sql")
    (output / "export.sql").write_text(query)
    with ExitStack() as stack:
        writers = {
            name: stack.enter_context(
                pq.ParquetWriter(output / f"{name}.parquet", SCHEMA, compression="zstd")
            )
            for name in ("articles", "good", "bad", "important")
        }
        async with conn.cursor(name="private_article_export") as cursor:
            await cursor.execute(query)
            if [column.name for column in cursor.description or ()] != SCHEMA.names:
                raise RuntimeError("Article query columns do not match the file schema")
            total = 0
            while rows := await cursor.fetchmany(BATCH_SIZE):
                table = pa.Table.from_pylist(rows, schema=SCHEMA)
                writers["articles"].write_table(table)
                for name in ("good", "bad", "important"):
                    subset = table.filter(pc.fill_null(table[f"is_{name}"], False))
                    if subset.num_rows:
                        writers[name].write_table(subset)
                total += len(rows)
                if total % 10000 == 0:
                    print(f"Exported {total} articles", flush=True)


def _verify_tables(
    output: Path, expected_counts: dict[str, int], expected_id_sum: int
) -> None:
    tables = {
        name: pq.ParquetFile(output / f"{name}.parquet") for name in expected_counts
    }
    for name, table in tables.items():
        if table.metadata.num_rows != expected_counts[name]:
            raise RuntimeError(f"Source and file counts differ for {name}")
        if table.schema_arrow != SCHEMA:
            raise RuntimeError(f"Unexpected file schema for {name}")
    positions = {name: 0 for name in ("good", "bad", "important")}
    previous_id = None
    id_sum = 0
    for group in range(tables["articles"].num_row_groups):
        table = tables["articles"].read_row_group(group)
        ids = table["article_id"].to_pylist()
        if any(a >= b for a, b in zip(ids, ids[1:])) or (
            previous_id is not None and previous_id >= ids[0]
        ):
            raise RuntimeError("Article IDs are duplicated or out of order")
        previous_id = ids[-1]
        id_sum += sum(ids)
        labels = {
            "good": pc.and_(
                pc.equal(table["status"], "read"), pc.greater_equal(table["vote"], 0)
            ),
            "bad": pc.equal(table["vote"], -1),
            "important": pc.and_(
                pc.greater_equal(table["vote"], 0),
                pc.or_(table["starred"], pc.equal(table["vote"], 1)),
            ),
        }
        for name, label in labels.items():
            if not label.equals(table[f"is_{name}"]):
                raise RuntimeError(f"Incorrect derived label for {name}")
            expected = table.filter(pc.fill_null(label, False))
            if expected.num_rows:
                actual = tables[name].read_row_group(positions[name])
                if not actual.equals(expected):
                    raise RuntimeError(f"Subset rows differ for {name}")
                positions[name] += 1
    if id_sum != expected_id_sum:
        raise RuntimeError("Source and file article ID sums differ")
    if any(positions[name] != tables[name].num_row_groups for name in positions):
        raise RuntimeError("A subset contains extra row groups")


async def export_snapshot(database_url: str, output_root: Path = OUTPUT_ROOT) -> Path:
    """Create a verified private snapshot from one read-only transaction.

    Args:
        database_url: PostgreSQL source connection URL.
        output_root: Parent directory for new snapshots.

    Returns:
        The completed snapshot directory.
    """
    if not database_url:
        raise ValueError("DATABASE_URL is not set")
    old_umask = os.umask(0o077)
    try:
        output = output_root / dt.datetime.now(dt.UTC).strftime(
            "snapshot_%Y%m%dT%H%M%S%fZ"
        )
        output.mkdir(parents=True, mode=0o700, exist_ok=False)
        async with await AsyncConnection.connect(
            database_url,
            autocommit=True,
            options=READ_ONLY_OPTIONS,
            application_name="feedoscope_readonly_snapshot",
            connect_timeout=10,
            row_factory=dict_row,
        ) as conn:
            await conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            try:
                cursor = await conn.execute(_query("get_article_snapshot_counts.sql"))
                source = await cursor.fetchone()
                if source is None or (
                    source["read_only"],
                    source["default_read_only"],
                    source["isolation"],
                ) != ("on", "on", "repeatable read"):
                    raise RuntimeError("The source transaction is not read-only")
                counts: dict[str, int] = {
                    name: source[name]
                    for name in ("articles", "good", "bad", "important")
                }
                metadata: dict[str, Any] = {
                    "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "transaction": {
                        key: source[key]
                        for key in (
                            "read_only",
                            "default_read_only",
                            "isolation",
                            "snapshot",
                        )
                    },
                    "snapshot_time_utc": source["snapshot_time"].isoformat(),
                    "source_counts": counts,
                    "important_read_count": source["important_read"],
                    "article_id_sum": int(source["article_id_sum"]),
                    "label_rules": LABEL_RULES,
                    "scope": "All articles. No date filter, sample, or training split.",
                    "warnings": [
                        "Unread neutral articles have no label. They are not negatives.",
                        "Importance does not require read status. Training uses both labels.",
                        "changed_at records the last change, not a dedicated read time.",
                        "Votes, stars, tags, status, and scores can reveal training labels.",
                        "The files contain private reading data. Keep them local.",
                    ],
                }
                await _write_tables(conn, output)
            finally:
                await conn.execute("ROLLBACK")
        _verify_tables(output, counts, metadata["article_id_sum"])
        metadata["transaction"]["end"] = "ROLLBACK"
        metadata["exported_counts"] = counts
        metadata["validation"] = (
            "All Parquet rows, counts, IDs, labels, and subsets verified."
        )
        manifest = json.dumps(metadata, indent=2) + "\n"
        checksums = []
        for path in sorted(output.iterdir()):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            checksums.append(f"{digest}  {path.name}\n")
        checksums.append(
            f"{hashlib.sha256(manifest.encode()).hexdigest()}  manifest.json\n"
        )
        (output / "SHA256SUMS").write_text("".join(checksums))
        (output / "manifest.json").write_text(manifest)
        print(f"Snapshot: {output}")
        print(f"Verified counts: {counts}")
        return output
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    asyncio.run(export_snapshot(os.environ.get("DATABASE_URL", "")))
