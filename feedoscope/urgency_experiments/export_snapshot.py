import argparse
import asyncio
import datetime as dt
import json
import logging
from pathlib import Path
import random

import pandas as pd  # type: ignore[import-untyped]

from custom_logging import init_logging
from feedoscope import config
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

SEED = 42
EVAL_FRACTION = 0.2
READ_TAGGED_URGENCY_SQL = (
    Path(__file__).with_name("sql") / "get_read_articles_with_urgency_tags.sql"
)


def _article_to_record(article: Article, label: int) -> dict[str, object]:
    """Serialize one labeled article row for the frozen urgency snapshot."""
    return {
        "article_id": article.article_id,
        "title": article.title,
        "content": article.content,
        "vote": article.vote,
        "starred": article.starred,
        "status": article.status,
        "feed_name": article.feed_name,
        "link": article.link,
        "author": article.author,
        "date_entered": article.date_entered.isoformat(),
        "last_read": article.last_read.isoformat() if article.last_read else None,
        "time_sensitivity_score": article.time_sensitivity_score,
        "tags": json.dumps(article.tags),
        "label": label,
    }


def _assign_eval_flags(
    df: pd.DataFrame, seed: int, eval_fraction: float
) -> pd.DataFrame:
    """Create one stratified frozen train/eval split over the exported rows."""
    rng = random.Random(seed)
    eval_ids: set[int] = set()

    for label in sorted(df["label"].unique()):
        class_ids = df.loc[df["label"] == label, "article_id"].tolist()
        rng.shuffle(class_ids)
        eval_count = max(1, round(len(class_ids) * eval_fraction))
        eval_ids.update(class_ids[:eval_count])

    out = df.copy()
    out["split"] = out["article_id"].apply(
        lambda article_id: "eval" if article_id in eval_ids else "train"
    )
    return out.sort_values(["label", "article_id"]).reset_index(drop=True)


async def _get_read_articles_with_urgency_tags() -> list[tuple[Article, int]]:
    """Fetch read urgency labels directly from the experiment-local SQL file."""
    query = READ_TAGGED_URGENCY_SQL.read_text().strip()

    async with dr.global_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(query)
        data = await cur.fetchall()

    results: list[tuple[Article, int]] = []
    for row in data:
        urgency_label = row.pop("urgency_label")
        article = Article(**row)
        results.append((article, urgency_label))

    return results


async def export_snapshot(output_dir: Path, seed: int, eval_fraction: float) -> None:
    """Export one frozen urgency snapshot from read-tagged Miniflux labels."""
    logger.info("Opening database pool for urgency snapshot export")
    await dr.global_pool.open(wait=True)
    try:
        labeled_articles = await _get_read_articles_with_urgency_tags()
    finally:
        await dr.global_pool.close()

    if not labeled_articles:
        raise RuntimeError("No read-tagged urgency articles found.")

    logger.info(f"Fetched {len(labeled_articles)} read-tagged urgency articles")

    records = [
        _article_to_record(article, label) for article, label in labeled_articles
    ]
    df = pd.DataFrame.from_records(records)
    df = df.sort_values("article_id").reset_index(drop=True)
    df = _assign_eval_flags(df, seed=seed, eval_fraction=eval_fraction)

    snapshot_id = dt.datetime.now(dt.UTC).strftime("urgency_snapshot_%Y%m%dT%H%M%SZ")
    snapshot_dir = output_dir / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    snapshot_path = snapshot_dir / "snapshot.parquet"
    train_path = snapshot_dir / "train.parquet"
    eval_path = snapshot_dir / "eval.parquet"
    metadata_path = snapshot_dir / "metadata.json"

    df.to_parquet(snapshot_path, index=False)
    df.loc[df["split"] == "train"].to_parquet(train_path, index=False)
    df.loc[df["split"] == "eval"].to_parquet(eval_path, index=False)

    metadata = {
        "snapshot_id": snapshot_id,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "database_url_present": bool(config.DATABASE_URL),
        "query_provenance": {
            "sql": "feedoscope/urgency_experiments/sql/get_read_articles_with_urgency_tags.sql",
            "semantics": "entries.status = 'read' and user tag in ('0-urgency', '1-urgency')",
        },
        "seed": seed,
        "eval_fraction": eval_fraction,
        "split_definition": "stratified random split by binary urgency label over the full exported read-tagged snapshot",
        "filtering_rules": {
            "labels": "Miniflux user tags 0-urgency and 1-urgency",
            "article_status": "read only",
        },
        "counts": {
            "total": int(len(df)),
            "train": int((df["split"] == "train").sum()),
            "eval": int((df["split"] == "eval").sum()),
            "evergreen_total": int((df["label"] == 0).sum()),
            "urgent_total": int((df["label"] == 1).sum()),
            "evergreen_eval": int(((df["label"] == 0) & (df["split"] == "eval")).sum()),
            "urgent_eval": int(((df["label"] == 1) & (df["split"] == "eval")).sum()),
        },
        "artifacts": {
            "snapshot": str(snapshot_path),
            "train": str(train_path),
            "eval": str(eval_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    latest_path = output_dir / "latest_snapshot.txt"
    latest_path.write_text(str(snapshot_dir) + "\n")

    print(f"SNAPSHOT_DIR={snapshot_dir}")
    print(f"TRAIN_PATH={train_path}")
    print(f"EVAL_PATH={eval_path}")
    print(f"METADATA_PATH={metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the frozen urgency snapshot to Parquet"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/urgency_autoresearch"),
        help="Directory where the snapshot directory should be created",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-fraction", type=float, default=EVAL_FRACTION)
    args = parser.parse_args()

    init_logging(config.LOGGING_CONFIG)
    asyncio.run(
        export_snapshot(
            output_dir=args.output_dir,
            seed=args.seed,
            eval_fraction=args.eval_fraction,
        )
    )


if __name__ == "__main__":
    main()
