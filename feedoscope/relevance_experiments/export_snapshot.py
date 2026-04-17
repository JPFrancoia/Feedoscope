import argparse
import asyncio
import datetime as dt
import json
import logging
from pathlib import Path
import random

import pandas as pd

from custom_logging import init_logging
from feedoscope import config
from feedoscope.data_registry import data_registry as dr

logger = logging.getLogger(__name__)

SEED = 42
EVAL_FRACTION = 0.2


def _article_to_record(article, label: int) -> dict[str, object]:
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


async def export_snapshot(output_dir: Path, seed: int, eval_fraction: float) -> None:
    logger.info("Opening read-only database pool for snapshot export")
    await dr.global_pool.open(wait=True)
    try:
        good_articles = await dr.get_read_articles_training(validation_size=0)
        bad_articles = await dr.get_published_articles(validation_size=0)
    finally:
        await dr.global_pool.close()

    logger.info(
        f"Fetched {len(good_articles)} good articles and {len(bad_articles)} bad articles"
    )

    records = [_article_to_record(article, 1) for article in good_articles]
    records.extend(_article_to_record(article, 0) for article in bad_articles)
    df = pd.DataFrame.from_records(records)
    df = df.sort_values("article_id").reset_index(drop=True)
    df = _assign_eval_flags(df, seed=seed, eval_fraction=eval_fraction)

    snapshot_id = dt.datetime.now(dt.UTC).strftime("relevance_snapshot_%Y%m%dT%H%M%SZ")
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
            "good_sql": "feedoscope/data_registry/sql/get_read_articles_training.sql",
            "bad_sql": "feedoscope/data_registry/sql/get_published_articles.sql",
            "good_semantics": "status = 'read' and vote >= 0",
            "bad_semantics": "vote = -1",
            "time_window": "published_at > now() - interval '1 year'",
        },
        "seed": seed,
        "eval_fraction": eval_fraction,
        "split_definition": "stratified random split by binary label over the full exported snapshot",
        "filtering_rules": {
            "good": "status = 'read' and vote >= 0",
            "bad": "vote = -1",
            "time_window": "last 1 year of published articles",
        },
        "counts": {
            "total": int(len(df)),
            "train": int((df["split"] == "train").sum()),
            "eval": int((df["split"] == "eval").sum()),
            "good_total": int((df["label"] == 1).sum()),
            "bad_total": int((df["label"] == 0).sum()),
            "good_eval": int(((df["label"] == 1) & (df["split"] == "eval")).sum()),
            "bad_eval": int(((df["label"] == 0) & (df["split"] == "eval")).sum()),
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
        description="Export the frozen relevance snapshot to Parquet"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/relevance_autoresearch"),
        help="Directory where the snapshot directory should be created",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-fraction", type=float, default=EVAL_FRACTION)
    args = parser.parse_args()

    init_logging(config.LOGGING_CONFIG)
    asyncio.run(
        export_snapshot(
            args.output_dir, seed=args.seed, eval_fraction=args.eval_fraction
        )
    )


if __name__ == "__main__":
    main()
