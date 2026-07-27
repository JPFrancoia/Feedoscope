import argparse
import asyncio
import csv
from pathlib import Path

from custom_logging import init_logging
from feedoscope import config, semantic_freshness_embedding
from feedoscope.data_registry import data_registry as dr


def load_labels(path: Path, source: str) -> list[tuple[int, int, str, str]]:
    """Load the accepted frozen teacher labels from a CSV export."""
    horizon_by_name = {
        horizon.replace("-", "_"): index
        for index, horizon in enumerate(semantic_freshness_embedding.HORIZONS)
    }
    labels: list[tuple[int, int, str, str]] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            horizon = horizon_by_name.get(row.get("horizon", ""))
            confidence = row.get("confidence", "")
            if horizon is None or confidence not in {"medium", "high"}:
                continue
            labels.append((int(row["article_id"]), horizon, confidence, source))
    return labels


async def main(path: Path, source: str) -> None:
    """Persist one frozen teacher-label export for future bootstrap training."""
    labels = load_labels(path, source)
    if not labels:
        raise RuntimeError("No medium/high semantic-freshness labels were found.")
    await dr.global_pool.open(wait=True)
    try:
        await dr.upsert_semantic_freshness_teacher_labels(labels)
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=Path)
    parser.add_argument("--source", default="autoresearch-teacher")
    arguments = parser.parse_args()
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main(arguments.labels_path, arguments.source))
