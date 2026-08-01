import argparse
import asyncio
import csv
from pathlib import Path

from custom_logging import init_logging
from feedoscope import config, semantic_freshness_embedding
from feedoscope.data_registry import data_registry as dr

EXPECTED_LABELS = 1200


def load_labels(path: Path, source: str) -> list[tuple[int, str, str]]:
    """Load and validate one complete three-label bootstrap CSV."""
    labels: list[tuple[int, str, str]] = []
    article_ids: set[int] = set()
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            article_id = int(row["article_id"])
            label = row.get("label", "")
            if article_id in article_ids:
                raise ValueError(f"Duplicate bootstrap article ID: {article_id}")
            if label not in semantic_freshness_embedding.HORIZONS:
                raise ValueError(f"Invalid freshness label for {article_id}: {label}")
            article_ids.add(article_id)
            labels.append((article_id, label, source))
    if len(labels) != EXPECTED_LABELS:
        raise RuntimeError(
            f"Expected {EXPECTED_LABELS} bootstrap rows, found {len(labels)}."
        )
    return labels


async def main(path: Path, source: str) -> None:
    """Persist the fixed Luna bootstrap labels used by weekly training."""
    labels = load_labels(path, source)
    await dr.global_pool.open(wait=True)
    try:
        await dr.replace_semantic_freshness_bootstrap_labels(labels)
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=Path)
    parser.add_argument("--source", default="gpt-5.6-luna")
    arguments = parser.parse_args()
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main(arguments.labels_path, arguments.source))
