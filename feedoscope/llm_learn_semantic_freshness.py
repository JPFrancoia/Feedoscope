import asyncio
import logging
import time

import numpy as np
import torch

from custom_logging import init_logging
from feedoscope import config, relevance_embedding, semantic_freshness_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.llm_infer import find_latest_model

logger = logging.getLogger(__name__)


async def main() -> None:
    """Train freshness from bootstrap labels and manually tagged read articles."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not config.ALLOW_TRAINING_WO_GPU:
        raise RuntimeError("GPU not available. Exiting")

    logger.info("Opening database pool for freshness training")
    await dr.global_pool.open(wait=True)
    try:
        conflicts = await dr.get_conflicting_semantic_freshness_labels()
        for article_id, title in conflicts:
            logger.warning(
                f"Skipping conflicting manual freshness labels for {article_id}: {title}"
            )

        labeled_data = await dr.get_semantic_freshness_training_data()
        if not labeled_data:
            raise RuntimeError("No effective freshness labels are available.")
        logger.info(f"Loaded {len(labeled_data)} effective freshness labels")

        articles = [article for article, _, _ in labeled_data]
        labels = np.asarray([label for _, label, _ in labeled_data], dtype=int)
        fingerprint = semantic_freshness_embedding.fingerprint_labels(
            [
                (article.article_id, label, source)
                for article, label, source in labeled_data
            ]
        )
        try:
            _, metadata = semantic_freshness_embedding.load_artifact(
                find_latest_model(
                    semantic_freshness_embedding.get_model_family_prefix(),
                    clean_old_models=False,
                )
            )
            if metadata.get("dataset_fingerprint") == fingerprint:
                logger.info("No new freshness labels; skipping training.")
                return
        except FileNotFoundError:
            pass

        tokenizer, encoder = relevance_embedding.load_encoder(
            device,
            pipeline_label="freshness",
        )
        start_time = time.time()
        embeddings = await relevance_embedding.encode_articles(
            articles,
            tokenizer,
            encoder,
            device,
            pipeline_label="freshness",
        )
        classifiers = semantic_freshness_embedding.fit_classifiers(embeddings, labels)

        train_counts = {
            horizon: int((labels == index).sum())
            for index, horizon in enumerate(semantic_freshness_embedding.HORIZONS)
        }
        label_source_counts = {
            source: sum(1 for _, _, row_source in labeled_data if row_source == source)
            for source in {source for _, _, source in labeled_data}
        }
        metadata = semantic_freshness_embedding.artifact_metadata(
            fingerprint,
            train_counts=train_counts,
            label_source_counts=label_source_counts,
        )
        model_path = semantic_freshness_embedding.build_model_path(fingerprint)
        semantic_freshness_embedding.save_artifact(model_path, classifiers, metadata)
        logger.info(
            f"Freshness training completed in {time.time() - start_time:.2f} seconds."
        )
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
