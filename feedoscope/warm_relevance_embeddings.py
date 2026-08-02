"""Populate the prompted embedding cache for every Feedoscope article."""

import asyncio
import logging

import torch

from custom_logging import init_logging
from feedoscope import config, relevance_embedding
from feedoscope.data_registry import data_registry as dr

logger = logging.getLogger(__name__)
WARM_BATCH_SIZE = 500


async def main() -> None:
    """Encode every entry in ascending batches under the active cache key."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not config.ALLOW_INFERENCE_WO_GPU:
        raise RuntimeError("GPU not available for embedding cache warm")

    await dr.global_pool.open(wait=True)
    try:
        tokenizer, encoder = relevance_embedding.load_encoder(
            device, pipeline_label="embedding cache warm"
        )
        after_article_id = 0
        total = 0
        while True:
            articles = await dr.get_articles_for_embedding_warm(
                after_article_id, WARM_BATCH_SIZE
            )
            if not articles:
                break
            await relevance_embedding.encode_articles(
                articles,
                tokenizer,
                encoder,
                device,
                pipeline_label="embedding cache warm",
            )
            after_article_id = articles[-1].article_id
            total += len(articles)
            logger.info(f"Warmed prompted embeddings for {total} articles")
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
