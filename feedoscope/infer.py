import asyncio
import logging
import os

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from custom_logging import init_logging
from feedoscope import config, utils
from feedoscope.data_registry import data_registry as dr

logger = logging.getLogger(__name__)


async def main() -> None:
    logger.debug("Loading SentenceTransformer model")
    embeddings_model = SentenceTransformer(
        config.EMBEDDINGS_MODEL_NAME, trust_remote_code=True, device="cpu"
    )
    logger.debug("Embeddings model loaded successfully.")

    # Load trained model (for inference) early. If this fails, we stop early.
    inference_model_path = f"saved_models/{config.INFERENCE_MODEL_NAME}"
    if os.path.exists(inference_model_path):
        logger.debug(f"Loading existing model from {inference_model_path}")

        # TODO: add error handling here, log if model loading fails
        pu_estimator = joblib.load(inference_model_path)
        logger.debug("Inference model loaded successfully.")
    else:
        raise RuntimeError(
            f"Model {config.INFERENCE_MODEL_NAME} not found in {inference_model_path}"
        )

    await dr.global_pool.open(wait=True)

    recent_unread_articles = await dr.get_previous_days_unread_articles()

    logger.debug(f"Collected {len(recent_unread_articles)} recent unread articles.")

    logger.debug("Computing embeddings for recent unread articles")
    recent_unread_embeddings = utils.compute_embeddings(
        embeddings_model,
        utils.prepare_articles_text(recent_unread_articles),
    )
    logger.debug(
        f"Computed embeddings for {len(recent_unread_embeddings)} "
        "recent unread articles."
    )

    predictions = pu_estimator.predict_proba(recent_unread_embeddings)[:, 1]
    predictions = np.round(predictions * 100).astype(int)

    article_ids = [article["article_id"] for article in recent_unread_articles]
    article_titles = [
        f"[{score}] {article['title']}"
        for score, article in zip(predictions, recent_unread_articles)
    ]

    await dr.update_scores(
        article_ids=article_ids,
        article_titles=article_titles,
        scores=predictions,
    )

    logger.debug(f"Scores updated in the database for {len(article_ids)} articles.")

    await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
