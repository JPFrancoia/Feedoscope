import asyncio
import logging
import time

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from custom_logging import init_logging
from feedoscope import config, utils
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article, UrgencyInferenceResults
from feedoscope.llm_infer import find_latest_model

logger = logging.getLogger(__name__)

URGENCY_MODEL_NAME = "urgency-ModernBERT-base"
MAX_LENGTH = 512
INFERENCE_BATCH_SIZE = 128


async def infer(articles: list[Article]) -> UrgencyInferenceResults:
    """Run urgency inference using the distilled ModernBERT model.

    Produces a continuous urgency probability (0.0 to 1.0) per article,
    where 0.0 = evergreen and 1.0 = highly urgent/ephemeral.

    Args:
        articles: List of articles to score.

    Returns:
        UrgencyInferenceResults with article IDs and urgency probabilities.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type != "cuda" and not config.ALLOW_INFERENCE_WO_GPU:
        mes = f"GPU not available. Device is '{device}'. Exiting"
        logger.critical(mes)
        raise RuntimeError(mes)

    model_path = find_latest_model(URGENCY_MODEL_NAME)

    logger.info(f"Loading urgency model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    articles_text = utils.prepare_articles_text(articles)

    logger.debug("Tokenizing articles for urgency inference...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(
        articles_text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    logger.debug("Articles tokenized successfully.")

    logger.debug("Running urgency inference...")
    all_probs: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(articles_text), INFERENCE_BATCH_SIZE):
            logger.debug(
                f"Processing batch {start // INFERENCE_BATCH_SIZE + 1} of "
                f"{len(articles_text) // INFERENCE_BATCH_SIZE + 1}"
            )
            end = start + INFERENCE_BATCH_SIZE
            batch_inputs = {k: v[start:end].to(device) for k, v in inputs.items()}
            preds = model(**batch_inputs).logits
            # Probability of class 1 (urgent)
            probs = torch.sigmoid(preds[:, 1]).cpu().numpy()
            all_probs.extend(probs.tolist())

    logger.debug("Urgency inference completed successfully.")

    article_ids = [article.article_id for article in articles]

    return UrgencyInferenceResults(
        article_ids=article_ids,
        urgency_scores=all_probs,
    )


async def main(number_of_days: int = 14) -> None:
    """Infer urgency for articles not yet cached, and write results to DB."""
    await dr.global_pool.open(wait=True)

    articles = await dr.get_articles_wo_urgency_inference(number_of_days=number_of_days)

    logger.info(
        f"Fetched {len(articles)} articles without urgency inference "
        f"from the last {number_of_days} days."
    )

    if not articles:
        logger.info("No articles to process. Exiting.")
        return

    start_time = time.time()

    results = await infer(articles)

    elapsed_time = time.time() - start_time
    logger.info(
        f"Urgency inference completed in {elapsed_time:.2f} seconds "
        f"for {len(articles)} articles."
    )

    await dr.register_urgency_inference(results)
    logger.info(f"Cached urgency scores for {len(results.article_ids)} articles.")

    await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
