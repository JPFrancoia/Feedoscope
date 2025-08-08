import asyncio
import logging
import os

import numpy as np
import torch
import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from custom_logging import init_logging
from feedoscope import config, utils
from feedoscope.data_registry import data_registry as dr

logger = logging.getLogger(__name__)

# https://huggingface.co/blog/modernbert

MODEL_NAME = "answerdotai-ModernBERT-base_512_2_epochs_16_batch_size"
MAX_LENGTH = 512  # Maximum length for the tokenizer
INFERENCE_BATCH_SIZE = 128


def preprocess_function(tokenizer, examples, max_length):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def find_latest_model(model_name: str) -> str:
    #iterate through the saved models directory. Look for the model starting with the model_name. sort by name and return the latest one.
    saved_models_dir = "saved_models"
    if not os.path.exists(saved_models_dir):
        raise FileNotFoundError(f"Directory {saved_models_dir} does not exist.")
    model_dirs = [
        d for d in os.listdir(saved_models_dir) if d.startswith(model_name)
    ]
    if not model_dirs:
        raise FileNotFoundError(f"No models found starting with {model_name} in {saved_models_dir}.")
    model_dirs.sort()  # Sort by name, assuming the latest model has the highest name
    latest_model = model_dirs[-1]
    return os.path.join(saved_models_dir, latest_model)


async def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_path = find_latest_model(MODEL_NAME.replace('/', '-'))

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)  # Move model to GPU if available

    await dr.global_pool.open(wait=True)

    recent_unread_articles = await dr.get_previous_days_unread_articles()
    articles_text = utils.prepare_articles_text(recent_unread_articles)

    logger.debug(f"Collected {len(recent_unread_articles)} recent unread articles.")

    logger.debug("Tokenizing articles for inference...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(
        articles_text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    logger.debug("Articles tokenized successfully.")

    logger.debug("Inferencing...")

    all_probs = []
    model.eval()
    with torch.no_grad():
        for start in tqdm.tqdm(
            range(0, len(articles_text), INFERENCE_BATCH_SIZE),
            total=(len(articles_text) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE,
            desc="Inferencing",
        ):
            end = start + INFERENCE_BATCH_SIZE
            batch_inputs = {k: v[start:end].to(device) for k, v in inputs.items()}
            preds = model(**batch_inputs).logits
            probs = torch.sigmoid(preds[:, 1]).cpu().numpy()
            all_probs.extend(probs)

    logger.debug("Inference completed successfully.")

    probs = np.round(np.array(all_probs) * 100).astype(int)

    article_ids = [article["article_id"] for article in recent_unread_articles]
    article_titles = [
        f"[{score}] {article['title']}"
        for score, article in zip(probs, recent_unread_articles)
    ]

    await dr.update_scores(
        article_ids=article_ids,
        article_titles=article_titles,
        scores=probs,
    )

    logger.debug(f"Scores updated in the database for {len(article_ids)} articles.")

    await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
