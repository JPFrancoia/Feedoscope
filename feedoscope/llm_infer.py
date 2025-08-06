import asyncio
import logging
import os

from datasets import Dataset
import evaluate
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from torch.nn.functional import softmax
import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from custom_logging import init_logging
from feedoscope import config, models, utils
from feedoscope.data_registry import data_registry as dr

logger = logging.getLogger(__name__)

# https://huggingface.co/blog/modernbert

# MODEL_NAME = "distilbert/distilbert-base-uncased"
MODEL_NAME = "answerdotai/ModernBERT-base"
# MODEL_NAME = "FacebookAI/roberta-base"
# MAX_LENGTH = 4096  # Maximum length for the tokenizer
MAX_LENGTH = 512  # Maximum length for the tokenizer

INFERENCE_BATCH_SIZE = 32


def preprocess_function(tokenizer, examples, max_length):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


async def main() -> None:

    model_path = f"saved_models/{MODEL_NAME.replace('/', '-')}"

    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        raise RuntimeError(f"Model {MODEL_NAME} not found in {model_path}")

    await dr.global_pool.open(wait=True)

    recent_unread_articles = await dr.get_previous_days_unread_articles()
    articles_text = utils.prepare_articles_text(recent_unread_articles)

    logger.debug(f"Collected {len(recent_unread_articles)} recent unread articles.")

    logger.debug("Tokenizing articles for validation...")

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
        for i in tqdm.tqdm(
            range(0, len(articles_text), INFERENCE_BATCH_SIZE),
            total=(len(articles_text) + INFERENCE_BATCH_SIZE - 1)
            // INFERENCE_BATCH_SIZE,
            desc="Inferencing",
        ):
            batch_texts = articles_text[i : i + INFERENCE_BATCH_SIZE]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            preds = model(**inputs).logits
            probs = torch.sigmoid(preds[:, 1]).cpu().numpy()
            all_probs.extend(probs)

    logger.debug("Inference completed successfully.")

    # breakpoint()

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
