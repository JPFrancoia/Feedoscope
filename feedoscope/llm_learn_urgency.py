import asyncio
import datetime
import logging
import os
import random
import shutil
import time
from typing import Any

from datasets import Dataset  # type: ignore[import]
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_utils import EvalPrediction

from custom_logging import init_logging
from feedoscope import config, utils
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

MODEL_NAME = "answerdotai/ModernBERT-base"
URGENCY_MODEL_PREFIX = "urgency-ModernBERT-base"
MAX_LENGTH = 512
EPOCHS = 2
BATCH_SIZE = 16


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float | np.floating]:
    """Compute evaluation metrics for the urgency classification model."""
    logits, labels = eval_pred
    # Apply sigmoid to convert logits to probabilities (class 1 = urgent)
    positive_class_probs = torch.sigmoid(torch.tensor(logits)[:, 1]).numpy()

    # Predictions based on 0.5 threshold
    preds = (positive_class_probs >= 0.5).astype(int)

    ap = average_precision_score(labels, positive_class_probs)
    roc_auc = roc_auc_score(labels, positive_class_probs)
    acc = accuracy_score(labels, preds)
    loss = log_loss(labels, positive_class_probs)

    return {
        "average_precision": ap,
        "roc_auc": roc_auc,
        "accuracy": acc,
        "eval_loss": loss,
    }


def preprocess_function(
    tokenizer: PreTrainedTokenizerBase, examples: dict[str, list[Any]], max_length: int
) -> BatchEncoding:
    """Tokenize text examples."""
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


class ProgressLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logger.info(
            f"Training progress: step {state.global_step}/{state.max_steps or '?'}"
        )


async def train_model(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    articles: list[Article],
    labels: list[int],
) -> Trainer:
    """Train a ModernBERT model for binary urgency classification.

    Args:
        model_path: Output directory for the trained model.
        tokenizer: Pre-trained tokenizer.
        articles: List of articles to train on.
        labels: List of binary labels (0=evergreen, 1=urgent).

    Returns:
        The trained Trainer instance.

    """
    all_texts = utils.prepare_articles_text(articles)

    dataset = Dataset.from_list(
        [{"text": text, "label": label} for text, label in zip(all_texts, labels)]
    )
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    tokenized = split_dataset.map(
        lambda x: preprocess_function(tokenizer, x, max_length=MAX_LENGTH),
        batched=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=model_path,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        metric_for_best_model="average_precision",
        logging_strategy="steps",
        logging_steps=2,
        disable_tqdm=True,
        logging_first_step=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            ProgressLoggingCallback(),
        ],
    )

    trainer.train()

    return trainer


async def main() -> None:
    """Train the distilled urgency model on LLM-generated binary labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type != "cuda" and not config.ALLOW_TRAINING_WO_GPU:
        mes = "GPU not available. Exiting"
        logger.critical(mes)
        raise RuntimeError(mes)

    start_time = time.time()

    logger.debug("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.debug("Tokenizer loaded successfully.")

    await dr.global_pool.open(wait=True)

    # Fetch all articles with urgency user tags (0-urgency or 1-urgency)
    labeled_data = await dr.get_articles_with_simplified_time_sensitivity()

    if not labeled_data:
        logger.error(
            "No articles with urgency user tags found. "
            "Run infer_simpler_time_sensitivity first."
        )
        return

    articles = [article for article, _ in labeled_data]
    labels = [label for _, label in labeled_data]

    urgent_count = sum(labels)
    evergreen_count = len(labels) - urgent_count
    logger.info(
        f"Fetched {len(labeled_data)} labeled articles: "
        f"{urgent_count} urgent, {evergreen_count} evergreen."
    )

    # Balance classes, prioritizing read articles (verified by user).
    # All read articles are always kept. Unread articles fill the remainder
    # up to the minority class count.
    read_urgent = [
        (a, l) for a, l in zip(articles, labels) if l == 1 and a.status == "read"
    ]
    read_evergreen = [
        (a, l) for a, l in zip(articles, labels) if l == 0 and a.status == "read"
    ]
    unread_urgent = [
        (a, l) for a, l in zip(articles, labels) if l == 1 and a.status != "read"
    ]
    unread_evergreen = [
        (a, l) for a, l in zip(articles, labels) if l == 0 and a.status != "read"
    ]

    logger.info(
        f"Read: {len(read_urgent)} urgent, {len(read_evergreen)} evergreen. "
        f"Unread: {len(unread_urgent)} urgent, {len(unread_evergreen)} evergreen."
    )

    target = min(urgent_count, evergreen_count)

    # For each class: keep all read, sample from unread to reach target
    def _select_class(
        read_items: list[tuple[Article, int]],
        unread_items: list[tuple[Article, int]],
        target_count: int,
    ) -> list[tuple[Article, int]]:
        selected = list(read_items)
        remaining = target_count - len(selected)
        if remaining > 0:
            sampled = random.sample(unread_items, min(remaining, len(unread_items)))
            selected.extend(sampled)
        return selected

    selected_urgent = _select_class(read_urgent, unread_urgent, target)
    selected_evergreen = _select_class(read_evergreen, unread_evergreen, target)

    balanced_data = selected_urgent + selected_evergreen
    articles = [a for a, _ in balanced_data]
    labels = [l for _, l in balanced_data]

    actual_urgent = sum(labels)
    actual_evergreen = len(labels) - actual_urgent
    logger.info(
        f"Balanced to {actual_urgent} urgent, {actual_evergreen} evergreen "
        f"({len(labels)} total)."
    )

    model_path = (
        f"models/{URGENCY_MODEL_PREFIX}"
        f"_{MAX_LENGTH}_{EPOCHS}_epochs_{BATCH_SIZE}_batch_size"
        f"_{datetime.date.today().strftime('%Y_%m_%d')}"
        f"_{actual_urgent}_urgent_{actual_evergreen}_evergreen"
    )

    if os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")

        try:
            AutoModelForSequenceClassification.from_pretrained(model_path)
        except ValueError as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Model not found or corrupted, deleting the model path.")
            shutil.rmtree(model_path)
            raise RuntimeError(
                f"Model not found or corrupted at {model_path}. "
                "The model's directory has been deleted."
            )
    else:
        logger.info("Training new urgency model...")
        trainer = await train_model(model_path, tokenizer, articles, labels)
        trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds.")

    await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
