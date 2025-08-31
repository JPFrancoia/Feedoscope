import asyncio
import logging
import os
import time
from typing import Any

from datasets import Dataset  # type: ignore[import]
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

# from torch.nn.functional import softmax
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

# https://huggingface.co/blog/modernbert

# TODO: try modernbert large or deberta v3 large

# MODEL_NAME = "distilbert/distilbert-base-uncased"
MODEL_NAME = "answerdotai/ModernBERT-base"
# MODEL_NAME = "FacebookAI/roberta-base"
# MODEL_NAME = "answerdotai/ModernBERT-large"
# MAX_LENGTH = 4096  # Maximum length for the tokenizer
MAX_LENGTH = 512  # Maximum length for the tokenizer
# MAX_LENGTH = 1024  # Maximum length for the tokenizer
VALIDATION_SIZE = 0
EPOCHS = 2  # Number of epochs for training
BATCH_SIZE = 16  # Batch size for training
# BATCH_SIZE = 8  # Batch size for training


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float | np.floating]:
    logits, labels = eval_pred
    # Apply sigmoid to convert logits to probabilities
    positive_class_probs = torch.sigmoid(torch.tensor(logits)[:, 1]).numpy()

    # Predictions for accuracy based on a 0.5 threshold
    preds = (positive_class_probs >= 0.5).astype(int)

    # Metrics
    ap = average_precision_score(labels, positive_class_probs)
    roc_auc = roc_auc_score(labels, positive_class_probs)
    acc = accuracy_score(labels, preds)
    # log_loss requires probabilities, not logits
    loss = log_loss(labels, positive_class_probs)

    return {
        "average_precision": ap,
        "roc_auc": roc_auc,
        "accuracy": acc,
        "eval_loss": loss,  # Will show up in logs
    }


def preprocess_function(
    tokenizer: PreTrainedTokenizerBase, examples: dict[str, list[Any]], max_length: int
) -> BatchEncoding:
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


class ProgressLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # state.global_step: current step
        # state.max_steps: total steps (if known)
        logger.info(
            f"Training progress: step {state.global_step}/{state.max_steps or '?'}"
        )


async def train_model(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    good_articles: list[Article],
    bad_articles: list[Article],
) -> Trainer:

    all_articles = utils.prepare_articles_text(good_articles + bad_articles)
    labels = [1] * len(good_articles) + [0] * len(bad_articles)

    # Create Hugging Face Dataset
    dataset = Dataset.from_list(
        [{"text": text, "label": label} for text, label in zip(all_articles, labels)]
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
        # stop training if no improvement for 2 epochs
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            ProgressLoggingCallback(),
        ],
    )

    trainer.train()

    return trainer


async def main() -> None:

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

    bad_articles = await dr.get_published_articles(validation_size=VALIDATION_SIZE)
    good_articles = await dr.get_read_articles_training(validation_size=VALIDATION_SIZE)

    # Ensure equal number of good and bad articles. Use the most recent good articles.
    good_articles = good_articles[-len(bad_articles):]

    logger.debug(f"Collected {len(good_articles)} good articles.")
    logger.debug(f"Collected {len(bad_articles)} bad articles.")

    model_path = f"saved_models/{MODEL_NAME.replace('/', '-')}_{MAX_LENGTH}_{EPOCHS}_epochs_{BATCH_SIZE}_batch_size_{len(good_articles)}_good_{len(bad_articles)}_not_good"

    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except ValueError as e:
            # handle this error: ValueError: Unrecognized model in saved_models/answerdotai-[..]
            # When this happens, it's likely because a previous run created the
            # directory but the training run didn't complete and left an empty directory.
            logger.error(f"Error loading model: {e}")
            logger.info("Model not found or corrupted, deleting the model path.")
            os.remove(model_path)
            raise RuntimeError(
                f"Model not found or corrupted at {model_path}. The model's directory has been deleted."
            )

    else:
        logger.info("Training new model...")
        trainer = await train_model(model_path, tokenizer, good_articles, bad_articles)
        trainer.save_model(model_path)
        model = trainer.model
        logger.info(f"Model saved to {model_path}")

    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds.")

    if VALIDATION_SIZE == 0:
        logger.info("No validation size set, skipping validation.")
        return

    logger.debug("Loading good and not good articles for validation...")
    good_articles = await dr.get_sample_good(validation_size=VALIDATION_SIZE)
    not_good_articles = await dr.get_sample_not_good(validation_size=VALIDATION_SIZE)
    good_texts = utils.prepare_articles_text(good_articles)
    not_good_texts = utils.prepare_articles_text(not_good_articles)
    logger.debug(
        f"Collected {len(good_texts)} good articles and {len(not_good_texts)} not good articles."
    )

    logger.debug("Tokenizing articles for validation...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs_good = tokenizer(
        good_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs_not_good = tokenizer(
        not_good_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    logger.debug("Articles tokenized successfully.")

    model.to(device)
    with torch.no_grad():
        inputs_good = {k: v.to(device) for k, v in inputs_good.items()}
        inputs_not_good = {k: v.to(device) for k, v in inputs_not_good.items()}
        good_preds = model(**inputs_good).logits
        not_good_preds = model(**inputs_not_good).logits

    # Sigmoid is an alternative to softmax when there are only two classes
    good_probs = torch.sigmoid(good_preds[:, 1]).cpu().numpy()
    not_good_probs = torch.sigmoid(not_good_preds[:, 1]).cpu().numpy()

    all_probs = np.concatenate([good_probs, not_good_probs])
    true_labels = np.concatenate(
        [np.ones(len(good_probs)), np.zeros(len(not_good_probs))]
    )
    pred_labels = (all_probs >= 0.5).astype(int)

    # Compute metrics
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    roc_auc = roc_auc_score(true_labels, all_probs)
    ap = average_precision_score(true_labels, all_probs)
    logloss = log_loss(true_labels, all_probs)

    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall: {recall:.2f}")
    logger.info(f"F1 score: {f1:.2f}")
    logger.info(f"ROC AUC: {roc_auc:.2f}")
    logger.info(f"Average Precision: {ap:.2f}")
    logger.info(f"Log Loss: {logloss:.2f}")

    await dr.global_pool.close()

    elapsed_time = time.time() - start_time
    logger.info(f"Training and validation completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
