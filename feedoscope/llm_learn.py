import asyncio
import logging
import os

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

from transformers import AutoTokenizer

from custom_logging import init_logging
from feedoscope import config, models, utils
from feedoscope.data_registry import data_registry as dr

from transformers import DataCollatorWithPadding
import evaluate

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
from torch.nn.functional import softmax


logger = logging.getLogger(__name__)

# https://huggingface.co/blog/modernbert

# MODEL_NAME = "distilbert/distilbert-base-uncased"
MODEL_NAME = "answerdotai/ModernBERT-base"
# MODEL_NAME = "FacebookAI/roberta-base"
# MAX_LENGTH = 4096  # Maximum length for the tokenizer
MAX_LENGTH = 512  # Maximum length for the tokenizer
VALIDATION_SIZE = 0


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_function(tokenizer, examples, max_length):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)
    # return tokenizer(examples["text"], truncation=True)


async def train_model(model_path: str, tokenizer: AutoTokenizer) -> Trainer:
    bad_articles = await dr.get_published_articles(validation_size=VALIDATION_SIZE)
    good_articles = await dr.get_read_articles_training(validation_size=VALIDATION_SIZE)
    # good_articles = good_articles[:len(bad_articles)]  # Ensure equal number of good and bad articles

    logger.debug(f"Collected {len(good_articles)} good articles.")
    logger.debug(f"Collected {len(bad_articles)} bad articles.")

    all_articles = utils.prepare_articles_text(good_articles + bad_articles)
    labels = [1] * len(good_articles) + [0] * len(bad_articles)

    # Create Hugging Face Dataset
    dataset = Dataset.from_list([{"text": t, "label": l} for t, l in zip(all_articles, labels)])
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    tokenized = split_dataset.map(
        lambda x: preprocess_function(tokenizer, x, max_length=MAX_LENGTH),
        batched=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=model_path,
        learning_rate=2e-5,
        # FIXME: set to 8 to see if it's faster
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
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
    )

    trainer.train()

    return trainer


async def main() -> None:

    logger.debug("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.debug("Tokenizer loaded successfully.")

    await dr.global_pool.open(wait=True)

    model_path = (
        f"saved_models/{MODEL_NAME.replace('/', '-')}"
    )

    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        logger.info("Training new model...")
        trainer = await train_model(model_path, tokenizer)
        trainer.save_model(model_path)
        model = trainer.model
        logger.info(f"Model saved to {model_path}")

    if VALIDATION_SIZE == 0:
        logger.info("No validation size set, skipping validation.")
        return

    logger.debug("Loading good and not good articles for validation...")
    good_articles = await dr.get_sample_good(validation_size=VALIDATION_SIZE)
    not_good_articles = await dr.get_sample_not_good(validation_size=VALIDATION_SIZE)
    good_texts = utils.prepare_articles_text(good_articles)
    not_good_texts = utils.prepare_articles_text(not_good_articles)
    logger.debug(f"Collected {len(good_texts)} good articles and {len(not_good_texts)} not good articles.")

    logger.debug("Tokenizing articles for validation...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs_good = tokenizer(good_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    inputs_not_good = tokenizer(not_good_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    logger.debug("Articles tokenized successfully.")

    model.eval()
    with torch.no_grad():
        good_preds = model(**inputs_good).logits
        not_good_preds = model(**inputs_not_good).logits

    # breakpoint()

    # Get probabilities and labels. Probabilities are normalized using softmax
    good_probs = softmax(good_preds, dim=1)[:, 1].cpu().numpy()
    not_good_probs = softmax(not_good_preds, dim=1)[:, 1].cpu().numpy()

    all_probs = np.concatenate([good_probs, not_good_probs])
    true_labels = np.concatenate([np.ones(len(good_probs)), np.zeros(len(not_good_probs))])
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


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
