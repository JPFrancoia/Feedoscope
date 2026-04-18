import argparse
import json
import logging
import math
from pathlib import Path
import shutil
import time
from typing import Any

from cleantext import clean  # type: ignore[import]
from datasets import Dataset  # type: ignore[import]
import numpy as np
import pandas as pd
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments,
)

from custom_logging import init_logging
from feedoscope import config
from feedoscope.llm_learn import WeightedTrainer
from feedoscope.utils import clean_title, strip_html_keep_text

logger = logging.getLogger(__name__)

DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_TRAIN_BALANCE_MODE = "full"
DEFAULT_EXCELLENT_WEIGHT = config.EXCELLENT_WEIGHT


class ProgressLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logger.info(
            f"Training progress: step {state.global_step}/{state.max_steps or '?'}"
        )


def _clean_text(text: str) -> str:
    return clean(" ".join(text.split()))


def prepare_single_blob(title: str, content: str) -> str:
    return _clean_text(strip_html_keep_text(f"{clean_title(title)} {content}"))


def _prepare_chunked_text(
    tokenizer: PreTrainedTokenizerBase,
    title: str,
    content: str,
    max_length: int,
    include_tail: bool,
) -> str:
    cleaned_title = _clean_text(clean_title(title))
    cleaned_body = _clean_text(strip_html_keep_text(content))

    title_ids = tokenizer.encode(cleaned_title, add_special_tokens=False)
    body_ids = tokenizer.encode(cleaned_body, add_special_tokens=False)
    special_buffer = 4

    if max_length <= special_buffer:
        return tokenizer.decode(title_ids[:1], skip_special_tokens=True).strip()

    title_budget = max(1, min(len(title_ids), max_length - special_buffer))
    kept_title_ids = title_ids[:title_budget]
    remaining_budget = max(0, max_length - special_buffer - len(kept_title_ids))

    head_ids: list[int] = []
    tail_ids: list[int] = []
    if len(body_ids) <= remaining_budget:
        head_ids = body_ids
    elif include_tail:
        head_budget = remaining_budget // 2
        tail_budget = remaining_budget - head_budget
        head_ids = body_ids[:head_budget]
        tail_ids = body_ids[-tail_budget:] if tail_budget > 0 else []
        if head_budget + tail_budget >= len(body_ids):
            head_ids = body_ids
            tail_ids = []
    else:
        head_ids = body_ids[:remaining_budget]

    sep = f" {tokenizer.sep_token} " if tokenizer.sep_token else " "
    parts = [tokenizer.decode(kept_title_ids, skip_special_tokens=True).strip()]
    if head_ids:
        parts.append(tokenizer.decode(head_ids, skip_special_tokens=True).strip())
    if tail_ids:
        parts.append(tokenizer.decode(tail_ids, skip_special_tokens=True).strip())

    return sep.join(part for part in parts if part)


def prepare_title_head_tail(
    tokenizer: PreTrainedTokenizerBase,
    title: str,
    content: str,
    max_length: int,
) -> str:
    return _prepare_chunked_text(
        tokenizer=tokenizer,
        title=title,
        content=content,
        max_length=max_length,
        include_tail=True,
    )


def prepare_title_head(
    tokenizer: PreTrainedTokenizerBase,
    title: str,
    content: str,
    max_length: int,
) -> str:
    return _prepare_chunked_text(
        tokenizer=tokenizer,
        title=title,
        content=content,
        max_length=max_length,
        include_tail=False,
    )


def load_snapshot(
    snapshot_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_df = pd.read_parquet(snapshot_dir / "train.parquet")
    eval_df = pd.read_parquet(snapshot_dir / "eval.parquet")
    metadata = json.loads((snapshot_dir / "metadata.json").read_text())
    return train_df, eval_df, metadata


def apply_training_balance(train_df: pd.DataFrame) -> pd.DataFrame:
    good_df = train_df.loc[train_df["label"] == 1].copy().sort_values("article_id")
    bad_df = train_df.loc[train_df["label"] == 0].copy().sort_values("article_id")

    min_count = min(len(good_df), len(bad_df))
    excellent_mask = (good_df["vote"] == 1) | good_df["starred"].astype(bool)
    excellent_df = good_df.loc[excellent_mask]
    regular_df = good_df.loc[~excellent_mask]

    remaining_slots = max(0, min_count - len(excellent_df))
    if remaining_slots > 0:
        regular_df = regular_df.tail(remaining_slots)
    else:
        regular_df = regular_df.iloc[0:0]

    balanced_good = pd.concat(
        [regular_df, excellent_df], ignore_index=True
    ).sort_values("article_id")
    balanced_bad = bad_df.tail(min_count).copy()

    return pd.concat([balanced_good, balanced_bad], ignore_index=True).sort_values(
        "article_id"
    )


def build_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    text_prep_mode: str,
) -> Dataset:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        if text_prep_mode == "single_blob":
            text = prepare_single_blob(row["title"], row["content"])
        elif text_prep_mode == "title_head_tail":
            text = prepare_title_head_tail(
                tokenizer,
                row["title"],
                row["content"],
                max_length=max_length,
            )
        elif text_prep_mode == "title_head":
            text = prepare_title_head(
                tokenizer,
                row["title"],
                row["content"],
                max_length=max_length,
            )
        else:
            raise ValueError(f"Unsupported text prep mode: {text_prep_mode}")

        weight = (
            DEFAULT_EXCELLENT_WEIGHT
            if (row["label"] == 1 and (row["vote"] == 1 or row["starred"]))
            else 1.0
        )
        records.append(
            {"text": text, "label": int(row["label"]), "weight": float(weight)}
        )

    dataset = Dataset.from_list(records)
    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )
    return tokenized


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    clipped_probs = np.clip(probs, 1e-7, 1 - 1e-7)
    preds = (clipped_probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, clipped_probs)),
        "average_precision": float(average_precision_score(labels, clipped_probs)),
        "log_loss": float(log_loss(labels, clipped_probs)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen-snapshot relevance experiment"
    )
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument(
        "--text-prep-mode",
        choices=["single_blob", "title_head_tail", "title_head"],
        default="single_blob",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument(
        "--train-balance-mode",
        choices=["balanced", "full"],
        default=DEFAULT_TRAIN_BALANCE_MODE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_logging(config.LOGGING_CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not config.ALLOW_TRAINING_WO_GPU:
        raise RuntimeError("GPU not available. Exiting")

    train_df, eval_df, metadata = load_snapshot(args.snapshot_dir)
    if args.train_balance_mode == "balanced":
        balanced_train_df = apply_training_balance(train_df)
    elif args.train_balance_mode == "full":
        balanced_train_df = train_df.copy().sort_values("article_id")
    else:
        raise ValueError(f"Unsupported train balance mode: {args.train_balance_mode}")

    logger.info(
        f"Loaded snapshot {metadata['snapshot_id']} with {len(train_df)} train rows and {len(eval_df)} eval rows"
    )
    logger.info(
        f"Training set has {len(balanced_train_df.loc[balanced_train_df['label'] == 1])} good and {len(balanced_train_df.loc[balanced_train_df['label'] == 0])} bad rows"
    )

    total_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    balanced_dataset = build_dataset(
        balanced_train_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_prep_mode=args.text_prep_mode,
    )
    internal_split = balanced_dataset.train_test_split(test_size=0.2, seed=args.seed)
    eval_dataset = build_dataset(
        eval_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_prep_mode=args.text_prep_mode,
    )

    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="average_precision",
        push_to_hub=False,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=True,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        report_to=[],
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=internal_split["train"],
        eval_dataset=internal_split["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(
            pred.label_ids,
            torch.sigmoid(torch.tensor(pred.predictions)[:, 1]).cpu().numpy(),
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            ProgressLoggingCallback(),
        ],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    train_start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - train_start

    predictions = trainer.predict(eval_dataset)
    probs = torch.sigmoid(torch.tensor(predictions.predictions)[:, 1]).cpu().numpy()
    labels = np.array(predictions.label_ids)
    metrics = compute_metrics(labels, probs)

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / math.pow(1024, 3)

    total_seconds = time.perf_counter() - total_start

    results = {
        **metrics,
        "peak_vram_gb": float(peak_vram_gb),
        "train_seconds": float(train_seconds),
        "total_seconds": float(total_seconds),
        "snapshot_id": metadata["snapshot_id"],
        "model_name": args.model_name,
        "max_length": args.max_length,
        "text_prep_mode": args.text_prep_mode,
        "seed": args.seed,
        "train_balance_mode": args.train_balance_mode,
        "train_rows": int(len(balanced_train_df)),
        "internal_train_rows": int(len(internal_split["train"])),
        "internal_dev_rows": int(len(internal_split["test"])),
        "eval_rows": int(len(eval_df)),
    }

    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    trainer.save_model(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")

    metric_order = [
        "average_precision",
        "roc_auc",
        "log_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "peak_vram_gb",
        "train_seconds",
        "total_seconds",
    ]
    for metric_name in metric_order:
        print(f"METRIC {metric_name}={results[metric_name]}")


if __name__ == "__main__":
    main()
