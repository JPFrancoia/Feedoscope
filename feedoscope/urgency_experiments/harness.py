import argparse
import json
import logging
import math
from pathlib import Path
import shutil
import time
from typing import Any

from datasets import Dataset  # type: ignore[import]
import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from custom_logging import init_logging
from feedoscope import config, relevance_text
from feedoscope.relevance_embedding import mean_pool
from feedoscope.urgency_experiments.metrics import (
    clip_probabilities,
    compute_binary_classification_metrics,
)

logger = logging.getLogger(__name__)

DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 2
DEFAULT_TRAIN_BALANCE_MODE = "full"
DEFAULT_CLASSIFIER_TYPE = "transformer"
DEFAULT_LINEAR_C = 1.0
DEFAULT_EMBED_POOLING = "mean"


class ProgressLoggingCallback(TrainerCallback):
    """Log coarse training progress without tqdm noise."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        logger.info(
            f"Training progress: step {state.global_step}/{state.max_steps or '?'}"
        )


class WeightedBinaryTrainer(Trainer):
    """Trainer variant with optional class-weighted cross entropy."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        del num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_snapshot(
    snapshot_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load one frozen urgency snapshot from disk."""
    train_df = pd.read_parquet(snapshot_dir / "train.parquet")
    eval_df = pd.read_parquet(snapshot_dir / "eval.parquet")
    metadata = json.loads((snapshot_dir / "metadata.json").read_text())
    return train_df, eval_df, metadata


def apply_training_balance(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample to a deterministic 50/50 subset for balance-mode experiments."""
    rng = np.random.default_rng(seed)
    class_zero = train_df.loc[train_df["label"] == 0].copy()
    class_one = train_df.loc[train_df["label"] == 1].copy()
    target = min(len(class_zero), len(class_one))

    if target == 0:
        raise RuntimeError("Need both urgency classes to build a balanced train split.")

    sampled_zero = class_zero.sample(n=target, random_state=seed)
    sampled_one = class_one.sample(n=target, random_state=seed + 1)
    balanced = pd.concat([sampled_zero, sampled_one], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
    return balanced.sort_values("article_id").reset_index(drop=True)


def build_texts(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    text_prep_mode: str,
) -> list[str]:
    """Prepare model-ready text from the frozen snapshot rows."""
    texts: list[str] = []
    for row in df.to_dict(orient="records"):
        if text_prep_mode == "single_blob":
            text = relevance_text.prepare_single_blob(row["title"], row["content"])
        elif text_prep_mode == "title_head":
            text = relevance_text.prepare_title_head(
                tokenizer=tokenizer,
                title=row["title"],
                content=row["content"],
                max_length=max_length,
            )
        else:
            raise ValueError(f"Unsupported text prep mode: {text_prep_mode}")
        texts.append(text)
    return texts


def build_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    text_prep_mode: str,
) -> Dataset:
    """Build a tokenized Hugging Face dataset for one dataframe split."""
    texts = build_texts(
        df, tokenizer=tokenizer, max_length=max_length, text_prep_mode=text_prep_mode
    )
    dataset = Dataset.from_list(
        [
            {"text": text, "label": int(label)}
            for text, label in zip(texts, df["label"].tolist())
        ]
    )
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for binary urgency training."""
    counts = np.bincount(labels.astype(int), minlength=2)
    if np.any(counts == 0):
        raise RuntimeError("Both urgency classes must be present in the train split.")
    total = int(counts.sum())
    weights = total / (2.0 * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def pool_hidden_states(
    outputs, attention_mask: torch.Tensor, pooling_mode: str
) -> torch.Tensor:
    """Pool token embeddings into one dense vector per article."""
    if pooling_mode == "mean":
        return mean_pool(outputs.last_hidden_state, attention_mask)
    if pooling_mode == "cls":
        return outputs.last_hidden_state[:, 0]
    if pooling_mode == "pooler":
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        raise RuntimeError("pooler_output requested but missing from model output")
    raise ValueError(f"Unsupported EMBED_POOLING: {pooling_mode}")


def encode_dense_texts(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    pooling_mode: str,
    apply_layer_norm: bool,
    truncate_dim: int,
) -> np.ndarray:
    """Encode text batches into normalized dense features."""
    embeddings: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            if not hasattr(outputs, "last_hidden_state"):
                raise RuntimeError(
                    "Embedding experiments require last_hidden_state in model output"
                )
            pooled = pool_hidden_states(outputs, inputs["attention_mask"], pooling_mode)
            if apply_layer_norm:
                pooled = torch.nn.functional.layer_norm(
                    pooled,
                    normalized_shape=(pooled.shape[1],),
                )
            if truncate_dim > 0 and truncate_dim < pooled.shape[1]:
                pooled = pooled[:, :truncate_dim]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def save_embedding_artifact(
    output_dir: Path,
    classifier: LogisticRegression,
    results: dict[str, Any],
) -> None:
    """Persist the embedding-linear classifier and run metadata."""
    artifact_dir = output_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, artifact_dir / "classifier.joblib")
    (artifact_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")


def run_embedding_experiment(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run one frozen-snapshot urgency experiment with dense embeddings."""
    total_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device)

    train_texts = build_texts(
        train_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_prep_mode=args.text_prep_mode,
    )
    eval_texts = build_texts(
        eval_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_prep_mode=args.text_prep_mode,
    )
    train_labels = train_df["label"].to_numpy(dtype=int)
    eval_labels = eval_df["label"].to_numpy(dtype=int)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_start = time.perf_counter()
    train_features = encode_dense_texts(
        model=model,
        tokenizer=tokenizer,
        texts=train_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        pooling_mode=args.embed_pooling,
        apply_layer_norm=args.embed_layer_norm,
        truncate_dim=args.embed_truncate_dim,
    )
    eval_features = encode_dense_texts(
        model=model,
        tokenizer=tokenizer,
        texts=eval_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        pooling_mode=args.embed_pooling,
        apply_layer_norm=args.embed_layer_norm,
        truncate_dim=args.embed_truncate_dim,
    )

    classifier = LogisticRegression(
        max_iter=4000,
        C=args.linear_c,
        random_state=args.seed,
        class_weight="balanced" if args.train_balance_mode == "full" else None,
    )
    classifier.fit(train_features, train_labels)
    train_seconds = time.perf_counter() - train_start

    probs = clip_probabilities(classifier.predict_proba(eval_features)[:, 1])
    metrics = compute_binary_classification_metrics(eval_labels, probs)
    total_seconds = time.perf_counter() - total_start
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated() / math.pow(1024, 3))
        if torch.cuda.is_available()
        else 0.0
    )

    results = {
        **metrics,
        "peak_vram_gb": peak_vram_gb,
        "train_seconds": float(train_seconds),
        "total_seconds": float(total_seconds),
        "snapshot_id": metadata["snapshot_id"],
        "model_name": args.model_name,
        "classifier_type": args.classifier_type,
        "max_length": args.max_length,
        "text_prep_mode": args.text_prep_mode,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "train_balance_mode": args.train_balance_mode,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "linear_c": args.linear_c,
        "embed_pooling": args.embed_pooling,
        "embed_layer_norm": args.embed_layer_norm,
        "embed_truncate_dim": args.embed_truncate_dim,
    }
    save_embedding_artifact(output_dir, classifier, results)
    return results


def run_transformer_experiment(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run one frozen-snapshot urgency experiment with a HF classifier."""
    del device  # Trainer manages device placement.
    total_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    internal_train_df, internal_dev_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=args.seed,
        stratify=train_df["label"],
    )
    internal_train_df = internal_train_df.sort_values("article_id").reset_index(
        drop=True
    )
    internal_dev_df = internal_dev_df.sort_values("article_id").reset_index(drop=True)

    train_dataset = build_dataset(
        internal_train_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_prep_mode=args.text_prep_mode,
    )
    dev_dataset = build_dataset(
        internal_dev_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_prep_mode=args.text_prep_mode,
    )
    eval_dataset = build_dataset(
        eval_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_prep_mode=args.text_prep_mode,
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
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
        report_to=[],
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        trust_remote_code=True,
    )
    class_weights = None
    if args.train_balance_mode == "full":
        class_weights = compute_class_weights(
            internal_train_df["label"].to_numpy(dtype=int)
        )

    trainer = WeightedBinaryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda pred: compute_binary_classification_metrics(
            np.asarray(pred.label_ids, dtype=int),
            torch.sigmoid(torch.tensor(pred.predictions)[:, 1]).cpu().numpy(),
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            ProgressLoggingCallback(),
        ],
        class_weights=class_weights,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - train_start

    predictions = trainer.predict(eval_dataset)
    probs = clip_probabilities(
        torch.sigmoid(torch.tensor(predictions.predictions)[:, 1]).cpu().numpy()
    )
    labels = np.asarray(predictions.label_ids, dtype=int)
    metrics = compute_binary_classification_metrics(labels, probs)
    total_seconds = time.perf_counter() - total_start
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated() / math.pow(1024, 3))
        if torch.cuda.is_available()
        else 0.0
    )

    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)

    results = {
        **metrics,
        "peak_vram_gb": peak_vram_gb,
        "train_seconds": float(train_seconds),
        "total_seconds": float(total_seconds),
        "snapshot_id": metadata["snapshot_id"],
        "model_name": args.model_name,
        "classifier_type": args.classifier_type,
        "max_length": args.max_length,
        "text_prep_mode": args.text_prep_mode,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "train_balance_mode": args.train_balance_mode,
        "train_rows": int(len(train_df)),
        "internal_train_rows": int(len(internal_train_df)),
        "internal_dev_rows": int(len(internal_dev_df)),
        "eval_rows": int(len(eval_df)),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI args for one urgency autoresearch run."""
    parser = argparse.ArgumentParser(
        description="Run one frozen-snapshot urgency experiment"
    )
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument(
        "--text-prep-mode",
        choices=["single_blob", "title_head"],
        default="single_blob",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifier-type",
        choices=["transformer", "embedding_linear"],
        default=DEFAULT_CLASSIFIER_TYPE,
    )
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
    parser.add_argument("--linear-c", type=float, default=DEFAULT_LINEAR_C)
    parser.add_argument(
        "--embed-pooling",
        choices=["mean", "cls", "pooler"],
        default=DEFAULT_EMBED_POOLING,
    )
    parser.add_argument(
        "--embed-layer-norm",
        action="store_true",
        help="Apply layer norm before L2 normalization in embedding mode",
    )
    parser.add_argument(
        "--embed-truncate-dim",
        type=int,
        default=0,
        help="Keep only the first N embedding dimensions, or 0 for full width",
    )
    return parser.parse_args()


def main() -> None:
    """Run one urgency experiment against the frozen snapshot."""
    args = parse_args()
    init_logging(config.LOGGING_CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not config.ALLOW_TRAINING_WO_GPU:
        raise RuntimeError("GPU not available. Exiting")

    train_df, eval_df, metadata = load_snapshot(args.snapshot_dir)
    if args.train_balance_mode == "balanced":
        train_df = apply_training_balance(train_df, seed=args.seed)
    elif args.train_balance_mode == "full":
        train_df = train_df.copy().sort_values("article_id").reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported train balance mode: {args.train_balance_mode}")

    logger.info(
        f"Loaded snapshot {metadata['snapshot_id']} with {len(train_df)} train rows and {len(eval_df)} eval rows"
    )
    logger.info(
        f"Training set has {int((train_df['label'] == 1).sum())} urgent and "
        f"{int((train_df['label'] == 0).sum())} evergreen rows"
    )

    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)

    if args.classifier_type == "embedding_linear":
        results = run_embedding_experiment(
            args=args,
            train_df=train_df,
            eval_df=eval_df,
            metadata=metadata,
            output_dir=output_dir,
            device=device,
        )
    else:
        results = run_transformer_experiment(
            args=args,
            train_df=train_df,
            eval_df=eval_df,
            metadata=metadata,
            output_dir=output_dir,
            device=device,
        )

    metric_order = [
        "average_precision",
        "roc_auc",
        "log_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "peak_vram_gb",
        "train_seconds",
        "total_seconds",
    ]
    for metric_name in metric_order:
        print(f"METRIC {metric_name}={results[metric_name]}")


if __name__ == "__main__":
    main()
