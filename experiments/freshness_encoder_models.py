"""Compare embedding encoders on one frozen three-label Freshness snapshot."""

import argparse
import asyncio
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from custom_logging import init_logging
from feedoscope import relevance_text

SEED = 42
VALIDATION_SIZE = 150
MAX_LENGTH = 2048
LINEAR_C = 20.0
WEIGHT_EXPONENT = 0.375
HORIZONS = ("fresh_d", "fresh_m", "fresh_y")
REPRESENTATIVE_DAYS = np.asarray((7.0, 90.0, 365.0))
DEFAULT_ARTIFACT_ROOT = Path("artifacts/freshness_encoder_models")
FRESHNESS_TASK = (
    "Classify the useful-life horizon of an RSS article as fresh_d for 0-29 days, "
    "fresh_m for 30 days through 6 months, or fresh_y for more than 6 months"
)


@dataclass(frozen=True)
class ModelContract:
    """Define one pinned encoder contract."""

    artifact_name: str
    model_name: str
    revision: str
    prefix: str
    pooling: Literal["mean", "last_token"]
    expected_dimension: int
    attention: Literal["sdpa"] | None = None


MODEL_CONTRACTS = {
    "embeddinggemma": ModelContract(
        "embeddinggemma-unprompted",
        "google/embeddinggemma-300m",
        "57c266a740f537b4dc058e1b0cda161fd15afa75",
        "",
        "mean",
        768,
    ),
    "jina-small": ModelContract(
        "jina-v5-small-classification",
        "jinaai/jina-embeddings-v5-text-small-classification",
        "4447914a9b5b2fb00db3ce0884602b47a08f9458",
        "Document: ",
        "last_token",
        1024,
    ),
    "harrier": ModelContract(
        "harrier-0.6b",
        "microsoft/harrier-oss-v1-0.6b",
        "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
        f"Instruct: {FRESHNESS_TASK}\nQuery: ",
        "last_token",
        1024,
    ),
    "qwen3-0.6b": ModelContract(
        "qwen3-embedding-0.6b",
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        f"Instruct: {FRESHNESS_TASK}\nQuery:",
        "last_token",
        1024,
        "sdpa",
    ),
    "bekko-a8m": ModelContract(
        "bekko-embedding-a8m",
        "hotchpotch/bekko-embedding-v1-a8m",
        "b24cde5de82214ada4c01f173b137c78160b13c6",
        "",
        "mean",
        384,
        "sdpa",
    ),
    "bekko-a25m": ModelContract(
        "bekko-embedding-a25m",
        "hotchpotch/bekko-embedding-v1-a25m",
        "e0f3136db1b823ccbc67c4bea7d29f295516535b",
        "",
        "mean",
        384,
        "sdpa",
    ),
    "nemotron": ModelContract(
        "nemotron-3-embed-1b",
        "nvidia/Nemotron-3-Embed-1B-BF16",
        "a5e0f804b9e90a1ca6784ecbf6e41595774fc834",
        "passage: ",
        "mean",
        2048,
        "sdpa",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_chronological(
    rows: pd.DataFrame, validation_size: int = VALIDATION_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort effective labels and hold out exactly the newest rows."""
    if validation_size <= 0 or len(rows) <= validation_size:
        raise ValueError("Freshness rows must exceed the positive validation size")
    ordered = rows.assign(
        published_at=pd.to_datetime(rows["published_at"], utc=True)
    ).sort_values(["published_at", "article_id"], kind="stable")
    if ordered["article_id"].duplicated().any():
        raise ValueError("Freshness snapshot contains duplicate article IDs")
    return (
        ordered.iloc[:-validation_size].reset_index(drop=True),
        ordered.iloc[-validation_size:].reset_index(drop=True),
    )


async def export_snapshot(output: Path, validation_size: int) -> None:
    """Export effective labels through a server-enforced read-only session."""
    os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"
    from feedoscope.data_registry import data_registry as dr

    await dr.global_pool.open(wait=True)
    try:
        async with dr.global_pool.connection() as connection:
            read_only = await connection.execute("show transaction_read_only")
            row = await read_only.fetchone()
            if row is None or row["transaction_read_only"] != "on":
                raise RuntimeError("Refusing export: PostgreSQL session is writable")
        labeled_data = await dr.get_semantic_freshness_training_data()
    finally:
        await dr.global_pool.close()

    records = [
        {
            "article_id": article.article_id,
            "title": article.title,
            "content": article.content,
            "published_at": article.date_entered.isoformat(),
            "label": label,
            "label_source": source,
        }
        for article, label, source in labeled_data
    ]
    train, evaluation = split_chronological(
        pd.DataFrame.from_records(records), validation_size
    )
    if set(train["article_id"]) & set(evaluation["article_id"]):
        raise RuntimeError("Freshness train and evaluation IDs overlap")

    output.mkdir(parents=True, exist_ok=False)
    train_path = output / "train.parquet"
    eval_path = output / "eval.parquet"
    train.to_parquet(train_path, index=False)
    evaluation.to_parquet(eval_path, index=False)
    metadata = {
        "snapshot_id": dt.datetime.now(dt.UTC).strftime(
            "freshness_encoder_models_%Y%m%dT%H%M%SZ"
        ),
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "transaction_read_only": "on",
        "split": "newest chronological effective labels",
        "validation_size": validation_size,
        "counts": {
            "train": len(train),
            "eval": len(evaluation),
            "train_by_label": _label_counts(train["label"].to_numpy()),
            "eval_by_label": _label_counts(evaluation["label"].to_numpy()),
        },
        "sha256": {
            "train.parquet": _sha256(train_path),
            "eval.parquet": _sha256(eval_path),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {len(train)} train and {len(evaluation)} evaluation rows")
    print("PostgreSQL transaction_read_only=on")


def _validate_labels(labels: np.ndarray) -> np.ndarray:
    """Return valid three-label Freshness targets."""
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or np.any((labels < 0) | (labels >= len(HORIZONS))):
        raise ValueError(f"Freshness labels must be in [0, {len(HORIZONS) - 1}]")
    return labels


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    labels = _validate_labels(labels)
    return {
        horizon: int(np.sum(labels == index)) for index, horizon in enumerate(HORIZONS)
    }


def load_snapshot(
    snapshot: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Load a snapshot only when its files match the export manifest."""
    metadata = json.loads((snapshot / "metadata.json").read_text())
    hashes = {
        name: _sha256(snapshot / name) for name in ("train.parquet", "eval.parquet")
    }
    if hashes != metadata.get("sha256"):
        raise ValueError("Freshness snapshot hashes do not match its manifest")
    train = pd.read_parquet(snapshot / "train.parquet")
    evaluation = pd.read_parquet(snapshot / "eval.parquet")
    required = {"article_id", "title", "content", "published_at", "label"}
    if not required.issubset(train.columns) or not required.issubset(
        evaluation.columns
    ):
        raise ValueError("Freshness snapshot is missing required columns")
    if set(train["article_id"]) & set(evaluation["article_id"]):
        raise ValueError("Freshness snapshot train and evaluation IDs overlap")
    _validate_labels(train["label"].to_numpy())
    _validate_labels(evaluation["label"].to_numpy())
    return metadata, train, evaluation


def last_token_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Pool the last non-padding token for left- or right-padded batches."""
    if bool(torch.all(attention_mask[:, -1] == 1)):
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    indexes = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[indexes, lengths]


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Average non-padding token representations."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def prepare_texts(
    data: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, prefix: str
) -> list[str]:
    """Prepare title and article head within the model-specific token budget."""
    prefix_length = len(tokenizer.encode(prefix, add_special_tokens=False))
    article_budget = MAX_LENGTH - prefix_length
    if article_budget <= 4:
        raise ValueError("Model prefix leaves no article token budget")
    return [
        prefix
        + relevance_text.prepare_title_head(
            tokenizer,
            str(row.title),
            str(row.content),
            article_budget,
        )
        for row in data.itertuples(index=False)
    ]


def encode_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    contract: ModelContract,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode and normalize text with one pinned model contract."""
    vectors: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            pooled = (
                mean_pool(hidden, inputs["attention_mask"])
                if contract.pooling == "mean"
                else last_token_pool(hidden, inputs["attention_mask"])
            )
            vectors.append(
                torch.nn.functional.normalize(pooled.float(), p=2, dim=1).cpu().numpy()
            )
            print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")
    result = np.concatenate(vectors).astype(np.float32, copy=False)
    if result.shape != (len(texts), contract.expected_dimension):
        raise RuntimeError(f"Unexpected embedding matrix shape: {result.shape}")
    if not np.isfinite(result).all() or not np.allclose(
        np.linalg.norm(result, axis=1), 1, atol=1e-4
    ):
        raise RuntimeError("Model produced invalid normalized embeddings")
    return result


def fit_classifiers(
    embeddings: np.ndarray, labels: np.ndarray
) -> list[LogisticRegression]:
    """Fit the active two cumulative Freshness heads."""
    labels = _validate_labels(labels)
    classifiers = []
    for boundary in range(len(HORIZONS) - 1):
        target = labels > boundary
        counts = np.bincount(target, minlength=2)
        if int(counts.min()) == 0:
            raise RuntimeError(f"Freshness boundary {boundary} needs both classes")
        weights = {
            value: (len(labels) / (2 * count)) ** WEIGHT_EXPONENT
            for value, count in enumerate(counts)
        }
        classifier = LogisticRegression(
            C=LINEAR_C,
            class_weight=weights,
            fit_intercept=False,
            max_iter=4000,
            random_state=SEED,
        )
        classifiers.append(classifier.fit(embeddings, target))
    return classifiers


def bucket_probabilities(
    embeddings: np.ndarray, classifiers: list[LogisticRegression]
) -> np.ndarray:
    """Convert cumulative-head output into ordered class probabilities."""
    tails = np.column_stack(
        [classifier.predict_proba(embeddings)[:, 1] for classifier in classifiers]
    )
    tails = -np.sort(-tails, axis=1)
    return np.column_stack(
        (1 - tails[:, 0], tails[:, :-1] - tails[:, 1:], tails[:, -1])
    )


def compute_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | None]:
    """Calculate the metrics used by the weekly Freshness evaluator."""
    labels = _validate_labels(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), len(HORIZONS)):
        raise ValueError("Freshness probability matrix has an unexpected shape")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Freshness probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Freshness probability rows must sum to one")
    predictions = probabilities.argmax(axis=1)
    observed = np.eye(len(HORIZONS))[labels]
    rps = np.mean(
        np.square(
            np.cumsum(probabilities, axis=1)[:, :-1]
            - np.cumsum(observed, axis=1)[:, :-1]
        )
    )
    predicted_days = probabilities @ REPRESENTATIVE_DAYS
    true_days = REPRESENTATIVE_DAYS[labels]
    long_lived = (labels == len(HORIZONS) - 1).astype(int)
    kappa = (
        cohen_kappa_score(labels, predictions, weights="quadratic")
        if np.unique(np.concatenate((labels, predictions))).size > 1
        else math.nan
    )
    return {
        "rps": float(rps),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(len(HORIZONS)),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_kappa": float(kappa) if math.isfinite(kappa) else None,
        "log_duration_mae": float(
            np.mean(np.abs(np.log(predicted_days) - np.log(true_days)))
        ),
        "long_lived_auc": (
            float(roc_auc_score(long_lived, probabilities[:, -1]))
            if np.unique(long_lived).size == 2
            else None
        ),
    }


def run_benchmark(
    snapshot: Path,
    model_key: str,
    output: Path,
    embeddings_dir: Path,
    batch_size: int,
    dtype_name: str,
    model_path: Path | None,
) -> None:
    """Encode one frozen snapshot and evaluate its ordered linear heads."""
    metadata, train, evaluation = load_snapshot(snapshot)
    contract = MODEL_CONTRACTS[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "auto": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    model_source = str(model_path) if model_path else contract.model_name
    revision = None if model_path else contract.revision
    model_source_kind = "local_unverified" if model_path else "huggingface_pinned"
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=revision,
        trust_remote_code=True,
        local_files_only=model_path is not None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    options = {"attn_implementation": contract.attention} if contract.attention else {}
    model = AutoModel.from_pretrained(
        model_source,
        revision=revision,
        trust_remote_code=True,
        local_files_only=model_path is not None,
        dtype=dtype,
        **options,
    ).to(device)
    model.config.use_cache = False
    combined = pd.concat((train, evaluation), ignore_index=True)
    texts = prepare_texts(combined, tokenizer, contract.prefix)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    embeddings = encode_texts(texts, tokenizer, model, contract, device, batch_size)
    encoding_seconds = time.perf_counter() - started

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = embeddings_dir / (
        f"{contract.artifact_name}-{metadata['snapshot_id']}.npz"
    )
    np.savez_compressed(
        embedding_path,
        embeddings=embeddings,
        article_ids=combined["article_id"].to_numpy(dtype=np.int64),
        labels=combined["label"].to_numpy(dtype=int),
    )
    train_count = len(train)
    classifiers = fit_classifiers(
        embeddings[:train_count], train["label"].to_numpy(dtype=int)
    )
    probabilities = bucket_probabilities(embeddings[train_count:], classifiers)
    model_details = asdict(contract)
    model_details.update(
        {
            "source": model_source_kind,
            "loaded_revision": revision,
        }
    )
    result = {
        "snapshot_id": metadata["snapshot_id"],
        "snapshot_sha256": metadata["sha256"],
        "model_key": model_key,
        "model": model_details,
        "settings": {
            "max_length": MAX_LENGTH,
            "text_prep": "title_head",
            "batch_size": batch_size,
            "dtype": str(dtype).removeprefix("torch."),
            "linear_c": LINEAR_C,
            "fit_intercept": False,
            "weight_exponent": WEIGHT_EXPONENT,
            "seed": SEED,
        },
        "counts": metadata["counts"],
        "metrics": compute_metrics(
            evaluation["label"].to_numpy(dtype=int), probabilities
        ),
        "evaluation": {
            "article_ids": evaluation["article_id"].to_numpy(dtype=np.int64).tolist(),
            "labels": evaluation["label"].to_numpy(dtype=int).tolist(),
            "probabilities": probabilities.tolist(),
        },
        "embedding_artifact": {
            "path": str(embedding_path),
            "sha256": _sha256(embedding_path),
        },
        "timing_seconds": {"encoding": encoding_seconds},
        "peak_vram_gb": (
            float(torch.cuda.max_memory_allocated() / math.pow(1024, 3))
            if device.type == "cuda"
            else 0.0
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {model_key} result to {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument(
        "--output", type=Path, default=DEFAULT_ARTIFACT_ROOT / "snapshot"
    )
    export.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    run = commands.add_parser("run")
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--model", choices=MODEL_CONTRACTS, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model-path", type=Path)
    run.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "embeddings",
    )
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    return parser


def main() -> None:
    init_logging(os.getenv("LOGGING_CONFIG", "logging.conf"))
    args = _parser().parse_args()
    if args.command == "export":
        asyncio.run(export_snapshot(args.output, args.validation_size))
    else:
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        run_benchmark(
            args.snapshot,
            args.model,
            args.output,
            args.embeddings_dir,
            args.batch_size,
            args.dtype,
            args.model_path,
        )


if __name__ == "__main__":
    main()
