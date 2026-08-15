"""Evaluate new embedding encoders on a frozen relevance snapshot."""

import argparse
import asyncio
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Literal
import zipfile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from custom_logging import init_logging
from feedoscope import relevance_text
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

SEED = 42
EVAL_SIZE_PER_CLASS = 200
MAX_LENGTH = 2048
LINEAR_C_VALUES = (0.01, 0.1, 1.0, 5.0, 10.0, 100.0)
CV_FOLDS = 5
BOOTSTRAP_SAMPLES = 10_000
RELEVANCE_TASK = "Classify RSS articles according to the reader's relevance preferences"
HARRIER_PREFIX = f"Instruct: {RELEVANCE_TASK}\nQuery: "
QWEN_PREFIX = f"Instruct: {RELEVANCE_TASK}\nQuery:"
FROZEN_SNAPSHOT_ID = "relevance_new_models_20260802T122707Z"
FROZEN_SNAPSHOT_SHA256 = {
    "train.parquet": "6486031c953ed4f7ddd3f741bb8c12190d918a132c7fa5bea423e2ce3f5bec64",
    "eval.parquet": "2112fb6b69ddeebdb450f43270612776ac959354469939aa2edc6f64755ea24c",
    "metadata.json": "e84e29354ad6522ee551365093bfc9939d836c2b7012b314fae5f9a062708372",
}
FROZEN_SNAPSHOT_COUNTS = {
    "total": 7700,
    "train": 7300,
    "eval": 400,
    "positive_train": 6142,
    "negative_train": 1158,
    "positive_eval": 200,
    "negative_eval": 200,
}
DEFAULT_EMBEDDINGS_DIR = Path("artifacts/relevance_new_models/embeddings")
ARTIFACT_ARRAY_KEYS = {
    "metadata",
    "train_embeddings",
    "eval_embeddings",
    "train_labels",
    "eval_labels",
    "train_article_ids",
    "eval_article_ids",
}


@dataclass(frozen=True)
class ModelContract:
    """Define the encoding contract for one benchmark model."""

    artifact_name: str
    model_name: str
    revision: str
    prefix: str
    pooling: Literal["mean", "last_token"]
    expected_dimension: int
    attention: Literal["sdpa"] | None = None


@dataclass(frozen=True)
class HeadContract:
    """Define one classifier head and its compact train-only search grid."""

    name: str
    candidates: tuple[dict[str, Any], ...]
    calibration: Literal["sigmoid"] | None = None


HEAD_CONTRACTS = {
    "logistic-regression": HeadContract(
        name="logistic-regression",
        candidates=tuple({"C": value} for value in LINEAR_C_VALUES),
    ),
    "linear-svc": HeadContract(
        name="linear-svc",
        candidates=tuple({"C": value} for value in LINEAR_C_VALUES),
        calibration="sigmoid",
    ),
    "rbf-svc": HeadContract(
        name="rbf-svc",
        candidates=(
            {"C": 1.0, "gamma": "scale"},
            {"C": 10.0, "gamma": "scale"},
            {"C": 1.0, "gamma": 0.1},
            {"C": 10.0, "gamma": 0.1},
        ),
        calibration="sigmoid",
    ),
    "mlp": HeadContract(
        name="mlp",
        candidates=(
            {"hidden_layer_size": 32, "alpha": 0.0001},
            {"hidden_layer_size": 64, "alpha": 0.0001},
            {"hidden_layer_size": 32, "alpha": 0.001},
        ),
    ),
    "extra-trees": HeadContract(
        name="extra-trees",
        candidates=(
            {"min_samples_leaf": 1, "max_features": 0.5},
            {"min_samples_leaf": 5, "max_features": 0.5},
            {"min_samples_leaf": 5, "max_features": 1.0},
        ),
    ),
}


MODEL_CONTRACTS = {
    "google/embeddinggemma-300m": ModelContract(
        artifact_name="embeddinggemma-unprompted",
        model_name="google/embeddinggemma-300m",
        revision="57c266a740f537b4dc058e1b0cda161fd15afa75",
        prefix="",
        pooling="mean",
        expected_dimension=768,
    ),
    "google/embeddinggemma-300m-classification": ModelContract(
        artifact_name="embeddinggemma-classification",
        model_name="google/embeddinggemma-300m",
        revision="57c266a740f537b4dc058e1b0cda161fd15afa75",
        prefix="task: classification | query: ",
        pooling="mean",
        expected_dimension=768,
    ),
    "jinaai/jina-embeddings-v5-text-small-classification": ModelContract(
        artifact_name="jina-v5-small-classification",
        model_name="jinaai/jina-embeddings-v5-text-small-classification",
        revision="4447914a9b5b2fb00db3ce0884602b47a08f9458",
        prefix="Document: ",
        pooling="last_token",
        expected_dimension=1024,
    ),
    "jinaai/jina-embeddings-v5-text-nano-classification": ModelContract(
        artifact_name="jina-v5-nano-classification",
        model_name="jinaai/jina-embeddings-v5-text-nano-classification",
        revision="a0129e9b8ea4c54f3dfd250793380d8d69058da3",
        prefix="Document: ",
        pooling="last_token",
        expected_dimension=768,
    ),
    "microsoft/harrier-oss-v1-0.6b": ModelContract(
        artifact_name="harrier-0.6b",
        model_name="microsoft/harrier-oss-v1-0.6b",
        revision="f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
        prefix=HARRIER_PREFIX,
        pooling="last_token",
        expected_dimension=1024,
    ),
    "Qwen/Qwen3-Embedding-0.6B": ModelContract(
        artifact_name="qwen3-embedding-0.6b",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        prefix=QWEN_PREFIX,
        pooling="last_token",
        expected_dimension=1024,
        attention="sdpa",
    ),
    "Qwen/Qwen3-Embedding-4B": ModelContract(
        artifact_name="qwen3-embedding-4b",
        model_name="Qwen/Qwen3-Embedding-4B",
        revision="5cf2132abc99cad020ac570b19d031efec650f2b",
        prefix=QWEN_PREFIX,
        pooling="last_token",
        expected_dimension=2560,
        attention="sdpa",
    ),
    "hotchpotch/bekko-embedding-v1-a8m": ModelContract(
        artifact_name="bekko-embedding-a8m",
        model_name="hotchpotch/bekko-embedding-v1-a8m",
        revision="b24cde5de82214ada4c01f173b137c78160b13c6",
        prefix="",
        pooling="mean",
        expected_dimension=384,
        attention="sdpa",
    ),
    "hotchpotch/bekko-embedding-v1-a25m": ModelContract(
        artifact_name="bekko-embedding-a25m",
        model_name="hotchpotch/bekko-embedding-v1-a25m",
        revision="e0f3136db1b823ccbc67c4bea7d29f295516535b",
        prefix="",
        pooling="mean",
        expected_dimension=384,
        attention="sdpa",
    ),
    "nvidia/Nemotron-3-Embed-1B-BF16": ModelContract(
        artifact_name="nemotron-3-embed-1b",
        model_name="nvidia/Nemotron-3-Embed-1B-BF16",
        revision="a5e0f804b9e90a1ca6784ecbf6e41595774fc834",
        prefix="passage: ",
        pooling="mean",
        expected_dimension=2048,
        attention="sdpa",
    ),
}


def assign_stratified_split(
    data: pd.DataFrame,
    eval_size_per_class: int = EVAL_SIZE_PER_CLASS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return a deterministic fixed-size holdout for every class."""
    if eval_size_per_class <= 0:
        raise ValueError("eval_size_per_class must be positive")
    if data.empty:
        raise ValueError("Cannot split an empty data set")

    rng = random.Random(seed)
    eval_ids: set[int] = set()
    for label in sorted(data["label"].unique()):
        article_ids = data.loc[data["label"] == label, "article_id"].tolist()
        if len(article_ids) <= eval_size_per_class:
            raise ValueError(
                f"Label {label} needs more than {eval_size_per_class} rows"
            )
        rng.shuffle(article_ids)
        eval_ids.update(article_ids[:eval_size_per_class])

    result = data.copy()
    result["split"] = result["article_id"].map(
        lambda article_id: "eval" if article_id in eval_ids else "train"
    )
    return result.sort_values(["label", "article_id"]).reset_index(drop=True)


def last_token_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool the last non-padding token for left-padded or right-padded batches."""
    if bool(torch.all(attention_mask[:, -1] == 1)):
        return last_hidden_state[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(
        last_hidden_state.shape[0], device=last_hidden_state.device
    )
    return last_hidden_state[batch_indices, sequence_lengths]


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average non-padding token representations."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _article_record(article: Article, label: int) -> dict[str, object]:
    return {
        "article_id": article.article_id,
        "title": article.title,
        "content": article.content,
        "vote": article.vote,
        "starred": article.starred,
        "label": label,
    }


async def export_snapshot(
    output: Path,
    eval_size_per_class: int = EVAL_SIZE_PER_CLASS,
) -> None:
    """Export current relevance rows into one immutable local snapshot."""
    from feedoscope.data_registry import data_registry as dr

    await dr.global_pool.open(wait=True)
    try:
        good_articles = await dr.get_read_articles_training(validation_size=0)
        bad_articles = await dr.get_published_articles(validation_size=0)
    finally:
        await dr.global_pool.close()

    records = [_article_record(article, 1) for article in good_articles]
    records.extend(_article_record(article, 0) for article in bad_articles)
    data = assign_stratified_split(
        pd.DataFrame.from_records(records),
        eval_size_per_class=eval_size_per_class,
    )
    train = data.loc[data["split"] == "train"].drop(columns="split")
    evaluation = data.loc[data["split"] == "eval"].drop(columns="split")

    output.mkdir(parents=True, exist_ok=False)
    train.to_parquet(output / "train.parquet", index=False)
    evaluation.to_parquet(output / "eval.parquet", index=False)

    metadata = {
        "snapshot_id": dt.datetime.now(dt.UTC).strftime(
            "relevance_new_models_%Y%m%dT%H%M%SZ"
        ),
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "seed": SEED,
        "eval_size_per_class": eval_size_per_class,
        "split": "stratified random fixed-size holdout by label",
        "query_provenance": {
            "positive": "get_read_articles_training(validation_size=0)",
            "negative": "get_published_articles(validation_size=0)",
            "window": "current repository SQL: three years",
        },
        "counts": {
            "total": len(data),
            "train": len(train),
            "eval": len(evaluation),
            "positive_train": int(train["label"].sum()),
            "negative_train": int((train["label"] == 0).sum()),
            "positive_eval": int(evaluation["label"].sum()),
            "negative_eval": int((evaluation["label"] == 0).sum()),
        },
        "train_article_ids": train["article_id"].tolist(),
        "eval_article_ids": evaluation["article_id"].tolist(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info(f"Exported frozen relevance snapshot to {output}")


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _prepare_texts(
    data: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    prefix: str,
) -> list[str]:
    prefix_length = len(tokenizer.encode(prefix, add_special_tokens=False))
    article_budget = MAX_LENGTH - prefix_length
    if article_budget <= 4:
        raise ValueError("Model prefix leaves no article token budget")
    return [
        prefix
        + relevance_text.prepare_title_head(
            tokenizer=tokenizer,
            title=str(row.title),
            content=str(row.content),
            max_length=article_budget,
        )
        for row in data.itertuples(index=False)
    ]


def _encode_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    contract: ModelContract,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
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
            if contract.pooling == "mean":
                pooled = mean_pool(hidden, inputs["attention_mask"])
            else:
                pooled = last_token_pool(hidden, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
            logger.info(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")

    result = np.concatenate(embeddings)
    if result.shape[1] != contract.expected_dimension:
        raise RuntimeError(
            f"Expected {contract.expected_dimension} dimensions, got {result.shape[1]}"
        )
    if not np.isfinite(result).all() or not np.allclose(
        np.linalg.norm(result, axis=1), 1, atol=1e-4
    ):
        raise RuntimeError("Model produced invalid normalized embeddings")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_snapshot(
    snapshot: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load only the approved balanced holdout snapshot."""
    hashes = {name: _sha256(snapshot / name) for name in FROZEN_SNAPSHOT_SHA256}
    if hashes != FROZEN_SNAPSHOT_SHA256:
        raise ValueError("Frozen snapshot hashes do not match the approved holdout")

    metadata = json.loads((snapshot / "metadata.json").read_text())
    train = pd.read_parquet(snapshot / "train.parquet")
    evaluation = pd.read_parquet(snapshot / "eval.parquet")
    if metadata.get("snapshot_id") != FROZEN_SNAPSHOT_ID:
        raise ValueError("Frozen snapshot ID does not match the approved holdout")
    if metadata.get("counts") != FROZEN_SNAPSHOT_COUNTS:
        raise ValueError("Frozen snapshot metadata counts do not match the holdout")
    if train["article_id"].tolist() != metadata.get("train_article_ids"):
        raise ValueError("Frozen snapshot train article IDs do not match its manifest")
    if evaluation["article_id"].tolist() != metadata.get("eval_article_ids"):
        raise ValueError(
            "Frozen snapshot evaluation article IDs do not match its manifest"
        )

    actual_counts = {
        "total": len(train) + len(evaluation),
        "train": len(train),
        "eval": len(evaluation),
        "positive_train": int(train["label"].sum()),
        "negative_train": int((train["label"] == 0).sum()),
        "positive_eval": int(evaluation["label"].sum()),
        "negative_eval": int((evaluation["label"] == 0).sum()),
    }
    if actual_counts != FROZEN_SNAPSHOT_COUNTS:
        raise ValueError("Frozen snapshot row counts do not match the approved holdout")
    return metadata, train, evaluation, hashes


def _weight_hashes(model_path: Path | None) -> dict[str, str]:
    """Hash local model weights when no Hub revision controls the loaded files."""
    if model_path is None:
        return {}
    hashes = {
        path.name: _sha256(path)
        for path in sorted(
            (*model_path.glob("*.safetensors"), *model_path.glob("*.bin"))
        )
    }
    if not hashes:
        raise ValueError(f"No model weights found under {model_path}")
    return hashes


def _local_support_file_hashes(model_path: Path | None) -> dict[str, str]:
    """Hash local tokenizer, configuration, and code files for artifact reuse."""
    if model_path is None:
        return {}
    return {
        str(path.relative_to(model_path)): _sha256(path)
        for path in sorted(model_path.rglob("*"))
        if path.is_file() and path.suffix not in {".bin", ".safetensors"}
    }


def _embedding_artifact_path(
    embeddings_dir: Path,
    snapshot_id: str,
    contract: ModelContract,
) -> Path:
    """Return the durable artifact path for one snapshot and model contract."""
    return embeddings_dir / f"{contract.artifact_name}-{snapshot_id}.npz"


def _artifact_identity(
    snapshot_id: str,
    snapshot_hashes: dict[str, str],
    contract: ModelContract,
    model_path: Path | None,
) -> dict[str, Any]:
    """Return the embedding-defining values that must match for reuse."""
    identity = {
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_hashes,
        "model": asdict(contract),
        "max_length": MAX_LENGTH,
        "text_prep": "title_head",
        "normalized": True,
        "tokenizer": {
            "name": contract.model_name,
            "revision": contract.revision,
        },
        "source": "local" if model_path else "huggingface",
        "weight_sha256": _weight_hashes(model_path),
    }
    if model_path:
        identity.update(
            {
                "local_model_path": str(model_path.resolve()),
                "local_support_file_sha256": _local_support_file_hashes(model_path),
            }
        )
    return identity


def _validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: int,
    expected_dimension: int,
) -> None:
    """Verify that one embedding matrix matches the saved contract."""
    if embeddings.dtype != np.float32:
        raise ValueError("Embedding artifact must contain float32 vectors")
    if embeddings.shape != (expected_rows, expected_dimension):
        raise ValueError("Embedding artifact has an unexpected matrix shape")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding artifact contains non-finite vectors")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4):
        raise ValueError("Embedding artifact contains non-normalized vectors")


def _save_embedding_artifact(
    artifact_path: Path,
    identity: dict[str, Any],
    train_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    train_article_ids: np.ndarray,
    eval_article_ids: np.ndarray,
    article_token_budget: int,
) -> str:
    """Atomically save one reusable embedding artifact and return its hash."""
    _validate_embeddings(
        train_embeddings,
        expected_rows=len(train_labels),
        expected_dimension=identity["model"]["expected_dimension"],
    )
    _validate_embeddings(
        eval_embeddings,
        expected_rows=len(eval_labels),
        expected_dimension=identity["model"]["expected_dimension"],
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "identity": identity,
        "article_token_budget": article_token_budget,
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    with tempfile.NamedTemporaryFile(
        dir=artifact_path.parent,
        prefix=f".{artifact_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        np.savez_compressed(
            stream,
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
            train_embeddings=train_embeddings,
            eval_embeddings=eval_embeddings,
            train_labels=train_labels,
            eval_labels=eval_labels,
            train_article_ids=train_article_ids,
            eval_article_ids=eval_article_ids,
        )
    temporary_path.replace(artifact_path)
    return _sha256(artifact_path)


def _load_embedding_artifact(
    artifact_path: Path,
    identity: dict[str, Any],
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    contract: ModelContract,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Load and validate one reusable embedding artifact."""
    with np.load(artifact_path, allow_pickle=False) as artifact:
        if set(artifact.files) != ARTIFACT_ARRAY_KEYS:
            raise ValueError("Embedding artifact has unexpected contents")
        try:
            metadata = json.loads(str(artifact["metadata"].item()))
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("Embedding artifact has invalid metadata") from error
        if metadata.get("identity") != identity:
            raise ValueError("Embedding artifact identity does not match")
        article_token_budget = metadata.get("article_token_budget")
        if not isinstance(article_token_budget, int) or article_token_budget <= 0:
            raise ValueError("Embedding artifact has an invalid token budget")
        train_embeddings = artifact["train_embeddings"]
        eval_embeddings = artifact["eval_embeddings"]
        train_labels = artifact["train_labels"]
        eval_labels = artifact["eval_labels"]
        train_article_ids = artifact["train_article_ids"]
        eval_article_ids = artifact["eval_article_ids"]

    expected_train_labels = train["label"].to_numpy(dtype=int)
    expected_eval_labels = evaluation["label"].to_numpy(dtype=int)
    expected_train_ids = train["article_id"].to_numpy(dtype=np.int64)
    expected_eval_ids = evaluation["article_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(train_labels, expected_train_labels) or not np.array_equal(
        eval_labels, expected_eval_labels
    ):
        raise ValueError("Embedding artifact labels do not match the snapshot")
    if not np.array_equal(train_article_ids, expected_train_ids) or not np.array_equal(
        eval_article_ids, expected_eval_ids
    ):
        raise ValueError("Embedding artifact article IDs do not match the snapshot")
    _validate_embeddings(
        train_embeddings,
        expected_rows=len(train),
        expected_dimension=contract.expected_dimension,
    )
    _validate_embeddings(
        eval_embeddings,
        expected_rows=len(evaluation),
        expected_dimension=contract.expected_dimension,
    )
    return (
        train_embeddings,
        eval_embeddings,
        _sha256(artifact_path),
        article_token_budget,
    )


def _quarantine_embedding_artifact(artifact_path: Path) -> Path:
    """Rename an invalid artifact so the next run recomputes it."""
    invalid_path = artifact_path.with_suffix(".npz.invalid")
    if invalid_path.exists():
        invalid_path.unlink()
    artifact_path.replace(invalid_path)
    return invalid_path


def _compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise RuntimeError("Classifier produced invalid probabilities")
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
    }


def _require_balanced_evaluation(evaluation: pd.DataFrame) -> None:
    """Require the approved 200-positive and 200-negative holdout."""
    counts = evaluation["label"].value_counts().to_dict()
    if len(evaluation) != EVAL_SIZE_PER_CLASS * 2 or counts != {
        0: EVAL_SIZE_PER_CLASS,
        1: EVAL_SIZE_PER_CLASS,
    }:
        raise ValueError(
            "Evaluation requires exactly 200 positive and 200 negative rows"
        )


def _new_classifier(head: HeadContract, parameters: dict[str, Any]) -> Any:
    """Return one permitted classifier head with deterministic settings."""
    if head.name == "logistic-regression":
        return LogisticRegression(C=parameters["C"], max_iter=4000, random_state=SEED)
    if head.name == "linear-svc":
        return CalibratedClassifierCV(
            LinearSVC(C=parameters["C"], random_state=SEED),
            method="sigmoid",
            cv=CV_FOLDS,
            ensemble=False,
        )
    if head.name == "rbf-svc":
        return CalibratedClassifierCV(
            SVC(C=parameters["C"], gamma=parameters["gamma"]),
            method="sigmoid",
            cv=CV_FOLDS,
            ensemble=False,
        )
    if head.name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(parameters["hidden_layer_size"],),
            alpha=parameters["alpha"],
            early_stopping=True,
            max_iter=300,
            random_state=SEED,
        )
    if head.name == "extra-trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=parameters["min_samples_leaf"],
            max_features=parameters["max_features"],
            n_jobs=1,
            random_state=SEED,
        )
    raise ValueError(f"Unsupported classifier head: {head.name}")


def _select_head_parameters(
    head: HeadContract,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select one classifier configuration with train rows only."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    results: list[dict[str, Any]] = []
    for parameters in head.candidates:
        scores: list[float] = []
        for fit_index, validation_index in folds.split(train_embeddings, train_labels):
            classifier = _new_classifier(head, parameters)
            classifier.fit(train_embeddings[fit_index], train_labels[fit_index])
            probabilities = classifier.predict_proba(
                train_embeddings[validation_index]
            )[:, 1]
            scores.append(
                float(
                    average_precision_score(
                        train_labels[validation_index], probabilities
                    )
                )
            )
        results.append(
            {
                "parameters": parameters,
                "mean_average_precision": float(np.mean(scores)),
                "std_average_precision": float(np.std(scores, ddof=0)),
            }
        )

    selected = max(
        results,
        key=lambda result: (
            result["mean_average_precision"],
            json.dumps(result["parameters"], sort_keys=True),
        ),
    )
    return dict(selected["parameters"]), results


def _paired_bootstrap_average_precision_difference(
    labels: np.ndarray,
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
) -> dict[str, float | int]:
    """Measure candidate AP minus reference AP with paired bootstrap samples."""
    if not np.array_equal(labels, labels.astype(int)):
        raise ValueError("Evaluation labels must be integers")
    rng = np.random.default_rng(SEED)
    differences = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        sample = rng.integers(0, len(labels), size=len(labels))
        differences[index] = average_precision_score(
            labels[sample], candidate_probabilities[sample]
        ) - average_precision_score(labels[sample], reference_probabilities[sample])
    return {
        "average_precision_difference": float(
            average_precision_score(labels, candidate_probabilities)
            - average_precision_score(labels, reference_probabilities)
        ),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_mean": float(np.mean(differences)),
        "ci_95_lower": float(np.quantile(differences, 0.025)),
        "ci_95_upper": float(np.quantile(differences, 0.975)),
    }


def _reference_probabilities(
    reference_result: Path,
    metadata: dict[str, object],
    snapshot_hashes: dict[str, str],
    evaluation: pd.DataFrame,
    contract: ModelContract,
) -> np.ndarray:
    """Load the matching logistic baseline for the approved holdout."""
    result = json.loads(reference_result.read_text())
    expected_labels = evaluation["label"].to_numpy(dtype=int).tolist()
    expected_article_ids = evaluation["article_id"].to_numpy(dtype=np.int64).tolist()
    reference_evaluation = result.get("evaluation")
    reference_model = result.get("model")
    reference_head = result.get("head")
    if (
        result.get("snapshot_id") != metadata["snapshot_id"]
        or result.get("snapshot_sha256") != snapshot_hashes
        or not isinstance(reference_model, dict)
        or reference_model.get("model_name") != contract.model_name
        or reference_model.get("contract_revision") != contract.revision
        or reference_model.get("prefix") != contract.prefix
        or not isinstance(reference_head, dict)
        or reference_head.get("name") != "logistic-regression"
        or not isinstance(reference_evaluation, dict)
        or reference_evaluation.get("labels") != expected_labels
        or reference_evaluation.get("article_ids") != expected_article_ids
    ):
        raise ValueError("Reference result is not the matching logistic evaluation")
    probabilities = np.asarray(reference_evaluation.get("probabilities"), dtype=float)
    if probabilities.shape != (len(evaluation),):
        raise ValueError("Reference result has invalid evaluation probabilities")
    _compute_metrics(evaluation["label"].to_numpy(dtype=int), probabilities)
    return probabilities


def run_benchmark(
    snapshot: Path,
    model_name: str,
    output: Path,
    batch_size: int,
    dtype_name: str,
    model_path: Path | None = None,
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    reference_result: Path | None = None,
    head_name: str = "logistic-regression",
) -> None:
    """Benchmark one allowed encoder and classifier head against the snapshot."""
    contract = MODEL_CONTRACTS[model_name]
    head = HEAD_CONTRACTS[head_name]
    metadata, train, evaluation, snapshot_hashes = _load_frozen_snapshot(snapshot)
    required_columns = {"article_id", "title", "content", "vote", "starred", "label"}
    if not required_columns.issubset(train) or not required_columns.issubset(
        evaluation
    ):
        raise ValueError("Snapshot is missing required columns")
    if set(train["article_id"]) & set(evaluation["article_id"]):
        raise ValueError("Snapshot train and evaluation article IDs overlap")
    _require_balanced_evaluation(evaluation)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(dtype_name, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    train_labels = train["label"].to_numpy(dtype=int)
    eval_labels = evaluation["label"].to_numpy(dtype=int)
    train_article_ids = train["article_id"].to_numpy(dtype=np.int64)
    eval_article_ids = evaluation["article_id"].to_numpy(dtype=np.int64)
    identity = _artifact_identity(
        snapshot_id=str(metadata["snapshot_id"]),
        snapshot_hashes=snapshot_hashes,
        contract=contract,
        model_path=model_path,
    )
    artifact_path = _embedding_artifact_path(
        embeddings_dir, str(metadata["snapshot_id"]), contract
    )
    artifact_reused = artifact_path.exists()
    article_token_budget: int
    if artifact_reused:
        try:
            (
                train_embeddings,
                eval_embeddings,
                artifact_sha256,
                article_token_budget,
            ) = _load_embedding_artifact(
                artifact_path,
                identity,
                train,
                evaluation,
                contract,
            )
            encode_seconds = 0.0
            logger.info(f"Reused embedding artifact {artifact_path}")
        except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
            invalid_path = _quarantine_embedding_artifact(artifact_path)
            artifact_reused = False
            logger.warning(
                f"Quarantined invalid embedding artifact {invalid_path}: {error}"
            )

    if not artifact_reused:
        model_source = str(model_path) if model_path else contract.model_name
        revision = None if model_path else contract.revision
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            revision=revision,
            trust_remote_code=True,
            local_files_only=model_path is not None,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_options = (
            {"attn_implementation": contract.attention} if contract.attention else {}
        )
        model = AutoModel.from_pretrained(
            model_source,
            revision=revision,
            trust_remote_code=True,
            local_files_only=model_path is not None,
            dtype=dtype,
            **model_options,
        ).to(device)
        model.config.use_cache = False
        train_texts = _prepare_texts(train, tokenizer, contract.prefix)
        eval_texts = _prepare_texts(evaluation, tokenizer, contract.prefix)
        article_token_budget = MAX_LENGTH - len(
            tokenizer.encode(contract.prefix, add_special_tokens=False)
        )

        encode_started = time.perf_counter()
        train_embeddings = _encode_texts(
            train_texts, tokenizer, model, contract, device, batch_size
        )
        eval_embeddings = _encode_texts(
            eval_texts, tokenizer, model, contract, device, batch_size
        )
        encode_seconds = time.perf_counter() - encode_started
        artifact_sha256 = _save_embedding_artifact(
            artifact_path,
            identity,
            train_embeddings,
            eval_embeddings,
            train_labels,
            eval_labels,
            train_article_ids,
            eval_article_ids,
            article_token_budget,
        )
        logger.info(f"Saved embedding artifact {artifact_path}")
    fit_started = time.perf_counter()
    selected_parameters, cross_validation = _select_head_parameters(
        head, train_embeddings, train_labels
    )
    classifier = _new_classifier(head, selected_parameters)
    classifier.fit(train_embeddings, train_labels)
    probabilities = classifier.predict_proba(eval_embeddings)[:, 1]
    fit_seconds = time.perf_counter() - fit_started
    reference_probabilities = (
        _reference_probabilities(
            reference_result, metadata, snapshot_hashes, evaluation, contract
        )
        if reference_result
        else None
    )

    model_details = asdict(contract)
    model_details["contract_revision"] = model_details.pop("revision")
    model_details.update(
        {
            "loaded_revision": None if model_path else contract.revision,
            "source": identity["source"],
            "weight_sha256": identity["weight_sha256"],
        }
    )
    result = {
        "snapshot_id": metadata["snapshot_id"],
        "snapshot_sha256": snapshot_hashes,
        "model": model_details,
        "settings": {
            "max_length": MAX_LENGTH,
            "article_token_budget": article_token_budget,
            "text_prep": "title_head",
            "batch_size": batch_size,
            "dtype": str(dtype).removeprefix("torch."),
            "head_name": head.name,
            "head_parameters": selected_parameters,
            "seed": SEED,
            "sample_weight": "none",
            "class_weight": "none",
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "embedding_artifact": {
            "path": str(artifact_path),
            "sha256": artifact_sha256,
            "reused": artifact_reused,
        },
        "counts": {
            "train": len(train),
            "eval": len(evaluation),
            "eval_positive": int(eval_labels.sum()),
            "eval_negative": int((eval_labels == 0).sum()),
        },
        "head": {
            "name": head.name,
            "calibration": head.calibration,
            "selected_parameters": selected_parameters,
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "seed": SEED,
            "candidates": cross_validation,
        },
        "metrics": _compute_metrics(eval_labels, probabilities),
        "evaluation": {
            "article_ids": eval_article_ids.tolist(),
            "labels": eval_labels.tolist(),
            "probabilities": probabilities.tolist(),
        },
        "comparison_to_logistic": (
            _paired_bootstrap_average_precision_difference(
                eval_labels, probabilities, reference_probabilities
            )
            if reference_probabilities is not None
            else None
        ),
        "timing_seconds": {
            "encoding": encode_seconds,
            "classifier": fit_seconds,
            "total": time.perf_counter() - started,
        },
        "peak_vram_gb": (
            float(torch.cuda.max_memory_allocated() / math.pow(1024, 3))
            if device.type == "cuda"
            else 0.0
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info(f"Wrote benchmark result to {output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument(
        "--eval-size-per-class",
        type=int,
        default=EVAL_SIZE_PER_CLASS,
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--snapshot", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--model", choices=MODEL_CONTRACTS, required=True)
    run_parser.add_argument(
        "--head", choices=HEAD_CONTRACTS, default="logistic-regression"
    )
    run_parser.add_argument("--model-path", type=Path)
    run_parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
    )
    run_parser.add_argument("--batch-size", type=int, default=4)
    run_parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    run_parser.add_argument(
        "--reference-result",
        type=Path,
        help="Matching logistic result JSON for paired average-precision bootstrap",
    )
    return parser


def main() -> None:
    init_logging(os.getenv("LOGGING_CONFIG", "logging.conf"))
    args = _build_parser().parse_args()
    if args.command == "export":
        asyncio.run(
            export_snapshot(
                args.output,
                eval_size_per_class=args.eval_size_per_class,
            )
        )
    else:
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        run_benchmark(
            snapshot=args.snapshot,
            model_name=args.model,
            output=args.output,
            batch_size=args.batch_size,
            dtype_name=args.dtype,
            model_path=args.model_path,
            embeddings_dir=args.embeddings_dir,
            reference_result=args.reference_result,
            head_name=args.head,
        )


if __name__ == "__main__":
    main()
