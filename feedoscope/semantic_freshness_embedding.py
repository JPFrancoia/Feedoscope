import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
from sklearn.linear_model import LogisticRegression

from feedoscope import config, relevance_embedding

logger = logging.getLogger(__name__)

HORIZONS = ("fresh_d", "fresh_m", "fresh_y")
THRESHOLDS = ("30d", "6m")
REPRESENTATIVE_DAYS = np.asarray((7.0, 90.0, 365.0))
ARTIFACT_FILENAME = "semantic_freshness.joblib"


def get_model_family_prefix() -> str:
    """Return the artifact family prefix for three-label freshness models."""
    return (
        f"freshness_3label_{config.RELEVANCE_EMBEDDING_KEY.replace('/', '-')}_"
        f"{config.RELEVANCE_MAX_LENGTH}_{config.RELEVANCE_TEXT_PREP_MODE}_"
        f"p{config.RELEVANCE_PREP_VERSION}_"
        f"embedding_linear_c{config.SEMANTIC_FRESHNESS_LINEAR_C}_"
        f"w{config.SEMANTIC_FRESHNESS_WEIGHT_EXPONENT}"
    )


def get_model_key(dataset_fingerprint: str | None = None) -> str:
    """Return a configuration key, optionally pinned to one label fingerprint."""
    key = (
        "freshness-3label-embedding_linear::"
        f"{config.RELEVANCE_EMBEDDING_KEY}::{config.RELEVANCE_MAX_LENGTH}::"
        f"{config.RELEVANCE_TEXT_PREP_MODE}::{config.RELEVANCE_PREP_VERSION}::"
        f"prompt={config.RELEVANCE_EMBEDDING_PROMPT}::"
        f"c={config.SEMANTIC_FRESHNESS_LINEAR_C}::"
        f"weight_exponent={config.SEMANTIC_FRESHNESS_WEIGHT_EXPONENT}"
    )
    if dataset_fingerprint is not None:
        return f"{key}::labels={dataset_fingerprint}"
    return key


def build_model_path(dataset_fingerprint: str) -> str:
    """Build a dated artifact path that identifies its effective label set."""
    return (
        f"models/{get_model_family_prefix()}_"
        f"{datetime.date.today():%Y_%m_%d}_{dataset_fingerprint[:12]}"
    )


def build_targets(labels: np.ndarray) -> np.ndarray:
    """Convert three ordered freshness labels into two cumulative targets."""
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or np.any((labels < 0) | (labels >= len(HORIZONS))):
        raise ValueError(
            f"Labels must be a one-dimensional array in [0, {len(HORIZONS) - 1}]."
        )
    return labels[:, None] > np.arange(len(THRESHOLDS))


def fingerprint_labels(rows: list[tuple[int, int, str]]) -> str:
    """Hash the effective labels and encoder configuration deterministically."""
    payload = {
        "rows": sorted(rows),
        "encoder": relevance_embedding.get_cache_config(),
        "linear_c": config.SEMANTIC_FRESHNESS_LINEAR_C,
        "weight_exponent": config.SEMANTIC_FRESHNESS_WEIGHT_EXPONENT,
        "thresholds": THRESHOLDS,
        "representative_days": REPRESENTATIVE_DAYS.tolist(),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def fit_classifiers(
    embeddings: np.ndarray, labels: np.ndarray
) -> list[LogisticRegression]:
    """Fit one independently weighted cumulative classifier per boundary."""
    targets = build_targets(labels)
    classifiers: list[LogisticRegression] = []
    for boundary, target in enumerate(targets.T):
        counts = np.bincount(target.astype(int), minlength=2)
        if int(counts.min()) == 0:
            raise RuntimeError(
                f"Freshness boundary {THRESHOLDS[boundary]} needs both target classes."
            )
        class_weight = {
            value: (len(target) / (2 * count))
            ** config.SEMANTIC_FRESHNESS_WEIGHT_EXPONENT
            for value, count in enumerate(counts)
        }
        classifier = LogisticRegression(
            max_iter=4000,
            C=config.SEMANTIC_FRESHNESS_LINEAR_C,
            class_weight=class_weight,
            fit_intercept=False,
            random_state=42,
        )
        classifier.fit(embeddings, target)
        classifiers.append(classifier)
    return classifiers


def bucket_probabilities(
    embeddings: np.ndarray,
    classifiers: list[LogisticRegression],
) -> np.ndarray:
    """Return ordered three-label probabilities from two cumulative heads."""
    if len(classifiers) != len(THRESHOLDS):
        raise ValueError(
            f"Expected {len(THRESHOLDS)} classifiers, got {len(classifiers)}."
        )
    tails = np.column_stack(
        [classifier.predict_proba(embeddings)[:, 1] for classifier in classifiers]
    )
    tails = -np.sort(-tails, axis=1)
    probabilities = np.column_stack(
        (1.0 - tails[:, 0], tails[:, :-1] - tails[:, 1:], tails[:, -1])
    )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-10):
        raise RuntimeError("Freshness classifiers produced invalid probabilities.")
    return np.clip(probabilities, 0.0, 1.0)


def expected_lifetime_days(probabilities: np.ndarray) -> np.ndarray:
    """Calculate expected useful lifetime from three-label probabilities."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(HORIZONS):
        raise ValueError(f"Expected probabilities with shape (n, {len(HORIZONS)}).")
    return probabilities @ REPRESENTATIVE_DAYS


def artifact_metadata(
    dataset_fingerprint: str,
    train_counts: dict[str, int],
    label_source_counts: dict[str, int],
) -> dict[str, Any]:
    """Build metadata defining artifact compatibility and provenance."""
    return {
        "backend": "embedding_ordinal_linear",
        "model_key": get_model_key(dataset_fingerprint),
        "configuration_key": get_model_key(),
        "thresholds": THRESHOLDS,
        "horizons": HORIZONS,
        "representative_days": REPRESENTATIVE_DAYS.tolist(),
        "encoder": relevance_embedding.get_cache_config(),
        "linear_c": config.SEMANTIC_FRESHNESS_LINEAR_C,
        "weight_exponent": config.SEMANTIC_FRESHNESS_WEIGHT_EXPONENT,
        "dataset_fingerprint": dataset_fingerprint,
        "train_counts": train_counts,
        "label_source_counts": label_source_counts,
    }


def save_artifact(
    model_path: str,
    classifiers: list[LogisticRegression],
    metadata: dict[str, Any],
) -> None:
    """Atomically save both classifiers and their metadata in one file."""
    path = Path(model_path)
    path.mkdir(parents=True, exist_ok=True)
    destination = path / ARTIFACT_FILENAME
    with tempfile.NamedTemporaryFile(dir=path, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump({"classifiers": classifiers, "metadata": metadata}, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    logger.info(f"Saved freshness artifact to {destination}")


def load_artifact(model_path: str) -> tuple[list[LogisticRegression], dict[str, Any]]:
    """Load a compatible three-label freshness artifact."""
    artifact = joblib.load(Path(model_path) / ARTIFACT_FILENAME)
    classifiers = artifact.get("classifiers")
    metadata = artifact.get("metadata")
    if not isinstance(classifiers, list) or len(classifiers) != len(THRESHOLDS):
        raise RuntimeError("Freshness artifact has an invalid classifier set.")
    if (
        not isinstance(metadata, dict)
        or metadata.get("configuration_key") != get_model_key()
    ):
        raise RuntimeError(
            "Freshness artifact is incompatible with the active configuration."
        )
    if tuple(metadata.get("thresholds", ())) != THRESHOLDS:
        raise RuntimeError("Freshness artifact has incompatible thresholds.")
    return classifiers, metadata
