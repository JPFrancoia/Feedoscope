from pathlib import Path
from typing import cast

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from feedoscope import semantic_freshness_embedding as freshness


def test_build_targets_covers_all_horizons() -> None:
    labels = np.arange(6)

    np.testing.assert_array_equal(
        freshness.build_targets(labels),
        np.array(
            [
                [False, False, False, False, False],
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, True, True, True],
            ]
        ),
    )


def test_bucket_probabilities_are_monotone_and_sum_to_one() -> None:
    class Classifier:
        def __init__(self, probability: float) -> None:
            self.probability = probability

        def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
            return np.tile(
                [1 - self.probability, self.probability], (len(embeddings), 1)
            )

    probabilities = freshness.bucket_probabilities(
        np.ones((2, 3)),
        cast(
            list[LogisticRegression],
            [Classifier(probability) for probability in (0.4, 0.9, 0.3, 0.8, 0.2)],
        ),
    )

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities[0], [0.1, 0.1, 0.4, 0.1, 0.1, 0.2])
    assert np.all(probabilities >= 0)


def test_artifact_round_trip_and_fingerprint_stability(tmp_path: Path) -> None:
    embeddings = np.arange(72, dtype=float).reshape(12, 6)
    labels = np.tile(np.arange(6), 2)
    classifiers = freshness.fit_classifiers(embeddings, labels)
    fingerprint = freshness.fingerprint_labels(
        [(3, 1, "reviewed", "high"), (1, 0, "teacher", "medium")]
    )
    metadata = freshness.artifact_metadata(
        fingerprint,
        train_counts={horizon: 2 for horizon in freshness.HORIZONS},
        validation_metrics={"rps": 0.1},
        label_source_counts={"reviewed": 1, "teacher": 1},
    )

    freshness.save_artifact(str(tmp_path), classifiers, metadata)
    loaded_classifiers, loaded_metadata = freshness.load_artifact(str(tmp_path))

    np.testing.assert_allclose(
        freshness.bucket_probabilities(embeddings, classifiers),
        freshness.bucket_probabilities(embeddings, loaded_classifiers),
    )
    assert loaded_metadata == metadata
    assert fingerprint == freshness.fingerprint_labels(
        [(1, 0, "teacher", "medium"), (3, 1, "reviewed", "high")]
    )


def test_fit_requires_both_classes_at_every_boundary() -> None:
    with pytest.raises(RuntimeError, match="24h"):
        freshness.fit_classifiers(np.ones((3, 2)), np.array([5, 5, 5]))
