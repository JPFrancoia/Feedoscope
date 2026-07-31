from pathlib import Path
from typing import cast

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from feedoscope import semantic_freshness_embedding as freshness

TAG_HORIZONS = ("lt-24h", "1-3d", "4-7d", "8-30d", "1-6m", "evergreen")
SQL_DIR = Path(__file__).parents[1] / "feedoscope" / "data_registry" / "sql"
MIGRATIONS_DIR = Path(__file__).parents[1] / "db" / "migrations"


def test_semantic_freshness_tag_queries_use_short_prefixes() -> None:
    """Keep reviewed and automatic freshness tag names aligned."""
    reviewed_names = {f"fresh-{horizon}" for horizon in TAG_HORIZONS}
    automatic_names = {f"fresh-auto-{horizon}" for horizon in TAG_HORIZONS}
    expected_names = {
        "upsert_semantic_freshness_user_tags.sql": reviewed_names | automatic_names,
        "get_semantic_freshness_user_tags.sql": reviewed_names | automatic_names,
        "set_semantic_freshness_auto_tag_for_entry.sql": automatic_names,
        "promote_read_auto_freshness_tags.sql": reviewed_names | automatic_names,
        "get_semantic_freshness_training.sql": reviewed_names,
        "get_conflicting_semantic_freshness_labels.sql": reviewed_names,
    }
    for filename, names in expected_names.items():
        text = (SQL_DIR / filename).read_text()
        assert all(name in text for name in names)
        assert "-freshness" not in text

    promotion_query = (SQL_DIR / "promote_read_auto_freshness_tags.sql").read_text()
    assert "replace(min(ut.title), 'fresh-auto-', 'fresh-')" in promotion_query
    assert "replace(c.reviewed_title, 'fresh-', 'fresh-auto-')" in promotion_query


def test_semantic_freshness_tag_migration_maps_every_tag() -> None:
    """Keep the reversible in-place tag rename complete and collision-safe."""
    up_query = (
        MIGRATIONS_DIR / "000009_rename_semantic_freshness_tags.up.sql"
    ).read_text()
    down_query = (
        MIGRATIONS_DIR / "000009_rename_semantic_freshness_tags.down.sql"
    ).read_text()
    for horizon in TAG_HORIZONS:
        assert f"when '{horizon}-freshness' then 'fresh-{horizon}'" in up_query
        assert (
            f"when '{horizon}-auto-freshness' then 'fresh-auto-{horizon}'" in up_query
        )
        assert f"when 'fresh-{horizon}' then '{horizon}-freshness'" in down_query
        assert (
            f"when 'fresh-auto-{horizon}' then '{horizon}-auto-freshness'" in down_query
        )
    assert "freshness tag rename blocked" in up_query
    assert "freshness tag rollback blocked" in down_query


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
        [(3, 1, "reviewed", "high"), (1, 0, "teacher", "high")]
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
        [(1, 0, "teacher", "high"), (3, 1, "reviewed", "high")]
    )


def test_fit_requires_both_classes_at_every_boundary() -> None:
    with pytest.raises(RuntimeError, match="24h"):
        freshness.fit_classifiers(np.ones((3, 2)), np.array([5, 5, 5]))
