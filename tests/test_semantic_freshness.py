import asyncio
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from sklearn.linear_model import LogisticRegression

from experiments.freshness import build_labels

from feedoscope import import_semantic_freshness_bootstrap_labels as bootstrap_import
from feedoscope import semantic_freshness_embedding as freshness
from feedoscope.data_registry import data_registry as dr

SQL_DIR = Path(__file__).parents[1] / "feedoscope" / "data_registry" / "sql"
MIGRATIONS_DIR = Path(__file__).parents[1] / "db" / "migrations"


def test_only_three_manual_freshness_tags_remain() -> None:
    migration = (MIGRATIONS_DIR / "000006_three_label_freshness.up.sql").read_text()
    training_query = (SQL_DIR / "get_semantic_freshness_training.sql").read_text()
    freshness_sql = "\n".join(
        path.read_text() for path in SQL_DIR.glob("*semantic_freshness*.sql")
    )

    for label in freshness.HORIZONS:
        assert label in migration
        assert label in training_query
    assert "auto" not in freshness_sql
    assert "freshness_inference" not in freshness_sql
    assert "e.status <> 'read'" in training_query
    assert "freshness migration blocked" in migration
    assert (
        "delete from freshness_bootstrap_labels"
        in (SQL_DIR / "delete_semantic_freshness_bootstrap_labels.sql").read_text()
    )
    assert (
        "on conflict"
        not in (SQL_DIR / "insert_semantic_freshness_bootstrap_labels.sql").read_text()
    )
    assert "freshness_inference" not in "\n".join(
        path.read_text() for path in MIGRATIONS_DIR.glob("*.sql")
    )


def test_bootstrap_validation_rejects_unquoted_evidence() -> None:
    batch = pd.DataFrame(
        [{"article_id": 1, "title": "A lasting reference", "content": "Useful text"}]
    )

    with pytest.raises(ValueError, match="evidence"):
        build_labels._validate_labels(
            [{"article_id": 1, "label": "fresh_y", "evidence": "invented quote"}],
            batch,
        )


@pytest.mark.parametrize("article_id", (True, 1.0, "1"))
def test_bootstrap_validation_requires_integer_ids(article_id: object) -> None:
    batch = pd.DataFrame(
        [{"article_id": 1, "title": "A lasting reference", "content": "Useful text"}]
    )

    with pytest.raises(ValueError, match="article ID"):
        build_labels._validate_labels(
            [{"article_id": article_id, "label": "fresh_y", "evidence": "Useful"}],
            batch,
        )


def test_bootstrap_sample_requires_unique_ids() -> None:
    sample = pd.DataFrame([{"article_id": 1}, {"article_id": 1}])

    with pytest.raises(RuntimeError, match="unique"):
        build_labels._validate_sample(sample, 2)


@pytest.mark.parametrize(
    "rows",
    (
        [{"article_id": 1, "label": "fresh_d", "evidence": "one"}],
        [
            {"article_id": 1, "label": "fresh_d", "evidence": "one"},
            {"article_id": 2, "label": "fresh_m", "evidence": "two"},
            {"article_id": 3, "label": "fresh_y", "evidence": "three"},
        ],
        [
            {"article_id": 1, "label": "fresh_d", "evidence": "one"},
            {"article_id": 1, "label": "fresh_m", "evidence": "one"},
        ],
    ),
)
def test_complete_bootstrap_must_exactly_match_sample(
    rows: list[dict[str, object]],
) -> None:
    sample = pd.DataFrame([{"article_id": 1}, {"article_id": 2}])

    with pytest.raises(RuntimeError, match="does not match"):
        build_labels._validate_complete_labels(pd.DataFrame(rows), sample, 2)


def test_generated_batch_rejects_missing_extra_and_duplicate_ids() -> None:
    batch = pd.DataFrame(
        [
            {"article_id": 1, "title": "one", "content": "first"},
            {"article_id": 2, "title": "two", "content": "second"},
        ]
    )
    invalid_batches = (
        [{"article_id": 1, "label": "fresh_d", "evidence": "one"}],
        [
            {"article_id": 1, "label": "fresh_d", "evidence": "one"},
            {"article_id": 2, "label": "fresh_m", "evidence": "two"},
            {"article_id": 3, "label": "fresh_y", "evidence": "three"},
        ],
        [
            {"article_id": 1, "label": "fresh_d", "evidence": "one"},
            {"article_id": 1, "label": "fresh_m", "evidence": "one"},
        ],
    )

    for rows in invalid_batches:
        with pytest.raises(ValueError):
            build_labels._validate_labels(rows, batch)


def test_bootstrap_import_rejects_duplicate_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "article_id,label,evidence\n1,fresh_d,one\n1,fresh_m,two\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bootstrap_import, "EXPECTED_LABELS", 2)

    with pytest.raises(ValueError, match="Duplicate"):
        bootstrap_import.load_labels(path, "gpt-5.6-luna")


def test_bootstrap_replacement_rolls_back_failed_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Cursor:
        async def __aenter__(self) -> "Cursor":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def execute(self, query: str) -> None:
            assert query == "delete_semantic_freshness_bootstrap_labels.sql"

        async def executemany(
            self, query: str, parameters: list[dict[str, object]]
        ) -> None:
            assert query == "insert_semantic_freshness_bootstrap_labels.sql"
            assert parameters
            raise RuntimeError("insert failed")

    class Connection:
        def __init__(self) -> None:
            self.rolled_back = False

        async def __aenter__(self) -> "Connection":
            return self

        async def __aexit__(self, exc_type: Any, *_: Any) -> None:
            self.rolled_back = exc_type is not None

        def cursor(self) -> Cursor:
            return Cursor()

    class Pool:
        def __init__(self, connection: Connection) -> None:
            self._connection = connection

        def connection(self) -> Connection:
            return self._connection

    connection = Connection()
    monkeypatch.setattr(dr, "global_pool", Pool(connection))
    monkeypatch.setattr(dr, "_get_query_from_file", lambda filename: filename)

    with pytest.raises(RuntimeError, match="insert failed"):
        asyncio.run(
            dr.replace_semantic_freshness_bootstrap_labels(
                [(1, "fresh_d", "gpt-5.6-luna")]
            )
        )

    assert connection.rolled_back


def test_build_targets_covers_all_labels() -> None:
    np.testing.assert_array_equal(
        freshness.build_targets(np.arange(3)),
        np.array(
            [
                [False, False],
                [True, False],
                [True, True],
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
            [Classifier(probability) for probability in (0.4, 0.9)],
        ),
    )

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities[0], [0.1, 0.5, 0.4])
    assert np.all(probabilities >= 0)


def test_expected_lifetime_uses_three_representative_values() -> None:
    np.testing.assert_allclose(
        freshness.expected_lifetime_days(np.eye(3)),
        [7.0, 90.0, 365.0],
    )


def test_artifact_round_trip_and_fingerprint_stability(tmp_path: Path) -> None:
    embeddings = np.arange(54, dtype=float).reshape(9, 6)
    labels = np.tile(np.arange(3), 3)
    classifiers = freshness.fit_classifiers(embeddings, labels)
    fingerprint = freshness.fingerprint_labels(
        [(3, 1, "manual"), (1, 0, "bootstrap:gpt-5.6-luna")]
    )
    metadata = freshness.artifact_metadata(
        fingerprint,
        train_counts={horizon: 3 for horizon in freshness.HORIZONS},
        label_source_counts={"manual": 1, "bootstrap:gpt-5.6-luna": 1},
    )

    freshness.save_artifact(str(tmp_path), classifiers, metadata)
    loaded_classifiers, loaded_metadata = freshness.load_artifact(str(tmp_path))

    np.testing.assert_allclose(
        freshness.bucket_probabilities(embeddings, classifiers),
        freshness.bucket_probabilities(embeddings, loaded_classifiers),
    )
    assert loaded_metadata == metadata
    assert fingerprint == freshness.fingerprint_labels(
        [(1, 0, "bootstrap:gpt-5.6-luna"), (3, 1, "manual")]
    )


def test_fit_requires_both_classes_at_every_boundary() -> None:
    with pytest.raises(RuntimeError, match="30d"):
        freshness.fit_classifiers(np.ones((3, 2)), np.array([2, 2, 2]))
