import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch

from experiments import freshness_encoder_models as experiment


def test_split_chronological_holds_out_newest_rows() -> None:
    rows = pd.DataFrame(
        {
            "article_id": [3, 1, 4, 2],
            "published_at": [
                "2026-01-03T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-04T00:00:00Z",
                "2026-01-02T00:00:00Z",
            ],
        }
    )

    train, evaluation = experiment.split_chronological(rows, validation_size=2)

    assert train["article_id"].tolist() == [1, 2]
    assert evaluation["article_id"].tolist() == [3, 4]


def test_split_chronological_rejects_duplicate_articles() -> None:
    rows = pd.DataFrame(
        {
            "article_id": [1, 1, 2],
            "published_at": pd.date_range("2026-01-01", periods=3, tz="UTC"),
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        experiment.split_chronological(rows, validation_size=1)


def test_last_token_pool_handles_right_padding() -> None:
    hidden = torch.tensor(
        [
            [[1.0], [2.0], [99.0]],
            [[3.0], [4.0], [5.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    pooled = experiment.last_token_pool(hidden, mask)

    assert pooled.tolist() == [[2.0], [5.0]]


def test_active_classifier_contract_returns_ordered_probabilities() -> None:
    embeddings = np.asarray(
        [
            [-2.0, 0.0],
            [-1.5, 0.2],
            [-1.0, -0.1],
            [0.0, 1.0],
            [0.1, 1.5],
            [-0.1, 2.0],
            [1.0, 0.0],
            [1.5, 0.2],
            [2.0, -0.1],
        ]
    )
    labels = np.repeat(np.arange(3), 3)

    classifiers = experiment.fit_classifiers(embeddings, labels)
    probabilities = experiment.bucket_probabilities(embeddings, classifiers)

    assert len(classifiers) == 2
    assert all(model.get_params()["C"] == 20.0 for model in classifiers)
    assert all(model.get_params()["fit_intercept"] is False for model in classifiers)
    assert probabilities.shape == (9, 3)
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_compute_metrics_rejects_invalid_labels() -> None:
    with pytest.raises(ValueError, match="Freshness labels"):
        experiment.compute_metrics(np.asarray([-1]), np.asarray([[0.0, 0.0, 1.0]]))


def test_compute_metrics_matches_perfect_three_label_predictions() -> None:
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[labels]

    metrics = experiment.compute_metrics(labels, probabilities)

    assert metrics == {
        "rps": 0.0,
        "macro_f1": 1.0,
        "weighted_kappa": 1.0,
        "log_duration_mae": 0.0,
        "long_lived_auc": 1.0,
    }


def test_instruction_contracts_name_all_current_boundaries() -> None:
    for model_key in ("harrier", "qwen3-0.6b"):
        prefix = experiment.MODEL_CONTRACTS[model_key].prefix
        assert "0-29 days" in prefix
        assert "30 days through 6 months" in prefix
        assert "more than 6 months" in prefix
