"""Fixed evaluation harness for intrinsic semantic-horizon experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from experiments.freshness.candidate import fit_predict

DATA_DIR = Path(".auto/data")
HORIZONS = ["lt_24h", "1_3d", "4_7d", "8_30d", "1_6m", "evergreen"]
HORIZON_TO_INDEX = {name: index for index, name in enumerate(HORIZONS)}
REPRESENTATIVE_DAYS = np.asarray([0.5, 2.0, 5.5, 19.0, 90.0, 365.0])


def _features(
    articles: pd.DataFrame, embeddings: np.ndarray, row_indexes: list[int]
) -> dict[str, object]:
    rows = articles.iloc[row_indexes]
    urgency = pd.to_numeric(rows["current_urgency_score"], errors="coerce").to_numpy()
    return {
        "article_ids": rows["article_id"].astype(int).to_numpy(),
        "embeddings": embeddings[row_indexes],
        "titles": rows["title"].astype(str).to_numpy(),
        "contents": rows["content"].astype(str).to_numpy(),
        "feed_names": rows["feed_name"].astype(str).to_numpy(),
        "published_at": rows["published_at"].astype(str).to_numpy(),
        "current_urgency_score": urgency,
    }


def _ranked_probability_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    observed = np.eye(len(HORIZONS), dtype=float)[labels]
    predicted_cdf = probabilities.cumsum(axis=1)[:, :-1]
    observed_cdf = observed.cumsum(axis=1)[:, :-1]
    return float(np.mean((predicted_cdf - observed_cdf) ** 2))


def main() -> None:
    articles = pd.read_csv(DATA_DIR / "articles.csv", keep_default_na=False)
    embeddings = np.load(DATA_DIR / "embeddings.npy", allow_pickle=False)
    labels = pd.read_csv(DATA_DIR / "teacher_labels.csv", keep_default_na=False)
    split = json.loads((DATA_DIR / "split.json").read_text(encoding="utf-8"))

    if len(articles) != len(embeddings):
        raise RuntimeError("Article and embedding row counts differ")
    id_to_row = {
        int(article_id): index
        for index, article_id in enumerate(articles["article_id"])
    }
    label_by_id = {
        int(row.article_id): HORIZON_TO_INDEX[str(row.horizon)]
        for row in labels.itertuples(index=False)
        if str(row.horizon) in HORIZON_TO_INDEX
        and str(row.confidence) in {"medium", "high"}
    }
    train_ids = [int(value) for value in split["train_ids"]]
    test_ids = [int(value) for value in split["test_ids"]]
    train_rows = [id_to_row[value] for value in train_ids]
    test_rows = [id_to_row[value] for value in test_ids]
    y_train = np.asarray([label_by_id[value] for value in train_ids], dtype=int)
    y_test = np.asarray([label_by_id[value] for value in test_ids], dtype=int)

    train = _features(articles, embeddings, train_rows)
    train["labels"] = y_train
    test = _features(articles, embeddings, test_rows)
    probabilities = np.asarray(fit_predict(train, test), dtype=float)

    expected_shape = (len(test_ids), len(HORIZONS))
    if probabilities.shape != expected_shape:
        raise RuntimeError(
            f"Candidate returned {probabilities.shape}; expected {expected_shape}"
        )
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise RuntimeError("Candidate probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise RuntimeError("Candidate probability rows must have positive mass")
    probabilities /= row_sums

    predictions = probabilities.argmax(axis=1)
    rps = _ranked_probability_score(probabilities, y_test)
    macro_f1 = float(
        f1_score(y_test, predictions, labels=range(len(HORIZONS)), average="macro")
    )
    kappa = float(cohen_kappa_score(y_test, predictions, weights="quadratic"))
    predicted_log_days = probabilities @ np.log1p(REPRESENTATIVE_DAYS)
    true_log_days = np.log1p(REPRESENTATIVE_DAYS[y_test])
    log_mae = float(np.mean(np.abs(predicted_log_days - true_log_days)))
    evergreen = (y_test == HORIZON_TO_INDEX["evergreen"]).astype(int)
    evergreen_auc = (
        float(roc_auc_score(evergreen, probabilities[:, -1]))
        if len(np.unique(evergreen)) == 2
        else 0.5
    )
    confidence = float(np.mean(probabilities.max(axis=1)))

    print(f"Evaluated {len(train_ids)} train and {len(test_ids)} temporal test rows")
    print(f"METRIC rps={rps:.8f}")
    print(f"METRIC macro_f1={macro_f1:.8f}")
    print(f"METRIC weighted_kappa={kappa:.8f}")
    print(f"METRIC log_duration_mae={log_mae:.8f}")
    print(f"METRIC evergreen_auc={evergreen_auc:.8f}")
    print(f"METRIC mean_confidence={confidence:.8f}")


if __name__ == "__main__":
    main()
