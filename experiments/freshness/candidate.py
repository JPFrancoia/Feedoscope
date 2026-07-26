"""Autoresearch candidate: ordinal classification over article embeddings."""

import numpy as np
from sklearn.linear_model import LogisticRegression

HORIZON_COUNT = 6


def fit_predict(train: dict[str, object], test: dict[str, object]) -> np.ndarray:
    """Fit one cumulative threshold classifier per ordered horizon boundary."""
    x_train = np.asarray(train["embeddings"], dtype=float)
    x_test = np.asarray(test["embeddings"], dtype=float)
    labels = np.asarray(train["labels"], dtype=int)
    tails = []
    for boundary in range(HORIZON_COUNT - 1):
        target = labels > boundary
        counts = np.bincount(target, minlength=2)
        class_weight = {
            value: (len(labels) / (2 * count)) ** 0.375
            for value, count in enumerate(counts)
        }
        model = LogisticRegression(
            C=20.0, class_weight=class_weight, fit_intercept=False
        )
        tails.append(model.fit(x_train, target).predict_proba(x_test)[:, 1])
    tails = np.minimum.accumulate(np.column_stack(tails), axis=1)
    return np.column_stack(
        [1.0 - tails[:, 0], tails[:, :-1] - tails[:, 1:], tails[:, -1]]
    )
