"""Autoresearch candidate: ordinal classification over article embeddings."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

HORIZON_COUNT = 6


def fit_predict(train: dict[str, object], test: dict[str, object]) -> np.ndarray:
    """Fit one cumulative threshold classifier per ordered horizon boundary."""
    x_train = normalize(np.asarray(train["embeddings"], dtype=float))
    x_test = normalize(np.asarray(test["embeddings"], dtype=float))
    labels = np.asarray(train["labels"], dtype=int)
    tails = np.column_stack(
        [
            LogisticRegression(C=20.0, max_iter=2000, random_state=0)
            .fit(x_train, labels > boundary)
            .predict_proba(x_test)[:, 1]
            for boundary in range(HORIZON_COUNT - 1)
        ]
    )
    tails = np.minimum.accumulate(tails, axis=1)
    return np.column_stack(
        [1.0 - tails[:, 0], tails[:, :-1] - tails[:, 1:], tails[:, -1]]
    )
