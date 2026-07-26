"""Autoresearch candidate: multinomial classification over article embeddings."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

HORIZON_COUNT = 6


def fit_predict(train: dict[str, object], test: dict[str, object]) -> np.ndarray:
    """Fit a regularized linear classifier over normalized embeddings."""
    x_train = normalize(np.asarray(train["embeddings"], dtype=float))
    x_test = normalize(np.asarray(test["embeddings"], dtype=float))
    labels = np.asarray(train["labels"], dtype=int)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
    model.fit(x_train, labels)
    probabilities = np.zeros((len(x_test), HORIZON_COUNT))
    probabilities[:, model.classes_] = model.predict_proba(x_test)
    return probabilities
