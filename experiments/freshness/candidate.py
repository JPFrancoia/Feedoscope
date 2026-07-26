"""Autoresearch candidate: map the existing urgency probability to horizons."""

import numpy as np

HORIZON_COUNT = 6


def fit_predict(train: dict[str, object], test: dict[str, object]) -> np.ndarray:
    """Return one probability distribution per test article.

    This baseline preserves the current production assumption: urgency 1.0 maps
    to the shortest horizon and urgency 0.0 maps to evergreen.
    """
    del train
    urgency = np.asarray(test["current_urgency_score"], dtype=float)
    urgency = np.nan_to_num(urgency, nan=0.5).clip(0.0, 1.0)
    centers = (1.0 - urgency) * (HORIZON_COUNT - 1)
    classes = np.arange(HORIZON_COUNT, dtype=float)
    logits = -0.5 * ((classes[None, :] - centers[:, None]) / 0.9) ** 2
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)
