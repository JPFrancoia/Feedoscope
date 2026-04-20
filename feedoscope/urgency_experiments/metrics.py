import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clamp probabilities away from 0 and 1 for stable metrics."""
    return np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)


def compute_binary_classification_metrics(
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
) -> dict[str, float]:
    """Compute probability-centric metrics for urgency model comparison."""
    probs = clip_probabilities(predicted_probs)
    labels = np.asarray(true_labels, dtype=int)
    pred_labels = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(labels, pred_labels)),
        "precision": float(precision_score(labels, pred_labels, zero_division=0)),
        "recall": float(recall_score(labels, pred_labels, zero_division=0)),
        "f1": float(f1_score(labels, pred_labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "log_loss": float(log_loss(labels, probs)),
        "brier_score": float(brier_score_loss(labels, probs)),
    }
