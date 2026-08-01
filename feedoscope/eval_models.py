"""Weekly evaluation of relevance, urgency, and freshness models.

For each model, trains on a subset of the data with a holdout set,
runs inference on the holdout, logs classification metrics, then discards
the eval model. Production models (trained on 100% of data) are NOT affected.

The holdout size is controlled by the VALIDATION_SIZE env var (via config.py).
If VALIDATION_SIZE is 0, the eval is skipped entirely.
"""

import asyncio
from collections.abc import Mapping
import datetime
import json
import logging
import math
import os
import random
import shutil
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import torch
from transformers import PreTrainedTokenizerBase

from custom_logging import init_logging
from feedoscope import (
    config,
    llm_learn,
    llm_learn_urgency,
    relevance_embedding,
    semantic_freshness_embedding,
    urgency_embedding,
)
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

# Eval model paths use these prefixes. They do NOT match any production model
# prefix (google-embeddinggemma-300m_* or urgency_google-embeddinggemma-300m_*),
# so find_latest_model() will never find or delete them.
EVAL_RELEVANCE_PREFIX = "eval_relevance"
EVAL_URGENCY_PREFIX = "eval_urgency"

MAX_LENGTH = 512
INFERENCE_BATCH_SIZE = 128
EVAL_HISTORY_PATH = "models/eval_history.json"
SUPER_IMPORTANT_SETTLING_DAYS = 40
SUPER_IMPORTANT_TRAIN_FRACTION = 0.6
SUPER_IMPORTANT_VALIDATION_FRACTION = 0.2
MIN_SUPER_IMPORTANT_EXAMPLES = 10
SUPER_IMPORTANT_RANKING_BUDGETS = (10, 25, 50)
SUPER_IMPORTANT_BONUS_GRID = tuple(index / 4 for index in range(13))
MAX_RELEVANCE_AP_DROP = 0.01
WEIGHTED_RELEVANCE_BASELINE_WEIGHT = 20.0


def _clean_stale_eval_dirs() -> None:
    """Remove leftover eval model directories from a previous interrupted run.

    If the eval script was killed (SIGKILL, OOM, etc.) before the ``finally``
    block could run, the temporary model directories may still exist on disk.
    Calling this at the start of each run ensures a clean slate.
    """
    for prefix in (EVAL_RELEVANCE_PREFIX, EVAL_URGENCY_PREFIX):
        path = f"models/{prefix}"
        if os.path.exists(path):
            logger.warning(
                f"Found stale eval directory {path} from a previous run. Removing."
            )
            shutil.rmtree(path, ignore_errors=True)


def compute_freshness_metrics(
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
) -> dict[str, float | None]:
    """Compute metrics for ordered three-label freshness predictions."""
    true_labels = np.asarray(true_labels, dtype=int)
    predicted_probs = np.asarray(predicted_probs, dtype=float)
    predicted_labels = predicted_probs.argmax(axis=1)
    true_probabilities = np.eye(len(semantic_freshness_embedding.HORIZONS))[true_labels]
    rps = np.mean(
        np.square(
            np.cumsum(predicted_probs, axis=1)[:, :-1]
            - np.cumsum(true_probabilities, axis=1)[:, :-1]
        )
    )
    predicted_lifetimes = semantic_freshness_embedding.expected_lifetime_days(
        predicted_probs
    )
    true_lifetimes = semantic_freshness_embedding.REPRESENTATIVE_DAYS[true_labels]
    long_lived = (true_labels == len(semantic_freshness_embedding.HORIZONS) - 1).astype(
        int
    )
    long_lived_auc = (
        float(roc_auc_score(long_lived, predicted_probs[:, -1]))
        if np.unique(long_lived).size == 2
        else None
    )
    weighted_kappa = (
        cohen_kappa_score(true_labels, predicted_labels, weights="quadratic")
        if np.unique(np.concatenate((true_labels, predicted_labels))).size > 1
        else math.nan
    )

    return {
        "rps": float(rps),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=np.arange(len(semantic_freshness_embedding.HORIZONS)),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_kappa": (
            float(weighted_kappa) if math.isfinite(weighted_kappa) else None
        ),
        "log_duration_mae": float(
            np.mean(np.abs(np.log(predicted_lifetimes) - np.log(true_lifetimes)))
        ),
        "long_lived_auc": long_lived_auc,
    }


def split_super_important_eval_articles(
    articles: list[Article],
    now: datetime.datetime | None = None,
) -> tuple[list[Article], list[Article], list[Article]]:
    """Split mature labels into chronological train, validation, and test sets."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=SUPER_IMPORTANT_SETTLING_DAYS)
    mature_articles = sorted(
        (
            article
            for article in articles
            if article.last_read is not None and article.last_read <= cutoff
        ),
        key=lambda article: (article.last_read, article.article_id),
    )
    train_end = int(len(mature_articles) * SUPER_IMPORTANT_TRAIN_FRACTION)
    validation_end = train_end + int(
        len(mature_articles) * SUPER_IMPORTANT_VALIDATION_FRACTION
    )
    return (
        mature_articles[:train_end],
        mature_articles[train_end:validation_end],
        mature_articles[validation_end:],
    )


def compute_super_important_ranking_metrics(
    articles: list[Article],
    scores: np.ndarray,
    budgets: tuple[int, ...] = SUPER_IMPORTANT_RANKING_BUDGETS,
) -> dict[str, float]:
    """Measure explicit-preference ranking and ordinary relevance guardrails."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(articles):
        raise ValueError(
            "Ranking scores must be one-dimensional and align with articles."
        )
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be non-empty and finite.")
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("Ranking budgets must be positive.")

    good_labels = np.asarray(
        [article.status == "read" and article.vote >= 0 for article in articles],
        dtype=int,
    )
    super_important_labels = np.asarray(
        [relevance_embedding.is_super_important(article) for article in articles],
        dtype=int,
    )
    read_mask = good_labels.astype(bool)
    read_labels = super_important_labels[read_mask]
    if np.unique(good_labels).size != 2 or np.unique(read_labels).size != 2:
        raise ValueError(
            "Ranking metrics need good, bad, super-important, and ordinary-read rows."
        )

    ranked_indices = np.argsort(-scores, kind="stable")
    positive_count = int(super_important_labels.sum())
    graded_labels = good_labels + super_important_labels
    metrics = {
        "positive_prevalence": float(read_labels.mean()),
        "super_important_average_precision": float(
            average_precision_score(read_labels, scores[read_mask])
        ),
        "relevance_average_precision": float(
            average_precision_score(good_labels, scores)
        ),
    }
    for budget in budgets:
        effective_budget = min(budget, len(articles))
        top_labels = super_important_labels[ranked_indices[:effective_budget]]
        metrics[f"precision_at_{budget}"] = float(top_labels.mean())
        metrics[f"recall_at_{budget}"] = float(top_labels.sum() / positive_count)
        metrics[f"ndcg_at_{budget}"] = float(
            ndcg_score(
                graded_labels[np.newaxis, :],
                scores[np.newaxis, :],
                k=effective_budget,
            )
        )
    return metrics


def super_important_rollout_gate_passes(
    baseline_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
) -> bool:
    """Return whether a ranker improves preference ranking within the AP guardrail."""
    recall_improved = any(
        candidate_metrics[f"recall_at_{budget}"]
        > baseline_metrics[f"recall_at_{budget}"]
        for budget in SUPER_IMPORTANT_RANKING_BUDGETS
    )
    return (
        candidate_metrics["relevance_average_precision"]
        >= baseline_metrics["relevance_average_precision"] - MAX_RELEVANCE_AP_DROP
        and candidate_metrics["super_important_average_precision"]
        > baseline_metrics["super_important_average_precision"]
        and recall_improved
    )


def select_bonus_passing_all_windows(
    window_results: list[
        tuple[Mapping[str, float], Mapping[float, Mapping[str, float]]]
    ],
) -> float | None:
    """Return the smallest bonus passing every chronological window."""
    if not window_results:
        return None
    eligible = [
        bonus
        for bonus in SUPER_IMPORTANT_BONUS_GRID
        if all(
            candidates[bonus]["relevance_average_precision"]
            >= baseline["relevance_average_precision"] - MAX_RELEVANCE_AP_DROP
            and candidates[bonus]["super_important_average_precision"]
            > baseline["super_important_average_precision"]
            for baseline, candidates in window_results
        )
        and any(
            candidates[bonus][f"recall_at_{budget}"] > baseline[f"recall_at_{budget}"]
            for baseline, candidates in window_results
            for budget in SUPER_IMPORTANT_RANKING_BUDGETS
        )
    ]
    return min(eligible, default=None)


def select_super_important_bonus(
    articles: list[Article],
    weighted_baseline_scores: np.ndarray,
    relevance_probabilities: np.ndarray,
    super_important_probabilities: np.ndarray,
) -> tuple[float | None, dict[float, dict[str, float]], dict[str, float]]:
    """Score the fixed bonus grid on one chronological validation window."""
    baseline_metrics = compute_super_important_ranking_metrics(
        articles,
        weighted_baseline_scores,
    )
    candidate_metrics = {
        bonus: compute_super_important_ranking_metrics(
            articles,
            relevance_embedding.combine_probabilities(
                relevance_probabilities,
                super_important_probabilities,
                bonus_strength=bonus,
            ),
        )
        for bonus in SUPER_IMPORTANT_BONUS_GRID
    }
    selected = select_bonus_passing_all_windows([(baseline_metrics, candidate_metrics)])
    return selected, candidate_metrics, baseline_metrics


def _super_important_partition_counts(articles: list[Article]) -> dict[str, int]:
    """Count relevance and preference labels in one benchmark partition."""
    good = sum(article.status == "read" and article.vote >= 0 for article in articles)
    super_important = sum(
        relevance_embedding.is_super_important(article) for article in articles
    )
    return {
        "good": good,
        "bad": len(articles) - good,
        "super_important": super_important,
        "ordinary_read": good - super_important,
    }


def _fit_super_important_rankers(
    embeddings: np.ndarray,
    articles: list[Article],
) -> tuple[LogisticRegression, LogisticRegression, LogisticRegression]:
    """Fit the weighted baseline and both unweighted ranker heads."""
    relevance_labels = np.asarray(
        [article.status == "read" and article.vote >= 0 for article in articles],
        dtype=int,
    )
    super_important_labels = np.asarray(
        [relevance_embedding.is_super_important(article) for article in articles],
        dtype=int,
    )
    weighted_classifier = relevance_embedding.fit_classifier(
        embeddings,
        relevance_labels,
        pipeline_label="weighted relevance baseline",
        sample_weights=np.where(
            super_important_labels == 1,
            WEIGHTED_RELEVANCE_BASELINE_WEIGHT,
            1.0,
        ),
    )
    relevance_classifier = relevance_embedding.fit_classifier(
        embeddings,
        relevance_labels,
        pipeline_label="unweighted relevance",
    )
    read_mask = relevance_labels.astype(bool)
    super_important_classifier = relevance_embedding.fit_classifier(
        embeddings[read_mask],
        super_important_labels[read_mask],
        pipeline_label="super-important",
    )
    return weighted_classifier, relevance_classifier, super_important_classifier


def compute_and_log_metrics(
    model_name: str,
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics and log them.

    Args:
        model_name: Name of the model being evaluated (for log prefixing).
        true_labels: Ground truth binary labels.
        predicted_probs: Predicted probabilities for the positive class.

    Returns:
        Dictionary of metric names to their float values.

    """
    pred_labels = (predicted_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "precision": float(precision_score(true_labels, pred_labels, zero_division=0)),
        "recall": float(recall_score(true_labels, pred_labels, zero_division=0)),
        "f1": float(f1_score(true_labels, pred_labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(true_labels, predicted_probs)),
        "average_precision": float(
            average_precision_score(true_labels, predicted_probs)
        ),
        "log_loss": float(log_loss(true_labels, predicted_probs)),
    }

    logger.info(f"[{model_name}] Evaluation results:")
    logger.info(f"[{model_name}]   Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"[{model_name}]   Precision:         {metrics['precision']:.4f}")
    logger.info(f"[{model_name}]   Recall:            {metrics['recall']:.4f}")
    logger.info(f"[{model_name}]   F1:                {metrics['f1']:.4f}")
    logger.info(f"[{model_name}]   ROC AUC:           {metrics['roc_auc']:.4f}")
    logger.info(
        f"[{model_name}]   Average Precision: {metrics['average_precision']:.4f}"
    )
    logger.info(f"[{model_name}]   Log Loss:          {metrics['log_loss']:.4f}")

    return metrics


async def save_eval_results(
    model_name: str,
    training_counts: dict[str, int],
    eval_counts: dict[str, int],
    metrics: Mapping[str, float | None],
) -> None:
    """Persist an evaluation record to JSON history and PostgreSQL.

    Creates the file if it does not exist. If the file exists but is
    corrupted, it is overwritten with a fresh list containing only the
    new record. The PostgreSQL insert is intentionally allowed to fail the
    eval job, because Miniflux's ``model_evals`` table is the durable history.

    Args:
        model_name: Name of the model evaluated.
        training_counts: Article counts used for training, keyed by class.
        eval_counts: Article counts used for evaluation, keyed by class.
        metrics: Metric name to value mapping from ``compute_and_log_metrics``.

    """
    eval_date = datetime.date.today()
    record = {
        "date": eval_date.isoformat(),
        "model": model_name,
        "training": training_counts,
        "eval": eval_counts,
        "metrics": metrics,
    }

    history: list[dict] = []  # type: ignore[type-arg]
    if os.path.exists(EVAL_HISTORY_PATH):
        try:
            with open(EVAL_HISTORY_PATH) as f:
                history = json.load(f)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                f"Could not parse {EVAL_HISTORY_PATH}. Starting fresh history."
            )
            history = []

    history.append(record)

    with open(EVAL_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"[{model_name}] Evaluation results saved to {EVAL_HISTORY_PATH}.")

    await dr.insert_model_eval(
        eval_date=eval_date,
        model_name=model_name,
        training_counts=training_counts,
        eval_counts=eval_counts,
        metrics=metrics,
    )


async def _run_relevance_inference(
    encoder: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    classifier: LogisticRegression,
    articles: list[Article],
    device: torch.device,
) -> np.ndarray:
    """Run held-out relevance inference with the relevance head only."""
    return await relevance_embedding.predict_probabilities(
        articles,
        tokenizer,
        encoder,
        classifier,
        device,
    )


async def _run_urgency_inference(
    encoder: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    classifier: LogisticRegression,
    articles: list[Article],
    device: torch.device,
) -> np.ndarray:
    """Run held-out urgency inference with the embedding-linear backend."""
    return await urgency_embedding.predict_probabilities(
        articles,
        tokenizer,
        encoder,
        classifier,
        device,
    )


async def eval_relevance(device: torch.device) -> None:
    """Evaluate the relevance model accuracy.

    Fetches ALL good/bad articles (no SQL-level holdout), then randomly
    samples VALIDATION_SIZE from each class for the eval set. Trains on
    the remaining articles with the same full-row training data as production,
    then evaluates on the held-out random sample.

    The eval model is discarded after metrics are computed.
    """
    validation_size = config.VALIDATION_SIZE
    logger.info(
        f"[Relevance] Starting evaluation with VALIDATION_SIZE={validation_size}"
    )

    start_time = time.time()

    # Fetch ALL articles (validation_size=0 means no SQL-level holdout).
    all_good = await dr.get_read_articles_training(validation_size=0)
    all_bad = await dr.get_published_articles(validation_size=0)

    logger.info(
        f"[Relevance] Fetched {len(all_good)} good, {len(all_bad)} bad articles total."
    )

    if len(all_good) < validation_size or len(all_bad) < validation_size:
        logger.warning(
            f"[Relevance] Not enough articles to hold out {validation_size} "
            f"per class (good={len(all_good)}, bad={len(all_bad)}). "
            "Skipping eval."
        )
        return

    # Randomly sample VALIDATION_SIZE articles from each class for eval.
    eval_good = random.sample(all_good, validation_size)
    eval_bad = random.sample(all_bad, validation_size)

    eval_good_ids = {a.article_id for a in eval_good}
    eval_bad_ids = {a.article_id for a in eval_bad}

    # Training set: everything NOT in the eval set.
    good_articles = [a for a in all_good if a.article_id not in eval_good_ids]
    bad_articles = [a for a in all_bad if a.article_id not in eval_bad_ids]

    logger.info(
        f"[Relevance] Eval set: {len(eval_good)} good, {len(eval_bad)} bad "
        f"(randomly sampled)."
    )

    logger.info(
        f"[Relevance] Training set: {len(good_articles)} good, {len(bad_articles)} bad."
    )

    # Train on a temp path.
    model_path = f"models/{EVAL_RELEVANCE_PREFIX}"

    try:
        (
            encoder,
            tokenizer,
            relevance_classifier,
            super_important_classifier,
        ) = await llm_learn.train_model(
            good_articles,
            bad_articles,
            model_path,
            device,
        )
        logger.info("[Relevance] Eval model trained successfully.")

        # Run inference on held-out set.
        good_probs = await _run_relevance_inference(
            encoder,
            tokenizer,
            relevance_classifier,
            eval_good,
            device,
        )
        bad_probs = await _run_relevance_inference(
            encoder,
            tokenizer,
            relevance_classifier,
            eval_bad,
            device,
        )

        all_probs = np.concatenate([good_probs, bad_probs])
        true_labels = np.concatenate(
            [np.ones(len(good_probs)), np.zeros(len(bad_probs))]
        )

        metrics = compute_and_log_metrics("Relevance", true_labels, all_probs)

        await save_eval_results(
            model_name="Relevance",
            training_counts={"good": len(good_articles), "bad": len(bad_articles)},
            eval_counts={"good": len(eval_good), "bad": len(eval_bad)},
            metrics=metrics,
        )

    finally:
        # Always clean up the eval model.
        shutil.rmtree(model_path, ignore_errors=True)
        logger.info(f"[Relevance] Cleaned up eval model at {model_path}.")

    elapsed_time = time.time() - start_time
    logger.info(f"[Relevance] Evaluation completed in {elapsed_time:.2f} seconds.")


async def eval_super_important(device: torch.device) -> None:
    """Select the smallest preference bonus passing two rolling windows."""
    all_articles = await dr.get_read_articles_training(validation_size=0)
    all_articles += await dr.get_published_articles(validation_size=0)
    train_articles, middle_articles, newest_articles = (
        split_super_important_eval_articles(all_articles)
    )
    partitions = {
        "oldest_60_percent": train_articles,
        "middle_20_percent": middle_articles,
        "newest_20_percent": newest_articles,
    }
    partition_counts = {
        name: _super_important_partition_counts(articles)
        for name, articles in partitions.items()
    }
    required_counts = [
        counts[label]
        for counts in partition_counts.values()
        for label in ("bad", "super_important", "ordinary_read")
    ]
    if min(required_counts) < MIN_SUPER_IMPORTANT_EXAMPLES or any(
        len(articles) < max(SUPER_IMPORTANT_RANKING_BUDGETS)
        for articles in partitions.values()
    ):
        logger.warning(
            "[Super-important] Not enough mature examples for stable tuning: "
            f"{partition_counts}. Skipping eval."
        )
        return

    logger.info(f"[Super-important] Chronological partition counts: {partition_counts}")
    for name, articles in partitions.items():
        logger.info(
            f"[Super-important] Fixed {name} article IDs: "
            f"{[article.article_id for article in articles]}"
        )

    tokenizer, encoder = relevance_embedding.load_encoder(
        device,
        pipeline_label="super-important evaluation",
    )
    ordered_articles = train_articles + middle_articles + newest_articles
    embeddings = await relevance_embedding.encode_articles(
        ordered_articles,
        tokenizer,
        encoder,
        device,
        pipeline_label="super-important evaluation",
    )
    train_end = len(train_articles)
    middle_end = train_end + len(middle_articles)
    windows = (
        (
            "window_1",
            train_articles,
            embeddings[:train_end],
            middle_articles,
            embeddings[train_end:middle_end],
        ),
        (
            "window_2",
            train_articles + middle_articles,
            embeddings[:middle_end],
            newest_articles,
            embeddings[middle_end:],
        ),
    )
    window_results: list[
        tuple[Mapping[str, float], Mapping[float, Mapping[str, float]]]
    ] = []
    fixed_bonus_metrics: dict[str, float] | None = None
    for name, fit_articles, fit_embeddings, eval_articles, eval_embeddings in windows:
        weighted_classifier, relevance_classifier, preference_classifier = (
            _fit_super_important_rankers(fit_embeddings, fit_articles)
        )
        weighted_scores = relevance_embedding.predict_probabilities_from_embeddings(
            eval_embeddings,
            weighted_classifier,
        )
        relevance_probabilities = (
            relevance_embedding.predict_probabilities_from_embeddings(
                eval_embeddings,
                relevance_classifier,
            )
        )
        preference_probabilities = (
            relevance_embedding.predict_probabilities_from_embeddings(
                eval_embeddings,
                preference_classifier,
            )
        )
        _, candidates, baseline = select_super_important_bonus(
            eval_articles,
            weighted_scores,
            relevance_probabilities,
            preference_probabilities,
        )
        window_results.append((baseline, candidates))
        logger.info(
            f"[Super-important][{name}][weighted_relevance_baseline] {baseline}"
        )
        for bonus, metrics in candidates.items():
            logger.info(f"[Super-important][{name}][bonus={bonus}] {metrics}")

        if name == "window_2":
            fixed_bonus_scores = relevance_embedding.combine_probabilities(
                relevance_probabilities,
                preference_probabilities,
                bonus_strength=config.SUPER_IMPORTANT_BONUS,
            )
            fixed_bonus_metrics = compute_super_important_ranking_metrics(
                eval_articles,
                fixed_bonus_scores,
            )
            logger.info(
                "[Super-important][window_2][fixed_bonus="
                f"{config.SUPER_IMPORTANT_BONUS}] {fixed_bonus_metrics}"
            )

    assert fixed_bonus_metrics is not None
    await save_eval_results(
        model_name="Super-important",
        training_counts=_super_important_partition_counts(
            train_articles + middle_articles
        ),
        eval_counts=_super_important_partition_counts(newest_articles),
        metrics={
            "super_important_average_precision": fixed_bonus_metrics[
                "super_important_average_precision"
            ],
            "relevance_average_precision": fixed_bonus_metrics[
                "relevance_average_precision"
            ],
            "recall_at_10": fixed_bonus_metrics["recall_at_10"],
            "recall_at_25": fixed_bonus_metrics["recall_at_25"],
            "recall_at_50": fixed_bonus_metrics["recall_at_50"],
            "super_important_bonus": config.SUPER_IMPORTANT_BONUS,
        },
    )

    selected_bonus = select_bonus_passing_all_windows(window_results)
    if selected_bonus is None:
        logger.warning("[Super-important] No bonus passed every rolling window.")
        logger.info("[Super-important] Rolling rollout gate passed: False")
        return

    positive_articles = [
        article
        for article in newest_articles
        if relevance_embedding.is_super_important(article)
    ]
    subgroup_counts = {
        "upvoted_only": sum(
            article.vote == 1 and not article.starred for article in positive_articles
        ),
        "starred_only": sum(
            article.vote != 1 and article.starred for article in positive_articles
        ),
        "both": sum(
            article.vote == 1 and article.starred for article in positive_articles
        ),
    }
    logger.info(
        f"[Super-important] Newest-window positive subgroups: {subgroup_counts}"
    )
    logger.info(
        "[Super-important] Rolling rollout gate passed: True; "
        f"selected_bonus={selected_bonus}"
    )


async def eval_urgency(device: torch.device) -> None:
    """Evaluate the urgency model accuracy.

    Trains on read-tagged urgency articles only and holds out a stratified
    subset of those trusted labels for evaluation.

    The eval model is discarded after metrics are computed.
    """
    validation_size = config.VALIDATION_SIZE
    logger.info(f"[Urgency] Starting evaluation with VALIDATION_SIZE={validation_size}")

    start_time = time.time()

    labeled_data = await dr.get_read_articles_with_urgency_tags()

    if not labeled_data:
        logger.warning("[Urgency] No tagged articles found. Skipping eval.")
        return

    logger.info(f"[Urgency] Total read-tagged articles: {len(labeled_data)}.")

    if len(labeled_data) <= validation_size:
        logger.warning(
            f"[Urgency] Not enough read-tagged articles ({len(labeled_data)}) "
            f"to hold out {validation_size}. Skipping eval."
        )
        return

    articles = [article for article, _ in labeled_data]
    labels = np.asarray([label for _, label in labeled_data], dtype=int)
    label_counts = np.bincount(labels, minlength=2)
    if validation_size < 2 or int(label_counts.min()) < 2:
        logger.warning(
            "[Urgency] Not enough examples per class to build a stratified eval split. "
            "Skipping eval."
        )
        return

    train_articles, eval_articles, train_labels, eval_labels = train_test_split(
        articles,
        labels,
        test_size=validation_size,
        random_state=42,
        stratify=labels,
    )

    train_articles = list(train_articles)
    eval_articles = list(eval_articles)
    train_labels = np.asarray(train_labels, dtype=int)
    eval_labels = np.asarray(eval_labels, dtype=int)

    eval_urgent = int(eval_labels.sum())
    eval_evergreen = int(len(eval_labels) - eval_urgent)
    logger.info(
        f"[Urgency] Eval set: {len(eval_articles)} read-tagged articles "
        f"({eval_urgent} urgent, {eval_evergreen} evergreen)."
    )
    logger.info(
        f"[Urgency] Training set: {len(train_articles)} read-tagged articles "
        f"({int(train_labels.sum())} urgent, "
        f"{int(len(train_labels) - train_labels.sum())} evergreen)."
    )

    # Train on a temp path.
    model_path = f"models/{EVAL_URGENCY_PREFIX}"

    try:
        encoder, tokenizer, classifier = await llm_learn_urgency.train_model(
            model_path,
            train_articles,
            train_labels,
            device,
        )
        logger.info("[Urgency] Eval model trained successfully.")

        # Run inference on held-out read articles.
        eval_probs = await _run_urgency_inference(
            encoder,
            tokenizer,
            classifier,
            eval_articles,
            device,
        )

        metrics = compute_and_log_metrics("Urgency", eval_labels, eval_probs)

        await save_eval_results(
            model_name="Urgency",
            training_counts={
                "urgent": int(train_labels.sum()),
                "evergreen": int(len(train_labels) - train_labels.sum()),
            },
            eval_counts={"urgent": eval_urgent, "evergreen": eval_evergreen},
            metrics=metrics,
        )

    finally:
        # Always clean up the eval model.
        shutil.rmtree(model_path, ignore_errors=True)
        logger.info(f"[Urgency] Cleaned up eval model at {model_path}.")

    elapsed_time = time.time() - start_time
    logger.info(f"[Urgency] Evaluation completed in {elapsed_time:.2f} seconds.")


async def eval_freshness(device: torch.device) -> None:
    """Evaluate freshness on the newest chronological label holdout."""
    validation_size = config.VALIDATION_SIZE
    logger.info(
        f"[Freshness] Starting evaluation with VALIDATION_SIZE={validation_size}"
    )
    labeled_data = await dr.get_semantic_freshness_training_data()
    if len(labeled_data) <= validation_size:
        logger.warning(
            f"[Freshness] Not enough labeled articles ({len(labeled_data)}) to hold "
            f"out {validation_size}. Skipping eval."
        )
        return

    training_data = labeled_data[:-validation_size]
    eval_data = labeled_data[-validation_size:]
    train_articles = [article for article, _, _ in training_data]
    eval_articles = [article for article, _, _ in eval_data]
    train_labels = np.asarray([label for _, label, _ in training_data], dtype=int)
    eval_labels = np.asarray([label for _, label, _ in eval_data], dtype=int)

    tokenizer, encoder = relevance_embedding.load_encoder(
        device, pipeline_label="freshness"
    )
    embeddings = await relevance_embedding.encode_articles(
        train_articles + eval_articles,
        tokenizer,
        encoder,
        device,
        pipeline_label="freshness",
    )
    try:
        classifiers = semantic_freshness_embedding.fit_classifiers(
            embeddings[: len(training_data)], train_labels
        )
    except RuntimeError as exc:
        logger.warning(f"[Freshness] {exc} Skipping eval.")
        return
    probabilities = semantic_freshness_embedding.bucket_probabilities(
        embeddings[len(training_data) :], classifiers
    )
    metrics = compute_freshness_metrics(eval_labels, probabilities)
    logger.info(f"[Freshness] Evaluation results: {metrics}")

    horizons = semantic_freshness_embedding.HORIZONS
    await save_eval_results(
        model_name="Freshness",
        training_counts={
            horizon: int((train_labels == index).sum())
            for index, horizon in enumerate(horizons)
        },
        eval_counts={
            horizon: int((eval_labels == index).sum())
            for index, horizon in enumerate(horizons)
        },
        metrics=metrics,
    )


async def main() -> None:
    """Run evaluation for all models sequentially."""
    validation_size = config.VALIDATION_SIZE

    if validation_size == 0:
        logger.info("VALIDATION_SIZE is 0. Skipping evaluation.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type != "cuda" and not config.ALLOW_TRAINING_WO_GPU:
        mes = "GPU not available. Exiting"
        logger.critical(mes)
        raise RuntimeError(mes)

    # Seed for reproducible random sampling within a single run.
    random.seed(42)

    # Remove any leftover eval model directories from a previous crashed run.
    _clean_stale_eval_dirs()

    await dr.global_pool.open(wait=True)

    try:
        await eval_relevance(device)
        await eval_super_important(device)
        await eval_urgency(device)
        await eval_freshness(device)
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
