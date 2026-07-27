import asyncio
import datetime
import logging
import math
import time

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
import torch

from custom_logging import init_logging
from feedoscope import config, relevance_embedding, semantic_freshness_embedding
from feedoscope.data_registry import data_registry as dr
from feedoscope.llm_infer import find_latest_model

logger = logging.getLogger(__name__)


def split_temporal_holdout(
    labels: np.ndarray,
) -> tuple[slice, slice]:
    """Return chronological train and validation slices for ordered labels."""
    validation_size = config.SEMANTIC_FRESHNESS_VALIDATION_SIZE
    if validation_size == 0:
        return slice(None), slice(0, 0)
    if validation_size >= len(labels):
        raise RuntimeError(
            "SEMANTIC_FRESHNESS_VALIDATION_SIZE must be smaller than the label count."
        )
    return slice(0, -validation_size), slice(-validation_size, None)


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Compute ordinal diagnostics for a chronological freshness holdout."""
    one_hot = np.eye(len(semantic_freshness_embedding.HORIZONS))[labels]
    cumulative_error = np.cumsum(probabilities - one_hot, axis=1)[:, :-1]
    predicted_labels = np.argmax(probabilities, axis=1)
    expected_days = semantic_freshness_embedding.expected_lifetime_days(probabilities)
    true_days = semantic_freshness_embedding.REPRESENTATIVE_DAYS[labels]
    metrics = {
        "rps": float(np.mean(np.sum(cumulative_error**2, axis=1) / 5)),
        "macro_f1": float(f1_score(labels, predicted_labels, average="macro")),
        "log_duration_mae": float(
            np.mean(np.abs(np.log(expected_days) - np.log(true_days)))
        ),
    }
    weighted_kappa = cohen_kappa_score(labels, predicted_labels, weights="quadratic")
    if math.isfinite(weighted_kappa):
        metrics["weighted_kappa"] = float(weighted_kappa)

    evergreen = labels == len(semantic_freshness_embedding.HORIZONS) - 1
    if len(np.unique(evergreen)) == 2:
        metrics["evergreen_auc"] = float(roc_auc_score(evergreen, probabilities[:, -1]))
    return metrics


async def main() -> None:
    """Train a changed-label semantic-freshness artifact."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not config.ALLOW_TRAINING_WO_GPU:
        raise RuntimeError("GPU not available. Exiting")

    await dr.global_pool.open(wait=True)
    try:
        await dr.ensure_semantic_freshness_user_tags()
        await dr.promote_read_auto_freshness_tags()
        conflicts = await dr.get_conflicting_semantic_freshness_labels()
        for article_id, title in conflicts:
            logger.warning(
                f"Skipping conflicting reviewed freshness labels for {article_id}: {title}"
            )

        labeled_data = await dr.get_semantic_freshness_training_data()
        if not labeled_data:
            raise RuntimeError("No effective semantic-freshness labels are available.")

        articles = [article for article, _, _, _ in labeled_data]
        labels = np.asarray([label for _, label, _, _ in labeled_data], dtype=int)
        fingerprint = semantic_freshness_embedding.fingerprint_labels(
            [
                (article.article_id, label, source, confidence)
                for article, label, source, confidence in labeled_data
            ],
            validation_size=config.SEMANTIC_FRESHNESS_VALIDATION_SIZE,
        )
        try:
            _, metadata = semantic_freshness_embedding.load_artifact(
                find_latest_model(
                    semantic_freshness_embedding.get_model_family_prefix(),
                    clean_old_models=False,
                )
            )
            if metadata.get("dataset_fingerprint") == fingerprint:
                logger.info("No new labels; skipping semantic-freshness training.")
                return
        except FileNotFoundError:
            pass

        train_slice, validation_slice = split_temporal_holdout(labels)
        train_articles = articles[train_slice]
        train_labels = labels[train_slice]
        tokenizer, encoder = relevance_embedding.load_encoder(
            device,
            pipeline_label="semantic freshness",
        )
        start_time = time.time()
        train_embeddings = await relevance_embedding.encode_articles(
            train_articles,
            tokenizer,
            encoder,
            device,
            pipeline_label="semantic freshness",
        )
        classifiers = semantic_freshness_embedding.fit_classifiers(
            train_embeddings, train_labels
        )

        metrics: dict[str, float] = {}
        validation_articles = articles[validation_slice]
        validation_labels = labels[validation_slice]
        if len(validation_articles):
            validation_embeddings = await relevance_embedding.encode_articles(
                validation_articles,
                tokenizer,
                encoder,
                device,
                pipeline_label="semantic freshness",
            )
            probabilities = semantic_freshness_embedding.bucket_probabilities(
                validation_embeddings, classifiers
            )
            metrics = compute_metrics(validation_labels, probabilities)
            logger.info(f"Freshness validation metrics: {metrics}")

        train_counts = {
            horizon: int((train_labels == index).sum())
            for index, horizon in enumerate(semantic_freshness_embedding.HORIZONS)
        }
        label_source_counts = {
            source: sum(
                1 for _, _, row_source, _ in labeled_data if row_source == source
            )
            for source in {source for _, _, source, _ in labeled_data}
        }
        metadata = semantic_freshness_embedding.artifact_metadata(
            fingerprint,
            train_counts=train_counts,
            validation_metrics=metrics,
            label_source_counts=label_source_counts,
        )
        model_path = semantic_freshness_embedding.build_model_path(fingerprint)
        semantic_freshness_embedding.save_artifact(model_path, classifiers, metadata)
        if metrics:
            await dr.insert_model_eval(
                eval_date=datetime.date.today(),
                model_name="Freshness",
                training_counts=train_counts,
                eval_counts={
                    horizon: int((validation_labels == index).sum())
                    for index, horizon in enumerate(
                        semantic_freshness_embedding.HORIZONS
                    )
                },
                metrics=metrics,
            )
        logger.info(
            f"Semantic freshness training completed in {time.time() - start_time:.2f} seconds."
        )
    finally:
        await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
