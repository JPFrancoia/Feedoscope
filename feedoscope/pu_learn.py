import asyncio
import logging
import os

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from custom_logging import init_logging
from feedoscope import config, models, utils
from feedoscope.data_registry import data_registry as dr

logger = logging.getLogger(__name__)


def chunk_compute_embeddings(
    model: SentenceTransformer, texts: list[str]
) -> np.ndarray:
    all_embeddings = []

    for idx, text in enumerate(texts):
        logger.debug(f"Processing text {idx + 1}/{len(texts)}")
        chunks = chunk_text(text, model.tokenizer, max_tokens=model.max_seq_length)
        chunk_embeddings = model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Weighted pooling, gives best results so far
        chunk_lengths = [len(model.tokenizer.tokenize(chunk)) for chunk in chunks]
        weights = np.array(chunk_lengths) / sum(chunk_lengths)
        text_embedding = np.average(chunk_embeddings, axis=0, weights=weights)
        all_embeddings.append(text_embedding)

    return np.vstack(all_embeddings)


def chunk_text(text, tokenizer, max_tokens):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.convert_tokens_to_string(tokens[i : i + max_tokens])
        chunks.append(chunk)
    return chunks


async def get_read_articles_embeddings(model):

    embeddings_path = f"embeddings/embeddings_{config.EMBEDDINGS_MODEL_NAME.replace('/', '-')}.npy"
    if os.path.exists(embeddings_path):
        logger.debug("Loading embeddings from file")
        embeddings = utils.load_embeddings(embeddings_path)
        return embeddings

    logger.debug("Collecting articles from the database")
    articles = await dr.get_read_articles_training()
    logger.debug(f"Collected {len(articles)} articles.")

    logger.debug("Computing embeddings for articles")
    embeddings = utils.compute_embeddings(model, utils.prepare_articles_text(articles))
    logger.debug(f"Computed embeddings for {len(embeddings)} articles.")

    utils.save_embeddings(embeddings_path, embeddings)

    return embeddings


async def get_unread_articles_embeddings(model):
    embeddings_path = f"embeddings/embeddings_unread_{config.EMBEDDINGS_MODEL_NAME.replace('/', '-')}.npy"
    if os.path.exists(embeddings_path):
        logger.debug("Loading unread articles embeddings from file")
        embeddings = utils.load_embeddings(embeddings_path)
        return embeddings

    logger.debug("Collecting unread articles from the database")
    unlabeled_articles = await dr.get_unread_articles_training()
    logger.debug(f"Collected {len(unlabeled_articles)} unread articles.")

    logger.debug("Computing embeddings for unread articles")
    unlabeled_embeddings = utils.compute_embeddings(
        model,
        utils.prepare_articles_text(unlabeled_articles),
    )
    logger.debug(
        f"Computed embeddings for {len(unlabeled_embeddings)} unread articles."
    )

    utils.save_embeddings(embeddings_path, unlabeled_embeddings)

    return unlabeled_embeddings


async def get_good_articles_embeddings(model):
    embeddings_path = f"embeddings/embeddings_good_{config.EMBEDDINGS_MODEL_NAME.replace('/', '-')}.npy"
    if os.path.exists(embeddings_path):
        logger.debug("Loading good articles embeddings from file")
        embeddings = utils.load_embeddings(embeddings_path)
        return embeddings
    logger.debug("Collecting good articles from the database")
    good_articles = await dr.get_sample_good()
    logger.debug(f"Collected {len(good_articles)} good articles.")
    logger.debug("Computing embeddings for good articles")
    good_embeddings = utils.compute_embeddings(
        model, utils.prepare_articles_text(good_articles)
    )
    logger.debug(f"Computed embeddings for {len(good_embeddings)} good articles.")
    utils.save_embeddings(embeddings_path, good_embeddings)

    return good_embeddings


async def get_not_good_articles_embeddings(model):
    embeddings_path = f"embeddings/embeddings_not_good_{config.EMBEDDINGS_MODEL_NAME.replace('/', '-')}.npy"
    if os.path.exists(embeddings_path):
        logger.debug("Loading not good articles embeddings from file")
        embeddings = utils.load_embeddings(embeddings_path)
        return embeddings

    logger.debug("Collecting not good articles from the database")
    not_good_articles = await dr.get_sample_not_good()
    logger.debug(f"Collected {len(not_good_articles)} not good articles.")

    logger.debug("Computing embeddings for not good articles")
    not_good_embeddings = utils.compute_embeddings(
        model,
        utils.prepare_articles_text(not_good_articles),
    )
    logger.debug(
        f"Computed embeddings for {len(not_good_embeddings)} not good articles."
    )

    utils.save_embeddings(embeddings_path, not_good_embeddings)

    return not_good_embeddings


async def main() -> None:

    logger.debug("Loading SentenceTransformer model")
    model = SentenceTransformer(config.EMBEDDINGS_MODEL_NAME, trust_remote_code=True, device="cpu")
    logger.debug("Model loaded successfully.")

    await dr.global_pool.open(wait=True)

    embeddings = await get_read_articles_embeddings(model)
    unlabeled_embeddings = await get_unread_articles_embeddings(model)
    good_embeddings = await get_good_articles_embeddings(model)
    not_good_embeddings = await get_not_good_articles_embeddings(model)

    # Combine embeddings and labels for PU learning
    X = np.concatenate([embeddings, unlabeled_embeddings], axis=0)
    y = np.concatenate(
        [np.ones(len(embeddings)), np.zeros(len(unlabeled_embeddings))], axis=0
    )

    models_to_test = [
        # models.svc_elkanoto_pu_classifier,
        # models.tuned_svc_elkanoto_pu_classifier,
        # models.svc_weighted_elkanoto_pu_classifier,
        # models.tuned_svc_weighted_elkanoto_pu_classifier,
        models.tuned_logistic_regression_weighted_elkanoto_pu_classifier,
        # models.svc_bagging,
        models.logistic_regression_bagging,
        models.tuned_logistic_regression_bagging,
        models.random_forest_bagging,
        models.tuned_random_forest_bagging,
        models.gradient_boosting_bagging,
        models.tuned_gradient_boosting_bagging,
        models.xgboost_bagging,
        models.tuned_xgboost_bagging,
    ]

    # True labels: 1 for good, 0 for not_good
    true_labels = np.concatenate(
        [np.ones(len(good_embeddings)), np.zeros(len(not_good_embeddings))]
    )

    for mod in models_to_test:
        mod_name = mod.__name__

        logger.debug(f"Fitting PU classifier {mod_name}")

        model_path = (
            f"saved_models/{mod_name}_{config.EMBEDDINGS_MODEL_NAME.replace('/', '-')}.pkl"
        )

        if os.path.exists(model_path):
            logger.debug(f"Loading existing model from {model_path}")
            pu_estimator = joblib.load(model_path)
            logger.debug(f"Model {mod_name} loaded successfully.")
        else:
            pu_estimator = mod(X, y, embeddings, unlabeled_embeddings)
            joblib.dump(pu_estimator, model_path)
            logger.info(f"Saved model {mod_name} to {model_path}")

        logger.debug(f"PU classifier {mod_name} fitted successfully.")

        good_preds = pu_estimator.predict_proba(good_embeddings)[:, 1]
        not_good_preds = pu_estimator.predict_proba(not_good_embeddings)[:, 1]
        all_probs = np.concatenate([good_preds, not_good_preds])
        pred_labels = (all_probs >= 0.5).astype(int)

        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        roc_auc = roc_auc_score(true_labels, all_probs)
        ap = average_precision_score(true_labels, all_probs)
        logloss = log_loss(true_labels, all_probs)

        logger.info(f"Precision for {mod_name}: {precision:.2f}")
        logger.info(f"Recall for {mod_name}: {recall:.2f}")
        logger.info(f"F1 score for {mod_name}: {f1:.2f}")
        logger.info(f"ROC AUC for {mod_name}: {roc_auc:.2f}")
        logger.info(f"Average Precision for {mod_name}: {ap:.2f}")
        logger.info(f"Log Loss for {mod_name}: {logloss:.2f}")

    await dr.global_pool.close()


if __name__ == "__main__":
    init_logging(config.LOGGING_CONFIG)
    asyncio.run(main())
