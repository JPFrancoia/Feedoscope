import asyncio
import logging
import os

import joblib
import numpy as np
from bs4 import BeautifulSoup
from cleantext import clean
from custom_logging import init_logging
from pulearn import (BaggingPuClassifier, ElkanotoPuClassifier,
                     WeightedElkanotoPuClassifier)
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

from feedoscope.data_registry import data_registry as dr

from .mlr import ModifiedLogisticRegression

logger = logging.getLogger(__name__)


# TODO: use another model, maybe with more dimensions?
MODEL_NAME = "all-MiniLM-L12-v2"
# MODEL_NAME = "all-MiniLM-L6-v2"  # Default model for embeddings

# MODEL_NAME = "all-mpnet-base-v2"


def strip_html_keep_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def compute_embeddings(model, texts: list[str]):
    embeddings = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True
    )
    return embeddings


def prepare_articles_text(articles) -> list[str]:
    texts = []
    for a in articles:
        text = clean(
            strip_html_keep_text(f"{a['feed_name']} {a['title']} {a['content']}")
        )
        texts.append(text)

    return texts


def save_embeddings(filepath, embeddings):
    np.save(filepath, embeddings)


def load_embeddings(filepath):
    return np.load(filepath)


def tune_pu_estimator(pu_estimator, X, y, param_grid):
    # Set up grid search
    grid_search = GridSearchCV(
        pu_estimator,
        param_grid,
        # scoring="accuracy",
        scoring="f1",
        # scoring="average_precision",  # better for ranking tasks
        cv=3,  # or another suitable value
        n_jobs=10,
        verbose=3,
    )

    grid_search.fit(X, y)

    logger.debug(f"Best parameters: {grid_search.best_params_}")
    logger.debug(f"Best score: {grid_search.best_score_}")

    # Use the best estimator for further predictions
    pu_estimator = grid_search.best_estimator_

    return pu_estimator


async def get_read_articles_embeddings(model):

    embeddings_path = f"embeddings_{MODEL_NAME}.npy"
    if os.path.exists(embeddings_path):
        logger.debug("Loading embeddings from file")
        embeddings = load_embeddings(embeddings_path)
        return embeddings

    logger.debug("Collecting articles from the database")
    articles = await dr.get_articles()
    logger.debug(f"Collected {len(articles)} articles.")

    logger.debug("Computing embeddings for articles")
    embeddings = compute_embeddings(model, prepare_articles_text(articles))
    logger.debug(f"Computed embeddings for {len(embeddings)} articles.")

    save_embeddings(embeddings_path, embeddings)

    return embeddings


async def get_unread_articles_embeddings(model):
    embeddings_path = f"embeddings_unread_{MODEL_NAME}.npy"
    if os.path.exists(embeddings_path):
        logger.debug("Loading unread articles embeddings from file")
        embeddings = load_embeddings(embeddings_path)
        return embeddings

    logger.debug("Collecting unread articles from the database")
    unlabeled_articles = await dr.get_unread_articles()
    logger.debug(f"Collected {len(unlabeled_articles)} unread articles.")

    logger.debug("Computing embeddings for unread articles")
    unlabeled_embeddings = compute_embeddings(
        model, prepare_articles_text(unlabeled_articles)
    )
    logger.debug(
        f"Computed embeddings for {len(unlabeled_embeddings)} unread articles."
    )

    save_embeddings(embeddings_path, unlabeled_embeddings)

    return unlabeled_embeddings


async def get_good_articles_embeddings(model):
    embeddings_path = f"embeddings_good_{MODEL_NAME}.npy"
    if os.path.exists(embeddings_path):
        logger.debug("Loading good articles embeddings from file")
        embeddings = load_embeddings(embeddings_path)
        return embeddings
    logger.debug("Collecting good articles from the database")
    good_articles = await dr.get_sample_good()
    logger.debug(f"Collected {len(good_articles)} good articles.")
    logger.debug("Computing embeddings for good articles")
    good_embeddings = compute_embeddings(model, prepare_articles_text(good_articles))
    logger.debug(f"Computed embeddings for {len(good_embeddings)} good articles.")
    save_embeddings(embeddings_path, good_embeddings)

    return good_embeddings


async def get_not_good_articles_embeddings(model):
    embeddings_path = f"embeddings_not_good_{MODEL_NAME}.npy"
    if os.path.exists(embeddings_path):
        logger.debug("Loading not good articles embeddings from file")
        embeddings = load_embeddings(embeddings_path)
        return embeddings

    logger.debug("Collecting not good articles from the database")
    not_good_articles = await dr.get_sample_not_good()
    logger.debug(f"Collected {len(not_good_articles)} not good articles.")

    logger.debug("Computing embeddings for not good articles")
    not_good_embeddings = compute_embeddings(
        model, prepare_articles_text(not_good_articles)
    )
    logger.debug(
        f"Computed embeddings for {len(not_good_embeddings)} not good articles."
    )

    save_embeddings(embeddings_path, not_good_embeddings)

    return not_good_embeddings


async def main() -> None:

    logger.debug("Loading SentenceTransformer model")
    model = SentenceTransformer(MODEL_NAME)
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
        # svc_elkanoto_pu_classifier,
        # tuned_svc_elkanoto_pu_classifier,
        # svc_weighted_elkanoto_pu_classifier,
        # tuned_svc_weighted_elkanoto_pu_classifier,
        # tuned_logistic_regression_weighted_elkanoto_pu_classifier,
        # svc_bagging,
        # logistic_regression_bagging,
        # tuned_logistic_regression_bagging,
        # modified_logistic_regression_bagging,
        # random_forest_bagging,
        # tuned_random_forest_bagging,
        # gradient_boosting_bagging,
        tuned_gradient_boosting_bagging,
        # xgboost_bagging,
        tuned_xgboost_bagging,
    ]

    # True labels: 1 for good, 0 for not_good
    true_labels = np.concatenate(
        [np.ones(len(good_embeddings)), np.zeros(len(not_good_embeddings))]
    )

    for mod in models_to_test:
        mod_name = mod.__name__

        logger.debug(f"Fitting PU classifier {mod_name}")

        model_path = f"saved_models/{mod_name}_{MODEL_NAME}.pkl"

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


def svc_elkanoto_pu_classifier(X, y, embeddings, unlabeled_embeddings):
    # Takes a while for 7k + 7k articles
    svc = SVC(C=10, kernel="rbf", gamma=0.4, probability=True)

    pu_estimator = ElkanotoPuClassifier(
        estimator=svc,
        hold_out_ratio=0.2,
    )
    pu_estimator.fit(X, y)

    return pu_estimator


def tuned_svc_elkanoto_pu_classifier(X, y, embeddings, unlabeled_embeddings):
    param_grid = {
        "estimator__C": [0.1, 1, 10],
        "estimator__gamma": [0.01, 0.1, 1],
        "estimator__kernel": ["rbf", "linear"],
    }
    estimator = SVC(probability=True)

    pu_estimator = ElkanotoPuClassifier(
        estimator=estimator,
        hold_out_ratio=0.2,
    )

    return tune_pu_estimator(pu_estimator, X, y, param_grid)


def svc_weighted_elkanoto_pu_classifier(X, y, embeddings, unlabeled_embeddings):
    svc = SVC(C=10, kernel="rbf", gamma=0.4, probability=True)

    pu_estimator = WeightedElkanotoPuClassifier(
        estimator=svc,
        labeled=len(embeddings),
        unlabeled=len(unlabeled_embeddings),
        hold_out_ratio=0.2,
    )
    pu_estimator.fit(X, y)

    return pu_estimator


def tuned_svc_weighted_elkanoto_pu_classifier(X, y, embeddings, unlabeled_embeddings):
    param_grid = {
        "estimator__C": [0.1, 1, 10],
        "estimator__gamma": [0.01, 0.1, 1],
        "estimator__kernel": ["rbf", "linear"],
    }
    estimator = SVC(probability=True)

    pu_estimator = WeightedElkanotoPuClassifier(
        estimator=estimator,
        labeled=len(embeddings),
        unlabeled=len(unlabeled_embeddings),
        hold_out_ratio=0.2,
    )

    return tune_pu_estimator(pu_estimator, X, y, param_grid)


def tuned_logistic_regression_weighted_elkanoto_pu_classifier(
    X, y, embeddings, unlabeled_embeddings
):
    param_grid = {
        "estimator__C": [0.01, 0.1, 1, 10],
        "estimator__penalty": ["l2"],
        "estimator__solver": ["lbfgs"],
        "estimator__max_iter": [100, 200],
    }
    estimator = LogisticRegression()

    pu_estimator = WeightedElkanotoPuClassifier(
        estimator=estimator,
        labeled=len(embeddings),
        unlabeled=len(unlabeled_embeddings),
        hold_out_ratio=0.2,
    )

    return tune_pu_estimator(pu_estimator, X, y, param_grid)


def svc_bagging(X, y, embeddings, unlabeled_embeddings):
    estimator = SVC(C=10, kernel="rbf", gamma=0.4, probability=True)
    pu_estimator = BaggingPuClassifier(
        estimator=estimator, random_state=42, n_estimators=15
    )
    pu_estimator.fit(X, y)

    return pu_estimator


def logistic_regression_bagging(X, y, embeddings, unlabeled_embeddings):
    estimator = LogisticRegression()
    pu_estimator = BaggingPuClassifier(
        estimator=estimator, random_state=42, n_estimators=15
    )
    pu_estimator.fit(X, y)

    return pu_estimator


def tuned_logistic_regression_bagging(X, y, embeddings, unlabeled_embeddings):
    param_grid = {
        "estimator__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "estimator__penalty": ["l1", "l2", "elasticnet", None],
        "estimator__solver": ["liblinear", "lbfgs", "saga"],
        "estimator__max_iter": [50, 100, 150, 200, 300],
        "n_estimators": [5, 10, 15, 20, 25, 30, 35],
    }

    estimator = LogisticRegression()
    pu_estimator = BaggingPuClassifier(
        estimator=estimator, random_state=42, n_estimators=15
    )

    return tune_pu_estimator(pu_estimator, X, y, param_grid)


def modified_logistic_regression_bagging(X, y, embeddings, unlabeled_embeddings):
    estimator = ModifiedLogisticRegression()
    pu_estimator = BaggingPuClassifier(
        estimator=estimator, random_state=42, n_estimators=15
    )
    pu_estimator.fit(X, y)

    return pu_estimator


def random_forest_bagging(X, y, embeddings, unlabeled_embeddings):
    estimator = RandomForestClassifier()
    pu_estimator = BaggingPuClassifier(
        estimator=estimator, random_state=42, n_estimators=15
    )
    pu_estimator.fit(X, y)

    return pu_estimator


def tuned_random_forest_bagging(X, y, embeddings, unlabeled_embeddings):
    # param_grid = {
    #     "estimator__n_estimators": [50, 100, 200],
    #     "estimator__max_depth": [None, 10, 20],
    #     "estimator__min_samples_split": [2, 5, 10],
    #     "estimator__min_samples_leaf": [1, 2, 4],
    #     "estimator__max_features": ["sqrt", "log2"],
    # }
    param_grid = {
        "estimator__n_estimators": [50, 100],
        "estimator__max_depth": [None, 10],
        "estimator__max_features": ["sqrt"],
    }
    estimator = RandomForestClassifier()
    pu_estimator = BaggingPuClassifier(
        estimator=estimator, random_state=42, n_estimators=15
    )
    return tune_pu_estimator(pu_estimator, X, y, param_grid)


def gradient_boosting_bagging(X, y, embeddings, unlabeled_embeddings):
    estimator = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_depth=6, random_state=42
    )
    pu_estimator = BaggingPuClassifier(
        estimator=estimator, random_state=42, n_estimators=15
    )
    pu_estimator.fit(X, y)
    return pu_estimator


def tuned_gradient_boosting_bagging(X, y, embeddings, unlabeled_embeddings):
    param_grid = {
        "estimator__learning_rate": [0.05, 0.1, 0.2],
        "estimator__max_iter": [100, 200],
        "estimator__max_depth": [3, 6, 9],
        "estimator__l2_regularization": [0.0, 1.0, 10.0],
        "n_estimators": [5, 10, 15, 20, 25, 30, 35],
    }
    estimator = HistGradientBoostingClassifier()
    pu_estimator = BaggingPuClassifier(estimator=estimator, random_state=42)
    return tune_pu_estimator(pu_estimator, X, y, param_grid)


def xgboost_bagging(X, y, embeddings, unlabeled_embeddings):
    estimator = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",  # Needed to avoid warnings
        verbosity=0,
        random_state=42,
    )
    pu_estimator = BaggingPuClassifier(
        estimator=estimator, random_state=42, n_estimators=15
    )
    pu_estimator.fit(X, y)
    return pu_estimator


def tuned_xgboost_bagging(X, y, embeddings, unlabeled_embeddings):
    param_grid = {
        "estimator__learning_rate": [0.05, 0.1, 0.2],
        "estimator__max_depth": [3, 6, 9],
        "estimator__n_estimators": [100, 200],
        "estimator__reg_lambda": [1, 5, 10],
        "n_estimators": [5, 10, 15, 20, 25, 30],  # bagging ensemble size
    }
    estimator = XGBClassifier()
    pu_estimator = BaggingPuClassifier(estimator=estimator, random_state=42)
    return tune_pu_estimator(pu_estimator, X, y, param_grid)


if __name__ == "__main__":
    init_logging("dev_logging.conf")
    asyncio.run(main())
