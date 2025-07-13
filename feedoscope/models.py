import logging

from pulearn import (
    BaggingPuClassifier,
    ElkanotoPuClassifier,
    WeightedElkanotoPuClassifier,
)
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def tune_pu_estimator(pu_estimator, X, y, param_grid):
    # Set up grid search
    # grid_search = GridSearchCV(
    grid_search = RandomizedSearchCV(
        pu_estimator,
        param_grid,
        scoring="average_precision",  # better for ranking tasks
        cv=4,
        n_jobs=10,
        verbose=3,
        n_iter=10,
    )

    grid_search.fit(X, y)

    logger.debug(f"Best parameters: {grid_search.best_params_}")
    logger.debug(f"Best score: {grid_search.best_score_}")

    # Use the best estimator for further predictions
    pu_estimator = grid_search.best_estimator_

    return pu_estimator


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
        "estimator__C": [0.5, 1, 1.5, 2, 3, 4, 5, 10],
        "estimator__penalty": ["l2"],
        "estimator__solver": ["lbfgs"],
        "estimator__max_iter": [50, 75, 100, 125, 150, 200],
        "n_estimators": [5, 10, 15, 20, 25, 30, 35],
    }

    estimator = LogisticRegression()
    pu_estimator = BaggingPuClassifier(estimator=estimator, random_state=42)

    return tune_pu_estimator(pu_estimator, X, y, param_grid)


def random_forest_bagging(X, y, embeddings, unlabeled_embeddings):
    estimator = RandomForestClassifier()
    pu_estimator = BaggingPuClassifier(
        estimator=estimator, random_state=42, n_estimators=15
    )
    pu_estimator.fit(X, y)

    return pu_estimator


def tuned_random_forest_bagging(X, y, embeddings, unlabeled_embeddings):
    param_grid = {
        "estimator__n_estimators": [50, 100, 200],
        "estimator__max_depth": [None, 10, 20],
        "estimator__min_samples_split": [2, 5, 10],
        "estimator__min_samples_leaf": [1, 2, 4],
        "estimator__max_features": ["sqrt", "log2"],
        "n_estimators": [5, 10, 15, 20, 25, 30, 35],
    }
    estimator = RandomForestClassifier()
    pu_estimator = BaggingPuClassifier(estimator=estimator, random_state=42)
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
        "n_estimators": [5, 10, 15, 20, 25, 30],
    }
    estimator = XGBClassifier(iuse_label_encoder=False, eval_metric="logloss")
    pu_estimator = BaggingPuClassifier(estimator=estimator, random_state=42)
    return tune_pu_estimator(pu_estimator, X, y, param_grid)
