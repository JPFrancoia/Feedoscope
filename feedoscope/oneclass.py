import asyncio

import numpy as np
from bs4 import BeautifulSoup
from cleantext import clean
from sentence_transformers import SentenceTransformer
# from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

from feedoscope.data_registry import data_registry as dr

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"


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


def normalize_scores(scores):
    scaler = MinMaxScaler()
    return scaler.fit_transform(scores.reshape(-1, 1)).flatten()


def ocsvm_score(estimator, X):
    # Higher decision_function means more inlier-like
    return np.mean(estimator.decision_function(X))


async def main() -> None:
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully.")

    print("Collecting articles from the database...")
    await dr.global_pool.open(wait=True)
    articles = await dr.get_articles()
    print(f"Collected {len(articles)} articles.")

    print("Computing embeddings for articles...")
    embeddings = compute_embeddings(model, prepare_articles_text(articles))
    print(f"Computed embeddings for {len(embeddings)} articles.")

    # Use best parameters directly
    ocsvm = OneClassSVM(kernel="linear", gamma="scale", nu=0.2)
    ocsvm.fit(embeddings)

    # # Hyperparameter tuning for OneClassSVM
    # param_grid = {
    #     "kernel": ["rbf", "linear", "sigmoid"],
    #     "gamma": ["scale", "auto", 0.01, 0.1, 1],
    #     "nu": [0.01, 0.05, 0.1, 0.2]
    # }
    # print("Tuning OneClassSVM hyperparameters...")
    # ocsvm = OneClassSVM()
    # grid = GridSearchCV(
    #     OneClassSVM(),
    #     param_grid,
    #     cv=3,
    #     n_jobs=-1,
    #     scoring=ocsvm_score
    # )
    # grid.fit(embeddings)
    # best_ocsvm = grid.best_estimator_
    # print("Best parameters:", grid.best_params_)

    not_good_sample = await dr.get_sample_not_good()
    not_good_embeddings = compute_embeddings(
        model, prepare_articles_text(not_good_sample)
    )
    raw_scores = ocsvm.decision_function(not_good_embeddings)
    scores = normalize_scores(raw_scores)

    correct_not_good, total_good = sum(s <= 0.5 for s in scores), len(scores)

    good_sample = await dr.get_sample_good()
    good_embeddings = compute_embeddings(model, prepare_articles_text(good_sample))
    raw_scores = ocsvm.decision_function(good_embeddings)
    scores = normalize_scores(raw_scores)

    correct_good, total_not_good = sum(s > 0.5 for s in scores), len(scores)

    print(
        f"Overall precision: {(correct_good + correct_not_good) / (total_good + total_not_good):.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
