import re

from bs4 import BeautifulSoup
from cleantext import clean
import numpy as np


def strip_html_keep_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def prepare_articles_text(articles) -> list[str]:
    """Sanitize and prepare article texts for embedding computation.

    WARNING: This function modifies the input articles in place by cleaning the
    title from any score computed in a previous evaluation.

    Args:
        articles: List of articles, where each article is a dictionary

    Returns:
        A blob of text for each article, ready for embedding computation.

    """
    texts = []
    for a in articles:
        # For articles that were evaluated previously, the title contains the score
        # in square brackets. Clean it up before processing.
        title = re.sub(r"^\[[^]]*\]\s*", "", a["title"])
        a["title"] = title
        text = clean(strip_html_keep_text(f"{title} {a['content']}"))
        texts.append(text)

    return texts


def save_embeddings(filepath, embeddings):
    np.save(filepath, embeddings)


def load_embeddings(filepath):
    return np.load(filepath)


def compute_embeddings(model, texts: list[str]):
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        truncate=True,
    )
    return embeddings
