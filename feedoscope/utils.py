from cleantext import clean
from bs4 import BeautifulSoup
import numpy as np


def strip_html_keep_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


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


def compute_embeddings(model, texts: list[str]):
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        truncate=True,
    )
    return embeddings
