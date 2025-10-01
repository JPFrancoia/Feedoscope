import re

from bs4 import BeautifulSoup
from cleantext import clean  # type: ignore[import]

from feedoscope.entities import Article


def strip_html_keep_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def prepare_articles_text(articles: list[Article]) -> list[str]:
    """Sanitize and prepare article texts for embedding computation.

    This function shouldn't mutate the articles in any way.

    Args:
        articles: List of articles

    Returns:
        A blob of text for each article, ready for embedding computation.

    """
    texts = []
    for a in articles:
        # For articles that were evaluated previously, the title contains the score
        # in square brackets. Clean it up before processing.

        # TODO: double check we're not nuking the content of html tags

        title = clean_title(a.title)
        text = clean(strip_html_keep_text(f"{title} {a.content}"))
        texts.append(text)

    return texts


def clean_title(title: str) -> str:
    """Clean the title from any score computed in a previous evaluation.

    Example:
        "[85] This is an article title (TS: 4)" -> "This is an article title"

    This also cleans multiple scores, if a previous inference run messed up the titles:
        "[85] [90] This is an article title (TS: 4) (TS: 5)" -> "This is an article title"

    Args:
        title: The article title.

    Returns:
        The cleaned title.
    """
    new_title = re.sub(r"\[\d+\]\s*", "", title)
    new_title = re.sub(r"\s*\(TS:\s*\d+\)\s*$", "", new_title)

    return new_title
