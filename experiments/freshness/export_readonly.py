"""Export a frozen local urgency dataset without permitting database writes."""

import csv
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import numpy as np
import psycopg
from psycopg.rows import dict_row

DATA_DIR = Path(".auto/data")
ARTICLES_PATH = DATA_DIR / "articles.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
MANIFEST_PATH = DATA_DIR / "export_manifest.json"
MODEL_KEY = (
    "urgency-embedding_linear::google/embeddinggemma-300m::"
    "2048::title_head::1::c=1.0"
)

QUERY = """
SELECT
    e.id AS article_id,
    e.title,
    f.title AS feed_name,
    e.published_at,
    e.content,
    COALESCE(e.tags, ARRAY[]::text[]) AS feed_tags,
    CASE ut.title WHEN '0-urgency' THEN 0 ELSE 1 END AS source_urgency_label,
    ui.urgency_score AS current_urgency_score,
    tss.score AS decoder_score,
    tss.explanation AS decoder_explanation,
    re.embedding
FROM entries e
JOIN feeds f ON f.id = e.feed_id
JOIN entry_user_tags eut ON eut.entry_id = e.id
JOIN user_tags ut ON ut.id = eut.user_tag_id
    AND ut.user_id = 1
    AND ut.title IN ('0-urgency', '1-urgency')
JOIN relevance_embeddings re ON re.article_id = e.id
    AND re.model_name = 'google/embeddinggemma-300m'
    AND re.max_length = 2048
    AND re.text_prep_mode = 'title_head'
    AND re.prep_version = 1
LEFT JOIN urgency_inference ui ON ui.article_id = e.id
    AND ui.model_key = %(model_key)s
LEFT JOIN time_sensitivity_simplified tss ON tss.article_id = e.id
WHERE e.status = 'read'
ORDER BY e.id
"""


def _clean_title(title: str) -> str:
    title = re.sub(r"\[\d+\]\s*", "", title)
    return re.sub(r"\s*\(TS:\s*\d+\)\s*$", "", title).strip()


def _clean_content(content: str | None) -> str:
    text = BeautifulSoup(content or "", "html.parser").get_text(" ", strip=True)
    return " ".join(text.split())[:6000]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Export article metadata and cached vectors to ignored local files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(
        host="localhost",
        port=5432,
        user="miniflux",
        dbname="miniflux",
        options="-c default_transaction_read_only=on",
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW transaction_read_only")
            setting = cur.fetchone()
            if setting is None:
                raise RuntimeError("Could not verify PostgreSQL transaction mode")
            read_only = setting["transaction_read_only"]
            if read_only != "on":
                raise RuntimeError(
                    "Refusing export: PostgreSQL transaction is writable"
                )
            cur.execute(QUERY, {"model_key": MODEL_KEY})
            rows = cur.fetchall()

    embeddings: list[np.ndarray] = []
    fields = [
        "article_id",
        "title",
        "feed_name",
        "published_at",
        "content",
        "feed_tags",
        "source_urgency_label",
        "current_urgency_score",
        "decoder_score",
        "decoder_explanation",
    ]
    with ARTICLES_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            raw_embedding = row.pop("embedding")
            embedding = np.frombuffer(raw_embedding, dtype=np.float32).copy()
            if embedding.shape != (768,):
                raise RuntimeError(
                    f"Article {row['article_id']} has embedding shape {embedding.shape}"
                )
            embeddings.append(embedding)
            writer.writerow(
                {
                    "article_id": row["article_id"],
                    "title": _clean_title(row["title"]),
                    "feed_name": row["feed_name"],
                    "published_at": row["published_at"].isoformat(),
                    "content": _clean_content(row["content"]),
                    "feed_tags": json.dumps(row["feed_tags"]),
                    "source_urgency_label": row["source_urgency_label"],
                    "current_urgency_score": row["current_urgency_score"],
                    "decoder_score": row["decoder_score"],
                    "decoder_explanation": row["decoder_explanation"] or "",
                }
            )

    matrix = np.vstack(embeddings).astype(np.float32)
    np.save(EMBEDDINGS_PATH, matrix, allow_pickle=False)
    manifest = {
        "database": "miniflux@localhost:5432/miniflux",
        "transaction_read_only": read_only,
        "rows": len(rows),
        "embedding_shape": list(matrix.shape),
        "embedding_config": {
            "model_name": "google/embeddinggemma-300m",
            "max_length": 2048,
            "text_prep_mode": "title_head",
            "prep_version": 1,
        },
        "urgency_model_key": MODEL_KEY,
        "files": {
            str(ARTICLES_PATH): _sha256(ARTICLES_PATH),
            str(EMBEDDINGS_PATH): _sha256(EMBEDDINGS_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} rows with {matrix.shape[1]}-dimensional embeddings")
    print("PostgreSQL transaction_read_only=on")


if __name__ == "__main__":
    main()
