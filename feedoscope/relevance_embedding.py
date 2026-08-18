import hashlib
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, NamedTuple

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
import joblib  # type: ignore[import-untyped]
import numpy as np
from scipy import sparse  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from feedoscope import config, relevance_text
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article
from feedoscope.utils import clean_title, strip_html_keep_text

logger = logging.getLogger(__name__)

EmbeddingField = Literal["title", "body"]

ARTIFACT_FILENAME = "relevance_title_body_word.joblib"
ARTIFACT_VERSION = 6
BACKEND = "embedding_title_body_word_logistic"
LINEAR_MAX_ITER = 4000
LINEAR_RANDOM_STATE = 42
TITLE_MAX_LENGTH = 512
EMBEDDING_FIELDS: tuple[EmbeddingField, ...] = ("title", "body")
TFIDF_PARAMETERS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.98,
    "sublinear_tf": True,
}
FEATURE_ORDER = ("title_embedding", "body_embedding", "title_word", "body_word")
LABEL_CONTRACT = {
    "relevance_positive": "read and vote >= 0",
    "relevance_negative": "vote = -1",
}
TRAIN_COUNT_KEYS = {"good", "bad"}
ENCODER_CACHE_ROOT = Path("models/relevance_encoder")
ENCODER_READY_FILENAME = ".snapshot_complete"


class RelevanceModel(NamedTuple):
    """Keep one classifier aligned with its text vectorizers and embeddings."""

    classifier: LogisticRegression
    title_vectorizer: TfidfVectorizer
    body_vectorizer: TfidfVectorizer
    embedding_dimensions: int


def _pipeline_name(pipeline_label: str) -> str:
    """Normalize pipeline labels for human-readable logging."""
    return pipeline_label.replace("_", " ")


def _pipeline_title(pipeline_label: str) -> str:
    """Return a capitalized pipeline label for log messages."""
    return _pipeline_name(pipeline_label).capitalize()


def spread_relevance_score(score: float) -> float:
    """Spread a final 0-100 score while preserving its ranking order."""
    normalized = score / 100
    if not 0 <= normalized <= 1:
        raise ValueError("score must be between 0 and 100")
    return (1 - math.cbrt(1 - normalized)) * 100


def prepare_scores_for_storage(scores: list[float]) -> list[int]:
    """Spread and round final scores for integer database storage."""
    return [round(spread_relevance_score(score)) for score in scores]


def get_encoder_cache_path() -> Path:
    """Return the shared on-disk cache path for the configured encoder."""
    return ENCODER_CACHE_ROOT / config.RELEVANCE_MODEL_NAME.replace("/", "--")


def get_model_family_prefix() -> str:
    """Return the versioned artifact family for relevance models."""
    return (
        f"relevance_{config.RELEVANCE_EMBEDDING_KEY.replace('/', '-')}_"
        f"title{TITLE_MAX_LENGTH}_body{config.RELEVANCE_MAX_LENGTH}_"
        f"p{config.RELEVANCE_PREP_VERSION}_word12_logistic_"
        f"c{config.RELEVANCE_LINEAR_C}_iw{config.IMPORTANT_ARTICLE_WEIGHT}"
    )


def has_local_encoder_snapshot(encoder_path: Path) -> bool:
    """Check whether the shared cache already contains a usable encoder snapshot."""
    has_config = (encoder_path / "config.json").exists()
    has_tokenizer = any(
        candidate.exists()
        for candidate in (
            encoder_path / "tokenizer.json",
            encoder_path / "tokenizer.model",
            encoder_path / "vocab.json",
        )
    )
    has_weights = (
        any(encoder_path.glob("*.safetensors"))
        or any(encoder_path.glob("*.bin"))
        or (encoder_path / "model.safetensors.index.json").exists()
    )
    return has_config and has_tokenizer and has_weights


def ensure_local_encoder(pipeline_label: str = "relevance") -> str:
    """Download the shared encoder snapshot once and reuse it afterward."""
    encoder_path = get_encoder_cache_path()
    ready_marker = encoder_path / ENCODER_READY_FILENAME
    pipeline_name = _pipeline_name(pipeline_label)
    pipeline_title = _pipeline_title(pipeline_label)

    if ready_marker.exists() or has_local_encoder_snapshot(encoder_path):
        if not ready_marker.exists():
            ready_marker.write_text("\n")
        logger.info(f"Using cached {pipeline_name} embedding encoder at {encoder_path}")
        return str(encoder_path)

    logger.info(
        f"Downloading {pipeline_name} embedding encoder {config.RELEVANCE_MODEL_NAME} to "
        f"{encoder_path}"
    )
    encoder_path.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=config.RELEVANCE_MODEL_NAME,
            local_dir=str(encoder_path),
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            f"Cannot download the gated {pipeline_name} embedding encoder. Populate the shared "
            f"cache at {encoder_path} from an authenticated machine or provide "
            f"Hugging Face credentials before running {pipeline_name} training or inference."
        ) from exc
    ready_marker.write_text("\n")
    logger.info(f"{pipeline_title} embedding encoder cached at {encoder_path}")
    return str(encoder_path)


def load_encoder(
    device: torch.device,
    pipeline_label: str = "relevance",
) -> tuple[PreTrainedTokenizerBase, torch.nn.Module]:
    """Load the shared embedding tokenizer and encoder onto the target device."""
    pipeline_name = _pipeline_name(pipeline_label)
    pipeline_title = _pipeline_title(pipeline_label)
    encoder_path = ensure_local_encoder(pipeline_label=pipeline_label)
    logger.info(f"Loading {pipeline_name} embedding encoder from {encoder_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        encoder_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    logger.info(f"{pipeline_title} embedding encoder loaded successfully")
    return tokenizer, model


def get_cache_config(field: EmbeddingField) -> dict[str, str | int]:
    """Return the configuration values that define one field embedding."""
    return {
        "model_name": config.RELEVANCE_EMBEDDING_KEY,
        "max_length": (
            TITLE_MAX_LENGTH if field == "title" else config.RELEVANCE_MAX_LENGTH
        ),
        "text_prep_mode": field,
        "prep_version": config.RELEVANCE_PREP_VERSION,
        "prompt": config.RELEVANCE_EMBEDDING_PROMPT,
    }


def get_tfidf_config() -> dict[str, object]:
    """Return the fixed Word TF-IDF feature contract."""
    return dict(TFIDF_PARAMETERS)


def get_encoder_output_dim(model: torch.nn.Module) -> int:
    """Return the embedding width produced by the current encoder path."""
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Relevance encoder config is missing hidden_size")
    return int(hidden_size)


def hash_prepared_text(text: str) -> str:
    """Build a stable hash for prepared article text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_articles_text(
    articles: list[Article],
    field: EmbeddingField,
    pipeline_label: str = "relevance",
) -> list[str]:
    """Prepare one independent article field before cache lookup or encoding."""
    pipeline_name = _pipeline_name(pipeline_label)
    logger.info(f"Preparing {pipeline_name} {field} text for {len(articles)} articles")
    prompt = config.RELEVANCE_EMBEDDING_PROMPT
    texts: list[str] = []
    for start in range(0, len(articles), 1000):
        batch = articles[start : start + 1000]
        if field == "title":
            prepared = [
                relevance_text.prepare_single_blob(article.title, "")
                for article in batch
            ]
        else:
            prepared = [
                relevance_text.prepare_single_blob("", article.content)
                for article in batch
            ]
        texts.extend(prompt + text for text in prepared)
        logger.info(
            f"Prepared {pipeline_name} {field} text for "
            f"{len(texts)}/{len(articles)} articles"
        )
    return texts


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average token embeddings across non-padding positions."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


async def encode_articles(
    articles: list[Article],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
    field: EmbeddingField,
    pipeline_label: str = "relevance",
) -> np.ndarray:
    """Prepare one article field, reuse cached vectors, and encode only misses."""
    if not articles:
        return np.empty((0, get_encoder_output_dim(model)), dtype=np.float32)

    texts = prepare_articles_text(
        articles,
        field,
        pipeline_label=pipeline_label,
    )
    text_hashes = [hash_prepared_text(text) for text in texts]
    article_ids = [article.article_id for article in articles]
    cache_config = get_cache_config(field)
    logger.info(
        f"Looking up {len(article_ids)} {_pipeline_name(pipeline_label)} embedding cache entries"
    )
    cached = await dr.get_relevance_embeddings(
        article_ids=article_ids,
        model_name=str(cache_config["model_name"]),
        max_length=int(cache_config["max_length"]),
        text_prep_mode=str(cache_config["text_prep_mode"]),
        prep_version=int(cache_config["prep_version"]),
    )

    expected_dim = get_encoder_output_dim(model)
    hits = 0
    stale = 0
    miss_indices: list[int] = []
    miss_texts: list[str] = []
    results: list[np.ndarray | None] = [None] * len(articles)

    for index, article_id in enumerate(article_ids):
        cached_row = cached.get(article_id)
        if cached_row is not None:
            cached_text_hash, cached_embedding = cached_row
            # The cache key is one row per article/config. A text hash mismatch
            # means the article content or prep output changed, so we overwrite
            # that row rather than creating another version.
            if (
                cached_text_hash == text_hashes[index]
                and cached_embedding.size == expected_dim
            ):
                results[index] = cached_embedding
                hits += 1
                continue
            stale += 1

        miss_indices.append(index)
        miss_texts.append(texts[index])

    logger.info(
        f"{_pipeline_title(pipeline_label)} embedding cache: "
        f"{hits} hits, {len(miss_indices)} misses"
        + (f" ({stale} stale)" if stale else "")
    )

    if miss_texts:
        fresh_embeddings = encode_texts(
            miss_texts,
            tokenizer,
            model,
            device,
            max_length=int(cache_config["max_length"]),
            pipeline_label=pipeline_label,
        )
        upsert_rows: list[tuple[int, str, np.ndarray]] = []

        for offset, index in enumerate(miss_indices):
            embedding = np.asarray(fresh_embeddings[offset], dtype=np.float32)
            results[index] = embedding
            upsert_rows.append((article_ids[index], text_hashes[index], embedding))

        await dr.upsert_relevance_embeddings(
            rows=upsert_rows,
            model_name=str(cache_config["model_name"]),
            max_length=int(cache_config["max_length"]),
            text_prep_mode=str(cache_config["text_prep_mode"]),
            prep_version=int(cache_config["prep_version"]),
        )

    assert all(
        embedding is not None for embedding in results
    ), "Internal error: missing relevance embeddings after cache fill"
    return np.stack([embedding for embedding in results if embedding is not None])


async def encode_article_fields(
    articles: list[Article],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
    pipeline_label: str = "relevance",
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned title and body embeddings for one article batch."""
    title_embeddings = await encode_articles(
        articles,
        tokenizer,
        model,
        device,
        field="title",
        pipeline_label=pipeline_label,
    )
    body_embeddings = await encode_articles(
        articles,
        tokenizer,
        model,
        device,
        field="body",
        pipeline_label=pipeline_label,
    )
    return title_embeddings, body_embeddings


def encode_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
    pipeline_label: str = "relevance",
) -> np.ndarray:
    """Encode raw text batches into normalized dense vectors."""
    if not texts:
        return np.empty((0, get_encoder_output_dim(model)), dtype=np.float32)

    embeddings: list[np.ndarray] = []
    total_batches = math.ceil(len(texts) / config.RELEVANCE_ENCODER_BATCH_SIZE)
    pipeline_name = _pipeline_name(pipeline_label)

    with torch.no_grad():
        for batch_index, start in enumerate(
            range(0, len(texts), config.RELEVANCE_ENCODER_BATCH_SIZE),
            start=1,
        ):
            batch = texts[start : start + config.RELEVANCE_ENCODER_BATCH_SIZE]
            logger.info(f"Encoding {pipeline_name} batch {batch_index}/{total_batches}")
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def is_important(article: Article) -> bool:
    """Return whether a read/good article received an explicit preference."""
    return (
        article.status == "read"
        and article.vote >= 0
        and (article.vote == 1 or article.starred)
    )


def build_relevance_sample_weights(articles: list[Article]) -> np.ndarray:
    """Return weights that emphasize explicitly preferred articles."""
    return np.array(
        [
            config.IMPORTANT_ARTICLE_WEIGHT if is_important(article) else 1.0
            for article in articles
        ]
    )


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Create one unfitted vectorizer with the tested feature contract."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )


def _word_texts(articles: list[Article], field: EmbeddingField) -> list[str]:
    """Prepare title or body text for Word TF-IDF."""
    if field == "title":
        return [clean_title(article.title) for article in articles]
    return [strip_html_keep_text(article.content) for article in articles]


def build_features(
    articles: list[Article],
    title_embeddings: np.ndarray,
    body_embeddings: np.ndarray,
    title_vectorizer: TfidfVectorizer,
    body_vectorizer: TfidfVectorizer,
    fit_vectorizers: bool = False,
    embedding_dimensions: int | None = None,
) -> sparse.csr_matrix:
    """Build aligned title, body, title-word, and body-word feature blocks."""
    title_embeddings = np.asarray(title_embeddings, dtype=np.float32)
    body_embeddings = np.asarray(body_embeddings, dtype=np.float32)
    if (
        title_embeddings.ndim != 2
        or body_embeddings.ndim != 2
        or title_embeddings.shape != body_embeddings.shape
        or len(title_embeddings) != len(articles)
        or not np.isfinite(title_embeddings).all()
        or not np.isfinite(body_embeddings).all()
    ):
        raise ValueError("Title and body embeddings must be finite and aligned")
    if (
        embedding_dimensions is not None
        and title_embeddings.shape[1] != embedding_dimensions
    ):
        raise ValueError("Embedding dimensions do not match the relevance artifact")

    transform = "fit_transform" if fit_vectorizers else "transform"
    title_words = getattr(title_vectorizer, transform)(_word_texts(articles, "title"))
    body_words = getattr(body_vectorizer, transform)(_word_texts(articles, "body"))
    return sparse.hstack(
        (
            sparse.csr_matrix(title_embeddings),
            sparse.csr_matrix(body_embeddings),
            title_words,
            body_words,
        ),
        format="csr",
    )


def fit_classifier(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    pipeline_label: str = "relevance",
    sample_weights: np.ndarray | None = None,
) -> LogisticRegression:
    """Fit the configured title/body plus Word TF-IDF logistic head."""
    pipeline_name = _pipeline_name(pipeline_label)
    logger.info(
        f"Fitting {pipeline_name} logistic regression on {len(labels)} rows with "
        f"C={config.RELEVANCE_LINEAR_C}"
    )
    classifier = LogisticRegression(
        C=config.RELEVANCE_LINEAR_C,
        max_iter=LINEAR_MAX_ITER,
        random_state=LINEAR_RANDOM_STATE,
    )
    classifier.fit(
        features,
        labels,
        sample_weight=sample_weights,
    )
    logger.info(f"{_pipeline_title(pipeline_label)} logistic regression fit completed")
    return classifier


def fit_relevance_model(
    articles: list[Article],
    title_embeddings: np.ndarray,
    body_embeddings: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
) -> RelevanceModel:
    """Fit both Word TF-IDF blocks and the weighted logistic classifier."""
    title_vectorizer = build_tfidf_vectorizer()
    body_vectorizer = build_tfidf_vectorizer()
    features = build_features(
        articles,
        title_embeddings,
        body_embeddings,
        title_vectorizer,
        body_vectorizer,
        fit_vectorizers=True,
    )
    classifier = fit_classifier(
        features,
        labels,
        sample_weights=sample_weights,
    )
    return RelevanceModel(
        classifier,
        title_vectorizer,
        body_vectorizer,
        int(title_embeddings.shape[1]),
    )


def build_metadata(
    train_counts: dict[str, int], embedding_dimensions: int
) -> dict[str, object]:
    """Build metadata that makes relevance artifacts safe to load."""
    return {
        "artifact_version": ARTIFACT_VERSION,
        "backend": BACKEND,
        "encoders": {field: get_cache_config(field) for field in EMBEDDING_FIELDS},
        "tfidf": get_tfidf_config(),
        "feature_order": FEATURE_ORDER,
        "embedding_dimensions": embedding_dimensions,
        "linear_c": config.RELEVANCE_LINEAR_C,
        "important_article_weight": config.IMPORTANT_ARTICLE_WEIGHT,
        "label_contract": LABEL_CONTRACT,
        "train_counts": train_counts,
    }


def save_relevance_artifact(
    model_path: str,
    relevance_model: RelevanceModel,
    train_counts: dict[str, int],
) -> None:
    """Persist the complete relevance model and compatibility metadata."""
    path = Path(model_path)
    path.mkdir(parents=True, exist_ok=True)
    artifact = {
        "relevance_classifier": relevance_model.classifier,
        "title_vectorizer": relevance_model.title_vectorizer,
        "body_vectorizer": relevance_model.body_vectorizer,
        "metadata": build_metadata(train_counts, relevance_model.embedding_dimensions),
    }
    destination = path / ARTIFACT_FILENAME
    with tempfile.NamedTemporaryFile(dir=path, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(artifact, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    logger.info(f"Saved relevance artifact to {model_path}")


def _vectorizer_is_compatible(vectorizer: object) -> bool:
    if not isinstance(vectorizer, TfidfVectorizer):
        return False
    params = vectorizer.get_params()
    return (
        all(params[name] == value for name, value in TFIDF_PARAMETERS.items())
        and hasattr(vectorizer, "vocabulary_")
        and hasattr(vectorizer, "idf_")
    )


def load_relevance_artifact(model_path: str) -> RelevanceModel:
    """Load a compatible title/body plus Word TF-IDF artifact."""
    artifact = joblib.load(Path(model_path) / ARTIFACT_FILENAME)
    if not isinstance(artifact, dict):
        raise RuntimeError("Relevance artifact is not compatible with this model.")
    relevance_classifier = artifact.get("relevance_classifier")
    title_vectorizer = artifact.get("title_vectorizer")
    body_vectorizer = artifact.get("body_vectorizer")
    metadata = artifact.get("metadata")
    train_counts = metadata.get("train_counts") if isinstance(metadata, dict) else None
    embedding_dimensions = (
        metadata.get("embedding_dimensions") if isinstance(metadata, dict) else None
    )
    if (
        not isinstance(relevance_classifier, LogisticRegression)
        or not _vectorizer_is_compatible(title_vectorizer)
        or not _vectorizer_is_compatible(body_vectorizer)
        or not isinstance(metadata, dict)
        or metadata.get("artifact_version") != ARTIFACT_VERSION
        or metadata.get("backend") != BACKEND
        or metadata.get("encoders")
        != {field: get_cache_config(field) for field in EMBEDDING_FIELDS}
        or metadata.get("tfidf") != get_tfidf_config()
        or metadata.get("feature_order") != FEATURE_ORDER
        or not isinstance(embedding_dimensions, int)
        or embedding_dimensions <= 0
        or metadata.get("linear_c") != config.RELEVANCE_LINEAR_C
        or metadata.get("important_article_weight") != config.IMPORTANT_ARTICLE_WEIGHT
        or metadata.get("label_contract") != LABEL_CONTRACT
        or not isinstance(train_counts, dict)
        or set(train_counts) != TRAIN_COUNT_KEYS
        or not all(isinstance(count, int) for count in train_counts.values())
    ):
        raise RuntimeError("Relevance artifact is not compatible with this model.")
    classifier_params = relevance_classifier.get_params()
    expected_features = (
        2 * embedding_dimensions
        + len(title_vectorizer.vocabulary_)  # type: ignore[union-attr]
        + len(body_vectorizer.vocabulary_)  # type: ignore[union-attr]
    )
    if (
        classifier_params["C"] != config.RELEVANCE_LINEAR_C
        or classifier_params["max_iter"] != LINEAR_MAX_ITER
        or classifier_params["random_state"] != LINEAR_RANDOM_STATE
        or getattr(relevance_classifier, "n_features_in_", None) != expected_features
    ):
        raise RuntimeError("Relevance artifact is not compatible with this model.")
    assert isinstance(title_vectorizer, TfidfVectorizer)
    assert isinstance(body_vectorizer, TfidfVectorizer)
    assert isinstance(embedding_dimensions, int)
    return RelevanceModel(
        relevance_classifier,
        title_vectorizer,
        body_vectorizer,
        embedding_dimensions,
    )


def predict_probabilities_from_features(
    features: sparse.csr_matrix,
    classifier: Any,
) -> np.ndarray:
    """Predict clipped positive-class probabilities from prepared features."""
    if not features.shape[0]:
        return np.array([], dtype=float)
    probs = classifier.predict_proba(features)[:, 1]
    return np.clip(probs, 1e-7, 1 - 1e-7)


async def predict_probabilities(
    articles: list[Article],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    relevance_model: RelevanceModel,
    device: torch.device,
    pipeline_label: str = "relevance",
) -> np.ndarray:
    """Predict positive-class probabilities for a batch of articles."""
    if not articles:
        return np.array([], dtype=float)

    title_embeddings, body_embeddings = await encode_article_fields(
        articles,
        tokenizer,
        model,
        device,
        pipeline_label=pipeline_label,
    )
    features = build_features(
        articles,
        title_embeddings,
        body_embeddings,
        relevance_model.title_vectorizer,
        relevance_model.body_vectorizer,
        embedding_dimensions=relevance_model.embedding_dimensions,
    )
    return predict_probabilities_from_features(features, relevance_model.classifier)


def peak_vram_gb() -> float:
    """Return the peak CUDA memory allocated in GiB for the current process."""
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / math.pow(1024, 3))
