import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
import joblib  # type: ignore[import-untyped]
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from feedoscope import config, relevance_text
from feedoscope.data_registry import data_registry as dr
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

CLASSIFIER_FILENAME = "classifier.joblib"
METADATA_FILENAME = "metadata.json"
TWO_HEAD_ARTIFACT_FILENAME = "relevance_two_head_mlp.joblib"
TWO_HEAD_ARTIFACT_VERSION = 3
TWO_HEAD_BACKEND = "embedding_prompted_mlp_two_head"
TWO_HEAD_LABEL_CONTRACT = {
    "relevance_positive": "read and vote >= 0",
    "relevance_negative": "vote = -1",
    "super_important_positive": "read and (vote = 1 or starred)",
    "super_important_negative": "read and vote = 0 and not starred",
}
TWO_HEAD_TRAIN_COUNT_KEYS = {
    "good",
    "bad",
    "super_important",
    "ordinary_read",
}
SUPER_IMPORTANT_DECISION_THRESHOLD = 0.5
ENCODER_CACHE_ROOT = Path("models/relevance_encoder")
ENCODER_READY_FILENAME = ".snapshot_complete"


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
    """Return the versioned artifact family for two-head relevance models."""
    return (
        f"relevance_two_head_{config.RELEVANCE_EMBEDDING_KEY.replace('/', '-')}_"
        f"{config.RELEVANCE_MAX_LENGTH}_{config.RELEVANCE_TEXT_PREP_MODE}_"
        f"p{config.RELEVANCE_PREP_VERSION}_prompted_mlp_"
        f"h{config.RELEVANCE_MLP_HIDDEN_LAYER_SIZE}_a{config.RELEVANCE_MLP_ALPHA}_"
        f"iw{config.IMPORTANT_ARTICLE_WEIGHT}"
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


def get_cache_config() -> dict[str, str | int]:
    """Return the configuration values that define the embedding output."""
    return {
        "model_name": config.RELEVANCE_EMBEDDING_KEY,
        "max_length": config.RELEVANCE_MAX_LENGTH,
        "text_prep_mode": config.RELEVANCE_TEXT_PREP_MODE,
        "prep_version": config.RELEVANCE_PREP_VERSION,
        "prompt": config.RELEVANCE_EMBEDDING_PROMPT,
    }


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
    tokenizer: PreTrainedTokenizerBase,
    pipeline_label: str = "relevance",
) -> list[str]:
    """Prepare article text once before cache lookup or encoding."""
    pipeline_name = _pipeline_name(pipeline_label)
    logger.info(
        f"Preparing {pipeline_name} text for {len(articles)} articles using "
        f"{config.RELEVANCE_TEXT_PREP_MODE}"
    )
    prompt = config.RELEVANCE_EMBEDDING_PROMPT
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    article_budget = config.RELEVANCE_MAX_LENGTH - prompt_tokens
    if article_budget <= 4:
        raise RuntimeError("Relevance embedding prompt leaves no article token budget")
    texts: list[str] = []
    for start in range(0, len(articles), 1000):
        texts.extend(
            prompt + text
            for text in relevance_text.prepare_articles_text(
                articles[start : start + 1000],
                tokenizer=tokenizer,
                max_length=article_budget,
                mode=config.RELEVANCE_TEXT_PREP_MODE,
            )
        )
        logger.info(
            f"Prepared {pipeline_name} text for {len(texts)}/{len(articles)} articles"
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
    pipeline_label: str = "relevance",
) -> np.ndarray:
    """Prepare article text, reuse cached vectors, and encode only misses."""
    if not articles:
        return np.empty((0, get_encoder_output_dim(model)), dtype=np.float32)

    texts = prepare_articles_text(
        articles,
        tokenizer,
        pipeline_label=pipeline_label,
    )
    text_hashes = [hash_prepared_text(text) for text in texts]
    article_ids = [article.article_id for article in articles]
    cache_config = get_cache_config()
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


def encode_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
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
                max_length=config.RELEVANCE_MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def is_super_important(article: Article) -> bool:
    """Return whether a read/good article received an explicit preference."""
    return (
        article.status == "read"
        and article.vote >= 0
        and (article.vote == 1 or article.starred)
    )


def build_relevance_sample_weights(articles: list[Article]) -> np.ndarray:
    """Return MLP weights that emphasize explicitly preferred articles."""
    return np.array(
        [
            config.IMPORTANT_ARTICLE_WEIGHT if is_super_important(article) else 1.0
            for article in articles
        ]
    )


def fit_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    pipeline_label: str = "relevance",
    sample_weights: np.ndarray | None = None,
) -> MLPClassifier:
    """Fit the configured prompted-embedding relevance MLP head."""
    pipeline_name = _pipeline_name(pipeline_label)
    logger.info(
        f"Fitting {pipeline_name} MLP on {len(labels)} rows with "
        f"{config.RELEVANCE_MLP_HIDDEN_LAYER_SIZE} hidden units"
    )
    classifier = MLPClassifier(
        hidden_layer_sizes=(config.RELEVANCE_MLP_HIDDEN_LAYER_SIZE,),
        alpha=config.RELEVANCE_MLP_ALPHA,
        early_stopping=True,
        max_iter=config.RELEVANCE_MLP_MAX_ITER,
        random_state=42,
    )
    classifier.fit(  # type: ignore[call-arg]  # sklearn-stubs omit sklearn 1.7 support
        embeddings,
        labels,
        sample_weight=sample_weights,
    )
    logger.info(f"{_pipeline_title(pipeline_label)} MLP fit completed")
    return classifier


def fit_logistic_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    pipeline_label: str,
    sample_weights: np.ndarray | None = None,
) -> LogisticRegression:
    """Fit a deterministic logistic head for a non-relevance pipeline."""
    classifier = LogisticRegression(
        max_iter=4000,
        C=config.RELEVANCE_LINEAR_C,
        random_state=42,
    )
    classifier.fit(embeddings, labels, sample_weight=sample_weights)
    logger.info(f"{_pipeline_title(pipeline_label)} logistic regression fit completed")
    return classifier


def build_two_head_metadata(
    train_counts: dict[str, int],
) -> dict[str, object]:
    """Build metadata that makes two-head relevance artifacts safe to load."""
    return {
        "artifact_version": TWO_HEAD_ARTIFACT_VERSION,
        "backend": TWO_HEAD_BACKEND,
        "encoder": get_cache_config(),
        "mlp_hidden_layer_size": config.RELEVANCE_MLP_HIDDEN_LAYER_SIZE,
        "mlp_alpha": config.RELEVANCE_MLP_ALPHA,
        "mlp_max_iter": config.RELEVANCE_MLP_MAX_ITER,
        "important_article_weight": config.IMPORTANT_ARTICLE_WEIGHT,
        "super_important_linear_c": config.RELEVANCE_LINEAR_C,
        "label_contract": TWO_HEAD_LABEL_CONTRACT,
        "train_counts": train_counts,
    }


def save_two_head_artifact(
    model_path: str,
    relevance_classifier: MLPClassifier,
    super_important_classifier: LogisticRegression,
    train_counts: dict[str, int],
) -> None:
    """Persist both relevance heads and compatibility metadata together."""
    path = Path(model_path)
    path.mkdir(parents=True, exist_ok=True)
    artifact = {
        "relevance_classifier": relevance_classifier,
        "super_important_classifier": super_important_classifier,
        "metadata": build_two_head_metadata(train_counts),
    }
    destination = path / TWO_HEAD_ARTIFACT_FILENAME
    with tempfile.NamedTemporaryFile(dir=path, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(artifact, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    logger.info(f"Saved two-head relevance artifact to {model_path}")


def load_two_head_artifact(
    model_path: str,
) -> tuple[MLPClassifier, LogisticRegression]:
    """Load a compatible two-head relevance artifact."""
    artifact = joblib.load(Path(model_path) / TWO_HEAD_ARTIFACT_FILENAME)
    if not isinstance(artifact, dict):
        raise RuntimeError(
            "Relevance artifact is not compatible with the two-head model."
        )
    relevance_classifier = artifact.get("relevance_classifier")
    super_important_classifier = artifact.get("super_important_classifier")
    metadata = artifact.get("metadata")
    train_counts = metadata.get("train_counts") if isinstance(metadata, dict) else None
    if (
        not isinstance(relevance_classifier, MLPClassifier)
        or not isinstance(super_important_classifier, LogisticRegression)
        or not isinstance(metadata, dict)
        or metadata.get("artifact_version") != TWO_HEAD_ARTIFACT_VERSION
        or metadata.get("backend") != TWO_HEAD_BACKEND
        or metadata.get("encoder") != get_cache_config()
        or metadata.get("mlp_hidden_layer_size")
        != config.RELEVANCE_MLP_HIDDEN_LAYER_SIZE
        or metadata.get("mlp_alpha") != config.RELEVANCE_MLP_ALPHA
        or metadata.get("mlp_max_iter") != config.RELEVANCE_MLP_MAX_ITER
        or metadata.get("important_article_weight") != config.IMPORTANT_ARTICLE_WEIGHT
        or metadata.get("super_important_linear_c") != config.RELEVANCE_LINEAR_C
        or metadata.get("label_contract") != TWO_HEAD_LABEL_CONTRACT
        or not isinstance(train_counts, dict)
        or set(train_counts) != TWO_HEAD_TRAIN_COUNT_KEYS
        or not all(isinstance(count, int) for count in train_counts.values())
    ):
        raise RuntimeError(
            "Relevance artifact is not compatible with the two-head model."
        )
    return relevance_classifier, super_important_classifier


def combine_probabilities(
    relevance_probabilities: np.ndarray,
    super_important_probabilities: np.ndarray,
    bonus_strength: float,
) -> np.ndarray:
    """Apply a bounded preference bonus above the classifier decision threshold."""
    if relevance_probabilities.shape != super_important_probabilities.shape:
        raise ValueError("Relevance and super-important probabilities must align.")
    if not math.isfinite(bonus_strength) or bonus_strength < 0:
        raise ValueError("bonus_strength must be finite and nonnegative.")
    preference_signal = np.clip(
        (super_important_probabilities - SUPER_IMPORTANT_DECISION_THRESHOLD)
        / (1 - SUPER_IMPORTANT_DECISION_THRESHOLD),
        0,
        1,
    )
    return (
        relevance_probabilities
        * (1 + bonus_strength * preference_signal)
        / (1 + bonus_strength)
    )


def save_artifact(
    model_path: str,
    classifier: LogisticRegression,
    train_counts: dict[str, int],
    pipeline_label: str = "relevance",
) -> None:
    """Persist the classifier and minimal metadata for later inference."""
    pipeline_name = _pipeline_name(pipeline_label)
    logger.info(f"Saving {pipeline_name} artifact to {model_path}")
    path = Path(model_path)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, path / CLASSIFIER_FILENAME)
    metadata = {
        "backend": "embedding_linear",
        "model_name": config.RELEVANCE_MODEL_NAME,
        "encoder_cache_path": str(get_encoder_cache_path()),
        "max_length": config.RELEVANCE_MAX_LENGTH,
        "text_prep_mode": config.RELEVANCE_TEXT_PREP_MODE,
        "prep_version": config.RELEVANCE_PREP_VERSION,
        "linear_c": config.RELEVANCE_LINEAR_C,
        "batch_size": config.RELEVANCE_ENCODER_BATCH_SIZE,
        "train_counts": train_counts,
    }
    (path / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info(f"Saved {pipeline_name} artifact to {model_path}")


def load_classifier(
    model_path: str,
    pipeline_label: str = "relevance",
) -> LogisticRegression:
    """Load a previously saved logistic-regression classifier."""
    logger.info(
        f"Loading {_pipeline_name(pipeline_label)} classifier from {model_path}"
    )
    return joblib.load(Path(model_path) / CLASSIFIER_FILENAME)


def predict_probabilities_from_embeddings(
    embeddings: np.ndarray,
    classifier: Any,
) -> np.ndarray:
    """Predict clipped positive-class probabilities from prepared embeddings."""
    if not len(embeddings):
        return np.array([], dtype=float)
    probs = classifier.predict_proba(embeddings)[:, 1]
    return np.clip(probs, 1e-7, 1 - 1e-7)


async def predict_probabilities(
    articles: list[Article],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    classifier: Any,
    device: torch.device,
    pipeline_label: str = "relevance",
) -> np.ndarray:
    """Predict positive-class probabilities for a batch of articles."""
    if not articles:
        return np.array([], dtype=float)

    embeddings = await encode_articles(
        articles,
        tokenizer,
        model,
        device,
        pipeline_label=pipeline_label,
    )
    return predict_probabilities_from_embeddings(embeddings, classifier)


def peak_vram_gb() -> float:
    """Return the peak CUDA memory allocated in GiB for the current process."""
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / math.pow(1024, 3))
