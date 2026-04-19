import json
import logging
import math
from pathlib import Path

from huggingface_hub import snapshot_download
import joblib  # type: ignore[import-untyped]
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from feedoscope import config, relevance_text
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

CLASSIFIER_FILENAME = "classifier.joblib"
METADATA_FILENAME = "metadata.json"
ENCODER_CACHE_ROOT = Path("models/relevance_encoder")
ENCODER_READY_FILENAME = ".snapshot_complete"


def get_encoder_cache_path() -> Path:
    """Return the shared on-disk cache path for the configured encoder."""
    return ENCODER_CACHE_ROOT / config.RELEVANCE_MODEL_NAME.replace("/", "--")


def ensure_local_encoder() -> str:
    """Download the shared encoder snapshot once and reuse it afterward."""
    encoder_path = get_encoder_cache_path()
    ready_marker = encoder_path / ENCODER_READY_FILENAME

    if ready_marker.exists():
        logger.info(f"Using cached relevance encoder at {encoder_path}")
        return str(encoder_path)

    logger.info(
        f"Downloading relevance encoder {config.RELEVANCE_MODEL_NAME} to "
        f"{encoder_path}"
    )
    encoder_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.RELEVANCE_MODEL_NAME,
        local_dir=str(encoder_path),
    )
    ready_marker.write_text("\n")
    logger.info(f"Relevance encoder cached at {encoder_path}")
    return str(encoder_path)


def load_encoder(
    device: torch.device,
) -> tuple[PreTrainedTokenizerBase, torch.nn.Module]:
    """Load the shared embedding tokenizer and encoder onto the target device."""
    encoder_path = ensure_local_encoder()
    logger.info(f"Loading relevance encoder from {encoder_path}")
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
    logger.info("Relevance encoder loaded successfully")
    return tokenizer, model


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average token embeddings across non-padding positions."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_articles(
    articles: list[Article],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Prepare article text and convert it into normalized embeddings."""
    logger.info(
        f"Preparing relevance text for {len(articles)} articles using "
        f"{config.RELEVANCE_TEXT_PREP_MODE}"
    )
    texts = relevance_text.prepare_articles_text(
        articles,
        tokenizer=tokenizer,
        max_length=config.RELEVANCE_MAX_LENGTH,
        mode=config.RELEVANCE_TEXT_PREP_MODE,
    )
    logger.info(f"Prepared relevance text for {len(texts)} articles")
    return encode_texts(texts, tokenizer, model, device)


def encode_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Encode raw text batches into normalized dense vectors."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embeddings: list[np.ndarray] = []
    total_batches = math.ceil(len(texts) / config.RELEVANCE_ENCODER_BATCH_SIZE)

    with torch.no_grad():
        for batch_index, start in enumerate(
            range(0, len(texts), config.RELEVANCE_ENCODER_BATCH_SIZE),
            start=1,
        ):
            batch = texts[start : start + config.RELEVANCE_ENCODER_BATCH_SIZE]
            logger.info(f"Encoding batch {batch_index}/{total_batches}")
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


def build_sample_weights(articles: list[Article]) -> np.ndarray:
    """Boost starred or upvoted articles using the configured weight."""
    return np.array(
        [
            config.EXCELLENT_WEIGHT if (article.vote == 1 or article.starred) else 1.0
            for article in articles
        ],
        dtype=float,
    )


def fit_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
) -> LogisticRegression:
    """Fit the logistic-regression head on top of frozen embeddings."""
    logger.info(
        f"Fitting logistic regression on {len(labels)} rows with "
        f"C={config.RELEVANCE_LINEAR_C}"
    )
    classifier = LogisticRegression(
        max_iter=4000,
        C=config.RELEVANCE_LINEAR_C,
        random_state=42,
    )
    classifier.fit(embeddings, labels, sample_weight=sample_weights)
    logger.info("Logistic regression fit completed")
    return classifier


def save_artifact(
    model_path: str,
    classifier: LogisticRegression,
    train_counts: dict[str, int],
) -> None:
    """Persist the classifier and minimal metadata for later inference."""
    logger.info(f"Saving relevance artifact to {model_path}")
    path = Path(model_path)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, path / CLASSIFIER_FILENAME)
    metadata = {
        "backend": "embedding_linear",
        "model_name": config.RELEVANCE_MODEL_NAME,
        "encoder_cache_path": str(get_encoder_cache_path()),
        "max_length": config.RELEVANCE_MAX_LENGTH,
        "text_prep_mode": config.RELEVANCE_TEXT_PREP_MODE,
        "linear_c": config.RELEVANCE_LINEAR_C,
        "batch_size": config.RELEVANCE_ENCODER_BATCH_SIZE,
        "train_counts": train_counts,
    }
    (path / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info(f"Saved relevance artifact to {model_path}")


def load_classifier(model_path: str) -> LogisticRegression:
    """Load a previously saved logistic-regression classifier."""
    logger.info(f"Loading relevance classifier from {model_path}")
    return joblib.load(Path(model_path) / CLASSIFIER_FILENAME)


def predict_probabilities(
    articles: list[Article],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    classifier: LogisticRegression,
    device: torch.device,
) -> np.ndarray:
    """Predict positive-class probabilities for a batch of articles."""
    if not articles:
        return np.array([], dtype=float)

    embeddings = encode_articles(articles, tokenizer, model, device)
    probs = classifier.predict_proba(embeddings)[:, 1]
    return np.clip(probs, 1e-7, 1 - 1e-7)


def peak_vram_gb() -> float:
    """Return the peak CUDA memory allocated in GiB for the current process."""
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / math.pow(1024, 3))
