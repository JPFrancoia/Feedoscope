"""Compare frozen relevance encoders without database access."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import time

# Never pass production credentials to imported configuration or database helpers.
os.environ.update(
    DATABASE_URL="postgresql://benchmark:benchmark@127.0.0.1:1/benchmark",
    RELEVANCE_EMBEDDING_PROMPT="task: classification | query: ",
    RELEVANCE_LINEAR_C="5",
    IMPORTANT_ARTICLE_WEIGHT="20",
    RELEVANCE_MAX_LENGTH="2048",
)

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from feedoscope import eval_models, relevance_embedding
from feedoscope.entities import Article

logger = logging.getLogger(__name__)

MODELS = {
    "gemma1": (
        "google/embeddinggemma-300m",
        "57c266a740f537b4dc058e1b0cda161fd15afa75",
    ),
    "gemma2": (
        "google/embeddinggemma-2",
        "f9972ac22e696a105c482202646979e33e4b5a1e",
    ),
}
MODELS["pplx"] = (
    "perplexity-ai/pplx-embed-v2-context-0.6b",
    "6d1fddab0a18478acc0a51edd1e81333fe441095",
)
MODEL_DIMENSIONS = {"gemma1": 768, "gemma2": 768, "pplx": 1024}
SOURCE_COLUMNS = [
    "article_id",
    "published_at",
    "title",
    "content",
    "status",
    "vote",
    "starred",
]
METRICS = ("average_precision", "roc_auc", "precision_at_50")


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write a new private JSON artifact without replacing another file."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def verify_snapshot(snapshot: Path) -> dict[str, str]:
    """Verify every source file before reading private article data."""
    hashes = {}
    for line in (snapshot / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = snapshot / name
        if path.resolve().parent != snapshot.resolve():
            raise ValueError("Snapshot checksum contains an unsafe path")
        if sha256(path) != expected:
            raise ValueError(f"Snapshot checksum mismatch: {name}")
        hashes[name] = expected
    if not {"articles.parquet", "manifest.json"}.issubset(hashes):
        raise ValueError("Snapshot checksums omit required files")
    return hashes


def frame_to_articles(frame: pd.DataFrame) -> list[Article]:
    """Convert the minimal benchmark fields to production article objects."""
    if frame.article_id.duplicated().any():
        raise ValueError("Article IDs must be unique")
    return [
        Article(
            article_id=int(row.article_id),
            date_entered=row.published_at.to_pydatetime(),
            title=row.title or "",
            content=row.content or "",
            status=row.status,
            vote=int(row.vote),
            starred=bool(row.starred),
            feed_name="",
            link="",
            author="",
            last_read=None,
            tags=[],
        )
        for row in frame.itertuples(index=False)
    ]


def select_rows(frame: pd.DataFrame, snapshot_time: pd.Timestamp) -> pd.DataFrame:
    """Apply the production labels and fixed three-year history."""
    if frame.article_id.duplicated().any():
        raise ValueError("Article IDs must be unique")
    if not frame.vote.isin([-1, 0, 1]).all():
        raise ValueError("Snapshot contains an unsupported vote")
    labeled = (frame.vote == -1) | ((frame.status == "read") & (frame.vote >= 0))
    recent = frame.published_at > snapshot_time - pd.Timedelta(days=1095)
    return frame.loc[labeled & recent, SOURCE_COLUMNS].sort_values("article_id")


def make_splits(
    articles: list[Article], validation_size: int = 150, fold_count: int = 3
) -> list[dict]:
    """Build disjoint newest-to-oldest holdouts with strict time boundaries."""
    good = [a for a in articles if a.status == "read" and a.vote >= 0]
    bad = [a for a in articles if a.vote == -1]
    folds = []
    for index in range(fold_count):
        good_fit, bad_fit, good_eval, bad_eval, cutoff = (
            eval_models.build_forward_holdout(good, bad, validation_size)
        )
        if not good_fit or not bad_fit:
            raise ValueError("Each fit partition must contain both classes")
        folds.append(
            {
                "name": "primary" if index == 0 else f"earlier-{index}",
                "cutoff": cutoff.isoformat(),
                "fit_ids": [a.article_id for a in good_fit + bad_fit],
                "eval_ids": [a.article_id for a in good_eval + bad_eval],
                "fit_positive": len(good_fit),
                "fit_negative": len(bad_fit),
                "eval_positive": len(good_eval),
                "eval_negative": len(bad_eval),
            }
        )
        good, bad = good_fit, bad_fit
    return folds


def prepare(snapshot: Path, output: Path, candidate: str = "gemma2") -> None:
    """Create a minimal immutable input for both encoders."""
    if output.resolve().is_relative_to(snapshot.resolve()):
        raise ValueError("Output must be outside the source snapshot")
    if candidate not in ("gemma2", "pplx"):
        raise ValueError("Unsupported candidate")
    hashes = verify_snapshot(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    frame = pd.read_parquet(snapshot / "articles.parquet", columns=SOURCE_COLUMNS)
    frame = select_rows(frame, pd.Timestamp(manifest["snapshot_time_utc"]))
    folds = make_splits(frame_to_articles(frame))
    used_ids = set().union(*(set(f["fit_ids"] + f["eval_ids"]) for f in folds))
    frame = frame[frame.article_id.isin(used_ids)].reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output / "articles.parquet", index=False)
    write_json(
        output / "protocol.json",
        {
            "source_snapshot": snapshot.name,
            "source_hashes": hashes,
            "snapshot_time_utc": manifest["snapshot_time_utc"],
            "articles_sha256": sha256(output / "articles.parquet"),
            "article_count": len(frame),
            "models": {key: MODELS[key] for key in ("gemma1", candidate)},
            "candidate": candidate,
            "candidate_contract": (
                {
                    "prompt": "[D] ",
                    "pooling": "document chunk mean and contextual_projection",
                    "output_quantization": "native tanh int8 returned as float32",
                    "normalized": True,
                    "empty_fields": "native zero vectors",
                    "vision": "removed on CPU before GPU transfer",
                    "dimension": 1024,
                }
                if candidate == "pplx"
                else {"dimension": 768}
            ),
            "folds": folds,
            "dtype": "float32",
            "prompt": "task: classification | query: ",
            "title_max_length": 512,
            "body_max_length": 2048,
            "classifier": {"C": 5, "max_iter": 4000, "random_state": 42},
            "important_positive_weight": 20,
            "tfidf": relevance_embedding.get_tfidf_config(),
            "bootstrap_samples": 10000,
            "bootstrap_seed": 42,
            "production_writes": False,
            "deployment_eligible": False,
        },
    )
    logger.info(f"Prepared {len(frame)} articles in {output}")


def read_input(root: Path) -> tuple[pd.DataFrame, dict]:
    """Read only the input that matches the recorded protocol."""
    protocol = json.loads((root / "protocol.json").read_text())
    if sha256(root / "articles.parquet") != protocol["articles_sha256"]:
        raise ValueError("Benchmark input checksum mismatch")
    candidate = protocol.get("candidate", "gemma2")
    if candidate not in ("gemma2", "pplx") or protocol["models"] != {
        key: list(MODELS[key]) for key in ("gemma1", candidate)
    }:
        raise ValueError("Model revisions differ from the benchmark protocol")
    frame = pd.read_parquet(root / "articles.parquet")
    if list(frame.columns) != SOURCE_COLUMNS or len(frame) != protocol["article_count"]:
        raise ValueError("Benchmark input schema or count mismatch")
    return frame, protocol


def text_only_config(config: object, model_key: str) -> object:
    """Disable the new model's unused encoders before construction."""
    if model_key == "gemma2":
        config.vision_config = None
        config.audio_config = None
    return config


def validate_vectors(
    vectors: np.ndarray,
    count: int,
    dimension: int = 768,
    empty_rows: list[int] | None = None,
) -> None:
    """Reject incomplete, nonfinite, or unnormalized embedding arrays."""
    if vectors.shape != (count, dimension) or not np.isfinite(vectors).all():
        raise ValueError(f"Embeddings must be finite {dimension}-value vectors")
    expected_norms = np.ones(count)
    if empty_rows:
        expected_norms[empty_rows] = 0
    if not np.allclose(np.linalg.norm(vectors, axis=1), expected_norms, atol=1e-4):
        raise ValueError("Embeddings must have unit length, except known empty fields")


def encode_pplx_batch(model: object, texts: list[str], limit: int) -> np.ndarray:
    """Use native document pooling and quantization with a bounded token span."""
    features = model.prepare_inputs([[text] for text in texts], task="document")
    features["input_ids"] = features["input_ids"][:, :limit]
    features["attention_mask"] = features["attention_mask"][:, :limit]
    features["chunk_token_spans"] = features["chunk_token_spans"].clamp(max=limit)
    features = {key: value.to(model.device) for key, value in features.items()}
    projected = model._pool(features, "document")
    vectors = torch.round(torch.tanh(projected) * 127).clamp(-128, 127)
    return torch.nn.functional.normalize(vectors, dim=-1)[:, 0].cpu().numpy()


def encode(
    root: Path, model_key: str, smoke: bool = False, batch_size: int = 4
) -> None:
    """Encode title and body text into isolated local artifacts."""
    if not torch.cuda.is_available():
        raise RuntimeError("The approved benchmark requires a CUDA GPU")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    frame, protocol = read_input(root)
    if smoke:
        # Include long bodies and empty fields before the complete GPU run.
        frame = pd.concat(
            [frame.head(2), frame.loc[frame.content.str.len().nlargest(2).index]]
        ).drop_duplicates("article_id")
    if model_key not in protocol["models"]:
        raise ValueError("Encoder is not part of this benchmark protocol")
    articles = frame_to_articles(frame)
    dimension = MODEL_DIMENSIONS[model_key]
    model_name, revision = MODELS[model_key]
    destination = root / (f"{model_key}-smoke" if smoke else model_key)
    destination.mkdir(exist_ok=False)
    cache = root / "huggingface"
    torch.cuda.reset_peak_memory_stats()
    load_start = time.monotonic()
    model_config = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache,
        trust_remote_code=model_key == "pplx",
    )
    model_config = text_only_config(model_config, model_key)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, cache_dir=cache
    )
    model, loading_info = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        config=model_config,
        cache_dir=cache,
        dtype=torch.float32,
        attn_implementation="sdpa",
        output_loading_info=True,
        trust_remote_code=model_key == "pplx",
    )
    if loading_info["missing_keys"] or loading_info.get("mismatched_keys"):
        raise ValueError("Model weights do not match the pinned architecture")
    if model_key == "gemma2" and (
        model.vision_tower is not None or model.audio_tower is not None
    ):
        raise ValueError("Unused modality encoders must remain disabled")
    removed_parameters = 0
    if model_key == "pplx":
        removed_parameters = sum(p.numel() for p in model.visual.parameters())
        model.visual = None
        model.tokenizer = tokenizer
        if any(name.startswith("visual.") for name, _ in model.named_parameters()):
            raise ValueError("Image encoder parameters must remain absent")
    model = model.to("cuda").eval()
    parameter_count = sum(p.numel() for p in model.parameters())
    load_seconds = time.monotonic() - load_start
    field_times = {}
    empty_fields = {}
    torch.cuda.synchronize()
    encode_start = time.monotonic()
    for field, limit in [("title", 512), ("body", 2048)]:
        texts = relevance_embedding.prepare_articles_text(articles, field)
        if model_key == "pplx":
            prompt = protocol["prompt"]
            if any(not text.startswith(prompt) for text in texts):
                raise ValueError("Unexpected field preparation prompt")
            texts = [text[len(prompt) :] for text in texts]
        empty_fields[field] = (
            [index for index, text in enumerate(texts) if not text]
            if model_key == "pplx"
            else []
        )
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        vectors = np.empty((len(texts), dimension), dtype=np.float32)
        field_start = time.monotonic()
        next_log = 0
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch = [texts[index] for index in indices]
                if model_key == "pplx":
                    vectors[indices] = encode_pplx_batch(model, batch, limit)
                else:
                    inputs = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=limit,
                        return_tensors="pt",
                    ).to("cuda")
                    outputs = model(**inputs)
                    pooled = relevance_embedding.mean_pool(
                        outputs.last_hidden_state.float(), inputs["attention_mask"]
                    )
                    pooled = torch.nn.functional.normalize(pooled, dim=1)
                    vectors[indices] = pooled.cpu().numpy()
                completed = start + len(indices)
                if completed >= next_log or completed == len(order):
                    elapsed = time.monotonic() - field_start
                    eta = elapsed * (len(order) - completed) / completed
                    logger.info(
                        f"{model_key} {field}: {completed}/{len(order)}, "
                        f"elapsed={elapsed:.1f}s, remaining={eta:.1f}s"
                    )
                    next_log = completed + max(1, len(order) // 20)
        torch.cuda.synchronize()
        field_times[field] = time.monotonic() - field_start
        validate_vectors(vectors, len(articles), dimension, empty_fields[field])
        np.save(destination / f"{field}.npy", vectors, allow_pickle=False)
    torch.cuda.synchronize()
    np.save(
        destination / "article_ids.npy", frame.article_id.to_numpy(), allow_pickle=False
    )
    write_json(
        destination / "metadata.json",
        {
            "model": model_name,
            "revision": revision,
            "protocol_sha256": sha256(root / "protocol.json"),
            "input_sha256": protocol["articles_sha256"],
            "parameter_count": parameter_count,
            "disabled_encoders": (
                ["vision", "audio"]
                if model_key == "gemma2"
                else ["vision"] if model_key == "pplx" else []
            ),
            "removed_parameters": removed_parameters,
            "dimension": dimension,
            "empty_fields": empty_fields,
            "batch_size": batch_size,
            "dtype": "float32",
            "gpu": torch.cuda.get_device_name(),
            "load_seconds": load_seconds,
            "encode_seconds": time.monotonic() - encode_start,
            "field_seconds": field_times,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["torch", "transformers", "numpy", "scikit-learn", "pandas"]
            },
            "hashes": {
                name: sha256(destination / name)
                for name in ["title.npy", "body.npy", "article_ids.npy"]
            },
        },
    )


def load_vectors(
    root: Path, key: str, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Verify model artifacts and their exact row order."""
    directory = root / key
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["protocol_sha256"] != sha256(root / "protocol.json"):
        raise ValueError("Embedding protocol checksum mismatch")
    if (metadata["model"], metadata["revision"]) != MODELS[key]:
        raise ValueError("Embedding model revision mismatch")
    for name, expected in metadata["hashes"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Embedding checksum mismatch: {name}")
    if not np.array_equal(np.load(directory / "article_ids.npy"), ids):
        raise ValueError("Embedding article IDs do not match the input order")
    title = np.load(directory / "title.npy", allow_pickle=False)
    body = np.load(directory / "body.npy", allow_pickle=False)
    dimension = MODEL_DIMENSIONS[key]
    empty_fields = metadata.get("empty_fields", {})
    if key == "pplx":
        frame, protocol = read_input(root)
        articles = frame_to_articles(frame)
        expected_empty = {}
        for field in ("title", "body"):
            texts = relevance_embedding.prepare_articles_text(articles, field)
            expected_empty[field] = [
                index for index, text in enumerate(texts) if text == protocol["prompt"]
            ]
        if empty_fields != expected_empty:
            raise ValueError("Empty vector rows differ from the input fields")
    validate_vectors(title, len(ids), dimension, empty_fields.get("title"))
    validate_vectors(body, len(ids), dimension, empty_fields.get("body"))
    return title, body, metadata


def bootstrap_differences(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    samples: int = 10000,
) -> dict:
    """Estimate paired metric differences with fixed class counts."""
    rng = np.random.default_rng(42)
    classes = [np.flatnonzero(labels == value) for value in [0, 1]]
    if not all(len(indices) for indices in classes):
        raise ValueError("Bootstrap requires both classes")
    values = {metric: [] for metric in METRICS}
    for _ in range(samples):
        indices = np.concatenate([rng.choice(c, size=len(c)) for c in classes])
        first = eval_models.compute_relevance_metrics(
            labels[indices], baseline[indices]
        )
        second = eval_models.compute_relevance_metrics(
            labels[indices], candidate[indices]
        )
        for metric in METRICS:
            values[metric].append(second[metric] - first[metric])
    return {
        metric: {
            "ci95_lower": float(np.quantile(differences, 0.025)),
            "ci95_upper": float(np.quantile(differences, 0.975)),
        }
        for metric, differences in values.items()
    }


def compare(root: Path, bootstrap_samples: int = 10000) -> dict:
    """Fit unchanged production classifiers and compare aligned predictions."""
    frame, protocol = read_input(root)
    articles = frame_to_articles(frame)
    ids = frame.article_id.to_numpy()
    position = {int(article_id): index for index, article_id in enumerate(ids)}
    results = {}
    predictions = {}
    candidate = protocol.get("candidate", "gemma2")
    for key in protocol["models"]:
        title, body, metadata = load_vectors(root, key, ids)
        results[key] = {"encoder": metadata, "folds": {}}
        predictions[key] = {}
        for fold in protocol["folds"]:
            fit = np.array([position[i] for i in fold["fit_ids"]])
            held = np.array([position[i] for i in fold["eval_ids"]])
            fit_articles = [articles[i] for i in fit]
            eval_articles = [articles[i] for i in held]
            labels = np.array([int(a.vote >= 0) for a in fit_articles])
            truth = np.array([int(a.vote >= 0) for a in eval_articles])
            start = time.monotonic()
            model = relevance_embedding.fit_relevance_model(
                fit_articles,
                title[fit],
                body[fit],
                labels,
                relevance_embedding.build_relevance_sample_weights(fit_articles),
            )
            if np.any(model.classifier.n_iter_ >= model.classifier.max_iter):
                raise ValueError("Classifier did not converge")
            features = relevance_embedding.build_features(
                eval_articles,
                title[held],
                body[held],
                model.title_vectorizer,
                model.body_vectorizer,
            )
            probabilities = model.classifier.predict_proba(features)[:, 1]
            if not np.isfinite(probabilities).all():
                raise ValueError("Classifier returned nonfinite probabilities")
            metrics = eval_models.compute_relevance_metrics(truth, probabilities)
            results[key]["folds"][fold["name"]] = {
                "metrics": metrics,
                "fit_seconds": time.monotonic() - start,
                "iterations": model.classifier.n_iter_.tolist(),
            }
            predictions[key][fold["name"]] = probabilities
            logger.info(f"{key} {fold['name']}: {metrics}")
    differences = {}
    for fold in protocol["folds"]:
        name = fold["name"]
        truth = np.array(
            [int(articles[position[i]].vote >= 0) for i in fold["eval_ids"]]
        )
        differences[name] = {
            metric: (
                results[candidate]["folds"][name]["metrics"][metric]
                - results["gemma1"]["folds"][name]["metrics"][metric]
            )
            for metric in METRICS
        }
        differences[name]["paired_bootstrap"] = bootstrap_differences(
            truth,
            predictions["gemma1"][name],
            predictions[candidate][name],
            bootstrap_samples,
        )
    with (root / "predictions.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            **{
                f"{key}_{fold['name']}": predictions[key][fold["name"]]
                for key in protocol["models"]
                for fold in protocol["folds"]
            },
        )
    result = {
        "protocol_sha256": sha256(root / "protocol.json"),
        "models": results,
        "candidate_minus_baseline": differences,
        "bootstrap_samples": bootstrap_samples,
        "production_writes": False,
        "deployment_eligible": False,
    }
    write_json(root / "results.json", result)
    return result


def main() -> None:
    """Run one isolated benchmark stage."""
    os.umask(0o077)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--snapshot", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument(
        "--candidate", choices=("gemma2", "pplx"), default="gemma2"
    )
    encode_parser = subparsers.add_parser("encode")
    encode_parser.add_argument("--root", type=Path, required=True)
    encode_parser.add_argument("--model", choices=MODELS, required=True)
    encode_parser.add_argument("--smoke", action="store_true")
    encode_parser.add_argument("--batch-size", type=int, default=4)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.snapshot, args.output, args.candidate)
    elif args.command == "encode":
        encode(args.root, args.model, args.smoke, args.batch_size)
    else:
        compare(args.root)


if __name__ == "__main__":
    main()
