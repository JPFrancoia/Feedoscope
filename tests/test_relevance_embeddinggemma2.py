import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments import relevance_embeddinggemma2 as benchmark

from feedoscope import relevance_embedding
from feedoscope.data_registry import data_registry as dr


def article_frame() -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01", tz="UTC")
    return pd.DataFrame(
        [
            {
                "article_id": day * 2 + label,
                "published_at": start + pd.Timedelta(days=day),
                "title": f"shared title {day} {'science' if day % 2 else 'history'}",
                "content": f"<p>shared body {day} {'science' if day % 2 else 'history'}</p>",
                "status": "read" if label else "unread",
                "vote": 0 if label else -1,
                "starred": label == 1 and day == 1,
            }
            for day in range(12)
            for label in [0, 1]
        ]
    )[benchmark.SOURCE_COLUMNS]


def test_label_and_history_boundaries() -> None:
    frame = article_frame()
    now = pd.Timestamp("2026-02-01", tz="UTC")
    frame.loc[0, "published_at"] = now - pd.Timedelta(days=1095)
    frame.loc[1, "published_at"] = (
        now - pd.Timedelta(days=1095) + pd.Timedelta(seconds=1)
    )
    frame.loc[2, ["status", "vote"]] = ["unread", 0]
    selected = benchmark.select_rows(frame, now)
    assert 0 not in selected.article_id.values
    assert 1 in selected.article_id.values
    assert 2 not in selected.article_id.values
    assert list(selected.columns) == benchmark.SOURCE_COLUMNS


def test_duplicate_ids_are_rejected() -> None:
    frame = article_frame()
    frame.loc[1, "article_id"] = frame.loc[0, "article_id"]
    with pytest.raises(ValueError, match="unique"):
        benchmark.frame_to_articles(frame)


def test_splits_are_strict_and_holdouts_are_disjoint() -> None:
    articles = benchmark.frame_to_articles(article_frame())
    folds = benchmark.make_splits(articles, validation_size=2)
    by_id = {a.article_id: a for a in articles}
    previous_eval = set()
    for fold in folds:
        fit, held = set(fold["fit_ids"]), set(fold["eval_ids"])
        assert fit.isdisjoint(held)
        assert previous_eval.isdisjoint(held)
        assert max(by_id[i].date_entered for i in fit) < min(
            by_id[i].date_entered for i in held
        )
        assert fold["eval_positive"] == fold["eval_negative"] == 2
        previous_eval.update(held)
    assert dt.datetime.fromisoformat(folds[0]["cutoff"]) > dt.datetime.fromisoformat(
        folds[1]["cutoff"]
    )


def test_gemma2_disables_unused_encoders() -> None:
    config = SimpleNamespace(vision_config=object(), audio_config=object())
    benchmark.text_only_config(config, "gemma2")
    assert config.vision_config is config.audio_config is None


def test_vector_validation_rejects_nan_zero_and_wrong_width() -> None:
    valid = np.ones((3, 768), dtype=np.float32) / np.sqrt(768)
    benchmark.validate_vectors(valid, 3)
    for invalid in [np.zeros((3, 768)), np.full((3, 768), np.nan), np.ones((3, 512))]:
        with pytest.raises(ValueError):
            benchmark.validate_vectors(invalid, 3)


def test_checksum_verification_and_output_boundary(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "articles.parquet").write_bytes(b"fixture")
    (source / "manifest.json").write_text("{}")
    sums = "\n".join(
        f"{benchmark.sha256(source / name)}  {name}"
        for name in ["articles.parquet", "manifest.json"]
    )
    (source / "SHA256SUMS").write_text(sums)
    assert len(benchmark.verify_snapshot(source)) == 2
    with pytest.raises(ValueError, match="outside"):
        benchmark.prepare(source, source / "output")
    (source / "articles.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        benchmark.verify_snapshot(source)


def test_embedding_id_order_must_match(tmp_path: Path) -> None:
    (tmp_path / "protocol.json").write_text("{}")
    destination = tmp_path / "gemma1"
    destination.mkdir()
    np.save(destination / "article_ids.npy", [2, 1])
    metadata = {
        "protocol_sha256": benchmark.sha256(tmp_path / "protocol.json"),
        "model": benchmark.MODELS["gemma1"][0],
        "revision": benchmark.MODELS["gemma1"][1],
        "hashes": {
            "article_ids.npy": benchmark.sha256(destination / "article_ids.npy")
        },
    }
    (destination / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="input order"):
        benchmark.load_vectors(tmp_path, "gemma1", np.array([1, 2]))


def test_head_uses_fit_only_vocabulary_and_never_opens_database(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Benchmark must not open a database connection")

    monkeypatch.setattr(dr.global_pool, "open", forbidden)
    monkeypatch.setattr(dr.global_pool, "connection", forbidden)
    articles = benchmark.frame_to_articles(article_frame())
    vectors = np.ones((len(articles), 768), dtype=np.float32) / np.sqrt(768)
    weights = relevance_embedding.build_relevance_sample_weights(articles)
    assert weights[3] == 20
    model = relevance_embedding.fit_relevance_model(
        articles[:-2],
        vectors[:-2],
        vectors[:-2],
        np.array([int(a.vote >= 0) for a in articles[:-2]]),
        weights[:-2],
    )
    articles[-1].title = "sealedholdouttoken"
    articles[-1].content = "<p>sealedbodytoken</p>"
    features = relevance_embedding.build_features(
        articles[-2:],
        vectors[-2:],
        vectors[-2:],
        model.title_vectorizer,
        model.body_vectorizer,
    )
    assert features.shape[0] == 2
    assert "sealedholdouttoken" not in model.title_vectorizer.vocabulary_
    assert "sealedbodytoken" not in model.body_vectorizer.vocabulary_


def test_paired_bootstrap_is_deterministic_and_zero_for_identical_scores() -> None:
    labels = np.array([0, 1] * 30)
    scores = np.linspace(0, 1, len(labels))
    intervals = benchmark.bootstrap_differences(labels, scores, scores, samples=10)
    assert all(
        value == {"ci95_lower": 0, "ci95_upper": 0} for value in intervals.values()
    )
    candidate = labels * 0.8 + scores * 0.1
    assert benchmark.bootstrap_differences(labels, scores, candidate, 10) == (
        benchmark.bootstrap_differences(labels, scores, candidate, 10)
    )


def test_complete_comparison_writes_only_experiment_results(
    tmp_path: Path, monkeypatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Benchmark must not use the database")

    monkeypatch.setattr(dr.global_pool, "open", forbidden)
    monkeypatch.setattr(dr.global_pool, "connection", forbidden)
    frame = article_frame()
    frame.to_parquet(tmp_path / "articles.parquet", index=False)
    protocol = {
        "articles_sha256": benchmark.sha256(tmp_path / "articles.parquet"),
        "article_count": len(frame),
        "models": benchmark.MODELS,
        "folds": benchmark.make_splits(
            benchmark.frame_to_articles(frame), validation_size=2
        ),
    }
    benchmark.write_json(tmp_path / "protocol.json", protocol)
    vectors = np.ones((len(frame), 768), dtype=np.float32) / np.sqrt(768)
    for key, (name, revision) in benchmark.MODELS.items():
        destination = tmp_path / key
        destination.mkdir()
        for field in ["title", "body"]:
            np.save(destination / f"{field}.npy", vectors)
        np.save(destination / "article_ids.npy", frame.article_id.to_numpy())
        benchmark.write_json(
            destination / "metadata.json",
            {
                "protocol_sha256": benchmark.sha256(tmp_path / "protocol.json"),
                "model": name,
                "revision": revision,
                "hashes": {
                    name: benchmark.sha256(destination / name)
                    for name in ["title.npy", "body.npy", "article_ids.npy"]
                },
            },
        )
    before = benchmark.sha256(tmp_path / "articles.parquet")
    result = benchmark.compare(tmp_path, bootstrap_samples=2)
    assert benchmark.sha256(tmp_path / "articles.parquet") == before
    assert result["production_writes"] is False
    assert result["deployment_eligible"] is False
    assert (tmp_path / "predictions.npz").is_file()
    assert json.loads((tmp_path / "results.json").read_text()) == result
    for differences in result["candidate_minus_baseline"].values():
        assert all(differences[metric] == 0 for metric in benchmark.METRICS)
