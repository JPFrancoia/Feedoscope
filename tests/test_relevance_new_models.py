import json

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch

from experiments import relevance_new_models
from experiments.relevance_new_models import (
    HEAD_CONTRACTS,
    MODEL_CONTRACTS,
    _artifact_identity,
    _embedding_artifact_path,
    _load_embedding_artifact,
    _load_frozen_snapshot,
    _local_support_file_hashes,
    _new_classifier,
    _paired_bootstrap_average_precision_difference,
    _reference_probabilities,
    _require_balanced_evaluation,
    _save_embedding_artifact,
    _select_head_parameters,
    _weight_hashes,
    assign_stratified_split,
    last_token_pool,
    mean_pool,
    run_benchmark,
)


def test_assign_stratified_split_is_deterministic() -> None:
    data = pd.DataFrame(
        {
            "article_id": list(range(20)),
            "label": [0] * 10 + [1] * 10,
        }
    )

    first = assign_stratified_split(data, eval_size_per_class=2)
    second = assign_stratified_split(data, eval_size_per_class=2)

    assert first.equals(second)
    assert first.loc[first["split"] == "eval", "article_id"].tolist() == [3, 7, 13, 15]
    train_ids = set(first.loc[first["split"] == "train", "article_id"])
    eval_ids = set(first.loc[first["split"] == "eval", "article_id"])
    assert train_ids.isdisjoint(eval_ids)
    assert first.groupby(["label", "split"]).size().to_dict() == {
        (0, "eval"): 2,
        (0, "train"): 8,
        (1, "eval"): 2,
        (1, "train"): 8,
    }


def test_planned_model_contracts_are_pinned() -> None:
    task = "Classify RSS articles according to the reader's relevance preferences"
    expected = {
        "google/embeddinggemma-300m-classification": (
            "57c266a740f537b4dc058e1b0cda161fd15afa75",
            "task: classification | query: ",
            "mean",
            768,
        ),
        "jinaai/jina-embeddings-v5-text-nano-classification": (
            "a0129e9b8ea4c54f3dfd250793380d8d69058da3",
            "Document: ",
            "last_token",
            768,
        ),
        "Qwen/Qwen3-Embedding-4B": (
            "5cf2132abc99cad020ac570b19d031efec650f2b",
            f"Instruct: {task}\nQuery:",
            "last_token",
            2560,
        ),
        "Qwen/Qwen3-Embedding-0.6B": (
            "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            f"Instruct: {task}\nQuery:",
            "last_token",
            1024,
        ),
        "hotchpotch/bekko-embedding-v1-a8m": (
            "b24cde5de82214ada4c01f173b137c78160b13c6",
            "",
            "mean",
            384,
        ),
        "hotchpotch/bekko-embedding-v1-a25m": (
            "e0f3136db1b823ccbc67c4bea7d29f295516535b",
            "",
            "mean",
            384,
        ),
        "nvidia/Nemotron-3-Embed-1B-BF16": (
            "a5e0f804b9e90a1ca6784ecbf6e41595774fc834",
            "passage: ",
            "mean",
            2048,
        ),
    }

    for model_name, values in expected.items():
        contract = MODEL_CONTRACTS[model_name]
        assert (
            contract.revision,
            contract.prefix,
            contract.pooling,
            contract.expected_dimension,
        ) == values
        if model_name in {
            "google/embeddinggemma-300m-classification",
            "jinaai/jina-embeddings-v5-text-nano-classification",
        }:
            assert contract.attention is None
        else:
            assert contract.attention == "sdpa"


def _embedding_artifact_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    contract = MODEL_CONTRACTS["google/embeddinggemma-300m"]
    train = pd.DataFrame(
        {
            "article_id": [1, 2, 3, 4],
            "title": ["a", "b", "c", "d"],
            "content": ["a", "b", "c", "d"],
            "vote": [0, 0, 0, 0],
            "starred": [False, False, False, False],
            "label": [0, 1, 0, 1],
        }
    )
    evaluation = pd.DataFrame(
        {
            "article_id": [5, 6],
            "title": ["e", "f"],
            "content": ["e", "f"],
            "vote": [0, 0],
            "starred": [False, False],
            "label": [0, 1],
        }
    )
    train_embeddings = np.eye(4, contract.expected_dimension, dtype=np.float32)
    eval_embeddings = np.eye(2, contract.expected_dimension, dtype=np.float32)
    identity = _artifact_identity(
        "test-snapshot",
        {"train.parquet": "train", "eval.parquet": "eval"},
        contract,
        None,
    )
    return train, evaluation, train_embeddings, eval_embeddings, identity


def test_embedding_artifact_round_trip(tmp_path) -> None:
    train, evaluation, train_embeddings, eval_embeddings, identity = (
        _embedding_artifact_inputs()
    )
    contract = MODEL_CONTRACTS["google/embeddinggemma-300m"]
    artifact_path = _embedding_artifact_path(tmp_path, "test-snapshot", contract)

    artifact_sha256 = _save_embedding_artifact(
        artifact_path,
        identity,
        train_embeddings,
        eval_embeddings,
        train["label"].to_numpy(dtype=int),
        evaluation["label"].to_numpy(dtype=int),
        train["article_id"].to_numpy(dtype=np.int64),
        evaluation["article_id"].to_numpy(dtype=np.int64),
        2048,
    )
    loaded = _load_embedding_artifact(
        artifact_path, identity, train, evaluation, contract
    )

    assert np.array_equal(loaded[0], train_embeddings)
    assert np.array_equal(loaded[1], eval_embeddings)
    assert loaded[2] == artifact_sha256
    assert loaded[3] == 2048


def test_embedding_artifact_rejects_changed_snapshot_rows(tmp_path) -> None:
    train, evaluation, train_embeddings, eval_embeddings, identity = (
        _embedding_artifact_inputs()
    )
    contract = MODEL_CONTRACTS["google/embeddinggemma-300m"]
    artifact_path = _embedding_artifact_path(tmp_path, "test-snapshot", contract)
    _save_embedding_artifact(
        artifact_path,
        identity,
        train_embeddings,
        eval_embeddings,
        train["label"].to_numpy(dtype=int),
        evaluation["label"].to_numpy(dtype=int),
        train["article_id"].to_numpy(dtype=np.int64),
        evaluation["article_id"].to_numpy(dtype=np.int64),
        2048,
    )

    with pytest.raises(ValueError, match="labels do not match"):
        _load_embedding_artifact(
            artifact_path,
            identity,
            train.assign(label=[1, 0, 1, 0]),
            evaluation,
            contract,
        )


def test_corrupt_embedding_artifact_is_quarantined(tmp_path, monkeypatch) -> None:
    train, evaluation, _, _, _ = _embedding_artifact_inputs()
    contract = MODEL_CONTRACTS["google/embeddinggemma-300m"]
    artifact_path = _embedding_artifact_path(tmp_path, "test-snapshot", contract)
    artifact_path.write_bytes(b"")
    metadata = {"snapshot_id": "test-snapshot"}
    snapshot_hashes = {"train.parquet": "train", "eval.parquet": "eval"}
    monkeypatch.setattr(
        relevance_new_models,
        "_load_frozen_snapshot",
        lambda _: (metadata, train, evaluation, snapshot_hashes),
    )
    monkeypatch.setattr(
        relevance_new_models.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("Artifact regeneration began"),
    )
    monkeypatch.setattr(
        relevance_new_models, "_require_balanced_evaluation", lambda _: None
    )

    with pytest.raises(pytest.fail.Exception, match="regeneration began"):
        run_benchmark(
            snapshot=tmp_path,
            model_name="google/embeddinggemma-300m",
            output=tmp_path / "result.json",
            batch_size=4,
            dtype_name="float32",
            embeddings_dir=tmp_path,
        )

    assert not artifact_path.exists()
    assert artifact_path.with_suffix(".npz.invalid").exists()


def test_embedding_artifact_reuse_bypasses_model_loading(tmp_path, monkeypatch) -> None:
    train, evaluation, train_embeddings, eval_embeddings, identity = (
        _embedding_artifact_inputs()
    )
    contract = MODEL_CONTRACTS["google/embeddinggemma-300m"]
    artifact_path = _embedding_artifact_path(tmp_path, "test-snapshot", contract)
    artifact_sha256 = _save_embedding_artifact(
        artifact_path,
        identity,
        train_embeddings,
        eval_embeddings,
        train["label"].to_numpy(dtype=int),
        evaluation["label"].to_numpy(dtype=int),
        train["article_id"].to_numpy(dtype=np.int64),
        evaluation["article_id"].to_numpy(dtype=np.int64),
        2048,
    )
    metadata = {"snapshot_id": "test-snapshot"}
    snapshot_hashes = {"train.parquet": "train", "eval.parquet": "eval"}
    monkeypatch.setattr(
        relevance_new_models,
        "_load_frozen_snapshot",
        lambda _: (metadata, train, evaluation, snapshot_hashes),
    )
    monkeypatch.setattr(
        relevance_new_models.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("Artifact reuse loaded a model"),
    )
    monkeypatch.setattr(
        relevance_new_models, "_require_balanced_evaluation", lambda _: None
    )
    monkeypatch.setattr(
        relevance_new_models,
        "_select_head_parameters",
        lambda *_: (
            {"C": 5.0},
            [
                {
                    "parameters": {"C": 5.0},
                    "mean_average_precision": 1.0,
                    "std_average_precision": 0.0,
                }
            ],
        ),
    )

    output = tmp_path / "result.json"
    run_benchmark(
        snapshot=tmp_path,
        model_name="google/embeddinggemma-300m",
        output=output,
        batch_size=4,
        dtype_name="float32",
        embeddings_dir=tmp_path,
    )

    result = json.loads(output.read_text())
    assert result["embedding_artifact"] == {
        "path": str(artifact_path),
        "sha256": artifact_sha256,
        "reused": True,
    }


def test_balanced_evaluation_requires_200_rows_per_class() -> None:
    evaluation = pd.DataFrame({"label": [0] * 200 + [1] * 200})

    _require_balanced_evaluation(evaluation)

    with pytest.raises(ValueError, match="200 positive and 200 negative"):
        _require_balanced_evaluation(evaluation.assign(label=[0] * 201 + [1] * 199))


def test_head_contracts_include_the_four_approved_candidates() -> None:
    assert set(HEAD_CONTRACTS) == {
        "logistic-regression",
        "linear-svc",
        "rbf-svc",
        "mlp",
        "extra-trees",
    }
    assert HEAD_CONTRACTS["linear-svc"].calibration == "sigmoid"
    assert HEAD_CONTRACTS["rbf-svc"].calibration == "sigmoid"


def test_all_heads_emit_valid_probabilities() -> None:
    embeddings = np.vstack(
        (np.zeros((10, 4), dtype=np.float32), np.ones((10, 4), dtype=np.float32))
    )
    labels = np.array([0] * 10 + [1] * 10)

    for head in HEAD_CONTRACTS.values():
        classifier = _new_classifier(head, head.candidates[0])
        classifier.fit(embeddings, labels)
        probabilities = classifier.predict_proba(embeddings)[:, 1]

        assert probabilities.shape == (20,)
        assert np.isfinite(probabilities).all()
        assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_select_head_parameters_uses_train_folds_only() -> None:
    train_embeddings = np.array(
        [[0.0], [0.1], [0.2], [0.3], [0.4], [0.6], [0.7], [0.8], [0.9], [1.0]],
        dtype=np.float32,
    )
    train_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    selected, results = _select_head_parameters(
        HEAD_CONTRACTS["logistic-regression"], train_embeddings, train_labels
    )

    assert selected in HEAD_CONTRACTS["logistic-regression"].candidates
    assert len(results) == len(HEAD_CONTRACTS["logistic-regression"].candidates)
    assert all("mean_average_precision" in result for result in results)


def test_reference_result_must_match_logistic_head(tmp_path) -> None:
    evaluation = pd.DataFrame({"article_id": [1, 2], "label": [0, 1]})
    reference = {
        "snapshot_id": "snapshot",
        "snapshot_sha256": {"eval.parquet": "eval"},
        "model": {
            "model_name": "google/embeddinggemma-300m",
            "contract_revision": "57c266a740f537b4dc058e1b0cda161fd15afa75",
            "prefix": "",
        },
        "head": {"name": "linear-svc"},
        "evaluation": {
            "article_ids": [1, 2],
            "labels": [0, 1],
            "probabilities": [0.1, 0.9],
        },
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))

    with pytest.raises(ValueError, match="matching logistic"):
        _reference_probabilities(
            path,
            {"snapshot_id": "snapshot"},
            {"eval.parquet": "eval"},
            evaluation,
            MODEL_CONTRACTS["google/embeddinggemma-300m"],
        )


def test_paired_bootstrap_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(relevance_new_models, "BOOTSTRAP_SAMPLES", 20)
    labels = np.array([0] * 100 + [1] * 100)
    candidate = np.linspace(0.1, 0.9, 200)
    reference = np.linspace(0.2, 0.8, 200)

    first = _paired_bootstrap_average_precision_difference(labels, candidate, reference)
    second = _paired_bootstrap_average_precision_difference(
        labels, candidate, reference
    )

    assert first == second
    assert first["bootstrap_samples"] == 20


def test_frozen_snapshot_rejects_changed_files(tmp_path) -> None:
    for name in ("train.parquet", "eval.parquet", "metadata.json"):
        (tmp_path / name).write_bytes(b"changed")

    with pytest.raises(ValueError, match="hashes do not match"):
        _load_frozen_snapshot(tmp_path)


def test_local_support_files_are_part_of_artifact_identity(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("config")
    (tmp_path / "tokenizer.json").write_text("tokenizer")

    identity = _artifact_identity(
        "test-snapshot",
        {"train.parquet": "train", "eval.parquet": "eval"},
        MODEL_CONTRACTS["google/embeddinggemma-300m"],
        tmp_path,
    )

    assert identity["local_model_path"] == str(tmp_path.resolve())
    assert identity["local_support_file_sha256"] == _local_support_file_hashes(tmp_path)


def test_weight_hashes_records_local_model_bytes(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    assert _weight_hashes(tmp_path) == {
        "model.safetensors": (
            "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"
        )
    }


def test_mean_pool_ignores_padding() -> None:
    pooled = mean_pool(
        torch.tensor([[[1.0], [3.0], [99.0]]]),
        torch.tensor([[1, 1, 0]]),
    )

    assert torch.equal(pooled, torch.tensor([[2.0]]))


def test_last_token_pool_supports_left_and_right_padding() -> None:
    hidden = torch.tensor(
        [
            [[1.0], [2.0], [9.0]],
            [[3.0], [4.0], [5.0]],
        ]
    )

    right_padded = last_token_pool(
        hidden,
        torch.tensor([[1, 1, 0], [1, 1, 1]]),
    )
    left_padded = last_token_pool(
        hidden,
        torch.tensor([[0, 1, 1], [1, 1, 1]]),
    )

    assert torch.equal(right_padded, torch.tensor([[2.0], [5.0]]))
    assert torch.equal(left_padded, torch.tensor([[9.0], [5.0]]))
