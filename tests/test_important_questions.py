from collections.abc import Callable
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from experiments import important_jev as base
from experiments import important_questions as q


def response(questions: list[dict], yes: float = 0.7) -> dict:
    probabilities = {"yes": yes, "no": 1 - yes, "insufficient": 0.0}
    return {
        "model": base.MODEL,
        "answers": {
            item["id"]: {
                "type": "choice",
                "choice": max(probabilities, key=lambda key: probabilities[key]),
                "probabilities": dict(probabilities),
                "confidence": (3 * max(probabilities.values()) - 1) / 2,
            }
            for item in questions
        },
    }


def test_payload_excludes_labels_profile_and_author_metadata() -> None:
    questions = json.loads(q.QUESTIONS.read_text())
    row = {
        "article_id": 1,
        "label": 1,
        "title": "Example",
        "body": "日本語 " * 30000 + " end",
        "author": "Secret author",
        "score": 99,
    }
    prompt = {"reader_profile": "Private preferences", "question": {"type": "choice"}}
    payload = q.feature_payload(row, prompt, questions)
    assert set(payload["state"]) == {"article"}
    assert set(payload["state"]["article"]) == {"title", "body"}
    assert "Private preferences" not in json.dumps(payload)
    assert (
        payload["state"]["article"]
        == base.request_payload(row, prompt)["state"]["article"]
    )
    assert len(json.dumps(payload, ensure_ascii=False).encode()) < 64000
    assert len(payload["questions"]) == 25


def test_ternary_probability_validation() -> None:
    questions = json.loads(q.QUESTIONS.read_text())[:2]
    answer = response(questions)
    assert q.validate_features(answer, questions) == [0.7, 0.0, 0.7, 0.0]
    answer["answers"][questions[0]["id"]]["confidence"] = 1
    with pytest.raises(ValueError, match="confidence"):
        q.validate_features(answer, questions)
    for invalid in (float("nan"), float("inf"), True, -0.2, 1.1):
        answer = response(questions)
        answer["answers"][questions[0]["id"]]["probabilities"]["yes"] = invalid
        with pytest.raises(ValueError):
            q.validate_features(answer, questions)
    answer = response(questions)
    del answer["answers"][questions[0]["id"]]
    with pytest.raises(ValueError, match="answers"):
        q.validate_features(answer, questions)


def test_compact_selection_recovers_signal_without_noise() -> None:
    rng = np.random.default_rng(42)
    y = np.tile([0, 1], 40)
    x = rng.uniform(size=(80, 6))
    x[:, 2] = y * 0.8 + 0.1
    selected = q.choose_compact(x, y, 3)
    assert selected["chosen"]["questions"] == [1]
    assert selected["chosen"]["mean_ap"] == 1
    assert q.choose_compact(x, y, 3) == selected


def test_export_matches_pipeline_and_extreme_logits_are_finite() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 5))
    y = np.tile([0, 1], 30)
    columns = [0, 2, 4]
    model = q.export_model(x, y, columns)
    expected = q.fit_model(x[:, columns], y).predict_proba(x[:, columns])[:, 1]
    np.testing.assert_allclose(q.predict(model, x), expected)
    np.testing.assert_array_equal(
        q.predict(
            {"columns": [0], "weights": [1], "intercept": 0}, np.array([[-1e6], [1e6]])
        ),
        [0, 1],
    )


def test_cv_scaling_uses_only_fit_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.arange(80, dtype=float).reshape(40, 2)
    y = np.tile([0, 1], 20)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(x, y))
    original = q.fit_model
    seen = []

    def spy(xx: np.ndarray, yy: np.ndarray) -> object:
        seen.append(xx.copy())
        return original(xx, yy)

    monkeypatch.setattr(q, "fit_model", spy)
    q.cv_result(x, y, [0], folds)
    for received, (train, valid) in zip(seen, folds, strict=True):
        np.testing.assert_array_equal(received, x[train][:, [0]])
        assert set(received[:, 0]).isdisjoint(x[valid, 0])


def test_cached_extraction_and_frozen_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    questions = json.loads(q.QUESTIONS.read_text())
    rows = [{"article_id": 1, "label": 1, "title": "Title", "body": "Body"}]
    prompt = {"reader_profile": "Profile", "question": {"type": "choice"}}
    for name, value in (
        ("train.json", rows),
        ("eval.json", rows),
        ("prompt.json", prompt),
        ("questions.json", questions),
    ):
        base.write_json(tmp_path / name, value)
    protocol = {
        "model": base.MODEL,
        "C": 1,
        "max_questions": q.MAX_QUESTIONS,
        "files": {
            name: base.digest(tmp_path / name)
            for name in ("train.json", "eval.json", "prompt.json", "questions.json")
        },
    }
    base.write_json(tmp_path / "protocol.json", protocol)
    calls = []

    def fake(payload: dict, key: str, validator: Callable) -> dict:
        calls.append(payload)
        if "importance" in payload["questions"]:
            answer = {
                "model": base.MODEL,
                "answers": {
                    "importance": {
                        "type": "choice",
                        "choice": "important",
                        "probabilities": {"important": 0.8, "good_only": 0.2},
                        "confidence": 0.6,
                    }
                },
            }
        else:
            answer = response(questions)
        validator(answer)
        return answer

    monkeypatch.setenv("JEV_KEY", "fake")
    monkeypatch.setattr(base, "call_jev", fake)
    q.extract(tmp_path, "train", None)
    assert len(calls) == 2
    q.extract(tmp_path, "train", None)
    assert len(calls) == 2
    x, y, names = q.matrix(tmp_path, "train")
    assert x.shape == (1, 52)
    assert y.tolist() == [1]
    base.write_json(
        tmp_path / "selection.json",
        {"input_hashes": {"protocol.json": base.digest(tmp_path / "protocol.json")}},
    )
    base.write_json(
        tmp_path / "selection_lock.json",
        {"selection_sha256": base.digest(tmp_path / "selection.json")},
    )
    with pytest.raises(ValueError, match="Incomplete frozen selection"):
        q.check_selection(tmp_path)
    (tmp_path / "selection.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen"):
        q.extract(tmp_path, "eval", None)
    (tmp_path / "questions.json").write_text("[]")
    with pytest.raises(ValueError, match="input changed"):
        q.extract(tmp_path, "train", None)


def test_prepare_balances_training_and_excludes_duplicate_text(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, snapshot = tmp_path / "source", tmp_path / "snapshot"
    source.mkdir()
    snapshot.mkdir()

    def row(i: int, label: int) -> dict:
        return {
            "article_id": i,
            "label": label,
            "title": f"Title {i}",
            "body": f"Body {i}",
        }

    train = [row(i, 1) for i in range(225)] + [row(i, 0) for i in range(300, 375)]
    evaluation = [row(i, 1) for i in range(225, 300)] + [
        row(i, 0) for i in range(375, 450)
    ]
    for name, values in (("profile.jsonl", train), ("eval.jsonl", evaluation)):
        (source / name).write_text("".join(json.dumps(r) + "\n" for r in values))
    all_rows = train + evaluation + [row(i, 0) for i in range(450, 700)]
    all_rows += [
        {"article_id": 1000, "label": 0, "title": "duplicate", "body": "Body 0"}
    ]
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "article_id": r["article_id"],
                    "title": r["title"],
                    "content": r["body"],
                    "is_important": bool(r["label"]),
                }
                for r in all_rows
            ]
        ),
        snapshot / "good.parquet",
    )
    base.write_json(
        source / "prompt.json", {"reader_profile": "Profile", "question": {}}
    )
    base.write_json(
        source / "frozen_prompt.json",
        {"prompt_sha256": base.digest(source / "prompt.json")},
    )
    (source / "predictions.jsonl").write_text("")
    base.write_json(
        source / "split_manifest.json",
        {
            "snapshot": str(snapshot),
            "source_sha256": {"good.parquet": base.digest(snapshot / "good.parquet")},
            "split_sha256": {
                name: base.digest(source / name)
                for name in ("profile.jsonl", "eval.jsonl")
            },
        },
    )
    output = tmp_path / "output"
    q.prepare(source, output)
    prepared = json.loads((output / "train.json").read_text())
    assert len(prepared) == 450
    assert sum(r["label"] for r in prepared) == 225
    assert 1000 not in {r["article_id"] for r in prepared}
    assert {base.text_key(r) for r in prepared}.isdisjoint(
        base.text_key(r) for r in evaluation
    )
    assert json.loads((output / "eval.json").read_text()) == evaluation


def test_synthetic_select_evaluate_report_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    questions = json.loads(q.QUESTIONS.read_text())
    prompt = {"reader_profile": "Synthetic profile", "question": {"type": "choice"}}

    def rows(start: int, count: int) -> list[dict]:
        return [
            {"article_id": i, "label": i % 2, "title": str(i), "body": f"Body {i}"}
            for i in range(start, start + count)
        ]

    train, evaluation = rows(0, 40), rows(100, 10)
    for name, value in [
        ("train.json", train),
        ("eval.json", evaluation),
        ("questions.json", questions),
        ("prompt.json", prompt),
    ]:
        base.write_json(output / name, value)

    def direct(probability: float) -> dict:
        return {
            "model": base.MODEL,
            "answers": {
                "importance": {
                    "type": "choice",
                    "choice": "important" if probability >= 0.5 else "good_only",
                    "probabilities": {
                        "important": probability,
                        "good_only": 1 - probability,
                    },
                    "confidence": abs(2 * probability - 1),
                }
            },
        }

    original = []
    for row in evaluation:
        original.append(
            {
                "article_id": row["article_id"],
                "request_sha256": q.fingerprint(base.request_payload(row, prompt)),
                "response": direct(0.9 if row["label"] else 0.1),
            }
        )
    (source / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in original)
    )
    base.write_json(
        output / "protocol.json",
        {
            "model": base.MODEL,
            "C": 1,
            "max_questions": q.MAX_QUESTIONS,
            "source": str(source),
            "source_hashes": {
                "predictions.jsonl": base.digest(source / "predictions.jsonl")
            },
            "files": {
                name: base.digest(output / name)
                for name in ("train.json", "eval.json", "prompt.json", "questions.json")
            },
        },
    )

    def fake(payload: dict, key: str, validator: Callable) -> dict:
        p = 0.9 if int(payload["state"]["article"]["title"]) % 2 else 0.1
        result = (
            direct(p)
            if "importance" in payload["questions"]
            else response(questions, p)
        )
        result["usage"] = {"input_tokens": 10}
        validator(result)
        return result

    monkeypatch.setenv("JEV_KEY", "fake")
    monkeypatch.setattr(base, "call_jev", fake)
    q.extract(output, "train", None)
    q.select(output)
    assert not (output / "eval_responses.jsonl").exists()
    lock = base.digest(output / "selection.json")
    q.extract(output, "eval", None)
    q.report(output)
    result = json.loads((output / "results.json").read_text())
    assert result["metrics"]["compact"]["average_precision"] == 1
    assert result["metrics"]["direct_raw"]["average_precision"] == 1
    assert result["metrics"]["compact"]["count"] == 10
    assert result["total_requests"] == 90
    assert base.digest(output / "selection.json") == lock
    with pytest.raises(ValueError, match="frozen"):
        q.select(output)
    path = output / "eval_responses.jsonl"
    with path.open("a") as handle:
        handle.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Reported input changed"):
        q.report(output)
    with pytest.raises(ValueError, match="duplicate"):
        q.matrix(output, "eval")


def test_ternary_ties_and_rounding() -> None:
    questions = json.loads(q.QUESTIONS.read_text())[:1]
    for probabilities, choice, confidence in [
        ({"yes": 0.5, "no": 0.5, "insufficient": 0}, "no", 0.25),
        ({"yes": 0.99, "no": 0.01, "insufficient": 0}, "yes", 0.98),
        ({"yes": 0.34, "no": 0.33, "insufficient": 0.33}, "yes", 0.01),
    ]:
        answer = response(questions)
        answer["answers"][questions[0]["id"]].update(
            probabilities=probabilities, choice=choice, confidence=confidence
        )
        assert q.validate_features(answer, questions)[0] == probabilities["yes"]


def test_failed_requests_preserve_successes_and_corrupt_cache_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qs = json.loads(q.QUESTIONS.read_text())
    row = {"article_id": 1, "label": 1, "title": "One", "body": "Text"}
    prompt = {"reader_profile": "Profile", "question": {"type": "choice"}}
    for name, value in [
        ("train.json", [row]),
        ("eval.json", [row]),
        ("prompt.json", prompt),
        ("questions.json", qs),
    ]:
        base.write_json(tmp_path / name, value)
    base.write_json(
        tmp_path / "protocol.json",
        {
            "model": base.MODEL,
            "C": 1,
            "max_questions": q.MAX_QUESTIONS,
            "files": {
                name: base.digest(tmp_path / name)
                for name in ("train.json", "eval.json", "prompt.json", "questions.json")
            },
        },
    )

    def fake(payload: dict, key: str, validator: Callable) -> dict:
        if "importance" in payload["questions"]:
            raise RuntimeError("Synthetic failure")
        answer = response(qs)
        validator(answer)
        return answer

    monkeypatch.setenv("JEV_KEY", "fake")
    monkeypatch.setattr(base, "call_jev", fake)
    with pytest.raises(RuntimeError, match="Completed responses were saved"):
        q.extract(tmp_path, "train", None)
    path = tmp_path / "train_responses.jsonl"
    saved = base.read_jsonl(path)
    assert len(saved) == 1 and saved[0]["kind"] == "features"
    with path.open("a") as handle:
        handle.write("{incomplete")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Preserve this file"):
        q.extract(tmp_path, "train", None)
    assert path.read_bytes() == before
