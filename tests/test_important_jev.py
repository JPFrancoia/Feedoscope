import hashlib
import json
from pathlib import Path

import pytest

from experiments.important_jev import (
    MODEL,
    REQUEST_BYTES,
    assign_split,
    clean_body,
    load_key,
    metrics,
    request_payload,
    text_key,
    validate_answer,
)


def article(article_id: int, label: int, body: str | None = None) -> dict:
    return {
        "article_id": article_id,
        "label": label,
        "title": f"Article {article_id}",
        "body": body if body is not None else f"Body {article_id}",
    }


def test_split_is_deterministic_disjoint_and_excludes_duplicate_text() -> None:
    important = [article(i, 1) for i in range(330)]
    good = [article(i, 0) for i in range(1000, 1200)]
    good.append(article(2000, 0, "Body 0"))
    important.append(article(4000, 1, "Body 1"))
    first = assign_split(important, good)
    assert first == assign_split(important, good)
    profile, evaluation, counts = first
    assert len(profile) == 300
    assert len(evaluation) == 150
    assert sum(row["label"] for row in profile) == 225
    assert sum(row["label"] for row in evaluation) == 75
    assert {row["article_id"] for row in profile}.isdisjoint(
        row["article_id"] for row in evaluation
    )
    assert {text_key(row) for row in profile}.isdisjoint(
        text_key(row) for row in evaluation
    )
    assert counts["conflicting_text_groups"] == 1
    assert not any(row["body"] == "Body 0" for row in profile + evaluation)


def test_prepare_checks_important_snapshot_subset(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from experiments.important_jev import prepare

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    pq.write_table(
        pa.table(
            {
                "article_id": [1],
                "is_important": [True],
                "is_good": [True],
                "title": ["Article"],
                "content": ["Text"],
            }
        ),
        snapshot / "important.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "article_id": [2],
                "is_important": [True],
                "is_good": [True],
                "title": ["Article"],
                "content": ["Text"],
            }
        ),
        snapshot / "good.parquet",
    )
    for name in ("manifest.json", "export.sql"):
        (snapshot / name).write_text("fixture")
    files = ("important.parquet", "good.parquet", "manifest.json", "export.sql")
    (snapshot / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((snapshot / name).read_bytes()).hexdigest()}  {name}\n"
            for name in files
        )
    )
    with pytest.raises(ValueError, match="IDs differ"):
        prepare(snapshot, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_split_rejects_insufficient_examples() -> None:
    with pytest.raises(ValueError, match="Not enough"):
        assign_split([article(1, 1)], [article(2, 0)])


def test_clean_body_preserves_text_and_removes_scripts() -> None:
    assert (
        clean_body("<p>One &amp; two</p><script>secret</script><p>Three</p>")
        == "One & two Three"
    )
    assert clean_body("") == ""


def test_request_strips_labels_and_bounds_unicode_input() -> None:
    row = article(1, 1, "Début " + "日本語 " * 40000 + " THE END")
    row.update(vote=1, starred=True, tags=["important"])
    payload = request_payload(
        row, {"reader_profile": "Profile", "question": {"type": "choice"}}
    )
    assert len(json.dumps(payload, ensure_ascii=False).encode()) <= REQUEST_BYTES
    assert set(payload["state"]) == {"reader_profile", "article"}
    assert set(payload["state"]["article"]) == {"title", "body"}
    assert payload["state"]["article"]["body"].startswith("Début")
    assert payload["state"]["article"]["body"].endswith("THE END")
    assert "Middle omitted" in payload["state"]["article"]["body"]


def answer(probability: float = 0.8) -> dict:
    return {
        "model": MODEL,
        "answers": {
            "importance": {
                "type": "choice",
                "choice": "important" if probability >= 0.5 else "good_only",
                "confidence": abs(2 * probability - 1),
                "probabilities": {
                    "important": probability,
                    "good_only": 1 - probability,
                },
            }
        },
    }


def test_valid_response() -> None:
    assert validate_answer(answer())["choice"] == "important"


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1])
def test_invalid_probability(value: float) -> None:
    with pytest.raises(ValueError):
        validate_answer(answer(value))


def test_wrong_model_and_inconsistent_choice() -> None:
    response = answer()
    response["model"] = "jev-latest"
    with pytest.raises(ValueError, match="model"):
        validate_answer(response)
    response = answer()
    response["answers"]["importance"]["choice"] = "good_only"
    with pytest.raises(ValueError, match="Choice"):
        validate_answer(response)


def test_confident_errors_cost_more() -> None:
    uncertain = metrics([0, 1], [0.51, 0.49], [0.02, 0.02])
    confident = metrics([0, 1], [0.99, 0.01], [0.98, 0.98])
    assert confident["accuracy"] == uncertain["accuracy"] == 0
    assert confident["brier"] > uncertain["brier"]
    assert confident["log_loss"] > uncertain["log_loss"]
    assert confident["confidently_wrong_0.9"] == 2
    assert uncertain["confidently_wrong_0.9"] == 0


def test_key_reads_only_named_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JEV_KEY", raising=False)
    env = tmp_path / ".envrc"
    env.write_text(
        "export DATABASE_URL=$(do_not_execute)\nexport JEV_KEY='fake-key' # comment\n"
    )
    assert load_key(env) == "fake-key"
    monkeypatch.setenv("JEV_KEY", "process-key")
    assert load_key(env) == "process-key"


def test_evaluate_resume_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.important_jev as experiment

    monkeypatch.setenv("JEV_KEY", "fake-key")
    rows = [article(i, int(i < 75)) for i in range(150)]
    for name, records in (("eval.jsonl", rows), ("profile.jsonl", [article(1000, 1)])):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in records))
    experiment.write_json(
        tmp_path / "split_manifest.json",
        {
            "split_sha256": {
                name: experiment.digest(tmp_path / name)
                for name in ("eval.jsonl", "profile.jsonl")
            }
        },
    )
    experiment.write_json(
        tmp_path / "prompt.json",
        {"reader_profile": "Synthetic profile", "question": {"type": "choice"}},
    )

    def fake_call(payload: dict, key: str) -> dict:
        assert key == "fake-key"
        article_id = int(payload["state"]["article"]["title"].split()[-1])
        response = answer(0.9 if article_id < 75 else 0.1)
        response["usage"] = {"input_tokens": 100, "output_tokens": 20}
        return response

    monkeypatch.setattr(experiment, "call_jev", fake_call)
    experiment.evaluate(tmp_path, None)

    def forbidden_call(*args: object) -> None:
        raise AssertionError("Resume must not repeat requests")

    monkeypatch.setattr(experiment, "call_jev", forbidden_call)
    experiment.evaluate(tmp_path, None)
    experiment.report(tmp_path)
    result = json.loads((tmp_path / "metrics.json").read_text())
    assert result["count"] == 150
    assert result["accuracy"] == 1
    assert result["brier"] == pytest.approx(0.01)
    assert result["average_precision"] == 1
    assert (
        len(
            result["stratified_bootstrap_95_percent_intervals_2000_samples"][
                "average_precision"
            ]
        )
        == 2
    )
    assert result["input_tokens"] == 15000
    assert len(result["errors_descending_confidence"]) == 0
    (tmp_path / "prompt.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen prompt changed"):
        experiment.evaluate(tmp_path, None)


def test_transient_http_error_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from email.message import Message
    import io
    from urllib.error import HTTPError

    import experiments.important_jev as experiment

    calls = []

    def fake_open(request: object, timeout: int) -> io.StringIO:
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError(
                "https://api.typesafe.ai/v1/systemone",
                429,
                "rate limit",
                Message(),
                None,
            )
        return io.StringIO(json.dumps(answer()))

    monkeypatch.setattr(experiment, "urlopen", fake_open)
    monkeypatch.setattr(experiment.time, "sleep", lambda seconds: None)
    assert experiment.call_jev({}, "fake-key")["model"] == MODEL
    assert len(calls) == 2
