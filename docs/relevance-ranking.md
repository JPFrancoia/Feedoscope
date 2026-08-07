# Relevance ranking

Status: current as of 2026-08-07.

## Pipeline

The relevance score has four steps:

1. EmbeddingGemma 300M encodes the article with the classification prompt.
2. One `MLPClassifier` head predicts the positive-class probability.
3. The probability is multiplied by 100.
4. A cube-root transform spreads the score before integer storage.

The label contract is simple. A positive row is `read and vote >= 0`. A negative row is `vote = -1`.

Training gives starred or upvoted reads a weight of 20. The weight comes from `IMPORTANT_ARTICLE_WEIGHT`. The function `is_important()` in `feedoscope/relevance_embedding.py` decides which rows get the weight. This weight is a training input only. No part of inference reads it.

## Removed components

The two-head design was removed on 2026-08-07. These parts are gone:

- the super-important logistic head
- the `combine_probabilities()` bonus function
- the bonus grid and the rolling-window bonus selection
- the `eval_super_important()` weekly evaluation
- the `super_important_inference` table
- the `important-auto` tag synchronization
- the `SUPER_IMPORTANT_INFERENCE_ENABLED` and `SUPER_IMPORTANT_BONUS` settings

The `important-auto` tags that Miniflux already holds stay in place. Nothing writes new ones. Nothing deletes the old ones.

The `model_evals` table keeps its `metrics_super_important_*` columns. These columns hold real history. New rows leave them empty.

## Artifact contract

| Item | Value |
|---|---|
| File name | `relevance_mlp.joblib` |
| Artifact version | 4 |
| Backend | `embedding_prompted_mlp` |
| Family prefix | `relevance_<encoder>_...` |
| Train count keys | `good`, `bad` |

The family prefix changed from `relevance_two_head_*`. Older artifacts on disk do not match the new prefix. Retrain before the next inference run.

The embedding cache key did not change. A retrain reuses every cached vector.

## Why the F1 metric fell

The weekly evaluation showed a lower F1 score after the change to the prompted encoder and the MLP head. Average precision increased at the same time.

This result is expected. The two metrics measure different things.

Average precision measures the ranked order. The pipeline produces a ranked order, so average precision measures the product.

F1 measures one decision at a fixed 0.5 threshold. The relevance pipeline has no threshold. `eval_models.py` applies `>= 0.5` for the F1 score alone.

Logistic regression is close to calibrated, so its 0.5 point sat near the best F1 point. F1 and average precision moved together for months. An `MLPClassifier` trained on log loss is not calibrated. It saturates its probabilities. The training set holds about 84% positive rows, so the probability mass moved past 0.5 and precision fell.

The 20x sample weight added a second positive tilt in the same release.

The F1 series is not comparable across this change. The name is the same. The scale is not.

Two known limits of the current head:

- The MLP uses `early_stopping=True`. Scikit-learn holds out 10% of the rows for that check and scores them without the sample weights. The weight and the stopping rule disagree.
- The benchmark in `experiments/relevance_new_models.py` selects a head on mean average precision only. F1 was never an objective.

The prompt is not the cause. On the same snapshot with the same logistic head, the prompted encoder scored F1 0.8970 against 0.8711 for the unprompted encoder.

## Related files

| Path | Role |
|---|---|
| `feedoscope/relevance_embedding.py` | encoder, cache, head, artifact |
| `feedoscope/llm_learn.py` | training entry point |
| `feedoscope/llm_infer.py` | inference entry point |
| `feedoscope/eval_models.py` | weekly evaluation |
| `db/migrations/000008_drop_super_important_inference.up.sql` | drops the dead table |
