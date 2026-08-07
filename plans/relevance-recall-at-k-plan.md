# Plan: replace F1 with recall@k in the weekly relevance evaluation

Status: open. Waiting for user input. Created 2026-08-07.

## Brief

The weekly evaluation reports an F1 score for relevance at a fixed 0.5
threshold. The relevance pipeline never applies a threshold. It ranks articles.
The F1 score therefore measures a decision that the product does not make. The
number lost its trend when the head changed to an MLP, because the MLP is not
calibrated. This plan replaces F1 with recall@k, which measures the ranked feed
that the reader sees.

## Current state

`feedoscope/eval_models.py` computes the relevance metrics in
`compute_and_log_metrics()`. That function applies `pred_labels =
(predicted_probs >= 0.5)` and then reports accuracy, precision, recall, and F1.
It also reports ROC AUC, average precision, and log loss, which need no
threshold.

`eval_relevance()` builds the holdout. It reads every good and bad article. It
then samples `VALIDATION_SIZE` rows from each class. The holdout is balanced.
The production feed is not balanced.

The `model_evals` table already holds `metrics_recall_at_10`,
`metrics_recall_at_25`, and `metrics_recall_at_50`. The old super-important
evaluation wrote them. Nothing writes them now. The columns are free to reuse.

The removed `compute_super_important_ranking_metrics()` function held a working
recall@k implementation. Commit `de69041` is the last commit that contains it.

## Proposed implementation

Add recall@k to the relevance metrics. Remove F1, accuracy, and precision at
the fixed threshold.

For a ranked list of holdout articles, recall@k is the count of positive rows
inside the top k, divided by the count of all positive rows.

Record the candidate count with every metric row. Recall@k depends on the size
of the candidate pool. Recall@10 over 50 candidates and recall@10 over 500
candidates are different numbers. Without the count, a later change to
`VALIDATION_SIZE` breaks the series without a visible cause. This failure mode
is the same one that F1 just showed.

## File-by-file impact

| Path | Change |
|---|---|
| `feedoscope/eval_models.py` | add recall@k to `compute_and_log_metrics()`, drop the 0.5 threshold metrics |
| `feedoscope/data_registry/data_registry.py` | map the new keys in `insert_model_eval()` |
| `feedoscope/data_registry/sql/insert_model_eval.sql` | add a candidate-count column |
| `db/migrations/000009_*` | add the candidate-count column |
| `tests/test_eval_models.py` | cover recall@k and the candidate count |

## Risks and edge cases

- The holdout is balanced 1:1. The production feed is not. Absolute recall@k
  values are optimistic. The trend across weeks is still valid.
- When k is more than the number of candidates, the metric saturates at 1.0.
  Clamp k to the candidate count, as the old code did.
- When the holdout holds no positive rows, the metric divides by zero. Return
  `None` for that row.
- The `metrics_f1` column keeps its history. A break in the series must stay
  visible. Do not backfill.

## Validation

- Unit test: a perfect ranking gives recall@k of 1.0 at k equal to the positive
  count.
- Unit test: a reversed ranking gives a lower value than the correct ranking.
- Unit test: k larger than the candidate count does not raise an error.
- Run one weekly evaluation and compare the new metric against the average
  precision for the same run.

## Step-by-step checklist

- [ ] Confirm the k values with the user.
- [ ] Add `recall_at_k()` to `eval_models.py`.
- [ ] Add the candidate count to the metric row.
- [ ] Write the migration for the candidate-count column.
- [ ] Update `insert_model_eval()` and its SQL file.
- [ ] Add the unit tests.
- [ ] Run `make lint` and `make format`.
- [ ] Update `docs/relevance-ranking.md` with the new metric set.

## Open questions

1. **Which k values?** The old code used 10, 25, and 50. The correct values
   depend on how far down the feed the reader goes in one session. This
   question is open. It blocks the first checklist item.

2. **Keep F1 or remove it?** Three options exist:
   - Remove F1. Recall@k measures the ranked product. This option is the
     recommendation.
   - Keep F1 and pick its threshold on the training folds instead of the fixed
     0.5. This option keeps a comparable threshold metric. It costs about five
     lines.
   - Keep F1 unchanged and add a note that the series broke at commit
     `6ca9e7d`.

3. **Should the holdout match the production class balance?** A balanced
   holdout inflates recall@k. An imbalanced holdout needs more rows for a
   stable measurement. This question is not urgent.

## Assumptions

- The positive label stays `read and vote >= 0`, the same target that the MLP
  trains on.
- `VALIDATION_SIZE` does not change during this work.
