# Plan: replace F1 with recall@k in the weekly relevance evaluation

Status: completed 2026-08-07.

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

Use k values 10, 25, and 50. The holdout has 300 candidates when
`VALIDATION_SIZE=150`, so these values measure coverage near the top of the
ranking. The existing `eval` JSON counts record the candidate pool size. No
schema change is necessary.

Keep the existing classification metrics for Urgency. For Relevance, stop
writing accuracy, precision, threshold recall, F1, and log loss. Keep average
precision and ROC AUC. Add Recall@10, Recall@25, and Recall@50.

## File-by-file impact

| Path | Change |
|---|---|
| `feedoscope/eval_models.py` | add relevance ranking metrics without changing Urgency metrics |
| `tests/test_eval_models.py` | cover Recall@10/25/50 and persistence mapping |
| Miniflux `internal/ui/ai_metrics.go` | mark Relevance views and emit ranking chart data |
| Miniflux `internal/template/templates/views/ai_metrics.html` | show average precision and Recall@10/25/50 for Relevance |
| Miniflux `internal/ui/static/js/app.js` | select the Relevance chart series |
| Miniflux focused tests | cover Relevance mapping, chart data, and chart series |

## Risks and edge cases

- The holdout is balanced 1:1. The production feed is not. Absolute Recall@k
  values do not represent production prevalence. The weekly trend remains useful.
- Values of k are less than the current positive count and candidate count.
  The metric still clamps k to the candidate count.
- When the holdout holds no positive rows, the metric divides by zero. Return
  `None` for that row.
- The `metrics_f1` column keeps its history. A break in the series must stay
  visible. Do not backfill.

## Validation

- Unit test: a perfect ranking gives recall@k of 1.0 at k equal to the positive
  count.
- Unit test: a reversed ranking gives a lower value than the correct ranking.
- Unit test: k larger than the candidate count does not raise an error.
- Miniflux tests: Relevance uses average precision and Recall@10/25/50.
- Run one weekly evaluation and compare Recall@k with average precision.

## Step-by-step checklist

- [x] Confirm k values 10, 25, and 50.
- [x] Add relevance ranking metrics to `eval_models.py`.
- [x] Add Feedoscope unit tests.
- [x] Update the Miniflux Relevance section and tests.
- [x] Run Feedoscope checks.
- [x] Run Miniflux checks.
- [x] Update `docs/reference/relevance-ranking.md` with the new metric set.

## Open questions

1. **Should the holdout match the production class balance?** A balanced
   holdout inflates recall@k. An imbalanced holdout needs more rows for a
   stable measurement. This question is not urgent.

## Assumptions

- The positive label stays `read and vote >= 0`, the same target that the MLP
  trains on.
- `VALIDATION_SIZE` stays at 150 during this work.
- Historical threshold metrics stay in their database columns. New Relevance
  rows leave those columns empty.

## Completion evidence

- Feedoscope: 88 tests passed. Mypy, Black, isort, and the diff check passed.
- Miniflux: `go test ./internal/ui` passed. Six JavaScript tests passed.
- Recall@k gives proportional credit when a score tie crosses k. Input order
  does not change the result.
- Historical Relevance rows show `-` for missing Recall@k values.
- No schema migration or new database column was necessary.
- Production evaluation job `feedoscope-eval-recall-e796b76` completed on
  2026-08-07 with 150 good and 150 bad holdout articles.
- The stored metrics are average precision 0.9334, ROC AUC 0.9456,
  Recall@10 0.0667, Recall@25 0.1667, and Recall@50 0.3133.
