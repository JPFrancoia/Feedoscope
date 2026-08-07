# Plan: replace threshold relevance metrics with ranking metrics

Status: approved. Precision@50 pivot started 2026-08-07.

## Brief

The weekly relevance evaluation must measure the ranked feed. Average precision
measures the complete ranking. Precision@50 measures the quality of the first
50 results.

The first rollout used Recall@10/25/50. On a holdout with a fixed 150 positive
rows, Recall@k is only Precision@k multiplied by `k / 150`. It gave no new
information and made strong results look weak. Replace it with Precision@50.

## Current state

`eval_relevance()` ranks a balanced holdout with 150 good and 150 bad articles.
The first ranking-metric run produced:

- average precision 0.9334
- ROC AUC 0.9456
- Recall@10 0.0667
- Recall@25 0.1667
- Recall@50 0.3133

These recall values mean that the first 10 contained 10 good articles, the first
25 contained 25, and the first 50 contained 47. Precision@50 expresses the last
result directly as `47 / 50 = 0.94`.

The `model_evals.metrics_precision` column can store Precision@50 for new
Relevance rows. Historical Relevance rows used this column for precision at the
old 0.5 threshold. New rows use an explicit evaluation-model label for the
Precision@50 metric contract. No schema change is necessary.

## Proposed implementation

1. Keep average precision and ROC AUC for Relevance.
2. Replace Recall@10/25/50 with tie-aware Precision@50.
3. Give proportional credit when a score tie crosses position 50.
4. Store Precision@50 in `metrics_precision` and leave the recall columns empty.
5. Keep Urgency classification metrics unchanged.
6. Show average precision and Precision@50 in the Miniflux Relevance section.
7. Use the evaluation-model label to identify new Precision@50 rows.
8. Show `-` for Precision@50 on historical threshold-based rows.

## File-by-file impact

| Path | Change |
|---|---|
| `feedoscope/eval_models.py` | calculate average precision, ROC AUC, and Precision@50 |
| `feedoscope/data_registry/data_registry.py` | map `precision_at_50` to the existing precision column |
| `tests/test_eval_models.py` | cover ties, edge cases, and persistence mapping |
| Miniflux `internal/ui/ai_metrics.go` | map Precision@50 without relabeling historical precision |
| Miniflux `internal/template/templates/views/ai_metrics.html` | show average precision and Precision@50 |
| Miniflux `internal/ui/static/js/app.js` | chart the two Relevance metrics |
| Miniflux focused tests | cover historical gaps and chart series |
| Workspace reference docs | document the final metric contract |

## Risks and edge cases

- The holdout is balanced. Precision@50 does not estimate production precision
  when production class prevalence differs.
- Precision@50 can saturate. If it stays near 1.0, evaluate Precision@100 before
  adding another persistent metric.
- A score tie can cross position 50. Proportional credit removes input-order
  bias.
- Historical `metrics_precision` values have different semantics. Miniflux uses
  the explicit evaluation-model label to identify Precision@50 rows.

## Validation

- A perfect top 50 gives Precision@50 of 1.0.
- A ranking with 47 good articles in the top 50 gives 0.94.
- Reordering articles inside a cutoff tie does not change the result.
- Single-class inputs do not produce NaN values.
- Urgency keeps its existing metric mapping and display.
- Historical Relevance rows show `-` for Precision@50.

## Step-by-step checklist

- [x] Approve Precision@50 and remove Recall@k.
- [x] Update Feedoscope metrics and tests.
- [x] Update Miniflux display and tests.
- [x] Update reference documentation.
- [ ] Run repository checks.
- [ ] Deploy the evaluator and UI.
- [ ] Run and record a new production evaluation.

## Decisions

- Average precision remains the primary relevance metric.
- Precision@50 is the only top-list metric.
- Precision@10 and Precision@25 are omitted because both saturated at 1.0.
- ROC AUC remains a secondary stored metric.
- No migration or new database column is necessary.

## Feedoscope validation

- 88 tests passed with five existing dependency warnings.
- Mypy, Black, isort, and the diff check passed.
