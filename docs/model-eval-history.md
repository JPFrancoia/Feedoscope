# Model Evaluation History

Feedoscope's weekly `make eval` job evaluates relevance, urgency, and
freshness, then stores each result in two places:

- `models/eval_history.json`, kept for the existing file-based history.
- Miniflux's `model_evals` PostgreSQL table, used as the durable history that
  Miniflux can display later.

## Database Ownership

The `model_evals` table is created and migrated by Miniflux, not Feedoscope.
Miniflux owns this schema because it reads the durable history on its AI Metrics
page. Deploy the Miniflux migration before deploying Feedoscope code that writes
the new freshness metrics; otherwise the insert cannot find the new columns.

## Stored Shape

Each eval run inserts one row per evaluated model. `training` and `eval` are
JSONB objects because the class names differ by model:

- Relevance uses `good` and `bad` counts.
- Urgency uses `urgent` and `evergreen` counts.
- Freshness uses `fresh_d`, `fresh_m`, and `fresh_y` counts.

Relevance and urgency retain these binary-classification metric columns:

- `metrics_accuracy`, `metrics_precision`, `metrics_recall`
- `metrics_f1`, `metrics_roc_auc`, `metrics_average_precision`,
  `metrics_log_loss`

Freshness reuses `metrics_f1` for Macro F1 and `metrics_roc_auc` for Long-lived
AUC. Its ordinal metrics use the new nullable columns:

- `metrics_rps`
- `metrics_weighted_kappa`
- `metrics_log_duration_mae`

All metric columns are nullable so a model writes only metrics that apply.
Freshness leaves binary-only metrics null; relevance and urgency leave the new
ordinal columns null. Long-lived AUC and weighted kappa are also null when they
cannot be calculated from the evaluation labels.

## Freshness Evaluation

Freshness labels are already ordered by publication time. Each weekly run trains
on the older labels and evaluates exactly the newest `VALIDATION_SIZE` labels.
It skips freshness evaluation when there are not more labels than the holdout
size or the classifier cannot be fitted.

| Metric | Meaning | Better direction |
|---|---|---|
| RPS | Ranked probability score across the two ordered lifetime boundaries. | Lower; 0 is perfect. |
| Macro F1 | Unweighted average F1 across the three lifetime labels; stored in `metrics_f1`. | Higher. |
| Quadratic weighted kappa | Ordered-label agreement; farther errors are penalized more. | Higher; it can be negative. |
| Log-duration MAE | Mean absolute error between predicted and representative lifetimes on a logarithmic day scale. | Lower. |
| Long-lived AUC | ROC AUC for `fresh_y` versus other labels, using its predicted probability; stored in `metrics_roc_auc`. | Higher. |

## AI Metrics page

Miniflux shows Freshness after Relevance and Urgency. Its Freshness section uses
RPS, Macro F1, and quadratic weighted kappa in the trend chart, and shows all
five freshness metrics in the latest-values and history table. Missing nullable
values display as `-` and are omitted from chart lines.

## Failure Behavior

The eval job writes the JSON history first, then inserts into `model_evals`.
PostgreSQL insert failures are not swallowed: the eval job fails visibly if the
Miniflux table is unavailable.
