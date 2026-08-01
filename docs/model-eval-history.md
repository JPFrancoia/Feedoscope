# Model Evaluation History

Feedoscope's weekly `make eval` job evaluates four models—Relevance,
Super-important, Urgency, and Freshness—and stores each result in two places:

- `models/eval_history.json`, kept for the existing file-based history.
- Miniflux's `model_evals` PostgreSQL table, used as the durable history that
  Miniflux can display later.

## Database Ownership and Deployment

Miniflux creates and migrates `model_evals` because it displays the durable
history on its AI Metrics page. Deploy Miniflux first so its migration adds the
Super-important metric columns, then deploy Feedoscope so the weekly job can
write those columns. Deploying Feedoscope first causes its insert to fail
against the older table.

## Stored Shape

Each eval run inserts one row per evaluated model. `training` and `eval` are
JSONB objects because the class names differ by model:

- Relevance uses `good` and `bad` counts.
- Super-important uses `good`, `bad`, `super_important`, and `ordinary_read`
  counts. Its row trains on the oldest 80% of mature labels and evaluates the
  newest 20%.
- Urgency uses `urgent` and `evergreen` counts.
- Freshness uses `fresh_d`, `fresh_m`, and `fresh_y` counts.

Relevance and urgency retain these binary-classification metric columns:

- `metrics_accuracy`, `metrics_precision`, `metrics_recall`
- `metrics_f1`, `metrics_roc_auc`, `metrics_average_precision`,
  `metrics_log_loss`

Super-important uses its own nullable columns:

- `metrics_super_important_average_precision`
- `metrics_relevance_average_precision`
- `metrics_recall_at_10`, `metrics_recall_at_25`, `metrics_recall_at_50`
- `metrics_super_important_bonus`

The bonus is the fixed `SUPER_IMPORTANT_BONUS` used for that evaluation, so old
rows remain interpretable if the configured value later changes.

Freshness reuses `metrics_f1` for Macro F1 and `metrics_roc_auc` for Long-lived
AUC. Its ordinal metrics use these nullable columns:

- `metrics_rps`
- `metrics_weighted_kappa`
- `metrics_log_duration_mae`

All metric columns are nullable so a model writes only metrics that apply.

## Super-important Metrics

| Metric | Meaning | Better direction |
|---|---|---|
| Preference AP | How well the combined score ranks explicitly preferred articles above ordinary read articles among read evaluation articles. | Higher. |
| Relevance AP | Whether the combined score still ranks readable articles above downvoted articles. | Higher. |
| Recall@10/25/50 | Share of all super-important evaluation articles in the first 10, 25, or 50 ranked results. | Higher. |
| Bonus | Fixed preference-probability adjustment used for this row. | Not a quality metric. |

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

Miniflux orders known sections as Relevance, Super-important, Urgency, and
Freshness; unknown model names follow in alphabetical order. The
Super-important section shows Preference AP, Relevance AP, and Recall@50 in its
trend chart. Its latest values and history table also show Recall@10, Recall@25,
the fixed bonus, and the training and evaluation class counts. The page gives a
plain-language explanation for every Super-important metric.

Freshness uses RPS, Macro F1, and quadratic weighted kappa in the trend chart,
and shows all five freshness metrics in its latest-values and history table.
Missing nullable values display as `-` and are omitted from chart lines.

## Failure Behavior

The eval job writes the JSON history first, then inserts into `model_evals`.
PostgreSQL insert failures are not swallowed: the eval job fails visibly if the
Miniflux table is unavailable or has not been migrated.
