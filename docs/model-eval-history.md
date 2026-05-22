# Model Evaluation History

Feedoscope's weekly eval job stores each model evaluation result in two places:

- `models/eval_history.json`, kept for the existing file-based history.
- Miniflux's `model_evals` PostgreSQL table, used as the durable history that
  Miniflux can display later.

## Database Ownership

The `model_evals` table is created by Miniflux migrations, not Feedoscope
migrations. Miniflux owns the schema because Miniflux is the future reader of the
table.

## Stored Shape

Each eval run inserts one row per model. `training` and `eval` are JSONB objects
because the class names differ by model:

- Relevance uses `good` and `bad` counts.
- Urgency uses `urgent` and `evergreen` counts.

The shared metrics are stored as flat columns:

- `metrics_accuracy`
- `metrics_precision`
- `metrics_recall`
- `metrics_f1`
- `metrics_roc_auc`
- `metrics_average_precision`
- `metrics_log_loss`

## Failure Behavior

The eval job writes the JSON history first, then inserts into `model_evals`.
PostgreSQL insert failures are not swallowed: the eval job fails visibly if the
Miniflux migration has not been deployed or the table is unavailable.
