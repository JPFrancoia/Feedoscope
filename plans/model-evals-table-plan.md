# Model Evals Table Plan

Status: Completed - 2026-05-22

## Brief

Create a Miniflux-owned `model_evals` table so weekly Feedoscope eval results can be stored in PostgreSQL and later displayed by Miniflux. Feedoscope will keep writing the JSON history file for now, but the database table must be created by Miniflux because Miniflux is the future reader of this data.

## Current State / Relevant Context

- `feedoscope/eval_models.py` currently writes eval history to `models/eval_history.json`.
- Each JSON record contains `date`, `model`, `training`, `eval`, and `metrics`.
- `training` and `eval` use model-specific count keys, for example `good`/`bad` for relevance and `urgent`/`evergreen` for urgency.
- Miniflux migrations are appended as Go functions in `/home/djipey/informatique/go/miniflux/internal/database/migrations.go`.

## Proposed Implementation

- Add a Miniflux migration that creates `model_evals`.
- Store `training` and `eval` as `jsonb`, because their keys differ by model.
- Store metrics as flat `double precision` columns because the metric set is shared and will be charted by Miniflux later.
- Add an index on `(eval_date DESC, model)` for future history views.

## File-By-File Impact

- `/home/djipey/informatique/go/miniflux/internal/database/migrations.go`: append a migration creating `model_evals`.
- `feedoscope/data_registry/sql/insert_model_eval.sql`: insert one eval row into `model_evals`.
- `feedoscope/data_registry/data_registry.py`: expose `insert_model_eval()` using JSONB adapters for the count blobs.
- `feedoscope/eval_models.py`: write the JSON history file and insert the same result into PostgreSQL.
- `docs/model-eval-history.md`: document the implemented persistence behavior.
- `docs/README.md`: index the eval history doc.

## Risks And Edge Cases

- Feedoscope must be deployed after Miniflux has run this migration, otherwise the DB insert will fail.
- Re-running an eval can create multiple rows for the same date and model. This matches the existing append-only JSON history behavior.
- Metrics are required, while `training` and `eval` remain flexible JSON objects.

## Validation / Testing

- Run `gofmt` on the modified Miniflux file.
- Run focused Go tests for the Miniflux database package if available.

## Step-By-Step Execution Checklist

- [x] Inspect Feedoscope eval JSON shape.
- [x] Inspect Miniflux migration convention.
- [x] Decide schema shape: `training`/`eval` jsonb, flat metrics columns.
- [x] Add Miniflux migration.
- [x] Validate Miniflux change with `go test ./internal/database`.
- [x] Add Feedoscope DB insert while preserving JSON output.
- [x] Document implemented eval history persistence.
- [x] Validate Feedoscope change with `uv run black . && uv run isort .` and `uv run --no-group infer mypy .`.

## Open Questions / Assumptions

- Assumption: duplicate rows for the same date/model are acceptable for now.
- Assumption: model names are limited to `Relevance` and `Urgency` for the initial table.
