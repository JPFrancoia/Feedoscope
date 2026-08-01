# Super-important AI metrics plan

**Status:** Completed — 2026-08-01

## 1. Brief

Add a dedicated **Super-important** section to Miniflux’s AI Metrics page and populate it from Feedoscope’s weekly evaluation. Each row will report the fixed-bonus model’s performance on the newest held-out articles, so the page shows how the actual model configuration performs over time rather than retuning or recommending a bonus.

## 2. Current state / relevant context

- Before this change, Feedoscope evaluated the super-important ranker with chronological data but only wrote its results to logs.
- The fixed production influence is `config.SUPER_IMPORTANT_BONUS`; this task does not change or tune it.
- The useful final evaluation is the existing second chronological window: fit on the oldest 80% of mature labels and evaluate on the newest 20%.
- The user selected the focused metric set: explicit-preference average precision, ordinary relevance average precision, and explicit-preference recall at 10, 25, and 50.
- Miniflux owns the `model_evals` table and already has model-specific Relevance/Urgency/Freshness sections.
- A row named `Super-important` would currently render through the generic binary UI with incorrect labels, so persistence and UI must change together.
- Feedoscope implementation is on the dedicated `super_important` branch. Miniflux implementation is in `/home/djipey/informatique/go/miniflux.super-important-ai-metrics` on branch `super-important-ai-metrics`, based on `freshness`.

## 3. Proposed implementation

### 3.1 Persist one weekly performance row

After the existing super-important evaluation fits the rankers on the oldest 80% and scores the newest 20%, calculate the final candidate with the fixed `config.SUPER_IMPORTANT_BONUS` and write one row:

```text
model = Super-important
training = oldest 80% class counts
eval = newest 20% class counts
metrics = fixed-bonus performance on the newest 20%
```

Persist these nullable values:

- `metrics_super_important_average_precision`
- `metrics_relevance_average_precision`
- `metrics_recall_at_10`
- `metrics_recall_at_25`
- `metrics_recall_at_50`
- `metrics_super_important_bonus`

The bonus is stored with the result so historical rows remain interpretable if configuration changes later. The existing rolling-window calculations and rollout logs remain unchanged; they are not displayed as extra rows. If the evaluator lacks enough mature examples, it continues to skip and writes no misleading row.

### 3.2 Extend the shared database row

Append a Miniflux migration adding the six nullable columns above to `model_evals`. Extend Miniflux’s typed model and storage SELECT/Scan path, then extend Feedoscope’s insert SQL and parameter mapping. Existing Relevance, Urgency, and Freshness rows write null for these fields and keep their current behavior.

Do not alias ranking values into F1, ROC AUC, or binary average-precision columns; the labels and meanings would be wrong.

### 3.3 Render a dedicated Miniflux section

Add `IsSuperImportant` view state and order sections as:

1. Relevance
2. Super-important
3. Urgency
4. Freshness
5. any unknown model names

The Super-important summary and chart show the three clearest trends:

- Preference AP
- Relevance AP
- Recall@50

The latest-values area and history table also show Recall@10, Recall@25, and the fixed bonus. Training/evaluation class counts continue using the existing JSON count formatter.

Add a Super-important chart mode to the existing lightweight JavaScript chart. All selected metrics are bounded from 0 to 1, so it reuses the normal 0–1 axis. Add concise English metric labels to every locale catalog, matching the existing fork-specific AI Metrics convention. Like the Freshness section, include one plain-language sentence explaining what each displayed metric means.

## 4. File-by-file impact

### Feedoscope

- `feedoscope/eval_models.py` — calculate and save fixed-bonus newest-window performance as `Super-important`.
- `feedoscope/data_registry/data_registry.py` — map the six new optional metrics.
- `feedoscope/data_registry/sql/insert_model_eval.sql` — insert the six new columns.
- `tests/test_eval_models.py` — check fixed-bonus persistence and nullable SQL bindings.
- `docs/model-eval-history.md` — document the new row, metrics, and schema ownership.
- `docs/super-important-ranker.md` — replace the log-only statement with the implemented weekly persistence behavior.
- `docs/README.md` only if its index wording needs adjustment.

### Miniflux

Implementation worktree: a new dedicated worktree based on `/home/djipey/informatique/go/miniflux` branch `freshness`.

- `internal/database/migrations.go` — append the six nullable columns.
- `internal/model/model_eval.go` — add typed nullable fields.
- `internal/storage/model_eval.go` — select and scan the new fields.
- `internal/ui/ai_metrics.go` — add canonical ordering, row fields, Super-important mode, and chart JSON.
- `internal/ui/ai_metrics_test.go` — check ordering and ranking chart data.
- `internal/template/templates/views/ai_metrics.html` — add the dedicated summary, latest metrics, chart legend, and table.
- `internal/ui/static/js/app.js` — select the three Super-important chart series while reusing the 0–1 axis.
- `internal/locale/translations/*.json` — add the new metric labels.
- `tests/ai_metrics_axis_test.js` — check selection of the three Super-important chart series.

### Infrastructure

- `/home/djipey/informatique/infra/manifests/perso/miniflux/base/feedoscope-eval-job.yaml` — set `SUPER_IMPORTANT_BONUS=0.5`, matching training and inference so weekly history measures the deployed fixed bonus.

## 5. Risks and edge cases

- **Cross-repository deployment order:** deploy the Miniflux migration before Feedoscope starts inserting the new columns.
- **Sparse positives:** the existing minimum-class checks remain the guard against unstable or undefined AP/recall values.
- **Historical meaning:** storing the fixed bonus beside metrics prevents later configuration changes from making old rows ambiguous.
- **Global row limit:** Miniflux fetches 200 evaluation rows across all models. One additional weekly row is acceptable; no pagination or retention change is needed now.
- **Chart semantics:** only coherent 0–1 ranking metrics are charted; class prevalence, precision, NDCG, and rolling tuner output remain logs to avoid a wide schema and noisy UI.

## 6. Validation / testing

### Feedoscope

- Add a focused test proving the saved row uses the oldest-80/newest-20 partition and `config.SUPER_IMPORTANT_BONUS`.
- Extend the SQL-binding test to verify new values and nulls for other models.
- Run `uv run --no-group infer pytest -q`.
- Run `uv run --no-group infer ruff check .` and `uv run --no-group infer mypy .`.
- Run Black/isort through `make format`, then confirm the diff is formatting-only where expected.

### Miniflux

- Add focused Go coverage for section ordering, metric mapping, and chart JSON.
- Run `gofmt` on changed Go files.
- Run `go test ./internal/storage ./internal/template ./internal/locale ./internal/ui` and the relevant JavaScript test if changed.
- Run `git diff --check` in both repositories.

### Integrated check

An isolated PostgreSQL 18 smoke test migrated Miniflux to schema version 136, confirmed all six columns, inserted a representative `Super-important` row, logged in as the admin, and rendered `/ai-metrics`. The rendered HTML contained the dedicated section, all metric labels and explanations, class counts, chart data, and expected values. ProofShot was unavailable on this machine, so no screenshot/video bundle was produced.

## 7. Step-by-step execution checklist

- [x] Inspect Feedoscope’s current super-important evaluator and persistence path.
- [x] Inspect Miniflux’s current AI Metrics schema, storage, view, template, chart, and tests.
- [x] Confirm each row represents fixed-bonus model performance after evaluation.
- [x] Confirm the focused metric set.
- [x] Obtain plan approval. The user also requested a one-sentence plain-language description for every displayed metric, matching the Freshness section.
- [x] Create a dedicated Miniflux task worktree from `freshness` at `/home/djipey/informatique/go/miniflux.super-important-ai-metrics`.
- [x] Add Miniflux schema/model/storage support.
- [x] Add Feedoscope fixed-bonus row persistence.
- [x] Add the dedicated Miniflux section, chart mode, and one-sentence metric explanations.
- [x] Add focused tests in both repositories.
- [x] Set the weekly eval CronJob to the same fixed bonus `0.5` used by training and inference.
- [x] Run repository validation: Feedoscope 43 tests, Ruff, mypy, Black/isort, and diff check; Miniflux targeted Go packages, five JavaScript tests, gofmt, locale/template checks, and diff check; infrastructure Kustomize render and diff check.
- [x] Run an isolated PostgreSQL migration/login/render smoke test against schema version 136.
- [x] Update durable Feedoscope documentation after implementation.
- [x] Run independent Python, Go, and cross-repository review passes; fix the eval-job bonus, Preference AP wording, docs, and newest-window test evidence; rerun affected validation.
- [x] Mark this plan completed with the final validation evidence.

## 8. Open questions / assumptions

- Confirmed: the bonus is fixed; the page reports model performance and does not recommend or tune it.
- Confirmed: display key signals rather than every precision/recall/NDCG diagnostic.
- Assumption: the exact persisted model name is `Super-important`, matching existing evaluator logs.
- Assumption: Recall@10/25/50 means the share of all super-important holdout articles captured within the first 10/25/50 ranked results.
- Assumption: the fixed bonus is displayed but not charted.
- Confirmed during review: the weekly eval CronJob must set `SUPER_IMPORTANT_BONUS=0.5`; otherwise its rows would measure the default zero bonus rather than deployed behavior.
