# Freshness AI metrics plan

**Status:** Completed — 2026-08-01

## Completion evidence

- Feedoscope implementation: `/home/djipey/informatique/python/feedoscope.freshness-ai-metrics` on `freshness`.
- Miniflux implementation: `/home/djipey/informatique/go/miniflux.freshness-ai-metrics` on `freshness`.
- Feedoscope: 23 focused tests passed; Black, isort, Ruff, mypy, and `git diff --check` passed.
- Miniflux: `go test ./...`, locale-key validation, gofmt, and `git diff --check` passed.
- An isolated PostgreSQL 17 smoke test migrated Miniflux to schema version 135, inserted Relevance/Urgency/Freshness rows, logged in as an admin, and verified the rendered AI Metrics HTML, nullable dashes, and gap-safe Freshness chart JSON with no server errors.
- Independent Python and Go reviewers approved the final diffs with no remaining findings.
- Durable Feedoscope docs now describe the implemented evaluator, schema contract, metrics, and deployment order.
- ProofShot was unavailable on this machine, so no screenshot/video bundle was produced. Real deployment still must migrate Miniflux before deploying the Feedoscope writer, then run `make eval`.

### Implementation deviation

The final UI also treats undefined weighted kappa as missing: it renders `-` and leaves a gap in the chart instead of coercing the value to zero. This is necessary because a single-class truth/prediction pair makes kappa undefined; the original plan explicitly required this behavior only for Long-lived AUC.

## 1. Brief

Add a real Freshness section to Miniflux's existing AI metrics page, backed by the active three-label Feedoscope model rather than an empty UI card. The weekly eval job will measure freshness on the newest labeled articles, store one `model='Freshness'` row beside Relevance and Urgency, and let Miniflux render ordinal metrics that match the `fresh_d` / `fresh_m` / `fresh_y` model.

## 2. Current state / relevant context

- Feedoscope's active freshness model predicts three ordered labels: `fresh_d`, `fresh_m`, and `fresh_y`.
- `feedoscope.eval_models` currently evaluates only Relevance and Urgency and writes seven binary-classification metrics to `model_evals`.
- The effective freshness-label query already returns articles in ascending publication order, so its newest rows can form a chronological holdout without a new query.
- Miniflux owns and creates the `model_evals` table. Its current schema requires every binary metric and has no ordinal-metric columns.
- Miniflux's AI metrics page groups rows by model but uses the same binary labels, table, and chart for every model.
- Historical Miniflux commit `96e5b18f` contains a useful Freshness-specific UI pattern, but it targets the superseded six-label design and assumes schema columns that are not currently deployed.
- Current durable docs explicitly say Freshness does not write model-evaluation rows; those docs must change only after the new path is implemented.

### Success criteria

- `make eval` writes one Freshness evaluation row in addition to Relevance and Urgency when enough labels exist.
- The Freshness evaluation uses a chronological holdout and never changes the production freshness artifact.
- The Miniflux page orders sections as Relevance, Urgency, Freshness, then any unknown models.
- Freshness displays RPS, Macro F1, quadratic weighted kappa, log-duration MAE, and long-lived (`fresh_y`) AUC; binary sections remain unchanged.
- Undefined long-lived AUC is stored as `NULL` and rendered as `-`, not `0`.
- Focused Python and Go tests, formatting, linting, and relevant migration checks pass.

## 3. Proposed implementation

### 3.1 Evaluate the active three-label model

Extend `feedoscope/eval_models.py` with one Freshness evaluation path:

1. Load the same effective labels used by production freshness training.
2. Use the newest `VALIDATION_SIZE` rows as the eval set and every older row as training data. The SQL is already ordered by `published_at, id`.
3. Fit the existing two cumulative logistic classifiers in memory with `semantic_freshness_embedding.fit_classifiers`.
4. Predict three-class probabilities for the held-out articles.
5. Persist class counts and five ordinal diagnostics:

| Stored/displayed metric | Calculation | Direction |
|---|---|---|
| RPS | Mean squared cumulative probability error across the two three-label boundaries, divided by 2 | Lower is better |
| Macro F1 | Macro-average F1 over predicted labels | Higher is better |
| Weighted kappa | Quadratic Cohen's kappa over ordered labels | Higher is better |
| Log-duration MAE | Mean absolute error between log predicted expected lifetime and log representative lifetime (`7`, `90`, `365`) | Lower is better |
| Long-lived AUC | `fresh_y` versus the other two labels using the `fresh_y` probability | Higher is better; nullable when the holdout has one side only |

Reuse `VALIDATION_SIZE` rather than adding another environment variable. The existing `make eval` value of 150 therefore applies consistently to all three models. Freshness fitting remains temporary and in memory, so production artifacts are untouched and no new cleanup path is needed.

### 3.2 Extend the shared evaluation row

Append a Miniflux database migration after the current `model_evals` creation migration. The migration will:

- make binary-only columns nullable (`accuracy`, `precision`, `recall`, `average_precision`, and `log_loss`);
- keep `metrics_f1` nullable and reuse it as Macro F1 for Freshness;
- keep `metrics_roc_auc` nullable and reuse it as long-lived AUC for Freshness;
- add nullable `metrics_rps`, `metrics_weighted_kappa`, and `metrics_log_duration_mae` columns.

Miniflux remains the sole schema owner. Feedoscope's migration version stays at 6; its insert SQL and Python binding will only be deployed after the Miniflux migration exists.

Generalize Feedoscope's `insert_model_eval()` binding with `metrics.get(...)`. Relevance and Urgency continue supplying all seven binary values; Freshness supplies `macro_f1`, `long_lived_auc`, `rps`, `weighted_kappa`, and `log_duration_mae`. The JSON history keeps the same record shape and naturally stores nullable metrics as JSON `null`.

### 3.3 Render a model-specific Freshness section

Adapt the small, proven parts of historical Miniflux commit `96e5b18f`:

- make metric fields nullable in the Go model and scan the three new columns;
- mark Freshness view data explicitly and place it after Urgency;
- show RPS, Macro F1, and weighted kappa in the summary card;
- show all five Freshness diagnostics in the detail panel and table;
- chart RPS, Macro F1, and weighted kappa, retaining a `-1..1` scale because weighted kappa may be negative;
- label RPS and log-duration MAE as lower-is-better so direction is not implied incorrectly;
- render missing long-lived AUC as `-`;
- leave Relevance/Urgency cards, tables, and `0..1` charts unchanged.

Add the five Freshness metric labels to every embedded locale catalog, following the current custom-page convention of using the English text in all catalogs.

## 4. File-by-file impact

### Feedoscope

- `feedoscope/eval_models.py` — three-label metric calculation, chronological holdout, temporary fit/inference, Freshness persistence, and main/docstring updates.
- `feedoscope/data_registry/data_registry.py` — optional/model-specific metric bindings.
- `feedoscope/data_registry/sql/insert_model_eval.sql` — insert the three new ordinal columns while reusing F1 and ROC AUC aliases.
- `tests/test_eval_models.py` — focused checks for perfect and imperfect three-label metrics, RPS normalization, nullable long-lived AUC, and chronological split behavior.
- `docs/model-eval-history.md` — document Freshness rows, nullable binary fields, aliases, and schema ownership.
- `docs/semantic-freshness.md` — document weekly chronological evaluation and metric meanings.
- `docs/README.md` — update only if its existing index text needs adjustment; no new durable doc is planned.

### Miniflux

- `internal/database/migrations.go` — append the `model_evals` extension migration.
- `internal/model/model_eval.go` — nullable shared metrics and three ordinal fields.
- `internal/storage/model_eval.go` — select/scan the new nullable columns.
- `internal/ui/ai_metrics.go` — Freshness ordering, row/view state, null handling, and chart JSON.
- `internal/template/templates/views/ai_metrics.html` — conditional Freshness summary/detail/chart/table.
- `internal/ui/static/js/app.js` — Freshness chart series and `-1..1` scale without changing binary charts.
- `internal/ui/ai_metrics_test.go` — ordering, chart-data, and nullable-AUC checks.
- `internal/locale/translations/*.json` — five Freshness labels in every catalog.

No dependency is added in either repository.

## 5. Risks and edge cases

- **Deployment order:** old Miniflux binaries cannot read the new columns, and new Feedoscope code cannot insert them until the schema migration runs. Deploy/migrate Miniflux first, then Feedoscope, then run `make eval`.
- **Small temporal holdout:** the training side must contain both classes at each cumulative boundary. If it does not, Freshness evaluation logs and skips rather than affecting Relevance/Urgency evaluation.
- **Missing eval classes:** long-lived AUC is undefined when the holdout contains only `fresh_y` or no `fresh_y`; store `NULL` and display `-`.
- **Teacher-label bias:** most labels are Luna bootstrap labels, so the metrics measure agreement with the effective label set, not independent human truth. The training/eval count maps keep the three class counts visible.
- **Mixed metric direction:** RPS and log-duration MAE improve downward while the other metrics improve upward. Labels must say so; no composite score or trend interpretation is added.
- **Historical code mismatch:** do not cherry-pick `96e5b18f` wholesale. Reuse only its model-specific UI structure and adapt wording/formulas to the active three-label model.
- **Schema rollback:** the Miniflux migration framework is forward-only. The new migration must be safe on the current schema and must not rewrite the existing table-creation migration.

## 6. Validation / testing

### Feedoscope

- Add a hand-checkable perfect-prediction test: RPS `0`, Macro F1 `1`, weighted kappa `1`, log-duration MAE `0`, long-lived AUC `1`.
- Add a one-sided holdout test proving long-lived AUC is `None`.
- Verify the chronological split uses the last `VALIDATION_SIZE` rows and leaves production artifacts untouched.
- Run:
  - `uv run --no-group infer pytest tests/test_eval_models.py tests/test_semantic_freshness.py`
  - `make format`
  - `uv run ruff check .`
  - `make lint`

### Miniflux

- Test Relevance/Urgency/Freshness ordering and Freshness chart JSON.
- Test nullable long-lived AUC view state.
- Run:
  - `gofmt` on changed Go files
  - `go test ./internal/ui ./internal/storage ./internal/model`
  - `make lint` if the full configured toolchain is available
  - `go test ./...` when practical

### Integrated check

1. Start/deploy Miniflux so its appended migration updates `model_evals`.
2. Run Feedoscope `make eval`.
3. Query the latest three rows and confirm Freshness has null binary-only values and populated ordinal values.
4. Open the admin AI metrics page and verify all three sections, labels, legends, tooltips, tables, and `-` rendering.

## 7. Step-by-step execution checklist

- [x] Recover recent three-label freshness decisions and the superseded six-label UI work.
- [x] Inspect both repositories, current schemas, eval paths, docs, tests, and worktree state.
- [x] Define the three-label metric contract and Miniflux-owned migration approach.
- [x] Obtain user approval for this plan.
- [x] Create one dedicated implementation worktree per repository from `semantic_freshness` (Feedoscope) and `main` (Miniflux).
- [x] Implement and test Feedoscope's Freshness evaluator and generalized insert.
- [x] Implement and test the Miniflux migration, reader, model-specific view, template, chart, and locales.
- [x] Run focused and broader validation in both worktrees.
- [x] Review the combined diffs for correctness, regression risk, and unnecessary complexity.
- [x] Update current-state docs in `docs/` after implementation passes.
- [x] Mark this plan completed with the date and record deviations/validation evidence.

## 8. Open questions / assumptions

- Assumption: the requested section should be end-to-end and populated, not a Miniflux-only empty card.
- Assumption: reuse the existing `VALIDATION_SIZE=150` for Freshness rather than add configuration.
- Assumption: display `fresh_y` discrimination as **Long-lived AUC**, not the superseded **Evergreen AUC** wording.
- Assumption: Miniflux continues to own `model_evals` schema changes, matching `docs/model-eval-history.md`.
- Assumption: the five historical ordinal diagnostics remain useful after adapting RPS normalization and wording to three labels.
