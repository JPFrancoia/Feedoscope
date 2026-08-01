# Three-label freshness redesign

**Status:** Completed — 2026-08-01

## Completion evidence

- Production was reset from migration 9 to migration 5, then migrated to
  `version=6, dirty=false` with `000006_three_label_freshness`.
- The preserved pre-reset export is
  `/home/djipey/backups/feedoscope/feedoscope-freshness-pre-reset-20260801T090612Z`.
- Production has exactly 1,200 `gpt-5.6-luna` bootstrap rows and only the three
  manual tags: `fresh_d`, `fresh_m`, and `fresh_y`. They were initially unlinked;
  subsequent links are user-created labels.
- The old freshness model-eval schema and UI are removed and not deployed.
- Controlled inference scored 8,795 articles without changing tag links.
  A repeat training run skipped because the effective-label fingerprint was
  unchanged.
- Focused tests (25 passed), Ruff, and mypy passed.
- All Feedoscope CronJobs use image `freshness-3label-20260801-r2`.

## 1. Brief

Replace the current six-horizon, automatically tagged freshness system with three manual labels: `fresh_d`, `fresh_m`, and `fresh_y`. Bootstrap 1,200 articles with `gpt-5.6-luna`, train two ordered classifiers from those bootstrap labels plus manually tagged read articles, and keep model predictions as the relevance-decay input without writing prediction tags back to Miniflux. Reset the deployed database from migration 9 to a clean migration 5, then apply one new migration 6 for this simpler design.

## 2. Current state / relevant context

- Production `schema_migrations` is currently `version=9, dirty=false`.
- Freshness migrations are the complete tail `000006`–`000009`:
  - `000006` creates teacher-label and inference tables.
  - `000007` changes `model_evals` for ordinal freshness metrics.
  - `000008` restricts teacher rows to high confidence.
  - `000009` renames six reviewed and six automatic tags.
- Production currently contains 899 bootstrap labels, 19,264 persisted inference rows, six reviewed tags, and six automatic tags. Automatic tags are attached to thousands of entries; only four reviewed freshness tags are attached.
- `feedoscope.main` predicts six freshness buckets, persists predictions, writes `fresh-auto-*` tags, and uses expected lifetime as the relevance-score half-life.
- Freshness training currently promotes read automatic tags into reviewed labels before fitting five cumulative logistic heads.
- The existing private sample at `.auto/data/teacher_sample.csv` already contains exactly 1,200 articles. The existing label-generation script can call Pi models but currently targets six horizons and defaults to `gpt-5.4`.
- A weekly Kubernetes CronJob already invokes `feedoscope.llm_learn_semantic_freshness` every Sunday at 05:00. No new scheduler is needed.

### Success criteria

- Miniflux exposes only `fresh_d`, `fresh_m`, and `fresh_y`; no freshness `auto` tags remain or are recreated.
- The bootstrap table contains exactly 1,200 labels produced by `openai-codex/gpt-5.6-luna`.
- Training data is the bootstrap set plus read articles carrying exactly one manual `fresh_*` label; a manual read label overrides its bootstrap label.
- Routine inference predicts freshness for score decay but never changes Miniflux tags.
- Repository migrations contain no old freshness migrations; the only post-`000005` migration is the new design's `000006` pair.
- Production ends at `schema_migrations.version=6, dirty=false` and matches a fresh replay of migrations 1–6.
- Focused tests, formatting, Ruff, and mypy pass.

## 3. Proposed implementation

### 3.1 Labels and model

Use three ordered labels with two cumulative boundaries:

| Label | Meaning used by Luna and training | Representative half-life |
|---|---|---:|
| `fresh_d` | Main claim remains useful for 0–29 days | 7 days |
| `fresh_m` | Main claim remains useful from 30 days through 6 months | 90 days |
| `fresh_y` | Main claim remains useful beyond 6 months | 365 days |

The 7/90/365-day values preserve the current expected-lifetime calculation with the smallest model change. They are representative values for decay, not hard expiry dates.

Keep the selected ordered-logistic approach, reduced from five heads to two:

1. `P(useful beyond 30 days)`
2. `P(useful beyond 6 months)`

Sort the two cumulative probabilities into descending order, convert them into three class probabilities, then calculate expected lifetime from `[7, 90, 365]`. This preserves ordinal behavior and keeps the existing in-memory relevance-decay path.

### 3.2 Bootstrap labels

Adapt `experiments/freshness/build_labels.py` to:

- use only `fresh_d`, `fresh_m`, and `fresh_y`;
- use the exact boundaries above in the prompt;
- default to `gpt-5.6-luna`;
- label the existing fixed 1,200-article sample;
- require one valid label for every article and retain evidence/provenance in the private CSV;
- write a new output filename so stale six-horizon labels cannot be reused accidentally.

Import all 1,200 accepted labels into the dedicated table with source `gpt-5.6-luna`. The database stores only the article ID, three-class label, source, and timestamp; confidence and experiment-only evidence stay in the private CSV.

### 3.3 Training data

Replace auto-tag promotion with one effective-label query:

1. Start with every bootstrap row, regardless of current read status.
2. Add every read article carrying exactly one of `fresh_d`, `fresh_m`, or `fresh_y`.
3. If a read article appears in both sets, use the manual tag.
4. If a read article has multiple freshness tags, exclude it and log a warning.
5. Ignore manual freshness tags on unread articles until they become read.

Fingerprint the effective article IDs, labels, sources, and encoder/model configuration. The existing weekly job skips fitting when the fingerprint has not changed and publishes a new artifact when it has.

### 3.4 Routine inference

Keep freshness inference in `feedoscope.main` because the user approved freshness-driven relevance decay. Remove all persistence and tag-writing side effects:

```text
active articles -> embeddings -> 3 freshness probabilities -> expected lifetime
               -> relevance decay in memory -> final score
```

Do not create an inference table. It has no current reader needed for scoring, and removing it avoids storing 19,000+ unused audit rows. The standalone `infer_freshness` command may remain as a prediction smoke run, but it will only log completion and will not mutate Miniflux.

### 3.5 Clean migration history and production database

Prepare and validate the code before touching production. During the maintenance step:

1. Suspend the freshness training, model evaluation, and hourly inference CronJobs so old code cannot write while schema history is reset.
2. Save a targeted pre-reset export of the old freshness tables, freshness tag rows/links, freshness evaluation rows, and `schema_migrations` state outside the repository.
3. Restore the old migration files from the pre-change Git revision into a temporary migration directory, then run four normal down migrations (`9 -> 5`) so golang-migrate performs each registered rollback and keeps its version table authoritative.
4. Verify `schema_migrations` is exactly `version=5, dirty=false`.
5. In one transaction, delete all legacy reviewed/automatic freshness tag links and tag rows left after migration 9's reverse rename.
6. Delete repository migration pairs `000006`–`000009` and add one new pair: `000006_three_label_freshness.{up,down}.sql`.
7. The new up migration creates `freshness_bootstrap_labels` with a label check covering `fresh_d`, `fresh_m`, and `fresh_y`, and creates the three Miniflux tag definitions for user 1. It does not create inference storage or modify `model_evals`.
8. Apply the new migration 6 and verify `version=6, dirty=false`, exact table constraints, and exactly the three new tag names.
9. Import the 1,200 Luna labels, build/push the new image, run a one-off bootstrap training job, and verify the new three-class artifact loads.
10. Resume the CronJobs and run one controlled full inference.

If any down migration fails, stop immediately, inspect `schema_migrations`, and repair with the smallest explicit SQL/`migrate force` operation needed to match the observed schema. Never continue to the replacement migration while the table is dirty.

## 4. File-by-file impact

### Migrations

- Delete `db/migrations/000006_semantic_freshness.{up,down}.sql`.
- Delete `db/migrations/000007_model_eval_freshness_metrics.{up,down}.sql`.
- Delete `db/migrations/000008_high_confidence_semantic_freshness_labels.{up,down}.sql`.
- Delete `db/migrations/000009_rename_semantic_freshness_tags.{up,down}.sql`.
- Add `db/migrations/000006_three_label_freshness.{up,down}.sql` for the bootstrap table and three manual tags.

### Model and training

- `feedoscope/semantic_freshness_embedding.py` — three labels, two thresholds, `[7, 90, 365]`, compatible artifact key/metadata, and three-bucket conversion.
- `feedoscope/llm_learn_semantic_freshness.py` — remove auto promotion and six-class metrics; train from the effective bootstrap/manual dataset.
- `feedoscope/import_semantic_freshness_bootstrap_labels.py` — validate and atomically replace all 1,200 Luna bootstrap rows.
- `experiments/freshness/build_labels.py` — three-label Luna prompt and fresh output path.
- `feedoscope/config.py` — remove six-horizon evaluation settings that are no longer operational; keep only classifier settings used by training.

### Database access

- `feedoscope/data_registry/data_registry.py` — remove automatic-tag, promotion, inference-storage, and inference-read functions; retain bootstrap import and effective training fetch.
- `feedoscope/data_registry/sql/get_semantic_freshness_training.sql` — merge bootstrap labels with read manual tags using manual precedence.
- `feedoscope/data_registry/sql/get_conflicting_semantic_freshness_labels.sql` — recognize only the three manual labels.
- Replace the old teacher-label upsert with separate delete/insert SQL so bootstrap imports atomically replace the fixed set.
- Delete SQL used only for tag creation/promotion, automatic assignment, and inference persistence/read.
- Restore `feedoscope/data_registry/sql/insert_model_eval.sql` to the pre-freshness schema.

### Inference and orchestration

- `feedoscope/entities.py` — change the result description/shape from six to three buckets.
- `feedoscope/llm_infer_semantic_freshness.py` — predict/log only; no DB persistence or tag writes.
- `feedoscope/main.py` — keep expected-lifetime decay, remove inference persistence and all freshness tag writes.
- `feedoscope/eval_models.py` — remove the old six-class freshness evaluation path because the replacement does not alter `model_evals`.
- `Makefile` and `run.sh` — keep the existing freshness training command; keep `infer_freshness` only as a smoke command.

### Tests and documentation

- `tests/test_semantic_freshness.py` — update target/probability/artifact checks for three classes and assert no `auto` tag/write SQL remains.
- `tests/test_main.py` — retain expected-lifetime decay and failure fallback checks with the new results.
- `docs/semantic-freshness.md` — document only the implemented three-label/manual-training behavior after code and DB rollout complete.
- `docs/model-eval-history.md` and `docs/README.md` — remove stale claims about freshness-specific evaluation columns while preserving the freshness doc index.
- Mark `plans/intrinsic-semantic-freshness-implementation-plan.md` and `plans/semantic-freshness-score-rollout-plan.md` as superseded by this plan; do not delete them.

No new Python dependency or new Kubernetes manifest is required. The existing weekly CronJob already has the requested cadence.

## 5. Risks and edge cases

- **Destructive reset:** migrations 6–9, old bootstrap labels, persisted predictions, freshness metric rows, and old tag links will be removed. A targeted pre-reset export provides a recovery point.
- **Migration dirtiness:** a failed `down` marks golang-migrate dirty. Every step must verify both schema and version before continuing.
- **Old jobs during reset:** current hourly inference expects the old tables/tags. Suspend jobs before migration rollback and resume only after the new image and artifact exist.
- **Artifact collision:** the new artifact family/configuration key must encode the new thresholds so the six-class artifact can never be loaded as a three-class model.
- **Manual/bootstrap overlap:** a manually tagged read article must replace, not duplicate, its bootstrap row.
- **Conflicting manual tags:** articles with more than one of the three tags are excluded and logged instead of guessed.
- **Class collapse:** both cumulative boundaries need positive and negative examples. Training fails clearly if Luna/manual labels do not cover all three useful ranges.
- **`fresh_y` wording:** the approved boundary is beyond six months even though the label says years; the prompt and docs must state this directly.
- **No prediction history:** removing the inference table gives up model-keyed prediction auditing. Add storage later only if an actual consumer needs it.

## 6. Validation / testing

- Verify the Luna CSV has exactly 1,200 unique article IDs and only the three allowed labels.
- Verify target construction for labels 0/1/2 and both cumulative boundaries.
- Verify three probabilities are finite, non-negative, and sum to one.
- Verify artifact round-trip and incompatibility with old six-class metadata.
- Verify training SQL behavior for bootstrap-only, manual-only read, manual override, unread manual tag, and conflicting manual tags.
- Run `uv run --no-group infer pytest tests/test_semantic_freshness.py tests/test_main.py`.
- Run `make format`.
- Run `uv run ruff check .`.
- Run `make lint`.
- Replay migrations 1–6 against a disposable PostgreSQL/Miniflux schema if available.
- In production, verify after reset:
  - migration `6`, `dirty=false`;
  - exactly 1,200 bootstrap rows;
  - only `fresh_d`, `fresh_m`, `fresh_y` freshness tags;
  - no semantic freshness inference table or freshness-specific model-eval columns;
  - one controlled inference updates scores without changing tag counts.

## 7. Step-by-step execution checklist

- [x] Inspect current freshness code, migrations, history, scheduler, and live database state.
- [x] Confirm three labels and no automatic freshness tags.
- [x] Confirm weekly training uses bootstrap labels plus manually tagged read articles.
- [x] Confirm freshness inference still drives relevance-score decay.
- [x] Confirm label boundaries: days 0–29, months 30 days–6 months, years beyond 6 months.
- [x] Obtain user approval for this plan.
- [x] Update the Luna bootstrap generator and produce/validate 1,200 private labels.
- [x] Add/update focused tests for the three-label contract.
- [x] Simplify the classifier artifact, training query, importer, and registry functions.
- [x] Remove automatic tag assignment and inference persistence from standalone/full inference.
- [x] Remove old freshness evaluation integration and obsolete SQL.
- [x] Replace migrations 6–9 in the repository with the new migration 6 pair.
- [x] Run formatting, tests, Ruff, mypy, and disposable migration replay.
- [x] Suspend production jobs and save the targeted pre-reset export.
- [x] Roll production migration 9 down to 5 and verify `dirty=false`.
- [x] Remove legacy freshness tags, apply new migration 6, and verify `dirty=false`.
- [x] Import exactly 1,200 Luna labels.
- [x] Build/push the image and run one-off three-label freshness training.
- [x] Run controlled inference, verify scores change without tag writes, and resume jobs.
- [x] Update durable docs and mark this plan completed with the completion date.

## 8. Open questions / assumptions

- Assumption: use 7, 90, and 365 days as representative half-lives for the approved ranges. These affect decay smoothness but not the Luna labels or classifier boundaries.
- Assumption: reuse the existing fixed 1,200-article private sample rather than selecting a different 1,200 articles.
- Assumption: all valid Luna labels are bootstrap training rows; there is no confidence filter in the replacement design.
- Assumption: user 1 remains the Miniflux owner of the three tag definitions, matching the existing deployment.
- Assumption: the existing Sunday 05:00 CronJob is the requested weekly training operator.
