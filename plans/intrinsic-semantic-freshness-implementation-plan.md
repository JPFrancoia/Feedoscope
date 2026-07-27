# Intrinsic Semantic Freshness Implementation Plan

**Status:** Grill completed — implementation-ready and awaiting user approval (2026-07-26)

## 1. Brief

Add a production sibling to the existing relevance and urgency pipelines that predicts how long an article's main claim remains useful. It will reuse the frozen EmbeddingGemma vectors, train five ordered logistic classifiers, and keep urgency as a separate concept; the implementation should initially prove and expose freshness without silently changing relevance decay unless that rollout is explicitly approved.

## 2. Current state / relevant context

- Relevance and urgency already share normalized 768-dimensional EmbeddingGemma vectors and a PostgreSQL embedding cache.
- Production urgency is binary: `0-urgency` versus `1-urgency`. Its probability is used to interpolate a 10–120 day relevance half-life in `feedoscope/main.py`.
- The autoresearch benchmark used six semantic-lifetime buckets and selected five cumulative logistic classifiers. On the frozen newest-30% split it reduced RPS from `0.21905157` for urgency-mapped lifetime to `0.08682275`.
- The selected candidate uses `C=20`, no intercept, fractional boundary class weights with exponent `3/8`, and descending sorting of the five boundary probabilities before deriving six bucket probabilities.
- The benchmark labels are teacher-generated and the test set has been consulted repeatedly. A new untouched, preferably human-reviewed temporal holdout is required before claiming production generalization.
- Master has no semantic-freshness label source, entity, artifact, database table, command, scheduler, or tests.
- This repository has manual Make/run entrypoints but no Kubernetes CronJob or other executable schedule. Scheduling must be wired in the external deployment repository or operator environment.
- A dedicated uncommitted implementation worktree exists at `/home/djipey/informatique/python/feedoscope.intrinsic-semantic-freshness` on branch `feature/intrinsic-semantic-freshness`.
- The requested plain-language explanation is already exported there as `explanation.md`.

## 3. Proposed implementation

### 3.1 Labels and training data

Reviewed labels and unreviewed model predictions use separate Miniflux tag families.

Reviewed tags, either set explicitly by the user or accepted when an article is read:

- `lt-24h-freshness`
- `1-3d-freshness`
- `4-7d-freshness`
- `8-30d-freshness`
- `1-6m-freshness`
- `evergreen-freshness`

Automatic tags, maintained only by freshness inference:

- `lt-24h-auto-freshness`
- `1-3d-auto-freshness`
- `4-7d-auto-freshness`
- `8-30d-auto-freshness`
- `1-6m-auto-freshness`
- `evergreen-auto-freshness`

Inference ensures one current auto tag per scored unread article by replacing only older auto tags. It never removes or edits a reviewed tag. A reviewed and auto tag may coexist; the reviewed tag takes precedence everywhere an effective horizon is needed.

Reading an article means its current freshness horizon is accepted for future training. An unread auto tag is not a training label. At the start of each freshness-training run, a preflight query finds read articles that have exactly one auto tag and no reviewed tag, replaces that auto tag with the equivalent reviewed tag in one transaction, and then loads training data. If a reviewed tag already exists, it is left untouched and takes precedence; the remaining auto tag may stay visible. Training uses a reviewed tag when present; otherwise it may use the initial medium/high-confidence bootstrap teacher label stored with provenance in a small `semantic_freshness_teacher_labels` table. If an article has multiple reviewed freshness tags, exclude that article from the training set and log its ID/title as a warning; do not fall back to its teacher label. The original frozen teacher labels are imported once into that table, not presented as reviewed tags.

Training flow:

1. Promote read auto-only tags to their equivalent reviewed tags, then fetch read articles with one effective label in deterministic publication order: reviewed tag first, otherwise bootstrap teacher label.
2. Build/reuse each article's cached EmbeddingGemma vector once.
3. Convert each six-bucket label into five binary targets:
   - useful beyond 24 hours;
   - useful beyond 3 days;
   - useful beyond 7 days;
   - useful beyond 30 days;
   - useful beyond 6 months.
4. Require both target classes at every boundary; fail clearly if labels are insufficient.
5. Fit five `LogisticRegression(C=20, fit_intercept=False)` heads with the selected fractional boundary weighting.
6. Validate on a chronological holdout with RPS as primary and macro F1, quadratic weighted kappa, log-duration MAE, and evergreen AUC as diagnostics.
7. Save all five heads atomically in one `joblib` artifact with metadata: thresholds, encoder/preparation config, class counts, label-source counts, dataset fingerprint, validation metrics, and model key.
8. After the new untouched holdout approves the fixed method, train the production artifact on the approved training pool. Do not repeatedly tune against that holdout.

### 3.2 Inference and outputs

Routine inference targets the existing **Active unread set**: every unread article from the last 40 days plus the rotating sample of older unread articles already used by full inference. Automatic tags remain on an article after it becomes read; the system does not scan or backfill the complete article database.

For each selected article:

1. Reuse the shared cached embedding.
2. Predict the five `P(useful beyond threshold)` values.
3. Sort them descending so longer-horizon probability cannot exceed shorter-horizon probability.
4. Subtract adjacent values to obtain six bucket probabilities that sum to one.
5. Optionally derive an expected lifetime in days from fixed representative bucket values `[0.5, 2, 5.5, 19, 90, 365]`.

Recommended storage: a model-keyed `semantic_freshness_inference` table containing:

- `article_id`
- `model_key`
- six bucket probabilities as a six-element `double precision[]`
- `expected_lifetime_days` as a derived convenience value
- `last_updated`
- primary key `(article_id, model_key)`

Store the full distribution, not only one score, because the six probabilities are the actual calibrated output and permit later UI display, auditing, remapping, and decay changes without rerunning the model. If there is no consumer for audit/UI and the user chooses minimum storage, the alternative is to compute the distribution in `main.py` and retain nothing.

### 3.3 Rollout and relevance decay

Recommended first rollout: shadow mode.

- Run freshness inference on the same article set as urgency/relevance.
- Persist freshness predictions under their model key.
- Leave the existing urgency-driven relevance decay unchanged initially.
- Compare predictions against newly reviewed labels and inspect how a lifetime-derived decay would affect ordering.

Only after acceptance should `main.py` replace urgency-based decay. The proposed mapping is to use expected lifetime days as the relevance half-life, with explicit configurable minimum/maximum clamps. Urgency remains available for urgency-specific features and is not deleted.

### 3.4 Retraining

Approved policy:

- Expose `make train_freshness` and `run.sh train_freshness` as idempotent commands.
- Compute a deterministic fingerprint from article IDs, labels, confidence/source, and encoder/preparation configuration.
- If the fingerprint matches the active artifact, log "no new labels" and skip training.
- If it changed, train and validate a new dated/model-keyed artifact. Structural failures (missing boundary classes, invalid probabilities, artifact write failure) stop publication, but metric regressions do not: every valid weekly candidate supersedes the previous active model.
- Run the command weekly in the external scheduler, but actual model fitting occurs only when effective labels changed.
- Retain at least the new active and immediately previous artifact for emergency rollback even though activation is automatic; do not use the current helper that deletes every older matching model.
- Record freshness metrics for monitoring rather than promotion gating. Regular evaluation should use labels newer than the training cutoff. The deployment scheduler itself cannot be implemented in this repository because no manifests or scheduler configuration are present.

### 3.5 Freshness metrics in Miniflux

Keep the existing `model_evals` history table and add freshness-specific support through Feedoscope migration `000007`:

- make binary-only metric columns nullable;
- reuse `metrics_f1` as Macro F1 for Freshness;
- reuse `metrics_roc_auc` as Evergreen AUC for Freshness;
- add nullable `metrics_rps`, `metrics_weighted_kappa`, and `metrics_log_duration_mae` columns.

Feedoscope writes one `model='Freshness'` row per weekly changed-label training/evaluation run. Metrics are informational and never block the new model from becoming active. Miniflux adds a Freshness card/graph/table that shows RPS, Macro F1, weighted kappa, log-duration MAE, and Evergreen AUC; Relevance and Urgency keep their current binary metrics.

## 4. File-by-file impact

### New files in the implementation worktree

- `explanation.md` — requested plain-language explanation (already written).
- `feedoscope/semantic_freshness_embedding.py` — five-head fitting, ordered probability conversion, artifact/model key, save/load, inference.
- `feedoscope/llm_learn_semantic_freshness.py` — DB label fetch, temporal validation, training, artifact publication.
- `feedoscope/llm_infer_semantic_freshness.py` — active artifact discovery and article scoring.
- `feedoscope/data_registry/sql/get_semantic_freshness_training.sql` — deterministic effective-label query with human-over-teacher precedence and auto-tag exclusion.
- `feedoscope/data_registry/sql/upsert_semantic_freshness_user_tags.sql` — create the six human and six auto Miniflux tag definitions.
- `feedoscope/data_registry/sql/set_semantic_freshness_auto_tag_for_entry.sql` — replace only the article's prior auto tag; preserve any reviewed tag.
- `feedoscope/data_registry/sql/promote_read_auto_freshness_tags.sql` — atomically replace one auto tag with its reviewed equivalent only for read articles without a reviewed tag.
- `feedoscope/data_registry/sql/register_semantic_freshness_inference.sql` — model-keyed probability upsert.
- `feedoscope/data_registry/sql/get_semantic_freshness_for_articles.sql` — model-keyed read.
- `db/migrations/000006_semantic_freshness.up.sql` / `.down.sql` — repository-owned bootstrap teacher-label table plus model-keyed inference table.
- `db/migrations/000007_model_eval_freshness_metrics.up.sql` / `.down.sql` — nullable binary fields plus RPS, weighted-kappa, and log-duration-MAE columns on `model_evals`.
- `tests/test_semantic_freshness.py` — focused check for target construction, monotone six-bucket output, probability sums, artifact round trip, and fingerprint stability.

### Modified production files

- `feedoscope/entities.py` — typed semantic-freshness inference result.
- `feedoscope/data_registry/data_registry.py` — label fetch and prediction upsert/read functions.
- `feedoscope/config.py` — freshness `C`, optional output mapping/clamps, and label-confidence policy only where configuration is truly operational.
- `feedoscope/main.py` — shadow inference/persistence and auto-tag assignment without changing current decay.
- `feedoscope/eval_models.py` — Freshness metrics calculation/history entry.
- `feedoscope/data_registry/data_registry.py` — model-eval insert support for nullable/model-specific metrics.
- `Makefile` — `train_freshness` and `infer_freshness` targets plus shadow inference in `full_infer`.
- `run.sh` — freshness training command for the external weekly scheduler.

### Miniflux repository changes

Create a separate uncommitted feature branch/worktree from `/home/djipey/informatique/go/miniflux` `main` after plan approval. Likely files:

- `internal/model/model_eval.go` — nullable/model-specific metric fields.
- `internal/storage/model_eval.go` — scan the extended `model_evals` columns.
- `internal/ui/ai_metrics.go` — explicit Freshness ordering and model-specific view data/chart series.
- `internal/template/templates/views/ai_metrics.html` — Freshness card/table labels.
- focused Go tests for Freshness view construction and chart data.
- locale keys required by the new Freshness metric labels.

### Documentation after implementation

- `docs/semantic-freshness.md` — implemented labels, training, artifact, output, storage, rollout, and retraining behavior.
- `docs/README.md` — index the new documentation.
- `architecture.d2` — only if freshness is added to the production pipeline/data store.

No new Python dependency is needed.

## 5. Risks and edge cases

- **Benchmark selection bias:** 75 experiments used one fixed 114-row test set. A fresh holdout is mandatory before changing production decay.
- **Teacher-label bias:** bootstrap labels are pseudo-gold. Import only medium/high-confidence rows with provenance; a human tag overrides the teacher row.
- **Accepted-auto feedback:** reading an article explicitly means accepting its current horizon for future training. Only read auto tags may be promoted; unread predictions remain ineligible.
- **Conflicting or missing labels:** exclude and warn about articles with multiple reviewed freshness tags; do not use their teacher fallback. Untagged articles without a teacher row remain outside training.
- **Boundary collapse:** every boundary needs both positive and negative training rows; fail before fitting if any is missing.
- **Probability ordering:** independent heads can cross; descending ordering is part of the selected model contract and needs a focused test.
- **Artifact compatibility:** validate metadata/thresholds/encoder configuration at load time rather than trusting the directory prefix alone.
- **Automatic regression acceptance:** every structurally valid changed-label candidate becomes active even when metrics worsen; the Miniflux Freshness metrics section is therefore required for visibility.
- **Rollback:** current model cleanup deletes all older family artifacts; freshness must retain the immediately previous artifact for emergency rollback.
- **Database growth:** model-keyed predictions preserve old versions. Add an explicit retention policy later only when growth is measured.
- **Decay semantics:** expected semantic lifetime is not automatically the same thing as relevance half-life. Shadow mode avoids silently changing ranking before the mapping is reviewed.
- **Migration ownership:** every freshness schema change, including `model_evals` extension, must have numbered up/down files in Feedoscope's `db/migrations/`; implementation is incomplete without them.
- **Cross-repo deployment order:** apply migrations 000006/000007 before deploying code that reads the new columns; coordinate Feedoscope and Miniflux changes.
- **Scheduling gap:** this repository cannot prove or implement weekly execution without the external deployment configuration.

## 6. Validation / testing

- Add one focused pytest module for non-trivial probability/serialization logic.
- Verify five target vectors for worked examples at all six horizons.
- Verify output shape `(n, 6)`, finite non-negative values, row sums of one, and ordered cumulative probabilities.
- Verify artifact save/load produces identical predictions and rejects incompatible metadata.
- Verify label/fingerprint changes cause retraining while unchanged data skips it.
- Verify SQL migration up/down and model-keyed upsert/read against PostgreSQL when available.
- Run `make format`.
- Run `make lint`.
- Run `uv run ruff check .`.
- Run `uv run --no-group infer pytest tests/test_semantic_freshness.py`.
- Run DB-backed training/inference only after labels are available.
- In Miniflux, run focused Go tests, `go test ./internal/ui ./internal/storage`, and `make lint` if practical.
- Verify the AI metrics page renders Relevance, Urgency, and Freshness rows with the correct model-specific labels.
- Before production decay changes, evaluate once on a new untouched temporal holdout and compare score ordering in shadow mode.

## 7. Step-by-step execution checklist

- [x] Create `feature/intrinsic-semantic-freshness` from `master` in a dedicated worktree.
- [x] Export the explanation to `explanation.md` without committing.
- [x] Inspect existing relevance, urgency, embedding cache, DB, artifact, and command patterns.
- [x] Decide label ownership: six reviewed `*-freshness` tags, six model `*-auto-freshness` tags, reviewed precedence, and bootstrap teacher labels kept separately with provenance.
- [x] Decide acceptance meaning: reading an article accepts its current freshness horizon for future training.
- [x] Decide the physical read-auto promotion mechanism: batch promotion before each freshness-training run.
- [x] Decide bootstrap-label persistence: PostgreSQL table created by repository migration 000006.
- [x] Decide output persistence level: six probabilities plus expected lifetime by model key.
- [x] Decide rollout: shadow mode before any relevance-decay replacement.
- [x] Decide auto-tag scope: the existing Active unread set; no complete-database backfill.
- [x] Decide retraining cadence/operator: external weekly invocation; promote read tags, then skip model fitting when the effective-label fingerprint is unchanged.
- [x] Decide candidate activation: every structurally valid changed-label weekly model supersedes the old model regardless of metric movement; metrics remain visible for monitoring.
- [x] Decide metrics storage/UI: extend `model_evals` via Feedoscope migration 000007 and add a model-specific Freshness section in Miniflux.
- [x] Complete grill and make the plan implementation-ready.
- [ ] Obtain user approval.
- [ ] Add focused tests for target/probability/artifact behavior.
- [ ] Implement the five-head artifact module.
- [ ] Implement label retrieval and training.
- [ ] Implement inference and selected persistence.
- [ ] Integrate the approved rollout into commands/orchestration.
- [ ] Run formatting, lint, types, tests, and available DB checks.
- [ ] Add/update durable docs for implemented behavior.
- [ ] Leave all feature-branch changes uncommitted for user review.

## 8. Open questions / assumptions

1. **Decided — labels:** use six reviewed tags named like `8-30d-freshness` and six model tags named like `8-30d-auto-freshness`. Both may remain checked; reviewed wins. Reading an article accepts its current horizon for future training.
2. **Decided — bootstrap source:** persist initial teacher labels, confidence, and source in a PostgreSQL table created by this repo's migration 000006; reviewed tags override those rows.
3. **Decided — storage:** persist six probabilities plus expected lifetime under a model key.
4. **Decided — rollout:** run in shadow mode first; do not change urgency-based relevance decay yet.
5. **Decided — read acceptance:** training preflight promotes read auto-only tags to reviewed tags before loading labels.
6. **Decided — retraining:** the external scheduler invokes training weekly; preflight promotes read tags, and unchanged effective labels skip model fitting.
7. **Decided — activation:** every structurally valid model trained from changed labels supersedes the old model even if evaluation metrics are worse; metrics are informational and shown in Miniflux.
8. **Open — approval:** implementation begins only after the user approves this completed plan.
9. Assumption: the approved model remains the autoresearch-selected five cumulative boundaries over shared EmbeddingGemma vectors, not five one-vs-rest class classifiers.
10. Assumption: urgency remains a separate feature and model even if freshness eventually replaces it for relevance decay.
11. Assumption: scheduling configuration lives outside this repository unless the user provides the deployment repository/path.

## 9. Decision log

### 2026-07-26 — Automatic tag scope

- **Question:** Which articles receive and refresh `*-auto-freshness` tags?
- **Accepted answer:** The existing **Active unread set**: all unread articles from the last 40 days plus the rotating older-unread sample.
- **Recommendation:** Accepted as recommended.
- **Rationale:** It covers the articles Feedoscope is actively ranking, reuses the current fetch/orchestration path, and avoids a new whole-database backfill. Tags remain visible after an article becomes read.

### 2026-07-26 — Reading accepts freshness

- **Question:** When does an automatic freshness prediction become eligible training data?
- **Accepted answer:** When the article is read, its current freshness horizon is considered correct; an explicitly set reviewed tag still takes precedence.
- **Recommendation:** Accepted as the product's review rule, with a visible source distinction between auto and reviewed tags.
- **Rationale:** Reading is the user's acceptance event. This grows the training set from normal reading behavior while preserving explicit corrections.

### 2026-07-26 — Read-tag promotion timing

- **Question:** When does a read article's auto tag become the equivalent reviewed tag?
- **Accepted answer:** At the start of each freshness-training run, before labels are loaded.
- **Recommendation:** Accepted as recommended.
- **Rationale:** Training already queries read articles, so batch promotion requires no database trigger or continuously running synchronization job.

### 2026-07-26 — Freshness migration ownership

- **Question:** Where are new freshness database migrations maintained?
- **Accepted answer:** In this repository under `db/migrations/`, with matching numbered up/down files.
- **Recommendation:** Accepted.
- **Rationale:** Feedoscope owns the freshness tables and must carry reproducible schema changes with the implementation.

### 2026-07-26 — Bootstrap teacher-label storage

- **Question:** Where do the original teacher-generated labels live after initial training?
- **Accepted answer:** A PostgreSQL table created by migration 000006 in this repository.
- **Recommendation:** Accepted as recommended.
- **Rationale:** It preserves horizon, confidence, and source durably without presenting teacher output as reviewed Miniflux tags; reviewed tags override it.

### 2026-07-26 — Retraining cadence

- **Question:** How often does the external scheduler start freshness training?
- **Accepted answer:** Weekly, with model fitting skipped when the effective-label fingerprint is unchanged.
- **Recommendation:** Accepted as recommended.
- **Rationale:** Weekly preflight steadily promotes newly read tags while the fingerprint prevents needless fitting; shared cached embeddings keep changed-label runs inexpensive.

### 2026-07-26 — Weekly candidate activation

- **Question:** When does a weekly candidate become the active freshness model?
- **Accepted answer:** Every structurally valid candidate trained from changed labels supersedes the old model, even when its metrics are worse.
- **Recommendation:** User overrode the recommended metric tolerance.
- **Rationale:** The model should continuously incorporate newly reviewed articles. Freshness metrics will be displayed in Miniflux for monitoring instead of blocking activation.

### 2026-07-26 — Freshness evaluation metrics

- **Question:** How does the binary-oriented `model_evals` table store and display ordinal Freshness metrics?
- **Accepted answer:** Extend the existing table through Feedoscope migration 000007; make irrelevant binary fields nullable, reuse F1/ROC AUC as Macro F1/Evergreen AUC for Freshness, and add RPS, weighted kappa, and log-duration MAE.
- **Recommendation:** Accepted as recommended.
- **Rationale:** One shared history table and a model-specific Miniflux Freshness card are smaller than a generic metrics rewrite or separate history stack.
