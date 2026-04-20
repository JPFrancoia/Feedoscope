Status: In Progress 2026-04-20 (implementation landed; DB-backed runtime validation pending)

# Plan: Urgency EmbeddingGemma Migration

## Brief

Replace the current urgency ModernBERT backend with the same
EmbeddingGemma-plus-logistic-regression approach already used by the relevance
pipeline. Train only on read articles manually tagged `0-urgency` or
`1-urgency`, and reuse the existing Postgres embedding cache as much as
possible by sharing the same encoder and text-preparation settings as
relevance.

## Current State / Relevant Context

- Branch: `gemma_urgency`, branched from `gemma`.
- Relevance already uses the embedding backend in
  `feedoscope/relevance_embedding.py`.
- Relevance embeddings are cached in Postgres in `relevance_embeddings`, keyed
  by:
  - `article_id`
  - `model_name`
  - `max_length`
  - `text_prep_mode`
  - `prep_version`
- Urgency still uses a transformer classifier in:
  - `feedoscope/llm_learn_urgency.py`
  - `feedoscope/llm_infer_urgency.py`
- Current urgency training data comes from
  `feedoscope/data_registry/sql/get_articles_with_simplified_time_sensitivity.sql`,
  which returns both read and unread tagged articles.
- Current urgency inference cache is `urgency_inference`, keyed only by
  `article_id`, which means scores from different urgency model versions would
  overwrite or mix implicitly.
- `main.py` already consumes urgency as a continuous probability in `[0, 1]`,
  which matches the embedding-linear logistic-regression output we want.
- V1 urgency autoresearch on `pi_research_urgency` showed that frozen
  embedding-plus-logistic-regression models beat the transformer baselines on
  average precision, and `google/embeddinggemma-300m` was a strong, simpler
  candidate to standardize on.

## Proposed Implementation

### 1. Reuse the existing shared embedding backend

Urgency should reuse `feedoscope/relevance_embedding.py` directly for:

- loading the shared encoder snapshot
- preparing text
- looking up cached embeddings in Postgres
- encoding only cache misses
- fitting and loading `LogisticRegression`
- predicting probabilities

This keeps the urgency pipeline aligned with the relevance pipeline and avoids a
second embedding implementation.

### 2. Share the embedding cache configuration intentionally

Urgency should intentionally share the same embedding configuration as
relevance so cached vectors can be reused across both pipelines.

Shared settings:

- `RELEVANCE_MODEL_NAME`
- `RELEVANCE_MAX_LENGTH`
- `RELEVANCE_TEXT_PREP_MODE`
- `RELEVANCE_PREP_VERSION`
- `RELEVANCE_ENCODER_BATCH_SIZE`

This means urgency will reuse cached rows in `relevance_embeddings` whenever an
article has already been embedded by the relevance pipeline under the same
configuration.

### 3. Add one urgency-specific classifier knob

Add `URGENCY_LINEAR_C` to `feedoscope/config.py`.

Reasoning:

- the embedding config should stay shared for cache reuse
- the logistic-regression head can still be tuned separately for urgency
- V1 urgency loop results suggest a default of `1.0` is a better starting point
  than relevance's current `5.0`

We should not reuse relevance's sample weighting for urgency. The urgency model
should not overweight starred or upvoted articles, because those are relevance
signals, not urgency signals.

### 4. Restrict urgency training data to read-tagged articles only

Add a dedicated SQL query and data-registry helper that return only:

- `entries.status = 'read'`
- articles tagged `0-urgency` or `1-urgency`

The tag value becomes the training label:

- `0-urgency` -> 0
- `1-urgency` -> 1

Unread tagged articles must not participate in urgency training anymore.

### 5. Replace urgency training with embedding-plus-logistic-regression

Rewrite `feedoscope/llm_learn_urgency.py` to mirror the relevance training flow:

1. fetch read-tagged urgency articles
2. optionally reserve a validation holdout when `VALIDATION_SIZE > 0`
3. load the shared encoder
4. embed the training articles via `relevance_embedding.encode_articles()`
5. fit a logistic-regression head
6. save `classifier.joblib` plus metadata

Training details:

- use all read-tagged rows
- do not downsample classes
- use `class_weight="balanced"` on the logistic-regression head
- preserve the current operator interface: `make train_urgency` should still be
  the entrypoint

Artifact naming should parallel the relevance artifact naming style, but with an
urgency-specific prefix and the urgency classifier hyperparameter.

### 6. Replace urgency inference with embedding-plus-logistic-regression

Rewrite `feedoscope/llm_infer_urgency.py` to mirror the relevance inference
path:

1. find the latest saved urgency embedding artifact
2. load `classifier.joblib`
3. load the shared embedding encoder
4. reuse cached embeddings from Postgres where available
5. compute urgency probabilities
6. write them to `urgency_inference`

The output contract should remain unchanged:

- `UrgencyInferenceResults`
- `urgency_scores` are probabilities in `[0, 1]`

### 7. Version the urgency inference cache by model

Add a new DB migration for `urgency_inference`.

Recommended schema change:

- add `model_key text not null`
- change the primary key from `article_id` to `(article_id, model_key)`

Why `model_key`, not just `model_name`:

- the urgency score depends on more than the base encoder name
- it must distinguish different logistic heads, text-prep modes, max lengths,
  prep versions, and future backend changes
- it must prevent stale ModernBERT scores and new EmbeddingGemma scores from
  mixing silently

Recommended `model_key` contents:

- backend family
- encoder model name
- max length
- text-prep mode
- prep version
- urgency linear `C`

Example shape:

- `urgency-embedding_linear::google/embeddinggemma-300m::2048::title_head::1::c=1.0`

### 8. Make urgency cache reads and writes model-aware

Update these SQL helpers and data-registry functions to require `model_key`:

- `get_articles_wo_urgency_inference.sql`
- `register_urgency_inference.sql`
- `get_urgency_scores_for_articles.sql`
- `feedoscope/data_registry/data_registry.py`

Then update:

- `feedoscope/llm_infer_urgency.py`
- `feedoscope/main.py`

so that urgency scores are always written and read for the currently active
urgency model only.

### 9. Refresh urgency on the exact same article set as relevance

Urgency refresh should mirror relevance refresh behavior exactly.

Instead of only scoring unread articles that are missing from
`urgency_inference`, `main.py` should:

1. build the active article set for scoring
2. run urgency inference for that exact set
3. upsert urgency scores for the active `model_key`
4. run relevance inference for the same set
5. apply time decay from the refreshed urgency probabilities

This means urgency will be refreshed for:

- the recent unread window
- the sampled older unread window

using the same article selection policy already used by relevance.

### 10. Update urgency evaluation to match production

`feedoscope/eval_models.py` should be updated in the same change.

Urgency eval should:

- hold out only read-tagged urgency articles for evaluation
- train on the remaining read-tagged urgency articles only
- use the same embedding-plus-logistic-regression backend as production urgency
- save metrics in the existing eval history format

If this is not updated in the same change, `make eval` will stop reflecting the
real urgency production path.

### 11. Update docs and comments that still reference ModernBERT urgency

At minimum update:

- `feedoscope/entities.py`
- `feedoscope/main.py`
- urgency-related comments in `feedoscope/data_registry/data_registry.py`
- `docs/README.md`

Add durable documentation in `docs/` for the urgency embedding backend once the
implementation lands.

## File-by-File Impact

New files:

- `plans/urgency-embeddinggemma-migration-plan.md`
- `feedoscope/data_registry/sql/get_read_articles_with_urgency_tags.sql`
- `db/migrations/<new>_urgency_inference_model_key.up.sql`
- `db/migrations/<new>_urgency_inference_model_key.down.sql`
- `docs/urgency-embedding-backend.md`

Updated files:

- `feedoscope/config.py`
- `feedoscope/llm_learn_urgency.py`
- `feedoscope/llm_infer_urgency.py`
- `feedoscope/eval_models.py`
- `feedoscope/main.py`
- `feedoscope/entities.py`
- `feedoscope/data_registry/data_registry.py`
- `feedoscope/data_registry/sql/get_articles_wo_urgency_inference.sql`
- `feedoscope/data_registry/sql/register_urgency_inference.sql`
- `feedoscope/data_registry/sql/get_urgency_scores_for_articles.sql`
- `docs/README.md`

Expected unchanged core files:

- `feedoscope/relevance_embedding.py`
- `feedoscope/llm_learn.py`
- `feedoscope/llm_infer.py`

## Risks and Edge Cases

- The biggest correctness risk is stale urgency cache reuse. If `model_key` is
  not added, old ModernBERT scores and new embedding-based scores may silently
  mix.
- If urgency refresh is left on the old "missing rows only" behavior, the live
  scoring path would continue to use stale urgency probabilities for older
  unread articles even after relevance refreshed the same articles.
- Sharing relevance embedding settings with urgency is intentional coupling. It
  maximizes cache reuse, but future relevance embedding-config changes will also
  affect urgency unless both are updated in sync.
- Cache reuse will be partial at first: some urgency training and inference
  articles may already be embedded by the relevance pipeline, while others will
  still need first-time encoding.
- Reusing relevance's sample weighting logic for urgency would be a mistake.
  Urgency should use balanced class weighting only.
- If `eval_models.py` remains on the old urgency backend, weekly evaluation will
  become misleading immediately.
- The current comments and docstrings still describe urgency as a distilled
  ModernBERT model. Those references will become stale after the migration.

## Validation / Testing

Minimum validation after implementation:

1. `make format`
2. `uv run --no-group infer mypy .`
3. `uv run ruff check .`
4. `uv run python -m compileall feedoscope custom_logging`
5. `VALIDATION_SIZE=100 make train_urgency`
6. confirm:
   - training uses only read-tagged urgency rows
   - the urgency artifact is classifier-based, not transformer-finetuned
   - embedding-cache logs show hits/misses as expected
7. `make infer_urgency`
8. confirm:
   - latest urgency embedding artifact is found
   - urgency scores are written with the active `model_key`
   - repeated inference reuses cached embeddings correctly
9. `make full_infer`
10. confirm:
    - `main.py` refreshes urgency for the exact same article set relevance scores
    - urgency rows are upserted for the active `model_key`
    - final decay uses refreshed urgency probabilities for that set
9. `VALIDATION_SIZE=100 make eval`
10. confirm:
    - urgency eval now matches the embedding production path
11. confirm:
    - `main.py` fetches urgency scores only for the active urgency model
    - final decayed relevance scoring still completes end-to-end

Completed in this session:

- added the `urgency_inference` `model_key` migration
- switched urgency training to read-tagged EmbeddingGemma + logistic regression
- switched urgency inference to the shared embedding backend
- aligned urgency refresh with the exact relevance article-selection path
- updated urgency eval to match production
- updated durable docs and the docs index
- ran `black` and `isort` on touched Python files
- ran targeted `mypy` on touched Python files
- ran targeted `ruff check` on touched Python files
- ran `python -m compileall feedoscope custom_logging`

Still pending:

- apply the new DB migration against the target database
- run DB-backed `make train_urgency`, `make infer_urgency`, `make eval`, and
  `make full_infer`

## Step-by-Step Execution Checklist

1. Add the approved plan file.
2. Add `URGENCY_LINEAR_C` to config.
3. Add the read-tagged urgency SQL query and helper.
4. Replace urgency training with the embedding-linear backend.
5. Replace urgency inference with the embedding-linear backend.
6. Add the `urgency_inference` migration for `model_key`.
7. Update urgency cache SQL helpers and data-registry functions.
8. Update `main.py` to refresh urgency on the exact relevance article set.
9. Update urgency evaluation in `eval_models.py`.
10. Update docstrings, comments, and durable docs.
11. Run formatting, lint, type-check, and compile validation.
12. Run DB-backed training, inference, eval, and full-pipeline smoke checks.

## Open Questions / Assumptions

- Assumption: the urgency pipeline should intentionally share the exact same
  embedding configuration as relevance to maximize cache reuse.
- Assumption: `URGENCY_LINEAR_C` should default to `1.0` based on the V1 loop
  results.
- Assumption: urgency should use full read-tagged rows with
  `class_weight="balanced"`, not downsampling.
- Assumption: adding `model_key` to `urgency_inference` is preferred over doing
  one-time cache wipes whenever the urgency backend changes.
