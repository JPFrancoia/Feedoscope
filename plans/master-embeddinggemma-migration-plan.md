Status: In Progress 2026-04-19 (implementation landed, runtime validation pending)

# Plan: Master EmbeddingGemma Migration

## Goal

Replace the current relevance backend on `master` with the validated
`google/embeddinggemma-300m` embedding + logistic-regression pipeline, using a
3-year article window and the full training set.

This plan is for implementation on the current `gemma` branch, which was branched
from `master`.

## Current State / Relevant Context

- `master` currently uses `answerdotai/ModernBERT-base` as a sequence classifier for
  relevance in `feedoscope/llm_learn.py` and `feedoscope/llm_infer.py`.
- `master` currently hardcodes `VALIDATION_SIZE = 0` in `feedoscope/llm_learn.py`.
- `master` currently balances the relevance training set by trimming good and bad
  rows to the same count.
- `master` currently uses a 1-year article window in the relevance SQL queries.
- `feedoscope/eval_models.py` currently evaluates the old transformer relevance path
  with the old balancing logic.

Validated production-style runs gathered from the other machine:

- `validate/relevance-embeddinggemma-prod`
  - 3-year window
  - full rows
  - `title_head`
  - `EmbeddingGemma-300m + logistic regression`
  - AP `0.96`
  - log loss `0.32`
- `validate/master-full-validation` with 3-year window
  - AP `0.94`
  - log loss `0.30`
- `validate/master-balanced-validation` with 3-year window
  - AP `0.94`
  - log loss `0.40`

Interpretation:

- If primary metric is `average_precision`, EmbeddingGemma is the winner.
- Full training rows are clearly better than balanced downsampling for the 3-year
  window.
- The 3-year window is now a fixed requirement.

User decisions already made:

- Keep inference-time pruning behavior as-is.
- Move `master` to the Gemma embedding backend.
- Use a 3-year article window.
- Cache the shared EmbeddingGemma repo once under `models/` and reuse it across
  all production artifacts rather than copying it into each artifact directory.

## Proposed Implementation

### 1. Add relevance-specific backend modules

Backport and clean up the validated Gemma branch helper modules:

- `feedoscope/relevance_text.py`
- `feedoscope/relevance_embedding.py`

Shared config will live in `feedoscope/config.py`, not a dedicated relevance config
module. These settings define:

- `MODEL_NAME = "google/embeddinggemma-300m"`
- `MAX_LENGTH = 2048`
- `TEXT_PREP_MODE = "title_head"`
- `LINEAR_C = 5.0`
- embedding batch sizes
- artifact naming inputs used directly in the relevance modules
- the shared local encoder cache location indirectly via `RELEVANCE_MODEL_NAME`

### 2. Replace relevance training in `llm_learn.py`

Convert relevance training from transformer fine-tuning to:

- fetch live DB data
- use full training rows
- build EmbeddingGemma embeddings
- fit a logistic-regression head
- save the artifact
- run held-out validation if `VALIDATION_SIZE > 0`

Required backports and fixes:

- replace local `VALIDATION_SIZE = 0` with `config.VALIDATION_SIZE`
- remove the balanced-downsampling block entirely
- keep excellent-article weighting
- remove training-side family cleanup
- close the DB pool on the no-validation early return

### 3. Replace relevance inference in `llm_infer.py`

Convert inference from `AutoModelForSequenceClassification` to the embedding backend:

- find latest artifact directory
- load `classifier.joblib`
- load the shared cached EmbeddingGemma encoder/tokenizer from `models/`
- encode unread articles
- compute probabilities
- convert to `0-100` integer scores

Important: keep inference-time artifact pruning behavior unchanged.

### 4. Update relevance eval in `eval_models.py`

Bring `make eval` in line with the new production relevance backend.

Relevance eval should:

- train an eval embedding backend, not a transformer classifier
- use full rows instead of balancing
- use the shared 3-year SQL window
- save metrics in the same `eval_history.json` format

Urgency eval should remain unchanged.

### 5. Change the relevance SQL window to 3 years

Update all four relevance SQL files:

- `feedoscope/data_registry/sql/get_read_articles_training.sql`
- `feedoscope/data_registry/sql/get_published_articles.sql`
- `feedoscope/data_registry/sql/get_sample_good.sql`
- `feedoscope/data_registry/sql/get_sample_not_good.sql`

Change:

- `interval '1 year'` -> `interval '3 years'`

### 6. Add explicit dependency support

Update:

- `pyproject.toml`
- `uv.lock`

Add explicit dependency:

- `scikit-learn`

### 7. Backport the production fixes found during validation

Backport these fixes before merging:

- no hardcoded `VALIDATION_SIZE`
- no balanced downsampling
- no training-side family cleanup
- preserve the trained model artifact after training
- improve the sparse logging in the embedding path
- fix the `title_head` tokenizer-warning path

The logging improvement should add clear progress logs for:

- encoder load
- text preparation start/end
- embedding batch progress
- logistic-regression fit start/end
- artifact save
- validation start/end

The `title_head` fix should avoid tokenizing the full article body before slicing to
the actual token budget.

## File-by-File Impact

New files:

- `feedoscope/relevance_text.py`
- `feedoscope/relevance_embedding.py`

Updated files:

- `AGENTS.md`
- `feedoscope/config.py`
- `feedoscope/llm_learn.py`
- `feedoscope/llm_infer.py`
- `feedoscope/eval_models.py`
- `feedoscope/data_registry/sql/get_read_articles_training.sql`
- `feedoscope/data_registry/sql/get_published_articles.sql`
- `feedoscope/data_registry/sql/get_sample_good.sql`
- `feedoscope/data_registry/sql/get_sample_not_good.sql`
- `pyproject.toml`
- `uv.lock`

## Risks and Edge Cases

- EmbeddingGemma won on the primary metric, but `master-full` on the 3-year window
  had slightly better displayed log loss in one comparison regime. This should be
  documented, but it is not a blocker given the AP decision.
- Replacing the relevance backend changes the artifact format from a Hugging Face
  model directory to a directory containing at least `classifier.joblib` and
  `metadata.json`.
- The shared EmbeddingGemma cache is separate from each trained classifier
  artifact; `models/` persistence is therefore required for cold-start avoidance.
- If `eval_models.py` is not updated in the same change, `make eval` will become
  misleading.
- The validated Gemma branch currently logs less than `master`. The migration should
  restore production-quality progress logging.
- The current `title_head` implementation emits a tokenizer warning; that should be
  fixed before merge.

## Validation / Testing

Minimum validation after implementation:

1. `VALIDATION_SIZE=100 make train`
2. confirm:
    - 3-year queries are used
    - full rows are used
    - artifact is saved and remains on disk
    - shared EmbeddingGemma cache is downloaded only once under `models/`
    - validation completes successfully
3. `make infer`
4. confirm:
   - latest embedding artifact is found
   - unread articles are encoded and scored
   - DB updates still work
5. `VALIDATION_SIZE=100 make eval`
6. confirm:
   - relevance eval uses the embedding backend
   - relevance eval uses full rows
   - urgency eval still works unchanged

Completed in this session:

- `uv lock`
- `make format`
- `uv run --no-group infer mypy .`
- `uv run ruff check feedoscope/config.py feedoscope/relevance_text.py feedoscope/relevance_embedding.py feedoscope/llm_learn.py feedoscope/llm_infer.py`
- `uv run python -m compileall feedoscope custom_logging`

Still pending:

- DB-backed `make train`, `make infer`, and `make eval` validation against a live `DATABASE_URL`
- Verify the shared encoder cache is reused on a second run without a re-download
- Metric comparison against the validated Gemma baseline after a real training run

Expected ballpark from the validated Gemma production-style run:

- AP around `0.96`
- ROC AUC around `0.97`
- log loss around `0.32`
- peak VRAM around `2.17 GB`

## Step-by-Step Execution Checklist

- [completed] Add `relevance_text.py` and `relevance_embedding.py`, and move relevance settings into `feedoscope/config.py`.
- [completed] Replace relevance training in `feedoscope/llm_learn.py`.
- [completed] Replace relevance inference in `feedoscope/llm_infer.py`.
- [completed] Update `feedoscope/eval_models.py` relevance path.
- [completed] Change the four relevance SQL queries from 1 year to 3 years.
- [completed] Add explicit `scikit-learn` dependency.
- [completed] Improve embedding-path progress logging.
- [completed] Fix the `title_head` tokenizer-warning path.
- [pending] Run train, infer, and eval validation passes.
- [pending] Review metrics against the validated Gemma baseline.

## Non-Goals

Not part of this migration:

- EmbeddingGemma prompt-mode experiments
- GTE / Arctic / BGE-M3 experiments
- sparse or hybrid embedding features
- urgency model changes
- score-decay changes in `main.py`

## Open Questions / Assumptions

- Assumption: the branch should replace the relevance backend directly, not introduce
  a backend toggle.
- Assumption: inference-time pruning behavior stays unchanged, per the user’s
  instruction.
