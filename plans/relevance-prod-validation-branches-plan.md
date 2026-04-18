Status: Completed 2026-04-18

# Plan: Relevance Production Validation Branches

## Goal

Create two dedicated validation branches from `pi_research` that convert the loop
findings into production-style relevance training/inference setups:

1. `validate/relevance-ettin68m-prod`
2. `validate/relevance-embeddinggemma-prod`

Each branch must support the user's real workflow of running relevance training with
`VALIDATION_SIZE=100` set in the environment, using full training rows and the
branch-specific best loop parameters.

## Current State / Relevant Context

- `make train` runs `feedoscope.llm_learn`.
- `feedoscope.llm_learn` currently hardcodes `VALIDATION_SIZE = 0`, so the env var is
  ignored for relevance training.
- Relevance production training still uses the old balanced downsampling logic.
- Relevance SQL queries currently only keep the last 1 year of articles; the
  validation branches should widen that window to 3 years.
- Production relevance inference still assumes a Hugging Face sequence classifier.
- The loop found two best real candidates worth validating in production style:
  - Ettin-68m transformer classifier with full rows, max length 1024, title-head prep
  - EmbeddingGemma-300m frozen embeddings + logistic regression, full rows,
    title-head prep, max length 2048, `C=5.0`
- The user no longer needs `make eval` adapted; they will run `llm_train` with
  `VALIDATION_SIZE` set.

## Proposed Implementation

### Shared changes for both branches

- Make relevance training read `VALIDATION_SIZE` from `config` instead of a local
  constant.
- Stop downsampling relevance training to a 50/50 balanced subset.
- Keep the existing per-sample excellent-article weighting.
- Extend the relevance train/validation article retention window from 1 year to 3
  years.
- Add relevance-specific text-prep helpers so relevance can diverge from urgency
  without changing global text prep behavior.

### Ettin branch

- Configure relevance training/inference for `jhu-clsp/ettin-encoder-68m`.
- Use `MAX_LENGTH = 1024`.
- Use deterministic `title_head` relevance text prep.
- Use production training args aligned with the loop's best realistic Ettin setup.

### EmbeddingGemma branch

- Add a production embedding-linear relevance backend.
- Train a logistic regression classifier on top of frozen
  `google/embeddinggemma-300m` embeddings.
- Use the latest kept loop configuration: `title_head`, mean pooling, `C=5.0`.
- Save and load classifier artifacts plus backend metadata from `models/`.
- Keep inference output identical at the interface level: 0-100 relevance scores.

## File-by-File Impact

- `plans/relevance-prod-validation-branches-plan.md`
  - This decision record.
- Ettin branch expected edits:
  - `feedoscope/llm_learn.py`
  - `feedoscope/llm_infer.py`
  - new relevance text-prep helper module
- EmbeddingGemma branch expected edits:
  - `feedoscope/llm_learn.py`
  - `feedoscope/llm_infer.py`
  - new relevance text-prep helper module
  - new embedding backend module
  - `pyproject.toml` for explicit sklearn dependency if needed

## Risks and Edge Cases

- Branch switching must not disturb unrelated user files; `results.md` is currently
  untracked and should be left untouched.
- EmbeddingGemma requires a new persisted artifact format rather than a Hugging Face
  classifier directory alone.
- Relevance-specific text prep must not change urgency/time-sensitivity behavior.

## Validation / Testing

- After each branch implementation:
  - run Python syntax checks on edited modules
  - run a quick import/compilation check for the branch backend
- Leave the branches ready for the user to run full training manually.

## Step-by-Step Execution Checklist

- [x] Create this plan file.
- [x] Create `validate/relevance-ettin68m-prod` and implement changes.
- [x] Verify Ettin branch syntax and summarize resulting files.
- [x] Create `validate/relevance-embeddinggemma-prod` from `pi_research` and implement changes.
- [x] Verify EmbeddingGemma branch syntax and summarize resulting files.
- [x] Mark plan completed with both branch names.

## Open Questions / Assumptions

- Assumption: the user wants both branches created locally now and does not need
  commits yet.
- Assumption: `results.md` is unrelated and should remain unmodified.

## Final Branches

- `validate/relevance-ettin68m-prod`
- `validate/relevance-embeddinggemma-prod`
