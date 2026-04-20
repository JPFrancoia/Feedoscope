Status: Completed 2026-04-20 (pi_research-style urgency autoresearch loop landed; fixed local benchmark files removed)

# Plan: Urgency Karpathy Loop

## Brief

We are adding the urgency equivalent of the `pi_research` autoresearch loop:
freeze the read-tagged urgency dataset into local Parquet files, then let an
external agent run one experiment at a time against that fixed snapshot. This
matters now because the trusted urgency supervision is in manually tagged read
articles, and the user explicitly wants experiment files rather than a local
fixed comparison script.

## Current State / Relevant Context

- `feedoscope/llm_learn_urgency.py` trains the current urgency classifier.
- `feedoscope/llm_infer_urgency.py` scores articles into continuous urgency
  probabilities for `urgency_inference`.
- Existing urgency labels come from Miniflux user tags via
  `feedoscope/data_registry/sql/get_articles_with_simplified_time_sensitivity.sql`.
- The current trainer uses both read and unread tagged articles, then
  down-samples unread data.
- The user wants to stop relying on one-shot Ministral labeling and instead use
  only read articles for urgency training.
- The repository already has a strong embedding baseline for relevance in
  `feedoscope/relevance_embedding.py`, which we can reuse for urgency model
  search.

## Proposed Implementation

### 1. Add a dedicated read-tagged urgency query

Create a new SQL query that returns only read articles carrying the
`0-urgency` or `1-urgency` tags and map the tag to a binary label.

### 2. Add a reusable urgency metrics module

Create a small helper that computes probability-centric binary metrics:

- average precision
- ROC AUC
- log loss
- Brier score
- accuracy
- precision
- recall
- F1

### 3. Export a frozen urgency snapshot

Mirror the relevance loop by exporting the read-tagged urgency dataset into a
local Parquet snapshot with fixed train/eval splits and metadata.

### 4. Add a single-run experiment harness

Create one harness that runs exactly one urgency experiment from the frozen
snapshot and prints structured `METRIC` lines. It should support both
transformer fine-tuning and frozen-embedding-plus-linear-head experiments.

### 5. Add root-level autoresearch loop files

Mirror the `pi_research` branch layout with:

- `program.md`
- `autoresearch.md`
- `autoresearch.ideas.md`
- `autoresearch.sh`
- `autoresearch.jsonl`
- `results.md`

### 6. Wire docs around the root experiment files

Document the workflow around the root loop files and direct commands rather than
adding extra convenience targets.

## File-by-File Impact

New files:

- `plans/urgency-karpathy-loop-plan.md`
- `feedoscope/urgency_experiments/__init__.py`
- `feedoscope/urgency_experiments/export_snapshot.py`
- `feedoscope/urgency_experiments/harness.py`
- `feedoscope/urgency_experiments/metrics.py`
- `feedoscope/urgency_experiments/sql/get_read_articles_with_urgency_tags.sql`
- `autoresearch.sh`
- `autoresearch.md`
- `autoresearch.ideas.md`
- `autoresearch.jsonl`
- `program.md`
- `results.md`
- `docs/urgency-karpathy-loop.md`

Updated files:

- `.gitignore`
- `docs/README.md`

Likely unchanged in this first pass:

- `feedoscope/llm_infer_urgency.py`
- `feedoscope/main.py`
- `db/migrations/`

## Risks and Edge Cases

- The read-tagged dataset is imbalanced, so candidate ranking should prioritize
  probability metrics over raw accuracy.
- A fixed validation split can be noisy on a small dataset, but it is a good
  first step and keeps the implementation simple.
- The loop intentionally stops short of production promotion, so a follow-up
  step will still be needed to wire the chosen model into `llm_infer_urgency.py`
  and to make `urgency_inference` model-aware.
- The embedding experiment path downloads models locally on the GPU machine and
  uses the frozen Parquet snapshot, matching the relevance autoresearch pattern.

## Validation / Testing

Planned validation:

1. format touched files with `black` and `isort`
2. run `mypy` on touched Python modules
3. run `ruff check` on touched Python modules
4. run `uv run python -m compileall feedoscope custom_logging`
5. run `uv run python -m feedoscope.urgency_experiments.export_snapshot`
6. run `LOGGING_CONFIG=dev_logging.conf ./autoresearch.sh` once a GPU machine is available

Completed in this session:

- removed the earlier fixed local comparison files
- implemented `pi_research`-style urgency autoresearch files
- added the read-only urgency training query and data-registry helper
- added the frozen snapshot exporter and direct loop commands
- updated durable docs

## Step-by-Step Execution Checklist

1. Save the implementation plan.
2. Add the read-tagged urgency SQL query.
3. Add the data-registry helper for read-tagged urgency labels.
4. Add shared urgency metrics.
5. Add the frozen snapshot exporter.
6. Add the one-run experiment harness.
7. Add the root-level autoresearch files.
8. Update docs and plan records.
9. Format and lint the touched files.

## Open Questions / Assumptions

- Assumption: experiment-only scaffolding is enough for this step; production
  promotion will happen later.
- Assumption: the current relevance embedding configuration is a good first
  embedding baseline for urgency as well.
- Assumption: one fixed stratified validation split is sufficient for initial
  model selection.
