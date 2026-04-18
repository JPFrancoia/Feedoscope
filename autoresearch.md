# Autoresearch: Relevance Model Optimization

Read `program.md` first. It is the authoritative full brief for this loop.

## Objective

Improve relevance evaluation on the existing frozen dataset snapshot.

This session is resumed and already bootstrapped. The snapshot, harness, and loop
infrastructure exist.

## Current Best To Beat

Current kept best:

- commit: `a27d7c7`
- model: `google/embeddinggemma-300m`
- classifier type: `embedding_linear`
- max length: `2048`
- text prep: `single_blob`
- train balance mode: `full`
- linear head `C`: `4.0`
- average precision: `0.9840544956495607`
- roc auc: `0.96794`
- log loss: `0.21510`
- peak vram: `1.69 GB`

Treat this as the control baseline.

## Metrics

- Primary: `average_precision` (higher is better)
- Secondary: `roc_auc`, `log_loss`, `accuracy`, `precision`, `recall`, `f1`,
  `peak_vram_gb`, `train_seconds`, `total_seconds`

## How to Run

Standardize on:

`./autoresearch.sh`

Rules for `autoresearch.sh`:

- run exactly one experiment configuration
- use `uv run ...` for Python commands
- print structured `METRIC` lines
- exit non-zero on crash

## Constraints

- Stay on the current dedicated experiment branch.
- Never create a new branch.
- Never switch branches.
- Never commit on `main` or `master`.
- If the current branch is `main` or `master`, stop and ask the human to restart
  from a dedicated branch.
- Local commits on the current dedicated branch are allowed.
- Never push.
- Never force-push.
- Database access is read-only. Allowed SQL is `SELECT` and `EXPLAIN SELECT` only.
- Keep using the local frozen snapshot only.
- Keep using the frozen eval split only.
- Use the `uv`-managed environment for all Python work.
- The machine has one 12 GB GPU. OOM is a crash.
- Default to `TRAIN_BALANCE_MODE=full` unless the explicit experiment is about
  balance mode.

## What Has Been Learned So Far

- The loop is no longer in bootstrap mode; the snapshot, harness, and append-only log
  already exist.
- Full-data training on the frozen train split outperformed the old balanced
  downsampling regime.
- Embedding models with a logistic-regression head have beaten the fine-tuned
  transformer baselines on this snapshot.
- `BAAI/bge-m3` was the strongest open embedding family before EmbeddingGemma access
  was granted.
- After access was granted, `google/embeddinggemma-300m @ 2048` became the best run.

## Required Next Run Order

Run these next, in this order:

1. control rerun of the current EmbeddingGemma winner after the harness changes
2. `Alibaba-NLP/gte-base-en-v1.5 @ 2048`, `single_blob`, embedding + linear head,
   full train split
3. `Alibaba-NLP/gte-large-en-v1.5 @ 2048`, `single_blob`, embedding + linear head,
   full train split
4. `nomic-ai/nomic-embed-text-v1.5 @ 2048`, `single_blob`, embedding + linear head,
   full train split
5. `Snowflake/snowflake-arctic-embed-m-v2.0 @ 2048`, `single_blob`, embedding +
   linear head, full train split
6. `mixedbread-ai/mxbai-embed-large-v1 @ 512`, `single_blob`, embedding + linear
   head, full train split

Do not reorder these first six runs.

## Model-Aware Defaults

- `google/embeddinggemma-300m`: `mean` pooling, no prefix, no layer norm, no dim
  truncation
- `Alibaba-NLP/gte-base-en-v1.5`: `cls` pooling, no prefix
- `Alibaba-NLP/gte-large-en-v1.5`: `cls` pooling, no prefix
- `nomic-ai/nomic-embed-text-v1.5`: `mean` pooling, `classification: ` prefix,
  layer norm on, truncate dim to `512`
- `Snowflake/snowflake-arctic-embed-m-v2.0`: `cls` pooling, no prefix
- `mixedbread-ai/mxbai-embed-large-v1`: `cls` pooling, no prefix

Log every embedding-specific knob in the run output.

## What's Been Tried

Keep this section updated as the loop progresses.

Current condensed history:

- best fine-tuned transformer before embeddings: Ettin-68m full-data run below the
  best embedding regime
- `BAAI/bge-m3 @ 2048` with embedding + logistic-regression head became the best open
  embedding family tried so far
- `google/embeddinggemma-300m @ 2048` overtook `bge-m3` once model access was
  granted

Keep updating this file with:

- best kept run so far
- dead ends
- crash causes
- promising combinations to try next
