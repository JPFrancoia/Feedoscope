# Autoresearch: Relevance Model Optimization

Read `program.md` first. It is the authoritative full brief for this loop.

## Objective

Improve relevance evaluation on the existing frozen dataset snapshot.

This session is resumed and already bootstrapped. The snapshot, harness, and loop
infrastructure exist.

## Current Best To Beat

Current kept best:

- commit: `ea778e6`
- model: `google/embeddinggemma-300m`
- classifier type: `embedding_linear`
- max length: `2048`
- text prep: `title_head`
- train balance mode: `full`
- linear head `C`: `5.0`
- average precision: `0.984098480367228`
- log loss: `0.21130`

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
- `google/embeddinggemma-300m @ 2048` overtook all prior open families, then improved
  again with `title_head` and `C=5.0`.
- Real-world validation suggests the current master path is still strong, so the next
  loop should be narrow and targeted.

## Required Next Run Order

Run these next, in this order:

1. control rerun of the current EmbeddingGemma winner after the harness changes
2. `google/embeddinggemma-300m @ 2048`, `title_head`, `EMBED_PROMPT_MODE=classification`
3. `google/embeddinggemma-300m @ 2048`, `title_head`, `EMBED_PROMPT_MODE=document`
4. `Alibaba-NLP/gte-large-en-v1.5 @ 2048`, `title_head`, dense embedding + linear head
5. `Snowflake/snowflake-arctic-embed-l-v2.0 @ 2048`, `title_head`, dense embedding +
   linear head
6. `BAAI/bge-m3 @ 2048`, `title_head`, `EMBED_FEATURE_MODE=hybrid`
7. `BAAI/bge-m3 @ 2048`, `title_head`, `EMBED_FEATURE_MODE=sparse`

Do not reorder these first seven runs.

## Model-Aware Defaults

- `google/embeddinggemma-300m`: `mean` pooling, prompt modes `none`, then
  `classification`, then `document`
- `Alibaba-NLP/gte-large-en-v1.5`: `cls` pooling, no prompt mode initially
- `Snowflake/snowflake-arctic-embed-l-v2.0`: `cls` pooling, no prompt mode initially
- `BAAI/bge-m3`: dense control uses `mean`; hybrid and sparse runs use hashed lexical
  features with `EMBED_SPARSE_HASH_DIM=8192`

Log every embedding-specific knob in the run output.

## What's Been Tried

Keep this section updated as the loop progresses.

Current condensed history:

- best fine-tuned transformer before embeddings: Ettin-68m full-data run below the
  best embedding regime
- `BAAI/bge-m3 @ 2048` with dense embeddings and a logistic-regression head became
  the best open embedding family tried so far
- `google/embeddinggemma-300m @ 2048` overtook `bge-m3`, then improved again with
  `title_head` and `C=5.0`

Keep updating this file with:

- best kept run so far
- dead ends
- crash causes
- promising combinations to try next
