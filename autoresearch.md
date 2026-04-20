# Autoresearch: Urgency Model Optimization

Read `program.md` first. It is the authoritative full brief for this loop.

## Objective

Improve urgency evaluation on the existing frozen read-tagged snapshot.

This session is in bootstrap mode for urgency autoresearch. The snapshot
exporter, single-run harness, and append-only run log now exist, but there is no
kept urgency winner yet.

## Training Data Policy

- Use only read articles tagged `0-urgency` or `1-urgency`.
- Ignore unread articles entirely for supervision.
- Ignore the old one-shot Ministral labeling path.

## Primary Metric

- `average_precision`

Higher is better.

## Secondary Metrics

- `roc_auc`
- `log_loss`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `brier_score`
- `peak_vram_gb`
- `train_seconds`
- `total_seconds`

## First Bootstrap Steps

1. Export the frozen urgency snapshot.
2. Run the control baseline.
3. Log the result to `autoresearch.jsonl` and `results.md`.
4. Continue with the required next runs in order.

## How to Export The Snapshot

Use:

`make urgency_snapshot`

Or:

`LOGGING_CONFIG=dev_logging.conf uv run python -m feedoscope.urgency_experiments.export_snapshot`

## How to Run One Experiment

Standardize on:

`./autoresearch.sh`

Rules for `autoresearch.sh`:

- run exactly one experiment configuration
- use `uv run ...` for Python commands
- print structured `METRIC` lines
- exit non-zero on crash

## Required Next Run Order

Run these next, in this order:

1. control bootstrap: `answerdotai/ModernBERT-base @ 512`, `title_head`, `transformer`
2. `answerdotai/ModernBERT-base @ 512`, `single_blob`, `transformer`
3. `distilbert-base-uncased @ 512`, `title_head`, `transformer`
4. `google/embeddinggemma-300m @ 2048`, `title_head`, `embedding_linear`
5. `BAAI/bge-m3 @ 2048`, `title_head`, `embedding_linear`
6. `Alibaba-NLP/gte-large-en-v1.5 @ 2048`, `title_head`, `embedding_linear`

Do not reorder these first six runs.

## Model-Aware Defaults

- `answerdotai/ModernBERT-base`
  - classifier type: `transformer`
  - max length: `512`
  - text prep: `title_head`
- `distilbert-base-uncased`
  - classifier type: `transformer`
  - max length: `512`
  - text prep: `title_head`
- `google/embeddinggemma-300m`
  - classifier type: `embedding_linear`
  - max length: `2048`
  - text prep: `title_head`
  - pooling: `mean`
- `BAAI/bge-m3`
  - classifier type: `embedding_linear`
  - max length: `2048`
  - text prep: `title_head`
  - pooling: `mean`
- `Alibaba-NLP/gte-large-en-v1.5`
  - classifier type: `embedding_linear`
  - max length: `2048`
  - text prep: `title_head`
  - pooling: `cls`

## What To Keep Updated

Keep these files current as the loop progresses:

- `autoresearch.jsonl`
- `results.md`
- `autoresearch.ideas.md`

Log every run with:

- keep / discard / crash status
- short rationale
- primary metric
- important secondary metrics
- model name and knobs
- what to try next
