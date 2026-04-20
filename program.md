# Program: Urgency Autoresearch Loop

This loop is inspired by `karpathy/autoresearch`, but it must stay on the
current dedicated branch instead of creating a new branch.

## Objective

Improve urgency evaluation using only manually labeled read articles.

Optimize the urgency model only. Ignore relevance.

This is a bootstrap loop. There is no kept urgency control yet, so the first job
is to establish one from the frozen read-tagged snapshot and then iterate from
that baseline.

## Data Definition

The supervision signal is:

- `entries.status = 'read'`
- Miniflux user tag `0-urgency` or `1-urgency`

Treat these as the trusted labels.

Do not use:

- unread urgency tags
- `time_sensitivity_simplified`
- one-shot Ministral classifications

## Primary Metric

- `average_precision`
- higher is better

## Secondary Metrics

Always log:

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

If two runs are effectively tied on `average_precision`, prefer:

1. lower `log_loss`
2. simpler implementation
3. lower runtime

## Constraints

- Use the `uv`-managed environment for all Python commands.
- Stay on the current dedicated experiment branch.
- Never create a new branch.
- Never switch branches.
- Never commit on `main` or `master`.
- Never push.
- Never force-push.
- Database access is read-only. Allowed SQL is `SELECT` and `EXPLAIN SELECT` only.
- Keep using the frozen urgency snapshot and frozen eval split only after export.
- The machine has one GPU. OOM is a crash.
- Default to `TRAIN_BALANCE_MODE=full` unless the explicit experiment is about
  balance mode.

## Files To Read First

- `program.md`
- `autoresearch.md`
- `autoresearch.ideas.md`
- `autoresearch.sh`
- `feedoscope/urgency_experiments/export_snapshot.py`
- `feedoscope/urgency_experiments/harness.py`

## Files In Scope

- `program.md`
- `autoresearch.md`
- `autoresearch.ideas.md`
- `autoresearch.jsonl`
- `autoresearch.sh`
- `results.md`
- `feedoscope/urgency_experiments/export_snapshot.py`
- `feedoscope/urgency_experiments/harness.py`
- small experiment-only helpers if needed

Avoid changing production urgency inference in this loop unless a very specific
experiment requires it.

## Snapshot Rules

Before the first run:

1. export the frozen urgency snapshot
2. record the snapshot path
3. keep reusing that exact snapshot for comparison

Do not re-export the snapshot between runs unless you intentionally want a new
campaign.

## Model Families In Scope

Start with these:

- `answerdotai/ModernBERT-base` as the transformer control
- `distilbert-base-uncased` as a smaller transformer baseline
- `google/embeddinggemma-300m` as the first frozen embedding baseline
- `BAAI/bge-m3` as the second frozen embedding baseline
- `Alibaba-NLP/gte-large-en-v1.5` as a stronger embedding follow-up

## Text Preparation Modes

Initial allowed urgency text prep modes:

- `title_head`
- `single_blob`

Prefer `title_head` first because it worked well in the relevance loop and
should better preserve title signal under fixed token budgets.

## How To Run One Experiment

Standardize on:

`./autoresearch.sh`

`autoresearch.sh` must:

1. fail fast on obvious errors
2. run exactly one experiment configuration
3. run Python commands via `uv run ...`
4. print structured `METRIC` lines
5. exit non-zero on crash

It must print at least:

- `METRIC average_precision=<value>`
- `METRIC roc_auc=<value>`
- `METRIC log_loss=<value>`
- `METRIC accuracy=<value>`
- `METRIC f1=<value>`
- `METRIC brier_score=<value>`
- `METRIC peak_vram_gb=<value>`
- `METRIC train_seconds=<value>`
- `METRIC total_seconds=<value>`

## Keep / Discard Policy

- Better `average_precision` on the same frozen eval split means `keep`.
- Worse `average_precision` means `discard`.
- If `average_precision` is effectively tied, use `log_loss`, simplicity, and
  runtime as tiebreakers.
- Crashes count as `crash` and must be logged with the reason.
- Do not keep complexity-heavy changes for tiny ambiguous gains.

## Logging

Every run should record:

- current branch name
- snapshot id or snapshot path
- model name
- classifier type
- max length
- text-prep mode
- train balance mode
- primary metric
- secondary metrics
- peak VRAM
- keep / discard / crash status
- short description
- what was learned

Use:

- `autoresearch.jsonl` for append-only machine-readable history
- `results.md` for the human-readable running summary

## Resume Rules

When resuming with no context:

1. verify the current branch is still the dedicated experiment branch
2. verify it is not `main` or `master`
3. read this file
4. read `autoresearch.md`
5. read `autoresearch.ideas.md`
6. read `autoresearch.jsonl` and `results.md`
7. identify the best kept run so far
8. continue from the next most justified experiment

## Final Deliverable

The loop is successful when it can answer:

- which urgency model family is best on the frozen read-tagged snapshot
- whether frozen embeddings beat transformer fine-tuning for urgency too
- whether `title_head` beats `single_blob` for urgency
- what the best urgency configuration is on the frozen snapshot
- what the tradeoffs are in AP, log loss, VRAM, and runtime
