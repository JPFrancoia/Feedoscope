# Program: Relevance Autoresearch Loop

This loop is inspired by `karpathy/autoresearch`, but it must stay on the current
dedicated branch instead of creating a new branch.

## Objective

Improve the relevance model evaluation on a fixed frozen dataset snapshot.

Optimize the relevance model only. Ignore urgency.

The first four candidate configurations are mandatory:

1. `ModernBERT-base @ 512`
2. `Ettin-68m @ 1024`
3. `Ettin-150m @ 512`
4. `ModernBERT-base @ 512` with `title+head+tail` chunking

Run those four first, in that order, before branching into combinations or follow-up
ideas.

## Current Repo Context

Relevant files in this repository:

- `feedoscope/llm_learn.py`: relevance training path, current model defaults, batch
  size, and sequence length.
- `feedoscope/llm_infer.py`: relevance inference path.
- `feedoscope/utils.py`: current text preparation (`title + content` as a single
  cleaned blob).
- `feedoscope/eval_models.py`: current relevance evaluation logic and metrics.
- `feedoscope/data_registry/data_registry.py`: DB access helpers.
- `feedoscope/data_registry/sql/get_read_articles_training.sql`: current positive
  label query.
- `feedoscope/data_registry/sql/get_published_articles.sql`: current negative label
  query.

Current label semantics in the repo should be the starting point for the experiment
snapshot unless the loop intentionally starts a new campaign:

- good / relevant: `status = 'read'` and `vote >= 0`
- bad / not relevant: `vote = -1`

## Primary Metric

Primary metric:

- `average_precision`
- higher is better

This matches the current training code's best-model metric and is the optimization
target for this loop.

## Secondary Metrics

Always log these secondary metrics:

- `roc_auc`
- `log_loss`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `peak_vram_gb`
- `train_seconds`
- `total_seconds`

If two runs are effectively tied on `average_precision`, prefer:

1. lower `log_loss`
2. simpler implementation
3. lower runtime

## Python Environment

This repository uses a `uv`-managed Python virtual environment.

Rules:

- Run all Python commands inside the project environment.
- Use `uv run ...` for Python scripts, modules, and CLIs.
- Do not use the system Python interpreter for project commands.
- Do not use system `pip`.
- If a Python dependency is required, install it with `uv` in this repository
  environment.
- New Python dependencies are allowed if they are needed for the Parquet snapshot or
  the experiment harness.

## Database Safety

Database access is strictly read-only.

Allowed:

- `SELECT`
- `EXPLAIN SELECT`

Forbidden:

- `INSERT`
- `UPDATE`
- `DELETE`
- `ALTER`
- `DROP`
- `CREATE`
- `TRUNCATE`
- `VACUUM`
- `ANALYZE`
- `GRANT`
- `REVOKE`
- any command that writes data, changes schema, changes permissions, changes stats,
  or modifies metadata

Never write experiment results back to PostgreSQL.
Never use the database as an artifact store.
Never use the database as a scratchpad.

Example connection:

`pgcli -h localhost -u miniflux -d miniflux -p 5432`

## First Step: Freeze the Dataset Locally

Before any baseline or model experiment, export the full relevance experiment dataset
from PostgreSQL into local Parquet file(s).

This is mandatory.

Rules:

- Use the database only to read the source data for the experiment snapshot.
- After the Parquet snapshot is created, all training and evaluation runs must use
  only the local Parquet data, not repeated live DB queries.
- Preserve the current relevance dataset definition and filtering behavior unless the
  loop explicitly decides to start a new experiment campaign.
- Record snapshot metadata:
  - snapshot timestamp
  - SQL / query provenance
  - random seed
  - holdout split definition
  - filtering rules used to create the snapshot

The snapshot must include all fields needed to reproduce the current relevance
pipeline, including at least:

- `article_id`
- `title`
- `content`
- `vote`
- `starred`
- `status`
- `feed_name` if needed
- timestamps if needed
- the binary relevance label or enough information to derive it deterministically

If Parquet support is missing, install `pyarrow` or `fastparquet` with `uv` and then
continue.

After the snapshot is created:

- all experiments must train and evaluate from the local Parquet snapshot only
- do not re-query the live DB for each run
- only refresh the snapshot when explicitly starting a new experiment campaign

## Keep the Evaluation Fixed

Use one frozen relevance eval split for the entire loop.

Requirements:

- same train / eval split across all runs
- same random seed across all runs
- same class-balancing logic across all runs unless the explicit experiment is about
  class-balancing
- same metric computation across all runs

Do not let the loop optimize against a moving target.

## Hardware Constraint

This machine has a single GPU with 12 GB VRAM.

Rules:

- A successful candidate must fit on this 12 GB GPU.
- Any configuration that OOMs on this hardware is a `crash`, not a valid result.
- Do not assume access to a larger GPU.
- Always log peak VRAM usage.

This constraint is part of the optimization target, not an incidental detail.

## Git Constraints

This loop runs on a human-prepared dedicated branch.

Rules:

- Never create a new branch.
- Never switch branches.
- Never commit on `main` or `master`.
- If the current branch is `main` or `master`, stop immediately and ask the human to
  restart the loop from a dedicated branch.
- Local commits on the current dedicated branch are allowed.
- Never merge `main` or `master` into the experiment branch.
- Never merge the experiment branch into anything.
- Never push.
- Never force-push.

These rules override any default branch-creation behavior in the autoresearch
tooling.

## Preferred Implementation Shape

Do not start by hacking production files back and forth.

Preferred path:

1. build a dedicated relevance experiment harness
2. point it at the frozen snapshot
3. parameterize:
   - model name
   - max length
   - text preparation mode
   - output / artifact path
   - seed
4. make `autoresearch.sh` call that harness and print structured metrics

Production files should remain mostly untouched during exploration.
Only backport the winning idea after the loop has produced a clear result.

## Files to Read First

Read these files fully before changing anything:

- `program.md`
- `autoresearch.md`
- `feedoscope/llm_learn.py`
- `feedoscope/llm_infer.py`
- `feedoscope/utils.py`
- `feedoscope/eval_models.py`
- `feedoscope/data_registry/data_registry.py`
- `feedoscope/data_registry/sql/get_read_articles_training.sql`
- `feedoscope/data_registry/sql/get_published_articles.sql`
- `plans/eval-job-plan.md`
- `plans/ettin-migration-plan.md`

## Files In Scope

Preferred in-scope surface:

- new experiment-only harness files
- new snapshot / export helpers
- new experiment text-prep helpers
- `program.md`
- `autoresearch.md`
- `autoresearch.sh`
- `autoresearch.ideas.md`
- local experiment artifact / log files

Avoid modifying these during exploration unless absolutely necessary:

- production training entrypoints
- production inference entrypoints
- deployment manifests
- urgency pipeline files

## Required Run Order

Step 0:

- export the experiment dataset from PostgreSQL into local Parquet
- define and freeze the eval split
- record snapshot metadata

Then run these four candidates first:

1. baseline: `ModernBERT-base @ 512` with the current single-blob text prep
2. `Ettin-68m @ 1024` with the same single-blob text prep
3. `Ettin-150m @ 512` with the same single-blob text prep
4. `ModernBERT-base @ 512` with `title+head+tail` chunking

Do not reorder these first four runs.

## Definition of title+head+tail

The `title+head+tail @ 512` experiment must preserve a fair token budget.

Default interpretation:

- clean the title
- clean the content body
- construct the input from:
  - full title
  - first chunk of the article body
  - last chunk of the article body
- do not exceed tokenizer `max_length`
- keep the method deterministic
- do not use random chunking

Prefer a simple deterministic budget split:

- always include the title
- split the remaining budget roughly evenly between head and tail
- reserve a small buffer for special tokens

Do not make the chunking logic overly clever in the first version.

## How to Run One Experiment

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
- `METRIC peak_vram_gb=<value>`
- `METRIC train_seconds=<value>`
- `METRIC total_seconds=<value>`

## Baseline Rule

The very first successful run must be the baseline:

- `ModernBERT-base @ 512`
- frozen snapshot
- frozen split
- current single-blob text preparation

Do not skip the baseline.

## Keep / Discard Policy

- Better `average_precision` on the same frozen eval split means `keep`.
- Worse `average_precision` means `discard`.
- If `average_precision` is effectively tied, use `log_loss`, simplicity, and
  runtime as tiebreakers.
- If confidence is low or the gain looks close to noise, rerun once on the same
  snapshot before deciding.
- Crashes count as `crash` and must be logged with the reason.
- Do not keep complexity-heavy changes for tiny ambiguous gains.

## Logging

Maintain append-only experiment history.

Every run should record:

- current branch name
- snapshot id or snapshot path
- model name
- max length
- text-prep mode
- primary metric
- secondary metrics
- peak VRAM
- keep / discard / crash status
- short description
- what was learned

The log must make it possible for a fresh agent to resume without re-reading the
whole repo.

## Experiment Strategy After the First Four

After the four required runs, continue with the most justified next moves.

Good next moves:

- combine the best encoder with `title+head+tail`
- if `Ettin-68m @ 1024` wins, test whether chunking still helps
- if chunking wins strongly at 512, test chunking with `Ettin-150m @ 512`
- try memory-saving implementation changes only if they enable longer context while
  preserving the evaluation protocol

Bad next moves:

- changing many things at once before the first four runs are isolated
- touching urgency
- swapping to decoder-only models for this classification task
- re-querying the DB every run
- rewriting the evaluation target mid-loop

## Resume Rules

When resuming with no context:

1. verify the current branch is still the dedicated experiment branch
2. verify it is not `main` or `master`
3. read this file
4. read `autoresearch.md`
5. read experiment logs
6. identify the best kept run so far
7. continue from the next most justified experiment
8. do not repeat dead ends unless there is a specific new reason

## Never Stop Rule

This is an autonomous loop.

After setup and baseline, continue until interrupted.

Do not ask whether to continue.
Do not stop after the first four runs.
Do not stop after finding one improvement.
Keep exploring justified combinations and adjacent ideas until explicitly stopped or
until a configured `maxIterations` limit is reached.

## Final Deliverable

The loop is successful when it can answer:

- which of the four required candidates won
- whether larger context helped more than encoder choice
- whether `title+head+tail` helped more than model swaps
- what the best relevance configuration is on the frozen snapshot
- what the measured tradeoffs are in AP, ROC AUC, log loss, VRAM, and runtime
