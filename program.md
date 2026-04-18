# Program: Relevance Autoresearch Loop

This loop is inspired by `karpathy/autoresearch`, but it must stay on the current
dedicated branch instead of creating a new branch.

## Objective

Improve the relevance model evaluation on the existing frozen dataset snapshot.

Optimize the relevance model only. Ignore urgency.

This is a resumed loop, not a bootstrap loop. The current control to beat is the
best kept run from the latest experiment history:

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

The next loop should treat that configuration as the control baseline.

## Current Repo Context

Relevant files in this repository:

- `feedoscope/llm_learn.py`: production relevance training path.
- `feedoscope/llm_infer.py`: production relevance inference path.
- `feedoscope/utils.py`: current production text preparation.
- `feedoscope/eval_models.py`: current evaluation job.
- `feedoscope/relevance_experiments/export_snapshot.py`: frozen snapshot export.
- `feedoscope/relevance_experiments/harness.py`: transformer experiment harness.
- `autoresearch.sh`: main control surface for one experiment run, including the
  embedding + linear-head path.
- `autoresearch.jsonl`: append-only experiment history.
- `autoresearch.ideas.md`: short list of next experiment ideas.

Current label semantics in the repo remain the starting point for the experiment
snapshot unless a future campaign explicitly changes them:

- good / relevant: `status = 'read'` and `vote >= 0`
- bad / not relevant: `vote = -1`

## Primary Metric

Primary metric:

- `average_precision`
- higher is better

This remains the optimization target for the loop.

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

## Frozen Snapshot Rules

The snapshot has already been exported and frozen locally.

Rules:

- Keep using the existing frozen snapshot unless the human explicitly starts a new
  campaign.
- Do not re-query the live DB for each run.
- Keep using the frozen eval split for all comparisons in this loop.
- Keep the snapshot metadata, seed, and split definition unchanged.

## Evaluation Rules

Use one frozen relevance eval split for the entire loop.

Requirements:

- same train / eval split across all runs
- same random seed across all runs unless the explicit experiment is about seed
- same metric computation across all runs
- compare only on the frozen snapshot

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

## Training Data Policy

Default training regime for this loop:

- use the full frozen training split
- do not downsample to a 50/50 balanced subset unless the explicit experiment is
  about balance mode
- keep the existing excellent-article weighting behavior

The recent experiment history strongly suggests that training on the full frozen
training split is better than the old balanced-downsample regime for this task.

## Preferred Implementation Shape

Do not start by hacking production files back and forth.

Preferred path:

1. keep using the dedicated relevance experiment harness
2. keep using the frozen snapshot
3. parameterize:
   - model name
   - max length
   - text preparation mode
   - classifier type
   - train balance mode
   - output / artifact path
   - seed
   - embedding-specific knobs if needed
4. make `autoresearch.sh` call that harness and print structured metrics

Production files should remain mostly untouched during exploration.
Only backport the winning idea after the loop has produced a clear result.

## Embedding Model Rules

Embedding models are now first-class candidates.

Allowed embedding-specific knobs:

- pooling mode, such as `mean` or `cls`
- optional text prefixing when recommended by the model card
- optional layer norm before final L2 normalization
- optional embedding-dimension truncation when recommended by the model family

Rules:

- If a model card recommends a specific pooling mode or prefix for classification,
  the loop may use it.
- Every embedding-specific knob must be logged in the run name and results so the
  comparison remains interpretable.
- After any harness change that could affect embeddings, rerun the current kept
  control before judging new models.

## Files to Read First

Read these files fully before changing anything:

- `program.md`
- `autoresearch.md`
- `autoresearch.ideas.md`
- `autoresearch.sh`
- `feedoscope/relevance_experiments/harness.py`
- `feedoscope/relevance_experiments/export_snapshot.py`
- `feedoscope/llm_learn.py`
- `feedoscope/llm_infer.py`
- `feedoscope/utils.py`
- `feedoscope/eval_models.py`
- `feedoscope/data_registry/data_registry.py`

## Files In Scope

Preferred in-scope surface:

- `program.md`
- `autoresearch.md`
- `autoresearch.ideas.md`
- `autoresearch.sh`
- `feedoscope/relevance_experiments/harness.py`
- new experiment-only helpers if needed
- local experiment artifact / log files

Avoid modifying these during exploration unless absolutely necessary:

- production training entrypoints
- production inference entrypoints
- deployment manifests
- urgency pipeline files

## Required Next Run Order

This is the required sequence for the next loop segment:

1. Control rerun of the current EmbeddingGemma winner after any harness changes.
2. `Alibaba-NLP/gte-base-en-v1.5 @ 2048`, `single_blob`, embedding + linear head,
   full train split.
3. `Alibaba-NLP/gte-large-en-v1.5 @ 2048`, `single_blob`, embedding + linear head,
   full train split.
4. `nomic-ai/nomic-embed-text-v1.5 @ 2048`, `single_blob`, embedding + linear head,
   full train split.
5. `Snowflake/snowflake-arctic-embed-m-v2.0 @ 2048`, `single_blob`, embedding +
   linear head, full train split.
6. `mixedbread-ai/mxbai-embed-large-v1 @ 512`, `single_blob`, embedding + linear
   head, full train split.

Do not reorder those first six runs.

## Model-Aware Defaults

Use these defaults unless the loop finds a reason to deviate:

- `google/embeddinggemma-300m`
  - pooling: `mean`
  - prefix: none
  - layer norm: off
  - truncate dim: off
- `Alibaba-NLP/gte-base-en-v1.5`
  - pooling: `cls`
  - prefix: none
  - layer norm: off
  - truncate dim: off
- `Alibaba-NLP/gte-large-en-v1.5`
  - pooling: `cls`
  - prefix: none
  - layer norm: off
  - truncate dim: off
- `nomic-ai/nomic-embed-text-v1.5`
  - pooling: `mean`
  - prefix: `classification: `
  - layer norm: on
  - truncate dim: `512`
- `Snowflake/snowflake-arctic-embed-m-v2.0`
  - pooling: `cls`
  - prefix: none
  - layer norm: off
  - truncate dim: off
- `mixedbread-ai/mxbai-embed-large-v1`
  - pooling: `cls`
  - prefix: none
  - layer norm: off
  - truncate dim: off

If the loop wants to compare a generic mean-pooling harness against a model-aware
setting, treat that as a separate explicit experiment.

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
- embedding-specific knobs if used

The log must make it possible for a fresh agent to resume without re-reading the
whole repo.

## Experiment Strategy After the Required Queue

After the required next-run queue, continue with the most justified moves.

Good next moves:

- compare one generic embedding configuration against one model-aware configuration
  for a promising family
- tune `C` locally around a promising new embedding winner
- compare `single_blob` versus `title_head` only if a new model is close to the
  control
- revisit transformer models only if a specific full-data hypothesis remains open

Bad next moves:

- changing many things at once before the required queue is complete
- touching urgency
- re-querying the DB every run
- rewriting the evaluation target mid-loop
- assuming a new embedding model is worse when it may simply need its recommended
  pooling / prefix

## Resume Rules

When resuming with no context:

1. verify the current branch is still the dedicated experiment branch
2. verify it is not `main` or `master`
3. read this file
4. read `autoresearch.md`
5. read `autoresearch.ideas.md`
6. read experiment logs
7. identify the best kept run so far
8. continue from the next most justified experiment
9. do not repeat dead ends unless there is a specific new reason

## Never Stop Rule

This is an autonomous loop.

After setup and the required queue, continue until interrupted.

Do not ask whether to continue.
Do not stop after one improvement.
Keep exploring justified combinations and adjacent ideas until explicitly stopped or
until a configured `maxIterations` limit is reached.

## Final Deliverable

The loop is successful when it can answer:

- whether the current EmbeddingGemma winner survives a control rerun after harness
  changes
- whether any other embedding family beats that control on the frozen snapshot
- whether model-aware pooling / prefixing matters materially for the best candidate
- whether full-data training should remain the default regime
- what the best relevance configuration is on the frozen snapshot
- what the measured tradeoffs are in AP, ROC AUC, log loss, VRAM, and runtime
