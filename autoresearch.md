# Autoresearch: Relevance Model Optimization

Read `program.md` first. It is the authoritative full brief for this loop.

## Objective

Improve relevance evaluation on a fixed frozen dataset snapshot.

This loop must optimize relevance only.

The first four required candidates are:

1. `ModernBERT-base @ 512`
2. `Ettin-68m @ 1024`
3. `Ettin-150m @ 512`
4. `ModernBERT-base @ 512` with `title+head+tail` chunking

Run them in that order before branching into combinations.

## Metrics

- Primary: `average_precision` (higher is better)
- Secondary: `roc_auc`, `log_loss`, `accuracy`, `precision`, `recall`, `f1`,
  `peak_vram_gb`, `train_seconds`, `total_seconds`

## How to Run

This session is not fully bootstrapped yet.

Initial setup requirements:

1. Export the relevance experiment data from PostgreSQL into local Parquet.
2. Freeze the eval split and record snapshot metadata.
3. Create the experiment harness.
4. Create `autoresearch.sh`.
5. Run the baseline.

Once bootstrapped, standardize on:

`./autoresearch.sh`

Rules for `autoresearch.sh`:

- run exactly one experiment configuration
- use `uv run ...` for Python commands
- print structured `METRIC` lines
- exit non-zero on crash

## Files In Scope

- new experiment-only harness files
- snapshot / export helpers
- experiment text-prep helpers
- `program.md`
- `autoresearch.md`
- `autoresearch.sh`
- `autoresearch.ideas.md`
- local experiment artifact / log files

## Off Limits

- any write to PostgreSQL
- urgency pipeline files unless absolutely required for shared utilities
- `main` and `master`
- creating branches
- switching branches
- merging any branch
- pushing or force-pushing
- deployment manifests unless a winning result is being backported later

## Constraints

- The current branch is a dedicated experiment branch. Stay on it.
- Never create a new branch.
- Never switch branches.
- Never commit on `main` or `master`.
- If the current branch is `main` or `master`, stop and ask the human to restart
  from a dedicated branch.
- Local commits on the current dedicated branch are allowed.
- Never push.
- Never force-push.
- Use the `uv`-managed environment for all Python work.
- If a dependency is needed for Parquet or the experiment harness, install it with
  `uv`.
- Database access is read-only. Allowed SQL is `SELECT` and `EXPLAIN SELECT` only.
- Freeze the dataset to Parquet before any model experiments.
- After the snapshot is created, all experiments must use the local snapshot only.
- The machine has one 12 GB GPU. OOM is a crash.
- Keep the evaluation target fixed across runs.
- Do not reorder the first four required runs.

## What's Been Tried

No experiments have been run yet.

Required initial sequence:

1. create the local Parquet snapshot
2. freeze the eval split
3. implement the experiment harness
4. implement `autoresearch.sh`
5. run the four required candidates in order

Keep this section updated with:

- best kept run so far
- dead ends
- crash causes
- promising combinations to try next
