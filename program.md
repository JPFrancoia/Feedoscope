# Program: Relevance Autoresearch Loop

This loop is inspired by `karpathy/autoresearch`, but it must stay on the current
dedicated branch instead of creating a new branch.

## Objective

Improve relevance evaluation on the existing frozen dataset snapshot.

Optimize the relevance model only. Ignore urgency.

This is a resumed loop, not a bootstrap loop. The current control to beat is the
best kept run from the latest experiment history:

- commit: `ea778e6`
- model: `google/embeddinggemma-300m`
- classifier type: `embedding_linear`
- max length: `2048`
- text prep: `title_head`
- train balance mode: `full`
- linear head `C`: `5.0`
- average precision: `0.984098480367228`
- log loss: `0.21130`

The next loop should stay narrow. Real-world validation suggests the current master
path remains strong, so this wave is only about improving the current embedding
approach or clearly beating it with one of a few larger alternatives.

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
- Keep using the existing frozen snapshot and frozen eval split only.
- The machine has one 12 GB GPU. OOM is a crash.
- Default to `TRAIN_BALANCE_MODE=full` unless the explicit experiment is about
  balance mode.

## Training Data Policy

Default training regime for this loop:

- use the full frozen training split
- do not downsample to a 50/50 balanced subset unless the explicit experiment is
  about balance mode
- keep the existing excellent-article weighting behavior

## Embedding Rules

Embedding models are first-class candidates.

Allowed embedding-specific knobs:

- pooling mode, such as `mean` or `cls`
- prefix mode, such as `classification` or `query`
- prompt mode, such as `classification` or `document`
- feature mode, such as `dense`, `sparse`, or `hybrid`
- optional layer norm before final L2 normalization
- optional embedding-dimension truncation
- sparse feature hash dimension for lexical features

Rules:

- If a model card recommends a specific pooling mode, prompt, or prefix, the loop may
  use it.
- Every embedding-specific knob must be logged in the run name and results.
- After any harness change that could affect embeddings, rerun the current control
  before judging new models.
- For BGE-M3 sparse or hybrid experiments, the loop may install `FlagEmbedding` with
  `uv` if it is missing.

## Files to Read First

- `program.md`
- `autoresearch.md`
- `autoresearch.ideas.md`
- `autoresearch.sh`
- `feedoscope/relevance_experiments/harness.py`
- `feedoscope/relevance_experiments/export_snapshot.py`

## Files In Scope

- `program.md`
- `autoresearch.md`
- `autoresearch.ideas.md`
- `autoresearch.sh`
- `feedoscope/relevance_experiments/harness.py`
- new experiment-only helpers if needed
- local experiment artifact / log files

Avoid modifying production training/inference entrypoints in this loop unless a very
specific experiment requires it.

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

Do not reorder those first seven runs.

## Model-Aware Defaults

Use these defaults unless the loop finds a reason to deviate:

- `google/embeddinggemma-300m`
  - pooling: `mean`
  - prompt modes to test: `none`, then `classification`, then `document`
  - prefix mode: `none`
  - layer norm: off
  - truncate dim: off
- `Alibaba-NLP/gte-large-en-v1.5`
  - pooling: `cls`
  - prompt mode: `none`
  - prefix mode: `none`
- `Snowflake/snowflake-arctic-embed-l-v2.0`
  - pooling: `cls`
  - prompt mode: `none`
  - prefix mode: `none`
- `BAAI/bge-m3`
  - dense control uses `mean` pooling
  - hybrid uses `EMBED_FEATURE_MODE=hybrid`
  - sparse-only uses `EMBED_FEATURE_MODE=sparse`
  - lexical features use `EMBED_SPARSE_HASH_DIM=8192` to start

Older candidates remain valid but are lower priority now:

- `Alibaba-NLP/gte-base-en-v1.5`
- `nomic-ai/nomic-embed-text-v1.5`
- `Snowflake/snowflake-arctic-embed-m-v2.0`
- `mixedbread-ai/mxbai-embed-large-v1`

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

Every run should record:

- current branch name
- snapshot id or snapshot path
- model name
- classifier type
- max length
- text-prep mode
- train balance mode
- prompt mode
- feature mode
- primary metric
- secondary metrics
- peak VRAM
- keep / discard / crash status
- short description
- what was learned

## Experiment Strategy After the Required Queue

Good next moves:

- tune `C` locally only if a prompt-aware or larger dense model comes close to the
  control
- compare BGE-M3 dense, hybrid, and sparse modes only after the larger dense runs are
  done
- revisit older queued dense models only if the new required queue disappoints

Bad next moves:

- changing many things at once before the required queue is complete
- touching urgency
- re-querying the DB every run
- rewriting the evaluation target mid-loop
- assuming a new embedding model is worse when it may simply need its recommended
  pooling, prompt, or hybrid mode

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

## Final Deliverable

The loop is successful when it can answer:

- whether the current EmbeddingGemma winner survives a control rerun after harness
  changes
- whether EmbeddingGemma improves further with classification or document prompting
- whether any larger dense embedding family beats that control on the frozen snapshot
- whether BGE-M3 hybrid dense+sparse features beat dense-only embeddings
- what the best relevance configuration is on the frozen snapshot
- what the measured tradeoffs are in AP, ROC AUC, log loss, VRAM, and runtime
