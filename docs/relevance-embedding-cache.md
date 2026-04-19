# Relevance Embedding Cache

## Overview

Feedoscope relevance scoring now uses a frozen EmbeddingGemma encoder plus a
logistic-regression classifier.

Two layers are cached:

1. The shared Hugging Face model files on disk under `models/relevance_encoder/`
2. Per-article relevance embeddings in Postgres in the `relevance_embeddings`
   table

This avoids redownloading the encoder and avoids recomputing article embeddings
for unchanged articles on every train, infer, and eval run.

## Current Components

Relevant files:

- `feedoscope/relevance_embedding.py`
- `feedoscope/relevance_text.py`
- `feedoscope/llm_learn.py`
- `feedoscope/llm_infer.py`
- `feedoscope/eval_models.py`
- `feedoscope/data_registry/data_registry.py`
- `db/migrations/000004_relevance_embeddings.up.sql`

## Shared Encoder Cache

The frozen encoder is stored once on disk under:

```text
models/relevance_encoder/<model-name-with-slashes-replaced>
```

For the current production configuration this is:

```text
models/relevance_encoder/google--embeddinggemma-300m
```

Runtime behavior:

- if the snapshot is already present, Feedoscope reuses it
- if `.snapshot_complete` is missing but the model files are complete,
  Feedoscope still accepts the cache and recreates the marker
- if the snapshot is missing, Feedoscope tries to download it
- if the Hugging Face repo is gated and credentials are missing, Feedoscope
  raises a clear runtime error instead of failing later with a less actionable
  stack trace

## Database Embedding Cache

Per-article embeddings are stored in Postgres in `relevance_embeddings`.

Schema summary:

- `article_id`
- `model_name`
- `max_length`
- `text_prep_mode`
- `prep_version`
- `text_hash`
- `embedding` (`bytea`)
- `last_updated`

The primary key is:

```text
(article_id, model_name, max_length, text_prep_mode, prep_version)
```

Important design choices:

- there is one current row per article and embedding-defining configuration
- `text_hash` is not part of the primary key
- when the prepared article text changes, the row is overwritten in place
- embeddings are stored as raw normalized `float32` bytes in `bytea`

The cache intentionally uses plain Postgres types instead of `pgvector`, because
the deployed Postgres image does not include the `vector` extension.

## Cache Key and Invalidation

An embedding is reused only when all of these match:

- `article_id`
- `RELEVANCE_MODEL_NAME`
- `RELEVANCE_MAX_LENGTH`
- `RELEVANCE_TEXT_PREP_MODE`
- `RELEVANCE_PREP_VERSION`
- the current prepared-text hash

`RELEVANCE_ENCODER_BATCH_SIZE` is not part of the key because it changes speed,
not embedding values.

`RELEVANCE_LINEAR_C` is also not part of the key because it changes only the
logistic-regression fit, not the embedding itself.

`RELEVANCE_PREP_VERSION` exists so preprocessing changes can invalidate the cache
cleanly without deleting rows manually.

## Runtime Flow

### Training

`feedoscope.llm_learn`:

1. loads the shared encoder snapshot
2. prepares article text
3. loads cached embeddings for matching rows
4. computes only cache misses or stale rows
5. upserts fresh embeddings
6. fits logistic regression on the combined embedding matrix

### Inference

`feedoscope.llm_infer`:

1. loads the latest saved classifier artifact
2. loads the shared encoder snapshot
3. prepares article text
4. reuses cached embeddings where possible
5. computes only misses or stale rows
6. predicts relevance probabilities with logistic regression

### Evaluation

`feedoscope.eval_models` uses the same cache-aware relevance path for holdout
evaluation.

This means train, infer, and eval all warm the same embedding cache.

## Text Preparation

The current production text-prep mode is `title_head`.

That path:

- keeps the cleaned title
- takes the first body tokens that fit within the configured token budget
- avoids tokenizing the full body just to slice it later

The prepared-text output is what gets hashed for cache invalidation.

## Storage Format

Embeddings are serialized as raw `float32` bytes with NumPy before being stored
in Postgres and are deserialized back into `np.ndarray` on read.

This format is optimized for cache reuse inside Feedoscope. It is not intended
for SQL-side similarity search.

## Operations

Apply the schema:

```bash
make up
```

Typical warm-cache validation flow:

```bash
make infer
make infer
```

Useful log signals:

- `Using cached relevance encoder`
- `Preparing relevance text for ...`
- `Relevance embedding cache: <hits> hits, <misses> misses`
- `Encoding batch ...`

## Current Limitations

- cold-cache runs still pay the full embedding cost
- cache rows accumulate across different embedding configurations
- similarity search is not implemented yet because embeddings are stored as
  `bytea`, not queryable vector types

## When to Update This Doc

Update this document when any of these change:

- the relevance model family
- the text-prep strategy
- the cache key shape
- the embedding storage format
- the training or inference call flow around cache reuse
