Status: In Progress 2026-04-19 (bytea implementation landed, migration applied, warm-cache validation pending)

# Plan: Relevance Embedding Cache

Decision update 2026-04-19:

- The implementation pivoted from `pgvector` to `bytea` after migration failed on
  the production-style Postgres image (`postgres:17-alpine`) because the `vector`
  extension is not installed there.
- The user explicitly chose not to change the database image.
- The cache now stores raw normalized `float32` bytes in Postgres, which removes
  the extension dependency while preserving the caching benefit.
- Future similarity search can still be added later with a separate migration once
  the database image supports `pgvector`.

## Brief

We will add a shared, config-aware database cache for relevance embeddings so
Feedoscope stops recomputing EmbeddingGemma vectors for unchanged articles on
every train and infer run. This is worth doing now because the frozen encoder is
the slow part of the new relevance backend, while the logistic-regression head is
cheap and unread articles are rescored repeatedly. The cache will be keyed by
article plus embedding-defining config, and it will invalidate automatically when
the prepared text changes.

## Current State / Relevant Context

- `feedoscope/relevance_embedding.py` always prepares article text and computes
  embeddings from scratch via `encode_articles()` and `encode_texts()`.
- Both `feedoscope/llm_learn.py` and `feedoscope/llm_infer.py` call that same
  path, so training and inference both pay the encoder cost repeatedly.
- `feedoscope/main.py` rescales scores for unread articles on every run, so the
  same unread backlog is embedded over and over.
- The shared EmbeddingGemma model files are already persisted under `models/`,
  but per-article embeddings are not cached anywhere yet.
- The repository currently has only scalar cache tables in migrations:
  `time_sensitivity`, `time_sensitivity_simplified`, and `urgency_inference`.
- The original design targeted pgvector, but the active implementation now uses
  `bytea` because the deployed Postgres image does not ship the extension.
- The user also approved a shared-by-config cache, meaning experiments with a
  different embedding configuration should coexist safely with production rows.

## Proposed Implementation

### 1. Add a bytea-backed relevance embedding table

Create a new migration that adds a cache table with one row
per article per embedding-defining configuration.

Implemented shape:

```sql
create table relevance_embeddings (
    article_id bigint not null references entries(id) on delete cascade,
    model_name text not null,
    max_length integer not null,
    text_prep_mode text not null,
    prep_version integer not null,
    text_hash text not null,
    embedding bytea not null,
    last_updated timestamp with time zone not null default now(),
    primary key (
        article_id,
        model_name,
        max_length,
        text_prep_mode,
        prep_version
    )
);
```

Implemented design choices:

- `text_hash` is not part of the primary key.
- There should be one current row per article/config.
- If the prepared text changes, we overwrite the row for that article/config.
- Embeddings are stored as raw normalized `float32` bytes to avoid any database
  extension dependency.
- Future similarity-search support remains possible, but it would require a
  separate migration to either introduce pgvector later or materialize vectors
  into a dedicated search table.

### 2. Make the cache key shared by embedding-defining config

The cache must be shared across production and future experiments, but only when
the embedding output would actually be identical.

The effective cache key is:

- `article_id`
- `model_name`
- `max_length`
- `text_prep_mode`
- `prep_version`

Notably excluded from the key:

- `RELEVANCE_ENCODER_BATCH_SIZE`, because it changes performance only
- logistic-regression parameters, because they do not affect the embedding vector

### 3. Introduce an explicit preprocessing version

Add `RELEVANCE_PREP_VERSION` to config so text-preparation changes can invalidate
the cache cleanly without deleting old rows or guessing from code history.

Initial value:

- `RELEVANCE_PREP_VERSION = 1`

Use this as part of the cache key everywhere the embedding cache is read or
written.

### 4. Refactor the relevance embedding flow around cache hits and misses

Refactor `feedoscope/relevance_embedding.py` so it does the expensive work only
for articles whose cached embedding is missing or stale.

Recommended flow:

1. prepare article text once using the current tokenizer and text prep mode
2. compute a deterministic `sha256` hash of each prepared text
3. fetch existing cache rows for the requested article IDs and config
4. mark rows as reusable only when the stored `text_hash` matches the current one
5. compute embeddings only for misses or stale rows
6. upsert the fresh embeddings and hashes
7. assemble the final `np.ndarray` in the original article order

This should become the single path used by both training and inference.

### 5. Add embedding-cache DB access helpers

Extend `feedoscope/data_registry/data_registry.py` with dedicated helpers for the
embedding cache.

Recommended new helpers:

- `get_relevance_embeddings(...)`
- `upsert_relevance_embeddings(...)`

Implemented notes:

- return cached vectors as raw `bytea` and parse them into `np.ndarray`
- use `executemany` plus `insert ... on conflict ... do update` for the first
  implementation rather than a more complex temp-table/COPY flow

This keeps the first version smaller while still being correct.

### 6. Reuse the cache in both training and inference

Update:

- `feedoscope/llm_learn.py`
- `feedoscope/llm_infer.py`

Both should call the same cache-aware embedding API instead of forcing a fresh
encoder pass every time.

Expected effect:

- repeated `make infer` and `make full_infer` runs become mostly cache reads
- repeated `make train` runs can reuse historical embeddings for unchanged rows

### 7. Keep similarity-search support as a future follow-up

The first version should focus on caching and stop short of adding vector-search
APIs or database search indexes.

Do not add in the first implementation:

- `ivfflat` indexes
- `hnsw` indexes
- nearest-neighbor queries

Reason:

- the current goal is cache reuse, not vector search
- enabling vector search now would also require changing the deployed Postgres
  image, which the user explicitly rejected

## File-by-File Impact

New files:

- `db/migrations/000004_relevance_embeddings.up.sql`
- `db/migrations/000004_relevance_embeddings.down.sql`
- `feedoscope/data_registry/sql/get_relevance_embeddings.sql`
- `feedoscope/data_registry/sql/upsert_relevance_embeddings.sql`

Updated files:

- `feedoscope/config.py`
- `feedoscope/relevance_embedding.py`
- `feedoscope/data_registry/data_registry.py`
- `feedoscope/llm_learn.py`
- `feedoscope/llm_infer.py`
- `feedoscope/eval_models.py`

Likely no changes needed:

- `feedoscope/main.py`
- existing relevance SQL selection queries
- urgency code paths

Actual implementation note:

- `pyproject.toml` and `uv.lock` still do not need changes because the cache now
  uses only standard Postgres `bytea` storage and NumPy serialization.

## Risks and Edge Cases

- If text-preparation logic changes and `RELEVANCE_PREP_VERSION` is not bumped,
  stale embeddings could be reused.
- If article content changes in Miniflux, the row must be overwritten when the
  prepared text hash changes.
- Cold-cache runs will still be slow. The payoff comes on repeated runs.
- Experiment rows will accumulate over time because the cache is shared by config.
  That is acceptable initially, but old configs may eventually need manual cleanup.
- The first implementation adds more DB reads and writes. This is expected, but
  warm-cache inference should still be much cheaper than repeated GPU embedding.
- Raw-byte storage is good for caching, but it does not provide immediate SQL-side
  similarity queries. That remains a future enhancement.

## Validation / Testing

Schema validation:

1. `make up`
2. confirm `relevance_embeddings` exists with the expected primary key

Static validation:

1. `make format`
2. `uv run --no-group infer mypy .`
3. `uv run ruff check` on touched files

Completed in this session:

- `make up`
- `make format`
- `uv run --no-group infer mypy .`
- `uv run ruff check feedoscope/config.py feedoscope/relevance_embedding.py feedoscope/data_registry/data_registry.py feedoscope/llm_learn.py feedoscope/llm_infer.py feedoscope/eval_models.py`
- `uv run python -m compileall feedoscope custom_logging`

Functional validation:

1. run `make infer` with a cold cache
2. confirm rows are inserted into `relevance_embeddings`
3. run `make infer` again immediately
4. confirm the second run reuses the cache and produces identical scores
5. run `make train`
6. confirm training also reuses cached embeddings where available

Performance validation:

1. measure cold-cache `make infer` wall time
2. measure warm-cache `make infer` wall time on the same article set
3. log cache hit and miss counts during both runs

Correctness validation:

1. compare a cold-cache and warm-cache inference run on unchanged data
2. confirm score outputs are identical

## Step-by-Step Execution Checklist

1. Confirm the actual final embedding dimension from the real Gemma output.
2. Add the embedding-cache migration.
3. Add `RELEVANCE_PREP_VERSION` to config.
4. Add SQL files for cache lookup and upsert.
5. Add DB access helpers for reading and writing cached embeddings.
6. Refactor `relevance_embedding.py` to:
   prepare texts once, hash them, fetch hits, compute misses, upsert misses, and
   return embeddings in original order.
7. Switch `llm_learn.py` to the cache-aware path.
8. Switch `llm_infer.py` to the cache-aware path.
9. Switch `eval_models.py` relevance eval to the same cache-aware path.
10. Run format, lint, and type checks.
11. Run cold-cache and warm-cache validation.
12. Record the measured speedup and decide whether batch-size tuning is still the
    next priority.

Completed in this session:

- [completed] Confirm the current Gemma hidden size from the cached model on the PVC (`768`).
- [completed] Add the bytea-based cache migration.
- [completed] Add `RELEVANCE_PREP_VERSION` to config.
- [completed] Add SQL files for cache lookup and upsert.
- [completed] Add DB access helpers for reading and writing cached embeddings.
- [completed] Refactor `relevance_embedding.py` to use cache hits and misses.
- [completed] Switch `llm_learn.py` to the cache-aware path.
- [completed] Switch `llm_infer.py` to the cache-aware path.
- [completed] Switch `eval_models.py` relevance eval to the cache-aware path.
- [completed] Apply the embedding-cache migration with `make up`.
- [completed] Run format, lint, and type checks.
- [pending] Run cold-cache and warm-cache validation.
- [pending] Record measured speedup and compare with batch-size tuning.

## Open Questions / Assumptions

- Assumption: shared-by-config cache behavior is approved.
- Assumption: lazy population is preferred over a dedicated backfill job.
- Assumption: no similarity-search index is needed in the first implementation.
- Resolved: the currently cached Gemma model reports `hidden_size = 768`, which
  matches the current mean-pooling path. The bytea layout remains agnostic to the
  exact dimension as long as the Python deserializer keeps using `float32`.
- Open question: whether to add more durable cache metrics beyond logs. For the
  first version, logs should be enough.
- Recommendation: keep batch-size tuning as a separate follow-up after the cache
  lands, so the performance gain from the cache is easy to measure on its own.
