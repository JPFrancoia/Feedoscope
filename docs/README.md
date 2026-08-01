# Docs

This directory contains durable documentation for implemented Feedoscope
systems.

## Index

- `relevance-embedding-cache.md`: current relevance embedding architecture,
  shared Gemma model cache, and Postgres bytea embedding cache.
- `urgency-embedding-backend.md`: urgency training/inference on shared Gemma
  embeddings, read-tagged labels, and model-keyed urgency score caching.
- `model-eval-history.md`: weekly relevance and urgency evaluation persistence
  to JSON history and Miniflux-owned PostgreSQL storage.
- `semantic-freshness.md`: three manual useful-lifetime labels, two-head
  training, and in-memory final-score decay.
