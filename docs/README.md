# Docs

This directory contains durable documentation for implemented Feedoscope
systems.

## Index

- `relevance-embedding-cache.md`: current relevance embedding architecture,
  shared Gemma model cache, and Postgres bytea embedding cache.
- `urgency-embedding-backend.md`: urgency training/inference on shared Gemma
  embeddings, read-tagged labels, and model-keyed urgency score caching.
- `model-eval-history.md`: weekly Relevance, Super-important, Urgency, and
  Freshness evaluation, Miniflux-owned PostgreSQL storage, and AI Metrics
  display behavior.
- `semantic-freshness.md`: three manual useful-lifetime labels, two-head
  training, chronological evaluation, and in-memory final-score decay.
- `super-important-ranker.md`: two-head relevance ranking, model-keyed
  explicit-preference probability storage, weekly fixed-bonus evaluation
  history, artifact compatibility, and the rolling rollout gate.
