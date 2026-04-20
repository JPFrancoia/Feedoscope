# Urgency Embedding Backend

The urgency pipeline now uses the same frozen embedding-plus-logistic-regression
approach as relevance.

## Summary

- Encoder: `google/embeddinggemma-300m` via the shared embedding backend
- Classifier: `LogisticRegression`
- Training labels: only read articles tagged `0-urgency` or `1-urgency`
- Embedding cache: shared Postgres `relevance_embeddings` table
- Score cache: `urgency_inference`, keyed by `(article_id, model_key)`

## Training Data

Urgency training reads from:

- `feedoscope/data_registry/sql/get_read_articles_with_urgency_tags.sql`

Only these rows are used:

- `entries.status = 'read'`
- articles tagged `0-urgency` or `1-urgency`

The tag value becomes the binary label:

- `0-urgency` -> evergreen
- `1-urgency` -> urgent

Unread tagged articles are not used for training.

## Shared Embedding Cache

Urgency intentionally shares the same embedding configuration as relevance:

- `RELEVANCE_MODEL_NAME`
- `RELEVANCE_MAX_LENGTH`
- `RELEVANCE_TEXT_PREP_MODE`
- `RELEVANCE_PREP_VERSION`
- `RELEVANCE_ENCODER_BATCH_SIZE`

Because of that, urgency can reuse cached vectors already written by the
relevance pipeline for the same article and text-preparation settings.

The urgency-specific knob is only:

- `URGENCY_LINEAR_C`

which affects the logistic-regression head but not the embedding cache key.

## Model Artifacts

Urgency artifacts are saved under `models/` and contain:

- `classifier.joblib`
- `metadata.json`

The metadata records the shared encoder config, the urgency classifier `C`, and
the derived `model_key`.

## Urgency Score Cache

Urgency probabilities are cached in `urgency_inference`.

The table is versioned by `model_key` so scores from different urgency backends
or configurations do not collide.

The current `model_key` includes:

- backend family
- encoder model name
- max length
- text prep mode
- prep version
- urgency classifier `C`

## Refresh Behavior

Urgency refresh now mirrors relevance refresh exactly.

`main.py` and `llm_infer_urgency.py` both refresh urgency on the same unread
article selection used for relevance scoring:

- unread articles from the recent lookback window
- a random sample of older unread articles

The refreshed urgency probabilities are then used immediately for time-decay.

This replaces the old behavior where urgency was only inferred for cache misses.
