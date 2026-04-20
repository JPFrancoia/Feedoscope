# Urgency Autoresearch Ideas

## Near-Term Queue

- confirm whether `title_head` beats `single_blob` for urgency like it did for relevance
- compare transformer fine-tuning against frozen embeddings quickly before doing any
  deeper hyperparameter work
- if an embedding model is close on AP, tune `LINEAR_C` before exploring more model
  families

## If The First Queue Underperforms

- try `ModernBERT-large` only if `ModernBERT-base` is already competitive
- try `gte-base-en-v1.5` if `gte-large-en-v1.5` is strong but too slow
- try `EMBED_TRUNCATE_DIM` only if VRAM or runtime become a real constraint

## Things To Avoid Early

- changing multiple knobs at once before the control is established
- mixing unread articles back into the supervision set
- wiring the winner into production before the loop stabilizes
