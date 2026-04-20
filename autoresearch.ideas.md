# Urgency Autoresearch Ideas

## Near-Term Queue

- confirm whether `title_head` beats `single_blob` for urgency like it did for relevance
- compare transformer fine-tuning against frozen embeddings quickly before doing any
  deeper hyperparameter work
- if an embedding model is close on AP, tune `LINEAR_C` before exploring more model
  families

## Current Observations

- bootstrap control is now established on the frozen read-tagged snapshot: `ModernBERT-base` transformer, `title_head`, `max_length=512`, `TRAIN_BALANCE_MODE=full` reached `average_precision=0.8707539292`
- the mandated `single_blob` comparison is now complete and lost to `title_head` on AP (`0.8695675423`), log loss, recall, and runtime, so `title_head` is the preferred urgency text prep for the current transformer baseline
- `distilbert-base-uncased` reduced peak VRAM to about `2.01 GB` and cut training time roughly in half, but its AP (`0.8651599302`) still trailed the ModernBERT control, so efficiency alone is not enough to beat the current baseline
- `google/embeddinggemma-300m` beat both transformer baselines and remains the strongest speed/quality tradeoff so far
- `BAAI/bge-m3` did not improve on embeddinggemma and was by far the slowest run at that point, so stronger multilingual embeddings are not automatically better on this English-heavy snapshot
- `Alibaba-NLP/gte-large-en-v1.5` is the best kept run with `average_precision=0.8744714509`; it slightly beats embeddinggemma on AP and calibration but is much slower
- moving `gte-large-en-v1.5` away from `LINEAR_C=1.0` in either direction (`0.5` or `2.0`) hurt AP, so its logistic regularization already looks near-optimal for the current snapshot
- increasing `LINEAR_C` on `embeddinggemma-300m` improved log loss / brier / recall / F1 but still hurt AP, which suggests ranking quality and thresholded quality are diverging on these embedding-linear runs
- the next promising branch is to complete a symmetric quick check with `embeddinggemma-300m LINEAR_C=0.5`, then decide whether to stop tuning `LINEAR_C` and explore lighter `gte` variants instead

## If The First Queue Underperforms

- try `ModernBERT-large` only if `ModernBERT-base` is already competitive
- try `gte-base-en-v1.5` if `gte-large-en-v1.5` is strong but too slow
- try `EMBED_TRUNCATE_DIM` only if VRAM or runtime become a real constraint

## Things To Avoid Early

- changing multiple knobs at once before the control is established
- mixing unread articles back into the supervision set
- wiring the winner into production before the loop stabilizes
