# Plan: Migrate Relevance Model from ModernBERT to Ettin Encoder

## Context

Ettin (jhu-clsp/ettin-encoder-150m) is a successor to ModernBERT released July 2025
by JHU-CLSP. It outperforms ModernBERT across all tasks (classification, retrieval,
code) while using fully open training data. Same architecture, same HuggingFace API,
drop-in replacement.

Blog post: https://huggingface.co/blog/ettin

## Scope

Ettin is used for **relevance classification only**. The urgency model stays on
ModernBERT for now.

## Changes

All changes are constant swaps -- no API or workflow modifications needed.

### Core files (functional changes)

- [x] `llm_learn.py`: MODEL_NAME -> `jhu-clsp/ettin-encoder-150m`
- [x] `llm_infer.py`: MODEL_NAME prefix -> `jhu-clsp-ettin-encoder-150m_512_2_epochs_16_batch_size`

### Urgency files (unchanged, kept on ModernBERT)

- `llm_learn_urgency.py`: MODEL_NAME stays `answerdotai/ModernBERT-base`
- `llm_infer_urgency.py`: URGENCY_MODEL_NAME stays `urgency-ModernBERT-base`

### Comment/docstring updates

- [x] `eval_models.py` -- production model prefix comment (relevance prefix updated)
- [x] `AGENTS.md` -- project description (mentions both Ettin and ModernBERT)
- [x] `llm_learn.py` -- blog link updated to Ettin

### Not changed (intentional)

- `saved_models/` -- historical artifact, left as-is
- No hyperparameter changes (lr, epochs, batch size, max_length)
- No dependency changes
- All urgency-related files, comments, and docstrings remain on ModernBERT

## Deployment notes

- Old relevance models in `models/answerdotai-ModernBERT-base_*` won't be found
  by the new prefix; first run will trigger a fresh relevance training
- Urgency models in `models/urgency-ModernBERT-base_*` are unaffected

## Status: COMPLETE
