Status: Completed 2026-04-18

# Plan: Next Embedding Experiment Wave

## Goal

Refresh the `pi_research` autoresearch files to run the next embedding-focused
experiment wave after the real-world validation runs. The new wave should answer
whether the current EmbeddingGemma setup can be improved through better prompting,
larger dense encoders, or a BGE-M3 hybrid dense+sparse feature path.

## Current State / Relevant Context

- The current loop files still point at an outdated control (`single_blob`, `C=4.0`) even
  though the latest kept winner is `EmbeddingGemma-300m @ 2048`, `title_head`,
  `C=5.0` on `pi_research` commit `ea778e6`.
- `autoresearch.sh` already supports dense embedding + logistic regression, model-aware
  pooling, prefixes, layer norm, and dimension truncation.
- It does not yet support prompt templates tailored to EmbeddingGemma’s model card,
  nor a BGE-M3 sparse/hybrid feature mode.
- Real-world validation suggested current `master` remains strong, so the next loop
  should be bounded and targeted rather than open-ended model hunting.

## Proposed Implementation

1. Update `program.md`, `autoresearch.md`, and `autoresearch.ideas.md` to:
   - treat `ea778e6` as the control
   - prioritize EmbeddingGemma prompt experiments first
   - then test larger dense models (`gte-large-en-v1.5`, `snowflake-arctic-embed-l-v2.0`)
   - then test `bge-m3` hybrid dense+sparse
2. Extend `autoresearch.sh` embedding mode with:
   - `EMBED_PROMPT_MODE` for prompt templates (`none`, `classification`, `document`)
   - `EMBED_FEATURE_MODE` (`dense`, `sparse`, `hybrid`)
   - `EMBED_SPARSE_HASH_DIM` for hashed lexical features
3. Add a BGE-M3 hybrid implementation using `FlagEmbedding` when sparse features are
   requested.
4. Keep the transformer harness unchanged; all new work stays in the experiment docs
   and `autoresearch.sh`.

## File-by-File Impact

- `plans/embedding-next-wave-plan.md`
  - This decision record.
- `program.md`
  - New current control and next-run queue.
- `autoresearch.md`
  - New resumed session instructions and candidate ordering.
- `autoresearch.ideas.md`
  - New experiment ideas with exact config knobs.
- `autoresearch.sh`
  - Prompt-mode support and BGE-M3 dense/sparse/hybrid execution.

## Risks and Edge Cases

- `FlagEmbedding` is not declared in `pyproject.toml`, so hybrid BGE-M3 runs may need
  `uv` installation before first use.
- Prompt templates can blur the meaning of `TEXT_PREP_MODE`, so the runner must log
  prompt mode explicitly.
- Sparse or hybrid features can increase CPU memory usage during logistic-regression
  fitting; hashed sparse features mitigate that risk.

## Validation / Testing

- Check shell syntax for `autoresearch.sh`.
- Compile the inline Python embedded in `autoresearch.sh`.
- Read the updated docs end-to-end for consistency with the new queue.

## Step-by-Step Execution Checklist

- [x] Create this plan file.
- [x] Update loop docs with the new control and experiment queue.
- [x] Extend `autoresearch.sh` with prompt modes and BGE-M3 sparse/hybrid features.
- [x] Validate shell and inline Python syntax.
- [x] Mark this plan complete with the final file list.

## Open Questions / Assumptions

- Assumption: the user wants the next loop to stay on `pi_research` and remain frozen-
  snapshot based, not switch to the real production validation path.

## Final File List

- `plans/embedding-next-wave-plan.md`
- `program.md`
- `autoresearch.md`
- `autoresearch.ideas.md`
- `autoresearch.sh`
