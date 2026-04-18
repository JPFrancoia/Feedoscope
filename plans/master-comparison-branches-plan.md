Status: Completed 2026-04-18

# Plan: Master Comparison Branches

## Goal

Create two comparable validation branches from `master` so the user can run the
current production-style training flow on the other machine and compare outcomes:

1. a branch that preserves `master`'s balanced downsampling behavior but makes
   `VALIDATION_SIZE` env-driven
2. a branch that keeps everything else close to `master` but uses the full
   training rows and also makes `VALIDATION_SIZE` env-driven

## Current State / Relevant Context

- `master` relevance training hardcodes `VALIDATION_SIZE = 0` in
  `feedoscope/llm_learn.py`, so `VALIDATION_SIZE=100 make train` does not actually
  change relevance validation behavior.
- `master` also balances the relevance training set by trimming good and bad rows
  to the same `min_count`.
- The user wants the smallest realistic comparison against `master`, so these new
  branches should stay close to `master` and avoid unrelated loop-driven changes.

## Proposed Implementation

- Branch A: start from `master`, change only the relevance validation size wiring.
- Branch B: start from `master`, change the relevance validation size wiring and
  remove the balanced downsampling block so all fetched training rows are used.
- Commit each branch locally, but do not push.

## File-by-File Impact

- `plans/master-comparison-branches-plan.md`
  - Local decision record.
- Branch A expected edits:
  - `feedoscope/llm_learn.py`
- Branch B expected edits:
  - `feedoscope/llm_learn.py`

## Risks and Edge Cases

- The current `pi_research` worktree has unrelated untracked files. The branch work
  should happen in separate worktrees so those files are not affected.
- Branch naming must avoid collisions with the earlier validation branches.

## Validation / Testing

- Check Python syntax on each branch after editing.
- Verify branch heads and cleanliness after committing.

## Step-by-Step Execution Checklist

- [x] Create this plan file.
- [x] Create balanced branch from `master` and commit the validation-size fix.
- [x] Create full-data branch from `master` and commit the validation-size + full-row change.
- [x] Verify branch state and report branch names / commits.

## Open Questions / Assumptions

- Assumption: the user wants normal committed branches that can be pushed and pulled,
  not uncommitted local-only branches.

## Final Branches

- `validate/master-balanced-validation`
- `validate/master-full-validation`
