Status: Completed 2026-04-18

# Plan: Relevance Next Loop Refresh

## Goal

Refresh the autonomous relevance-research instructions and harness defaults so the
next loop starts from the current best embedding configuration instead of the stale
bootstrap-era plan.

## Current State / Relevant Context

- The repository already contains a working frozen-snapshot experiment harness.
- `program.md` and `autoresearch.md` still describe the original four-run bootstrap
  loop and say the session is not yet bootstrapped.
- Recent git history shows the current best configuration is
  `google/embeddinggemma-300m @ 2048`, `single_blob`, embedding + logistic
  regression, `TRAIN_BALANCE_MODE=full`, `C=4.0` with AP `0.98405`.
- `feedoscope/relevance_experiments/harness.py` now supports
  `--train-balance-mode`, but still defaults to `balanced`.
- `autoresearch.sh` supports an embedding-linear path but does not yet expose
  model-aware embedding options like pooling choice, prefixes, or optional layer
  norm / dimension truncation.

## Proposed Implementation

Update the loop artifacts and harness defaults:

1. Rewrite `program.md` as the authoritative brief for the next loop.
2. Rewrite `autoresearch.md` as a resumed session document with the current best,
   recent findings, and explicit next candidates.
3. Rewrite `autoresearch.ideas.md` with the next embedding-model queue.
4. Change `autoresearch.sh` defaults so new runs use the full frozen training split
   and expose model-aware embedding controls.
5. Change `feedoscope/relevance_experiments/harness.py` so transformer experiments
   also default to `train_balance_mode=full`.

## File-by-File Impact

- `plans/relevance-next-loop-refresh-plan.md`
  - Decision record for this refresh.
- `program.md`
  - Updated Karpathy-loop brief for the next campaign.
- `autoresearch.md`
  - Updated `pi-autoresearch` session brief.
- `autoresearch.ideas.md`
  - New embedding-focused experiment queue.
- `autoresearch.sh`
  - New embedding-specific knobs and default full-data regime.
- `feedoscope/relevance_experiments/harness.py`
  - Default transformer training regime changed from balanced to full.

## Risks and Edge Cases

- Changing the default training balance mode changes future behavior; the docs and
  harness must stay aligned.
- Model-aware embedding handling can bias comparisons if not logged, so every new
  embedding knob must be written to results and run names.
- Nomic and other embedding models have model-specific expectations; the loop should
  still run a control rerun of the current winner after harness changes.

## Validation / Testing

- Read all edited files end-to-end for consistency.
- Verify `autoresearch.sh` still parses and exposes the new env vars.
- Verify the transformer harness default changed to `full` and still accepts the
  explicit `balanced` override.
- Review the final git diff / status.

## Step-by-Step Execution Checklist

- [x] Create this plan file.
- [x] Rewrite `program.md`.
- [x] Rewrite `autoresearch.md`.
- [x] Rewrite `autoresearch.ideas.md`.
- [x] Update `autoresearch.sh`.
- [x] Update `feedoscope/relevance_experiments/harness.py`.
- [x] Review the edits and provide the exact next-loop command.

## Open Questions / Assumptions

- Assumption: The user wants the next loop to continue from the current frozen
  snapshot and branch, not start a fresh campaign.

## Final File List

- `plans/relevance-next-loop-refresh-plan.md`
- `program.md`
- `autoresearch.md`
- `autoresearch.ideas.md`
- `autoresearch.sh`
- `feedoscope/relevance_experiments/harness.py`
