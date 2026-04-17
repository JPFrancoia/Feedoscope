Status: Completed 2026-04-17

# Plan: Relevance Autoresearch Loop Setup

## Goal

Add the instruction files needed to run an autonomous relevance-model experiment loop
from another machine. The loop will optimize relevance evaluation on a fixed dataset
snapshot while respecting strict database, git, hardware, and environment constraints.

## Current State / Relevant Context

- The repository already contains the relevance training/eval code paths in
  `feedoscope/llm_learn.py`, `feedoscope/llm_infer.py`, `feedoscope/utils.py`, and
  `feedoscope/eval_models.py`.
- There is no existing root-level `program.md` or `autoresearch.md`.
- The user will run the loop on another machine using `pi-coding-agent` with the
  `pi-autoresearch` skill.
- The user wants a Karpathy-loop style brief, but the target agent ecosystem also
  benefits from a `pi-autoresearch`-compatible `autoresearch.md`.
- Key constraints from the user:
  - Export the dataset from PostgreSQL first and freeze it locally as Parquet.
  - Database access must be strictly read-only.
  - The machine has a 12 GB GPU.
  - The loop runs on a dedicated branch, must not create branches, must not touch
    `main`/`master`, and must not push.
  - Python commands must run inside the `uv`-managed environment.

## Proposed Implementation

Created two root files:

1. `program.md`
   - Full Karpathy-loop brief.
   - Includes objective, first-run order, snapshot requirements, metrics,
     git/database rules, and hardware constraints.

2. `autoresearch.md`
   - `pi-autoresearch`-friendly session document.
   - Mirrors the same constraints while using the structure expected by the tool.

Do not create an `autoresearch.sh` yet because the experiment harness does not exist
in this repository today; the loop agent on the other machine should create the
benchmark script together with the harness it decides to build.

## File-by-File Impact

- `plans/relevance-autoresearch-loop-plan.md`
  - Decision record for this setup task.
- `program.md`
  - Authoritative Karpathy-loop instruction brief.
- `autoresearch.md`
  - `pi-autoresearch` session brief for resume/startup behavior.

## Risks and Edge Cases

- If `program.md` and `autoresearch.md` diverge, the remote agent may follow the
  wrong rule set. The files should therefore be intentionally aligned.
- Parquet support is not currently declared in `pyproject.toml`; the instructions
  must explicitly allow installing `pyarrow` or `fastparquet` with `uv`.
- The existing repo code evaluates on live DB data. The loop brief must make it
  explicit that the experiment campaign uses a frozen local snapshot instead.

## Validation / Testing

- Read both generated files end-to-end.
- Confirm they encode all user-provided constraints.
- Confirm they name the first four required experiments in the intended order.

## Step-by-Step Execution Checklist

- [x] Create this plan file.
- [x] Write `program.md`.
- [x] Write `autoresearch.md`.
- [x] Review both files for consistency.
- [x] Mark this plan completed with the final file list.

## Open Questions / Assumptions

- Assumption: The user only needs the instruction files now; the remote agent will
  create the actual experiment harness and `autoresearch.sh` on the other machine.

## Final File List

- `plans/relevance-autoresearch-loop-plan.md`
- `program.md`
- `autoresearch.md`
