# Semantic Freshness Score Rollout Plan

**Status:** Superseded by [`three-label-freshness-redesign-plan.md`](three-label-freshness-redesign-plan.md) — 2026-08-01

> Historical record only. The former six-label `fresh-*`/`fresh-auto-*`
> rollout, prediction persistence, and freshness model-evaluation integration
> are no longer deployed.

## Brief

Replace urgency probability with semantic freshness's `expected_lifetime_days` when applying the final relevance-score decay. Keep urgency inference intact, but make the selected decay source explicit through a single environment flag so rollback is a one-value manifest change. Rename the existing freshness label tags to the shorter `fresh-*` names without losing their links to Miniflux entries.

## Current state / relevant context

- `feedoscope.main` runs semantic freshness in shadow mode, writes predictions and tags, then always derives final score decay from urgency probability.
- The successful bootstrap artifact contains five cumulative classifiers and inference is now writing `semantic_freshness_inference` rows.
- The existing tag names are `<horizon>-freshness` and `<horizon>-auto-freshness`; their `user_tags` IDs are attached to existing entries.
- User selected the tag names `fresh-<horizon>` and `fresh-auto-<horizon>`, semantic freshness as the default decay backend, and raw relevance score when a freshness prediction is unavailable.
- The inference CronJob is managed in `/home/djipey/informatique/infra/manifests/perso/miniflux/base/feedoscope-infer-job.yaml`.

## Proposed implementation

1. Add migration `000009_rename_semantic_freshness_tags` that renames the twelve existing `user_tags` rows in place:
   - reviewed: `<horizon>-freshness` -> `fresh-<horizon>`
   - automatic: `<horizon>-auto-freshness` -> `fresh-auto-<horizon>`
   Renaming preserves each tag ID and therefore preserves every existing `entry_user_tags` relationship. The down migration restores the current names.
2. Replace the old tag literals in all six freshness SQL files and `assign_semantic_freshness_auto_tags()` so tag creation, retrieval, auto-tag replacement, read promotion, training-label lookup, and conflict diagnostics all use the new names. Rewrite read promotion for the `fresh-auto-` to `fresh-` prefix mapping rather than the current suffix replacement.
3. Add `RELEVANCE_DECAY_BACKEND` in `feedoscope/config.py`, constrained to `semantic_freshness` or `urgency`, defaulting to `semantic_freshness`.
4. Make the final-score decay accept a concrete, finite, positive half-life in days. When the backend is `semantic_freshness`, pass the in-memory `expected_lifetime_days` produced for the same active article set; at that many elapsed days, the relevance score halves. When the backend is `urgency`, retain the existing probability-to-half-life interpolation exactly. Clamp future publication dates to zero elapsed days so score never increases.
5. If semantic freshness inference fails or an active article lacks a valid expected lifetime while that backend is selected, leave its relevance score unchanged and emit a warning. Do not silently fall back to urgency.
6. Set `RELEVANCE_DECAY_BACKEND=semantic_freshness` explicitly in the inference CronJob, so rollback is one manifest-value change to `urgency` followed by `kubectl apply -k`.
7. Update `docs/semantic-freshness.md` to describe the active score path, renamed tags, flag, raw-score fallback, and rollback command. Update the documentation index only if a new document is needed (not expected).

## File-by-file impact

### Feedoscope repository

- `db/migrations/000009_rename_semantic_freshness_tags.up.sql` — rename existing tag rows in place.
- `db/migrations/000009_rename_semantic_freshness_tags.down.sql` — restore existing names.
- `feedoscope/data_registry/sql/upsert_semantic_freshness_user_tags.sql` — create only `fresh-*` tags.
- `feedoscope/data_registry/sql/get_semantic_freshness_user_tags.sql` — read `fresh-*` tags.
- `feedoscope/data_registry/sql/set_semantic_freshness_auto_tag_for_entry.sql` — replace only `fresh-auto-*` tags.
- `feedoscope/data_registry/sql/promote_read_auto_freshness_tags.sql` — promote `fresh-auto-*` to `fresh-*` after an entry is read.
- `feedoscope/data_registry/sql/get_semantic_freshness_training.sql` — recognize reviewed `fresh-*` labels.
- `feedoscope/data_registry/sql/get_conflicting_semantic_freshness_labels.sql` — retain conflict diagnostics for reviewed `fresh-*` labels.
- `feedoscope/data_registry/data_registry.py` — build `fresh-auto-<horizon>` lookup keys.
- `feedoscope/config.py` — parse and validate `RELEVANCE_DECAY_BACKEND`.
- `feedoscope/main.py` — select urgency or expected-lifetime half-life for final score decay and log missing freshness predictions.
- `tests/test_semantic_freshness.py` (or a focused new `tests/test_main.py`) — cover renamed tag-key construction, both decay backends, and raw-score fallback.
- `docs/semantic-freshness.md` — update implemented behavior.

### Manifests repository

- `perso/miniflux/base/feedoscope-infer-job.yaml` — set the new feature flag to `semantic_freshness`.
- `perso/miniflux/overlays/miniflux/kustomization.yaml` — update Feedoscope's image tag after building the implementation.

## Risks and edge cases

- `expected_lifetime_days` is an expected useful lifetime, not a score. It will become the exponential-decay half-life: a score falls to 50% after that many days.
- A tag migration without the new image can let an old CronJob recreate the old tag names. Suspend the inference CronJob and wait for active work to finish, migrate, deploy the new image, then resume. Roll back in reverse order while suspended.
- The migration must first check whether manually-created `fresh-*` tags already exist. If none exist (expected), rename in place; otherwise stop for a user decision rather than silently merging tag IDs.
- Do not silently fall back to urgency in semantic mode; raw-score fallback is the requested observable degradation. Logs must identify affected article IDs.
- Keep the inference result map only after the model produced valid aligned article IDs and lifetimes. Tag-assignment failure must not discard valid model output; prediction failure leaves every semantic-mode score raw.
- Keep urgency inference running. It may still supply stored urgency predictions and dashboard data; only final relevance decay changes.
- Existing `semantic_freshness_inference` rows and their model keys need no migration.

## Validation / testing

1. Run focused pytest coverage for half-life decay math, urgency compatibility, expected-lifetime selection by article ID, and missing/invalid freshness raw-score behavior.
2. Run `make lint` and `make format` in Feedoscope.
3. Apply migration in a safe environment and verify existing tag IDs remain attached to entries while names become `fresh-*`.
4. Build and push the Feedoscope image; update the manifest image tag and validate with `kubectl kustomize` and server dry-run.
5. Suspend `feedoscope-infer-job`, wait for no active Job, apply migration `000009`, deploy the overlay with the new image, then resume it.
6. Trigger one inference Job and verify:
   - final score changes follow expected-lifetime half-lives;
   - `semantic_freshness_inference` still records predictions;
   - only `fresh-*` tags are assigned;
   - changing `RELEVANCE_DECAY_BACKEND=urgency` restores the old score calculation without rebuilding.

## Step-by-step execution checklist

- [x] Preflight found no `fresh-*` user tags, then applied the collision-safe tag-renaming migration.
- [x] Updated all freshness tag SQL and Python lookup keys.
- [x] Added the validated decay-backend setting and switched score-decay data flow.
- [x] Added focused tests; formatter, 14 tests, and mypy pass.
- [x] Updated semantic freshness documentation.
- [x] Built/pushed Feedoscope image `88a577e` and updated the manifest image tag plus feature flag.
- [x] Suspended inference, applied migration `000009`, deployed the manifest, and resumed inference.
- [x] Completed `feedoscope-infer-freshness-score-bootstrap`: it processed 8,774 articles, wrote freshness predictions, applied `fresh-*` tags, and wrote final scores.
- [x] Verified urgency remains selectable by `RELEVANCE_DECAY_BACKEND=urgency`; the focused test covers the backend switch.

## Open questions / assumptions

- `fresh-<horizon>` and `fresh-auto-<horizon>` are the exact desired tag names.
- Semantic freshness stays the manifest default; `urgency` remains the rollback option.
- A missing or failed freshness prediction keeps raw relevance score rather than using urgency.
- The migration and new image were deployed while inference was suspended, preventing old tag names from being recreated.
