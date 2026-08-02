# Automatic super-important tag plan

**Status:** Validation and rollout in progress — 2026-08-01

## 1. Brief

When relevance inference gives an article a super-important probability above 50%, Feedoscope will attach the visible Miniflux user tag `important-auto`. For every article processed again, Feedoscope will remove that tag when the latest probability is at or below 50%, so the tag remains an accurate view of the current model decision.

## 2. Current state / relevant context

- Relevance inference already returns aligned article IDs and raw super-important probabilities in `RelevanceInferenceResults`.
- `register_super_important_inference()` is the shared write path used by both full and standalone relevance inference.
- Miniflux stores visible tags in `user_tags` and `entry_user_tags`; `entries.tags` is not the active user-tag relationship.
- There is one Miniflux user (`admin`, ID 1), but the SQL can derive each entry's user instead of hard-coding it.
- The current model has 29 stored predictions above 50%; 28 are unread and one is read.
- The user chose synchronized lifecycle, immediate backfill, and push plus production deployment.

## 3. Proposed implementation

1. Add one idempotent SQL query that receives all processed article IDs and the subset above 50%.
2. In that query, create `important-auto` for each affected Miniflux user if missing, remove the tag from every processed article, then add it back only to the above-50% subset. This updates only the automation-owned tag and preserves every other user tag.
3. Run the tag sync in the same database transaction as the cached super-important prediction upsert by extending `register_super_important_inference()`.
4. Log how many processed articles are currently above the threshold.
5. Backfill the 29 current-model matches directly from stored predictions after deployment; no model rerun is needed.

The threshold comparison will be strictly `> 0.5`, matching ranker behavior.

## 4. File-by-file impact

- `feedoscope/data_registry/data_registry.py`: derive above-threshold IDs and execute tag synchronization after prediction upsert.
- `feedoscope/data_registry/sql/sync_important_auto_tags.sql`: create/synchronize the Miniflux user tag for processed entries.
- `tests/test_relevance_ranker.py`: verify the raw predictions are still upserted and only probabilities above 50% are passed to tag sync.
- `docs/super-important-ranker.md`: document the visible tag and its synchronized lifecycle after implementation.
- `/home/djipey/informatique/infra/manifests/perso/miniflux/overlays/miniflux/kustomization.yaml`: point CronJobs to the new commit-tagged image.

## 5. Risks and edge cases

- Exactly 50% must not receive the tag; the code uses strict greater-than comparison.
- Re-inference can change a classification; removing then re-adding within the transaction prevents stale `important-auto` tags.
- A user manually removing `important-auto` from an above-threshold article will be overridden on its next inference because this tag is explicitly automation-owned.
- An inference failure before transaction commit leaves both cached predictions and tags unchanged.
- Existing unrelated Miniflux tags are never modified.

## 6. Validation / testing

- Focused test with probabilities below, exactly at, and above 50%.
- Full pytest suite, Ruff, mypy, Black, isort, and `git diff --check`.
- Build and push the commit-tagged image; deploy it to all Feedoscope CronJobs.
- Backfill from the active model and verify exactly 29 `important-auto` relationships, including 28 unread entries.
- Verify the registry and live CronJob reference the new image.

## 7. Step-by-step execution checklist

- [x] Add atomic tag synchronization SQL.
- [x] Extend the shared prediction registration transaction.
- [x] Add focused threshold and parameter tests.
- [x] Run the full validation suite.
- [x] Update implemented-behavior documentation.
- [ ] Commit, build, and push the new image.
- [ ] Deploy the image through the infrastructure repository.
- [ ] Backfill current above-threshold predictions and verify counts.
- [ ] Mark this plan completed with rollout evidence.

## 8. Open questions / assumptions

- `important-auto` is owned by Feedoscope; manual edits to that specific tag may be replaced during later inference.
- Only articles processed in a run are synchronized during that run. Other articles retain their last model-derived tag until reprocessed.
- The direct backfill uses only predictions from the currently active model artifact.
