# Thresholded super-important bonus rollout plan

**Status:** Age-block rollout in progress — 2026-08-01

## 1. Brief

Change the super-important ranking signal so predictions at or below the classifier's 50% decision boundary have no effect. Above 50%, the signal will rise smoothly to the existing full bonus at 100%, then the existing semantic-freshness decay will continue unchanged. Build and deploy this narrowly scoped scoring change immediately after local validation.

## 2. Current state / relevant context

- Production runs the two-head ranker with `SUPER_IMPORTANT_BONUS=0.5`.
- The super-important head is an unweighted binary logistic-regression classifier trained on explicitly preferred versus ordinary read articles.
- Current combination uses every raw preference probability:

  ```text
  rank = P(good) × (1 + bonus × P(preference)) / (1 + bonus)
  ```

- The last controlled run observed a maximum preference score around 0.65, so a 50% threshold keeps only the strongest current predictions active; an 80% threshold would disable the bonus entirely.
- Existing unrelated AI-metrics changes are uncommitted in the current Feedoscope checkout. Implementation will use a clean dedicated worktree from `a363e45` and will not touch those changes.

## 3. Proposed implementation

Use a thresholded linear ramp:

```text
preference signal = clip((P(preference) - 0.5) / 0.5, 0, 1)
rank = P(good) × (1 + bonus × preference signal) / (1 + bonus)
```

This gives no preference-based reordering through 50%, 20% of the maximum signal at 60%, 60% at 80%, and the full signal at 100%. Keep the existing global denominator, bonus strength, raw probability persistence, model artifact, training process, and freshness decay unchanged.

The 50% threshold will be a constant beside the shared combiner rather than a new environment variable: this is the classifier decision boundary, not a deployment tuning surface.

After the first 365-day refresh exhausted the 2 GiB PostgreSQL volume during the final all-article score update, write `entries.score` in fixed batches of 1,000 rows and commit each batch before starting the next. Earlier committed batches remain valid if a later batch fails, and PostgreSQL can checkpoint/recycle WAL between transactions. Keep the shared `update_scores` interface so both standalone and full inference use the same bounded write path.

## 4. File-by-file impact

- `feedoscope/relevance_embedding.py` — transform the raw preference probability through the 50% ramp inside `combine_probabilities`.
- `feedoscope/data_registry/data_registry.py` — split final article-score updates into 1,000-row transactions.
- `tests/test_relevance_ranker.py` — check the threshold/ramp behavior and prove score batches are bounded and committed independently.
- `docs/super-important-ranker.md` — document the implemented thresholded formula after deployment.
- `plans/thresholded-super-important-bonus-plan.md` — record validation and rollout outcome.
- `/home/djipey/informatique/infra/manifests/perso/miniflux/overlays/miniflux/kustomization.yaml` — update only the Feedoscope image tag after the image is built.

No model retraining, schema migration, new dependency, or new configuration is required.

## 5. Risks and edge cases

- Logistic-regression outputs are not guaranteed to be calibrated probabilities; 50% is used as the classifier's positive decision boundary, not a claim of exact real-world certainty.
- A discontinuous hard gate would cause a large ranking jump at 50%; the smooth ramp avoids that.
- Old articles may still rank highly when base relevance and semantic-freshness lifetime are high. If that remains after rollout, inspect freshness rather than increasing this threshold blindly.
- Rollout verification found a separate stale-score problem: articles beyond the 40-day full-refresh window are sampled randomly, so current first-page articles can retain pre-rollout scores and have no current super-important prediction. A one-time 60-day refresh moved the stale first page from ages 40–49 to ages 60–69, confirming the threshold change itself is live but cannot update articles the sampler does not select.
- Changing the combination formula changes weekly benchmark values. The evaluation code already calls the shared combiner, so it will automatically evaluate the deployed behavior.

## 6. Validation / testing

- Focused test proves scores at 20%, 30%, and 50% receive identical preference signal; 60% receives 20% of the maximum; 100% receives the full existing bonus.
- Run `uv run pytest tests/test_relevance_ranker.py -q`.
- Run the full test suite, Ruff, and mypy if the focused check passes.
- Build and push the commit-tagged Docker image, including the already completed local Super-important AI-metrics changes and inference startup logging.
- Update and apply the production manifest, retry the 365-day refresh, and verify bounded score-batch progress, successful Job completion, database free space, and current top scores.

## 7. Step-by-step execution checklist

- [x] Create a clean dedicated Feedoscope worktree from `a363e45` at `/home/djipey/informatique/python/feedoscope.thresholded-super-important-bonus`.
- [x] Implement the shared 50% ramp and focused test.
- [x] Run formatting, focused/full tests, Ruff, and mypy. Result: 42 tests passed; formatting, Ruff, and mypy passed; independent Python review approved with no findings.
- [x] Commit the initial threshold change as `ff3c181` and build/push its image.
- [x] Update, commit, and apply production image `ff3c181` in infrastructure commit `5b37f73`.
- [x] Trigger controlled inference and verify live completion for 8,737 articles, then a 60-day refresh for 11,384 articles. Verification exposed separate stale old scores beyond the full-refresh window.
- [x] Batch final article-score writes in 1,000-row committed transactions and merge the pending local AI-metrics/startup-log changes with the threshold change. Validation: 44 tests, Ruff, mypy, Black/isort, and diff check passed.
- [x] Commit the combined local changes, build/push image `3a5dc33`, and update production.
- [x] Keep the already-running 365-day `3a5dc33` job alive while improving the next image. It completed all 55,003 articles in 44 minutes and committed 56 score batches without filling PostgreSQL.
- [x] Add 30-second/1,000-article progress logging and non-overlapping `--min-age-days`/`--max-age-days` controlled inference, validated with 46 tests, Ruff, mypy, Black/isort, and CLI checks.
- [x] Delete 13 superseded Feedoscope manifests from the remote registry and run offline garbage collection, reducing registry storage from 21.2 GiB to 4.6 GiB while preserving the running image.
- [x] Build/push image `6475d27` and deploy it to every Feedoscope CronJob.
- [x] Verify image `6475d27` through the next scheduled 8,715-article inference: text preparation logged every 10–30 seconds, score writes committed in nine batches, and PostgreSQL remained at 40% usage.
- [x] Clear 904 stale positive scores from unread downvoted articles immediately; add startup cleanup so future downvotes cannot retain old rank scores after inference runs.
- [ ] Run one controlled age block on the final image to verify the new age-range path; the completed 365-day refresh makes rerunning every block unnecessary now.
- [ ] Mark this plan completed with final rollout evidence.

## 8. Open questions / assumptions

- Confirmed by the user: use 50% as the no-bonus boundary and ramp smoothly above it.
- Assumption: retain `SUPER_IMPORTANT_BONUS=0.5`; only the input signal changes.
- Assumption: deploy without retraining because classifier outputs and artifact format are unchanged.
- No open implementation decision remains.
