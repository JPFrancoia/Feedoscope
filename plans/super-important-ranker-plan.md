# Dedicated super-important ranker plan

**Status:** Rolling-window bonus tuning in progress after the one-time confirmation failed — 2026-08-01

## 1. Brief

Feedoscope currently learns whether an article is generally worth reading; starred or upvoted articles only receive a larger loss weight during that training. Add a second lightweight classifier that directly predicts the existing explicit-preference signal, persist that probability by model version, then use it to reorder the current relevance score before semantic-freshness decay. This lets the system prioritize articles like the ones you later call super-important without adding another encoder, label workflow, or scheduler.

## 2. Current state / relevant context

- The active task branch is `super_important` at `c1d3fd1`; before the rolling-window edit, the full suite passes (42 tests).
- Current relevance labels are read, non-downvoted entries as positive and `vote=-1` entries as negative. `vote=1 OR starred` currently only produces `EXCELLENT_WEIGHT` during fitting.
- Production currently sets `EXCELLENT_WEIGHT=20` in the external Feedoscope training CronJob, although the code default is 3.
- The current three-label semantic-freshness model supplies the half-life that decays the final relevance score. It does not write prediction tags or prediction rows.
- The shared EmbeddingGemma vectors are already cached in PostgreSQL by prepared article text, so another logistic-regression head needs no encoder training or new cache.
- Confirmed after the initial plan: preserve each super-important probability so it can be inspected and later drive a dedicated queue.
- Live label audit from the current three-year window: 6,310 read/good entries, 1,352 downvoted entries, and 186 starred-or-upvoted entries (134 upvoted, 73 starred, 21 both). No downvoted entry is starred.

### Label contract

This first version preserves the existing user behavior rather than creating a new workflow:

```text
super-important positive = read AND (vote = 1 OR starred)
ordinary-read negative   = read AND vote = 0 AND NOT starred
```

Downvoted and unread entries are excluded from the second head. The resulting score means “resembles an article that received an explicit preference after being read,” not an objective measure of importance or a calibrated probability over all unread articles.

## 3. Proposed implementation

### Two heads, one embedding pass

Train two `LogisticRegression` heads from the same cached embedding matrix:

1. **Relevance head:** read/good vs. downvoted, with unit sample weights. This replaces the current excellent-weighted objective so the explicit-preference signal is not counted twice.
2. **Super-important head:** explicit-preference positive vs. ordinary-read negative, using only the read/good rows.

At inference, calculate the raw ranking score in floating point:

```text
raw rank = P(good) × P(explicit preference | read)
final rank = existing semantic-freshness decay(raw rank)
```

Round only when the existing database-write path needs an integer. `feedoscope.main` writes the super-important probability before it decays and writes the final combined score.

### Inference storage

Create a Miniflux-owned `super_important_inference` table, mirroring the proven urgency-score cache:

```text
article_id                  → entry being scored
model_key                   → two-head model/configuration identity
super_important_score       → unrounded probability in [0, 1]
last_updated                → inference time
```

`(article_id, model_key)` is the primary key; `article_id` references `entries(id)` with cascade delete. Each refresh upserts the current model's probability. This records the importance-head output only; `entries.score` remains the final combined-and-decayed integer used by the existing UI.

Feedoscope owns the migration in `db/migrations/000007_super_important_inference.{up,down}.sql`, consistent with its existing urgency-inference and embedding-cache tables. Feedoscope adds one SQL upsert and registry function, returns importance probabilities alongside the combined result, and calls that function from both standalone relevance inference and the full pipeline.

### Artifacts and compatibility

- Give the new relevance artifact family a `two_head` prefix that includes model name, maximum length, text-preparation mode, preparation version, and linear C.
- Save both heads and metadata in one new relevance artifact format. Metadata records artifact version, encoder-cache configuration, training counts for relevance and super-important classes, and the label contract.
- Validate metadata on load and reject one-head artifacts. This guarantees inference cannot silently load an older weighted artifact.
- Keep urgency’s standalone generic classifier artifact untouched.

### Training and inference

- Refactor the relevance-only helper in `feedoscope/relevance_embedding.py` into small reusable operations for fitting and predicting the two heads; do not duplicate embedding/cache code.
- Update `feedoscope/llm_learn.py` to encode the training rows once, fit both heads, log the four class counts, and save the versioned artifact.
- Update `feedoscope/llm_infer.py` to load both heads, score the same active articles once, multiply the two probabilities, and return the existing `RelevanceInferenceResults` shape.
- Retire `EXCELLENT_WEIGHT` from Feedoscope config and remove it from the external training CronJob because it no longer represents the active model.

### Evaluation before rollout

Extend `feedoscope/eval_models.py` with a separate super-important evaluation that uses the same two-head scorer but does not change the existing `model_evals` schema.

Implementation outcome: the benchmark uses the newest 20% of mature labels as the chronological holdout, requires at least 10 examples for each required class, and logs the exact holdout IDs. The previous production weight of 20 is retained only as the offline baseline. The preference head itself uses natural class prevalence, and inference ignores incomplete artifact directories left by interrupted training.

- Split read labels chronologically, reserving a fixed later holdout and excluding the most recent 40 days so unread/read outcomes have time to settle.
- Require enough explicit-preference examples in both partitions; otherwise skip with a clear log message rather than reporting unstable results.
- Compare the current weighted one-head baseline, unweighted relevance alone, and the two-head ranker on the same fixed holdout.
- Report positive prevalence, average precision for explicit preference among read entries, precision and recall at top 10/25/50, graded NDCG@10/25/50 (`downvoted=0`, ordinary read=1`, explicit preference=2`), and normal good-vs-bad average precision as a guardrail.
- Report counts for upvoted-only, starred-only, and both-positive subgroups. Treat this as an offline benchmark; do not apply current age/freshness decay to historical articles.

A successful benchmark is required before changing the production image or manifest. The first live run showed that direct multiplication improved explicit-preference AP from 0.0821 to 0.3164 and top-50 recall from 0 to 0.1525, but reduced general relevance AP from 0.9615 to 0.8521, so rollout stopped.

### Deterministic bonus tuning

Replace direct multiplication with a bounded bonus:

```text
rank = P(good) × (1 + bonus × P(explicit preference | read)) / (1 + bonus)
```

The denominator keeps the score in `[0, 1]` and does not change ordering. Tune only the single nonnegative `bonus` value; no autoresearch loop is needed.

- Use a fixed chronological train/validation/test split over mature labels.
- Evaluate a small fixed bonus grid on validation probabilities from one model fit.
- Keep only candidates whose general relevance AP is within 0.01 of the weighted baseline and whose explicit-preference AP and at least one top-K recall improve.
- Choose the highest explicit-preference AP; break ties toward the smaller bonus.
- Evaluate that one selected bonus once on the newest test partition. Deploy only if the same guardrail and improvement requirements pass there.
- Freeze the selected bonus in the inference manifest. Do not retune automatically on each training run; rerun after a material model/label-policy change or a substantial increase in explicit-preference labels.

The validation slice selected bonus `3.0`, but its one-time confirmation failed: explicit-preference AP improved from `0.0821` to `0.3176` while relevance AP fell from `0.9615` to `0.9412`, exceeding the `0.01` loss limit. Production remained unchanged. The large shift in explicit-preference prevalence between validation (`2.4%`) and confirmation (`11.1%`) makes a single split unstable.

### Rolling-window replacement

Use the already observed periods as rolling validation rather than trying a second bonus against the failed confirmation:

1. Fit on the oldest 60% and evaluate on the next 20%.
2. Refit on the oldest 80% and evaluate on the newest 20%.
3. Evaluate a fixed quarter-step grid from `0.0` through `3.0` in both windows.
4. A bonus is eligible only if every window keeps relevance AP within `0.01`, improves explicit-preference AP, and improves recall at one of top 10/25/50.
5. Select the smallest eligible bonus. This deliberately favors the least production influence that consistently works across time.

There is no untouched historical test after this redesign. Before rollout, run a no-write comparison on current unread articles and then a controlled inference. Freeze the selected bonus in the versioned infrastructure manifest and retune only after a substantial increase in explicit-preference labels, expected in a few months.

The code change itself makes the two-head model the only artifact family inference accepts, so deployment must train it before scheduling inference.

## 4. File-by-file impact

- `feedoscope/config.py` — remove `EXCELLENT_WEIGHT`; add only a format/version constant if code organization requires it.
- `feedoscope/relevance_embedding.py` — define the versioned two-head relevance artifact, fitting/loading validation, and shared two-head probability helpers; retain the embedding cache and urgency-compatible single-classifier helpers.
- `feedoscope/llm_learn.py` — train and save the unweighted relevance and super-important heads from one embedding matrix.
- `feedoscope/llm_infer.py` — locate the new artifact family, return both the super-important probability and the combined ranking score, and multiply before the existing integer result is built.
- `feedoscope/entities.py` — carry the unrounded super-important probabilities with relevance inference results.
- `feedoscope/data_registry/data_registry.py` and `feedoscope/data_registry/sql/upsert_super_important_inference.sql` — upsert per-article probabilities by model key.
- `feedoscope/main.py` — persist the probability before applying existing decay and writing final scores.
- `db/migrations/000007_super_important_inference.{up,down}.sql` — create/drop the Feedoscope-owned prediction table.
- `feedoscope/eval_models.py` — add fixed chronological evaluation and rank-based metrics for the three candidate scorers; leave freshness evaluation out.
- `tests/test_relevance_ranker.py` (new) — focused synthetic checks for label assignment, probability multiplication, artifact compatibility, and ranking metrics.
- Existing relevance tests, if any — update artifacts/imports only where the new format requires it.
- `docs/super-important-ranker.md` (new) and `docs/README.md` — document the implemented label contract, score combination, and evaluation interpretation after validation succeeds.
- `/home/djipey/informatique/infra/manifests/perso/miniflux/base/feedoscope-learn-job.yaml` — remove stale `EXCELLENT_WEIGHT=20` after the two-head artifact has passed evaluation.

No Miniflux UI change, additional model, dependency, or CronJob is needed. One Feedoscope schema migration and the matching upsert path are required.

## 5. Risks and edge cases

- Ordinary read articles are implicit negatives: “not explicitly preferred” can mean forgotten rather than unimportant. The score must remain a ranking signal, not an asserted fact.
- Starred may mean bookmark while upvote may mean preference. Keep the combined existing contract for now, but report each subgroup to detect divergence.
- Multiplication can compress scores. It intentionally keeps low-relevance articles from reaching the top solely from a high preference-head output; validate the normal relevance guardrail.
- A two-head artifact must be trained and present before inference; deployment must run training before the hourly inference CronJob uses the new image.
- Historical scores/freshness predictions were not retained, so the offline comparison evaluates raw ranking only. The new table retains super-important predictions, not historical relevance/freshness values.
- The Feedoscope migration and upsert must deploy together. Feedoscope should fail clearly if the expected table is absent rather than silently discard predictions.
- The deployment manifest is in the separate infrastructure checkout, so its worktree and status must be checked independently before any change.

## 6. Validation / testing

- Add the smallest unit tests that fail for incorrect label partitioning, one-head artifact loading, reversed multiplication/ranking, and malformed rank metrics.
- Run `uv run pytest -q`.
- Run `uv run ruff check .`, `make lint`, and `make format`.
- Run the new evaluation against the live database through the deployed environment; preserve its fixed holdout IDs/log output for comparison.
- Acceptance: two-head ranking improves explicit-preference average precision and at least one realistic top-K recall without a material regression in the ordinary good-vs-bad guardrail. If it does not, keep the current artifact/deployment and report the result rather than force rollout.
- Before deploying: build the image, run the training command once, verify the new artifact loads, then run one controlled full inference and confirm final scores update without any freshness tag writes.

## 7. Step-by-step execution checklist

- [x] Re-read the current branch, recent commits, freshness rollout, training/inference path, current manifests, and live label counts.
- [x] Confirm the existing explicit-preference proxy remains the first-version label contract.
- [x] Obtain approval for the score-combination and benchmark plan. The raw super-important probability is persisted separately; the existing final `entries.score` remains the combined score.
- [x] Create a dedicated task worktree for the Feedoscope implementation at `/home/djipey/informatique/python/feedoscope.super-important-ranker` on branch `super-important-ranker`.
- [x] Implement the versioned two-head artifact, one-pass training/inference path, and model-keyed probability upsert.
- [x] Add focused tests and the offline chronological benchmark.
- [x] Run format, tests, Ruff, and mypy. Final local result: 39 tests passed; Ruff and mypy passed.
- [x] Run the first live benchmark. Direct multiplication improved explicit-preference ranking but failed the 0.01 relevance-AP guardrail, so it was not deployed.
- [x] Implement deterministic bounded-bonus tuning and run chronological validation/test benchmarks. Bonus `3.0` passed validation but failed the one-time confirmation guardrail; production remained unchanged.
- [ ] Replace the single confirmation with the predeclared two-window expanding evaluation and select the smallest bonus passing every window.
- [ ] If a bonus clears both rolling windows, run a no-write current-article comparison, update the infrastructure manifest, train once, and perform controlled inference.
- [x] Update durable documentation for the implemented behavior in `docs/super-important-ranker.md` and `docs/README.md`.
- [ ] After the benchmark and rollout decision, mark this plan completed.

## 8. Open questions / assumptions

- Assumption: normal read, unstarred, neutral-vote articles are acceptable pragmatic negatives for “super-important.” No new user labels are introduced in version one.
- Confirmed: the current `entries.score` ordering is the intended consumer. The two-head result replaces the raw relevance ranking in memory; the separate importance probability is stored for inspection and possible future UI use, but is not displayed in this version.
- Assumption: top-10, top-25, and top-50 are useful initial review budgets; they are benchmark reporting points, not hard product limits.
- Confirmed: a general relevance AP drop larger than 0.01 versus the weighted baseline blocks rollout.
- The implementation should not proceed to production deployment if every rolling window does not show a measured improvement within that guardrail.
- Confirmed: freeze the deployed bonus and retune in a few months after substantially more super-important labels have accumulated.
