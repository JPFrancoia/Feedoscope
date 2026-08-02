# Super-important Ranker

The relevance scorer now uses two lightweight logistic-regression heads on the
same cached EmbeddingGemma vectors. The first estimates whether an article is
generally worth reading; the second estimates whether it resembles an article
that later received an explicit preference.

## Labels

The relevance head keeps the existing labels:

- positive: a read article with `vote >= 0`
- negative: an article with `vote = -1`

The super-important head is trained only on read/good articles:

- positive: `read AND (vote = 1 OR starred)`
- negative: `read AND vote = 0 AND NOT starred`

Unread and downvoted articles are excluded from the second head. This score is
a ranking signal based on explicit preference, not an objective importance
rating or a probability calibrated across all unread articles.

## Training and scoring

`make train` encodes the relevance training rows once, then fits both heads:

1. the unweighted relevance head on read/good versus downvoted articles
2. the super-important head on explicitly preferred versus ordinary read
   articles

The previous relevance sample weighting for starred or upvoted articles is not
used. Training requires both explicitly preferred and ordinary read examples.

Inference calculates the score in floating point before any freshness decay:

```text
preference signal = max(0, (P(explicit preference | read) - 0.5) / 0.5)
raw rank = P(good) × (1 + bonus × preference signal) / (1 + bonus)
final rank = semantic-freshness decay(raw rank)
```

Predictions at or below the classifier's 50% decision boundary have no ranking
effect. Above 50%, the preference signal rises linearly to its full value at
100%, avoiding a hard score jump. The bounded bonus cannot raise the raw rank
above `P(good)`.

Feedoscope keeps the rank in floating point through semantic-freshness decay,
then rounds and writes final integer `entries.score` values in transactions of
1,000 articles. Each transaction commits before the next batch starts, limiting
the WAL and rollback cost of large refreshes.

### Controlled age-block inference

Every inference run first resets stale scores on unread downvoted articles to
zero; those articles are excluded from model scoring afterward. Scheduled
inference then keeps its existing 40-day full refresh plus 1,500 sampled older
articles. Large controlled refreshes can instead select one complete,
non-overlapping age block:

```bash
python -m feedoscope.main --min-age-days 0 --max-age-days 30
python -m feedoscope.main --min-age-days 30 --max-age-days 90
python -m feedoscope.main --min-age-days 90 --max-age-days 120
```

The minimum age is included and the maximum age is excluded, so adjacent blocks
do not overlap. Both bounds are required together. During text preparation,
Feedoscope reports progress at least every 30 seconds and after each 1,000
articles so large runs do not appear stalled.

## Artifact compatibility

Two-head artifacts use a versioned `relevance_two_head_` model family. Its name
includes the encoder model, maximum length, text-preparation mode and version,
and logistic-regression `C` value.

Each artifact stores both classifiers in `relevance_two_head.joblib` with
metadata for the artifact version, backend, encoder configuration, linear `C`,
label contract, and counts for `good`, `bad`, `super_important`, and
`ordinary_read`. Inference validates all of this metadata and rejects old
one-head or otherwise incompatible artifacts. It selects only complete
two-head artifact directories, so an interrupted newer training run cannot
replace the last complete model.

## Persisted super-important probabilities

Migration `000007_super_important_inference` creates the Feedoscope-owned
`super_important_inference` table. Both `make infer` and `make full_infer`
upsert the raw second-head probability before final-score writing or decay.

| Column | Meaning |
|---|---|
| `article_id` | The Miniflux entry; references `entries(id)` and cascades on delete. |
| `model_key` | The exact artifact directory name used for inference. |
| `super_important_score` | Unrounded second-head probability, constrained to 0 through 1. |
| `last_updated` | Time of the latest upsert. |

`(article_id, model_key)` is the primary key. Scores from different artifacts
remain separate; the table stores the preference-head output, not the final
combined-and-decayed `entries.score`.

## Automatic important tag

Every relevance inference synchronizes the visible Miniflux user tag
`important-auto` in the same database transaction as the probability upsert.
Articles above 50% receive the tag; articles at or below 50% lose it when they
are processed again. Other tags are unchanged.

`important-auto` is owned by Feedoscope, so manual changes to that tag may be
replaced by later inference. Articles outside a run keep their previous tag
until they are processed again.

## Weekly benchmark, history, and rollout gate

`make eval` uses mature labels only: labels read in the last 40 days are
excluded. It orders the rest by `last_read` and article ID, then splits them
into oldest 60%, middle 20%, and newest 20% partitions. It skips the
super-important evaluation when any partition lacks the required `bad`,
`super_important`, or `ordinary_read` examples, or is smaller than the largest
ranking budget.

Each successful run writes one `Super-important` row to Miniflux's
`model_evals` history. The row fits the rankers on the oldest 80% (the first
two partitions), evaluates the newest 20%, and records the fixed
`SUPER_IMPORTANT_BONUS` used for that score. It includes preference AP,
relevance AP, Recall@10, Recall@25, Recall@50, the bonus, and training and
evaluation counts for `good`, `bad`, `super_important`, and `ordinary_read`.

The rolling tuner remains separate from this history row. It compares the old
weighted-relevance baseline and every quarter-step bonus from 0 through 3
across two chronological windows (60%→20% and 80%→20%). It logs AP, precision,
recall, and NDCG diagnostics, and logs whether a bonus passes the rollout gate.
It selects the smallest passing bonus; it does not change
`SUPER_IMPORTANT_BONUS` or write additional history rows. A passing bonus must
improve preference AP in both windows, keep relevance AP within 0.01 of the
baseline in both windows, and improve Recall@10, @25, or @50 in at least one
window.

## Recorded rollout and future retuning

The deployed image is `220a648` with `SUPER_IMPORTANT_BONUS=0.5`, the 50%
decision-boundary ramp, batched score writes, progress logging, age-block
inference, and stale downvote cleanup. Training used 6,323 good, 1,357 bad, and
194 super-important labels. A controlled 365-day refresh scored 55,003 articles,
and a final `[0, 30)` block verified the deployed operational path.

Bonus 0.5 was the smallest value that passed the rolling gate. In the older
window, preference AP changed from 0.0173 to 0.0175 and relevance AP from
0.9706 to 0.9721. In the newer window, preference AP changed from 0.0821 to
0.2660, relevance AP from 0.9615 to 0.9603, and the top 50 changed from zero
to 20 super-important articles. The newer-window recall gain is sufficient for
the gate; the older window had only 17 preference positives for training, so
requiring a top-50 gain in each window was not stable.

Do not retune during scheduled training. Retune only after a substantial
increase in explicit-preference labels: run `make eval`, review the logged
results for both chronological windows against the gate above, and use its
smallest passing bonus. If the selected value changes, set
`SUPER_IMPORTANT_BONUS` to that value in the versioned deployment manifest,
then run controlled training and inference before deploying it. Record the new
value, label counts, and controlled-inference row count in this section.
