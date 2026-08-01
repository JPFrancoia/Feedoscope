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
raw rank = P(good) × P(explicit preference | read)
final rank = semantic-freshness decay(raw rank)
```

The raw multiplication prevents a high super-important score from lifting an
article whose general relevance score is low. Feedoscope rounds only when it
writes the final integer `entries.score`; semantic freshness therefore receives
the unrounded combined score.

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

## Offline benchmark and rollout gate

`make eval` also runs a super-important offline benchmark. It combines read and
downvoted training articles, excludes labels read within the most recent 40
days, orders the remaining labels by `last_read` and article ID, and holds out
the newest 20%. It skips the benchmark when either partition lacks at least 10
required examples or the holdout has fewer than 50 articles.

The fixed holdout compares three raw, undecayed scorers:

- the old weighted relevance baseline (weight 20 for explicit preferences)
- unweighted relevance alone
- the two-head multiplied score

It logs explicit-preference prevalence and average precision among read
articles; precision, recall, and graded NDCG at 10, 25, and 50; relevance
average precision as a guardrail; and counts for upvoted-only, starred-only,
and both-positive holdout examples. Graded NDCG uses downvoted = 0, ordinary
read = 1, and explicitly preferred = 2. These benchmark results are log output
only and do not change the shared `model_evals` schema.

The rollout gate is a live chronological benchmark that improves
explicit-preference average precision and at least one realistic top-K recall
without a material regression in relevance average precision. The live
benchmark has **not** yet been completed. The production manifest rollout has
also **not** yet been completed; do not schedule two-head inference until a
new two-head artifact has been trained and the gate has passed.
