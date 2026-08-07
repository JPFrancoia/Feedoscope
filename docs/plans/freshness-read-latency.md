# Freshness from read latency

Status: proposal, not implemented. Written 2026-08-07.

Nothing in this plan is in production. The only new file is
`experiments/read_latency_halflife.py`. It runs read-only SQL. No module in
`feedoscope/` imports it. It is not in the `Makefile`. The `pyproject.toml`
mypy config excludes `experiments/`, so it does not affect `make lint`.

## The problem

Every freshness approach so far degrades week after week. The list includes
binary urgency with probabilities, three cumulative heads for days, months, and
years, and six lifetime buckets.

The cause is the label, not the model.

All these approaches predict an unobservable quantity: how long an article
stays useful. No ground truth exists for it. The labels come from the opinion
of an LLM on a frozen sample of 1,200 articles. The script is
`experiments/freshness/build_labels.py`. The holdout is the newest 150 labels,
in `eval_models.eval_freshness`.

This setup degrades for three reasons:

- The holdout drifts away from the frozen bootstrap distribution each week.
- Manual `fresh_d`, `fresh_m`, and `fresh_y` tags start to disagree with the
  bootstrap labels. The query `get_conflicting_semantic_freshness_labels.sql`
  already reports this disagreement.
- The accuracy ceiling is the agreement between the LLM and the user. Nothing
  in the pipeline raises that ceiling.

More heads, more buckets, and better encoders cannot correct a target that
nobody observes.

## The proposal

Measure the decay instead of a prediction of the decay.

Miniflux already records the event. The column `published_at` holds the time of
publication. The column `changed_at` holds the time of the read. The delay
between the two is observed data. It is self-labeling. It grows every week.

The arithmetic is convenient. For an exponential decay, the median delay
between publication and read is the half-life. That is the exact number that
`main.decay_relevance_score` needs. There are no labels, no classifier, and no
evaluation metric that can degrade. The estimator gets tighter with more reads.

The first hypothesis to test is that the feed predicts the decay, and the text
does not. Hacker News decays in hours. A mathematics blog does not decay.

## The experiment

```bash
DATABASE_URL=... uv run python -m experiments.read_latency_halflife
uv run python -m experiments.read_latency_halflife --self-check
```

The script prints the global median delay, the per-feed median delay for feeds
with 30 or more reads, and the fraction of log-latency variance that the feed
explains.

The SQL removes two sources of noise:

- A read second with more than 30 entries is a "mark all as read" burst. That
  action measures a click, not a decay of interest.
- Delays below one minute or above 365 days are dropped.

## How to read the result

If the feed explains more than 40% of the variance, delete the freshness
pipeline. Replace `main.get_decay_half_life` with a lookup of the per-feed
median and a global fallback. This is one SQL query. It needs no GPU, no
labels, and no weekly retrain.

If the feed explains less, the feed alone is not sufficient. The next step is a
survival model on the same observed target, for example an accelerated failure
time model from `lifelines`, on the embeddings that are already cached in
`relevance_embeddings`. The target stays observed. There is still no labeling
step.

## The known bias

The delay is only observable for articles that were read. An article that was
never read has no delay. It is censored data, not an article with an infinite
lifetime. The median over read articles is therefore too short.

This bias is known and bounded. The target is still better than a target that
nobody observes. If the bias is too large, a Kaplan-Meier estimator corrects it
exactly. Unread articles enter as right-censored at their current age. The same
query supplies the data.

## What this plan does not cover

Censoring, the survival model, and the embedding features are out of scope.
Add them if the per-feed medians do not separate.
