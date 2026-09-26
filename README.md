# Feedoscope

Feedoscope ranks unread Miniflux articles by personal relevance. It trains from explicit Miniflux feedback and writes derived scores back to PostgreSQL.

## Current pipeline

Feedoscope uses these relevance labels:

- Positive: a read article with vote `0` or `1`.
- Negative: an article with vote `-1`.
- Important positive: a positive article that is starred or upvoted.

The model uses prompted `google/embeddinggemma-300m` title and body vectors. It also uses separate Word TF-IDF title and body features.

Weighted logistic regression fits the combined features. Important positive rows receive a weight of 20.

Routine inference selects unread, unstarred articles without a downvote. It applies fixed seven-day age decay and a cube-root score transform.

Feedoscope stores the final integer score in Miniflux `entries.score`. Miniflux displays and sorts the score.

Urgency, semantic-freshness, and separate preference heads are not active production pipelines.

## Commands

Install dependencies:

```bash
make install
```

Warm title and body embeddings:

```bash
make warm_embeddings
```

Train the relevance model:

```bash
make train
```

Run relevance inference without age decay:

```bash
make infer
```

Run full inference with fixed age decay:

```bash
make full_infer
```

Run the strict forward evaluation:

```bash
make eval
```

Set `DATABASE_URL` before a command that uses PostgreSQL. Training and inference require a GPU unless their explicit CPU fallback setting is enabled.

Read `AGENTS.md` for repository commands, tests, and code conventions. Read `../docs/feedoscope-context.md` for current Feedoscope terms.

## Historical background

The original system used Tiny Tiny RSS and positive-unlabeled learning. The current system uses the Miniflux fork and explicit downvotes.

The [original Feedoscope article](https://jpfrancoia.github.io/2025/08/17/feedoscope.html) describes an earlier architecture.
