# Semantic Freshness

Semantic freshness estimates how long an article's main claim remains useful. It
is separate from urgency: shadow-mode freshness predictions do not change the
existing urgency-based relevance decay.

## Labels

The six ordered horizons are `lt-24h`, `1-3d`, `4-7d`, `8-30d`, `1-6m`, and
`evergreen`. Teacher CSV horizons use the same names with hyphens replaced by
underscores (for example, `lt_24h`).

- Reviewed Miniflux tags end in `-freshness` and override every other label.
- Inference maintains separate `-auto-freshness` tags. It replaces only its own
  automatic tag and never removes a reviewed tag.
- At training start, an article that has been read with exactly one automatic
  tag and no reviewed tag is promoted to the equivalent reviewed tag.
- Read articles with one reviewed tag train the model. A medium/high-confidence
  row in `semantic_freshness_teacher_labels` is a bootstrap fallback. Articles
  with multiple reviewed tags are excluded and logged.

Import the frozen teacher CSV once after migration. It must include
`article_id`, `horizon`, and `confidence` columns; only `medium` and `high`
confidence rows are imported.

```bash
LOGGING_CONFIG=dev_logging.conf uv run python -m \
  feedoscope.import_semantic_freshness_teacher_labels path/to/teacher_labels.csv
```

## Training and artifacts

`make train_freshness` reuses the shared EmbeddingGemma cache and fits five
`LogisticRegression(C=20, fit_intercept=False)` heads for usefulness beyond
24 hours, 3 days, 7 days, 30 days, and 6 months. Boundary weights use the
selected `0.375` exponent.

The trainer hashes the effective labels, their source/confidence, and encoder
configuration. If that fingerprint matches the active artifact, it logs `No
new labels` and skips fitting. Valid changed-label artifacts are dated and kept;
the previous artifact is not deleted.

Set `SEMANTIC_FRESHNESS_VALIDATION_SIZE` to hold out that many newest labels.
The chronological holdout records ranked probability score, macro F1, quadratic
weighted kappa, log-duration MAE, and (when possible) evergreen AUC. Metrics
are informational; they do not block a structurally valid changed-label model.

## Inference

`make infer_freshness` scores the same active unread set as urgency: every
eligible unread article from the last 40 days plus the existing sample of older
unread articles. It sorts the five cumulative probabilities descending, converts
them into six bucket probabilities, stores them in
`semantic_freshness_inference` by `(article_id, model_key)`, and assigns one
automatic freshness tag.

`make full_infer` runs this same inference in shadow mode before relevance
scoring. It persists predictions and tags but leaves `main.py`'s urgency decay
unchanged.

## Schema and rollout

Apply Feedoscope migrations `000006` and `000007` before deploying this code.
`000006` creates the teacher-label and model-keyed prediction tables. `000007`
extends Miniflux's shared `model_evals` table for ordinal freshness metrics.
Apply those migrations before deploying the Miniflux reader change.

An untouched temporal, preferably human-reviewed holdout is still required
before using expected lifetime as a relevance half-life.
