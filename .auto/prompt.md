# Autoresearch: intrinsic semantic freshness

## Objective

Find the simplest local model that predicts how long an RSS article's main current/actionable claim remains useful. This is intrinsic semantic lifetime: never use when the user read the article, click/pageview popularity, backlog behavior, votes, status, or urgency tags as input features.

The frozen labels come from an article-only teacher that assigns one of six ordered horizons:

1. `<24 hours`
2. `1–3 days`
3. `4–7 days`
4. `8–30 days`
5. `1–6 months`
6. `evergreen (>6 months)`

The test set is the newest 30% of usable labeled articles. It is fixed. Improve generalization to newer articles, not memorization.

## Metrics

- **Primary:** `rps` (Ranked Probability Score, lower is better). It measures calibration across ordered horizons and penalizes distant horizon errors more than adjacent errors.
- **Secondary:** `macro_f1`, `weighted_kappa`, `log_duration_mae`, `evergreen_auc`, `mean_confidence`.

Primary metric is king. Prefer less code when RPS is equal within 0.001.

## How to Run

```bash
./.auto/measure.sh
```

The script verifies frozen-file hashes and emits `METRIC name=value` lines.

## Files in Scope

- `experiments/freshness/candidate.py` — the only model implementation to modify.

Keep the public function:

```python
def fit_predict(train: dict[str, object], test: dict[str, object]) -> np.ndarray:
    ...
```

Return an `(n_test, 6)` non-negative finite probability matrix. Rows are normalized by the evaluator.

Available training fields:

- `embeddings`: frozen 768-dimensional EmbeddingGemma vectors
- `titles`, `contents`, `feed_names`, `published_at`
- `current_urgency_score`: production binary urgency probability, useful only as a baseline feature
- `labels`: teacher horizon indexes `0..5`

The test dictionary has the same fields except `labels`.

## Off Limits

Do not modify or read around the evaluator to discover test labels. These are fixed benchmark infrastructure:

- `.auto/measure.sh`
- `.auto/checks.sh`
- `.auto/frozen.sha256`
- `.auto/data/`
- `experiments/freshness/evaluate.py`
- `experiments/freshness/export_readonly.py`
- `experiments/freshness/build_labels.py`
- every file under `feedoscope/`, `db/`, `docs/`, and `plans/`

Do not connect to PostgreSQL, the network, external APIs, or model services. Every iteration is local. Do not add dependencies.

Never use or reconstruct:

- publication-to-read delay or `last_read`
- read/unread status
- `0-urgency` / `1-urgency` source tags
- votes, stars, clicks, pageviews, or popularity
- decoder-generated labels or explanations
- test labels

Publication time itself is allowed only for textual/date reasoning, not as a proxy for whether the user eventually read something.

## Constraints

- Python 3.12 and already-installed NumPy, pandas, and scikit-learn only.
- CPU-only benchmark; keep each run comfortably below 5 minutes.
- Deterministic seeds for every stochastic method.
- Fit preprocessing on training data only.
- Do not persist model artifacts or predictions.
- A primary improvement must pass `.auto/checks.sh` before keeping.
- Simpler is better; do not build abstractions for one candidate.

## Hypotheses Backlog

Try structurally different ideas rather than only hyperparameter sweeps:

1. Current urgency probability mapped to ordered horizon buckets (baseline).
2. Multinomial logistic regression over normalized EmbeddingGemma vectors.
3. Ordinal threshold classifiers over embeddings.
4. Title/body TF-IDF with linear classification as a cheap text control.
5. Explicit temporal features: dates, deadlines, durations, future tense, advisories, scheduled events, and developing-story phrases.
6. Embeddings plus explicit temporal features.
7. Two-stage mutable/evergreen classification followed by short-horizon classification.
8. Class weighting and probability calibration using training-only cross-validation.
9. Nearest-neighbor or centroid models for rare horizon buckets.
10. Small calibrated ensembles when components make different errors.
11. Abstention-like smoothing for genuinely uncertain rows; RPS rewards honest distributions.

Do not implement same-story/newer-context reassessment in this loop. It needs a different corpus and benchmark after the best article-only baseline is known.

## What's Been Tried

- Urgency-to-horizon baseline: `rps=0.21905157`; useful evergreen signal but poor horizon predictions.
- Normalized EmbeddingGemma multinomial logistic regression roughly halved RPS; weaker regularization helped up to `C=10`.
- Five cumulative logistic boundaries aligned better with ordered RPS. The retained model uses `C=20`, quarter-to-three-eighths fractional class balancing, and training-only preprocessing.
- Proper least-squares isotonic projection of crossing boundary probabilities is the current best: `rps=0.08792380`, deterministic on repeat.
- Word/character TF-IDF, title-only text, fixed temporal phrases, parsed durations/dates, urgency, and feed identity did not improve RPS. Explicit temporal features often improved kappa but not calibration.
- RBF SVMs, k-nearest neighbors, ExtraTrees, PCA/PLS, ridge cumulative regression, latent-score regression, L1 sparsity, bagging, and cross-validated sigmoid calibration all regressed.
- Two-stage evergreen/mutable classification and centered-embedding blends produced sub-0.001 nominal gains but were discarded under the simplicity and anti-overfitting rule.
- Do not resume decimal hyperparameter tuning or text-only variants. Prefer structurally different, training-only ideas and require material gains for added complexity.
