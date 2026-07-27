# Intrinsic Freshness Autoresearch Plan

**Status:** Prepared for interactive autoresearch — 2026-07-26

## 1. Brief

Build an isolated autoresearch loop that tests ways to predict an article's intrinsic semantic lifetime without using publication-to-read delay or modifying PostgreSQL. The loop will iterate against a frozen local dataset and fixed evaluation harness, so experiments are fast, reproducible, and cannot affect production scoring.

## 2. Current state / relevant context

- Production urgency is a binary EmbeddingGemma plus logistic-regression classifier trained on read articles tagged `0-urgency` or `1-urgency`.
- Its probability is converted linearly into a 10–120 day half-life, although the training target contains no duration.
- PostgreSQL currently contains 2,520 read-tagged urgency articles; all have cached 768-dimensional EmbeddingGemma vectors.
- The database also contains decoder-generated binary urgency explanations and current urgency probabilities, but these are weak comparison signals rather than intrinsic-expiry truth.
- Personal read delay is explicitly excluded because backlog and periods without feed reading would bias it.
- Research found no mature full-article benchmark with authoritative semantic-expiry dates. The closest established tasks predict sentence/fact duration, mutable versus evergreen facts, factual-update risk, and validity changes after newer context arrives.

## 3. Proposed implementation

### Frozen local dataset

Create a read-only exporter that connects with PostgreSQL's `default_transaction_read_only=on` setting and writes local ignored files only:

- cleaned title and bounded article text
- feed and publication timestamp
- existing urgency probability and decoder explanation for baselines
- existing EmbeddingGemma vector
- user urgency tag for sampling/comparison only, never as a model input

No experiment will connect to PostgreSQL. A manifest will record row count, query/config provenance, and hashes.

### Semantic-horizon labels

Select a fixed, source-diverse and class-balanced sample from the 2,520 rows. Use a strong offline LLM teacher once to assign ordered horizon labels:

1. `<24 hours`
2. `1–3 days`
3. `4–7 days`
4. `8–30 days`
5. `1–6 months`
6. `evergreen (>6 months)`
7. `unknown` when the article alone does not support a defensible horizon

Each label must include confidence, reason type, and a direct evidence quote. Low-confidence/unknown rows remain available for coverage reporting but do not control the primary metric.

### Fixed evaluation harness

Use a deterministic temporal/group-aware split that keeps near-duplicate titles and related stories together. The candidate receives training labels and test features but does not use read time, status, user tags, vote, or test labels.

Primary metric:

- **Ranked Probability Score (RPS), lower is better** across the six ordered horizon buckets. RPS rewards calibrated probability distributions and penalizes distant horizon errors more than adjacent errors.

Secondary metrics:

- quadratic weighted kappa
- macro F1
- log-duration MAE
- evergreen AUROC
- prediction coverage after abstention
- per-feed and per-horizon diagnostics

The harness and frozen-data hashes are off limits to experiments.

### Autoresearch hypotheses

Start with the current urgency probability mapped into horizon buckets, then test:

1. EmbeddingGemma vectors with multinomial logistic regression.
2. Ordinal threshold models over the same embeddings.
3. Text-only TF-IDF models as a cheap control.
4. Explicit temporal features: deadlines, relative dates, future tense, event-completion and developing-story language.
5. Embedding plus temporal-feature models.
6. Two-stage mutable/evergreen then short-horizon prediction.
7. Calibrated ensembles and abstention for uncertain articles.
8. Same-story/newer-context invalidation as a separate follow-up once a reliable article-only baseline exists.

Prefer simpler models when performance is equal. Popularity, clicks, pageviews, read delay, and feed-reading cadence are out of scope.

### Autoresearch operation

- Work in dedicated branch/worktree `autoresearch/intrinsic-freshness-2026-07-26`.
- Store session control files under `.auto/`.
- Limit candidate changes to the experiment implementation, not production code.
- Run a baseline, log it, then continue autonomous measured iterations.
- Cap the initial segment to 30 iterations to bound model/API cost; it can be resumed explicitly.

## 4. File-by-file impact

Current branch:

- `plans/intrinsic-freshness-autoresearch-plan.md` — durable decision and progress record.

Dedicated autoresearch worktree:

- `.auto/prompt.md` — complete experiment playbook and hypotheses.
- `.auto/measure.sh` — fixed benchmark entry point.
- `.auto/checks.sh` — verifies frozen-file hashes and candidate output contract.
- `.auto/config.json` — initial iteration cap.
- `.auto/.gitignore` — excludes local article data, labels, predictions, logs, and caches.
- `experiments/freshness/export_readonly.py` — one-shot local exporter with enforced read-only transactions.
- `experiments/freshness/build_labels.py` — reproducible frozen teacher-label generation.
- `experiments/freshness/evaluate.py` — fixed split, metrics, and output contract.
- `experiments/freshness/candidate.py` — the only primary model file mutated by the loop.
- `.auto/data/*` — ignored local CSV/NumPy/JSON files.

Production files under `feedoscope/` remain untouched.

## 5. Risks and edge cases

- **Pseudo-gold bias:** an LLM teacher is not human ground truth. Keep provenance/evidence, exclude uncertain cases from the primary metric, and treat results as model-selection evidence rather than proof.
- **Label leakage:** user urgency tags, read timestamps, status, votes, and decoder labels must not enter candidate features.
- **Event leakage:** near-duplicate or same-story articles must remain in one split.
- **Mixed-lifetime articles:** background facts can be evergreen while the main development expires quickly. Labels target the article's main actionable/current claim and permit `unknown`.
- **Explicit dates:** a mentioned historical date is not necessarily an expiry date; temporal rules need semantic context.
- **Database safety:** every export connection sets `default_transaction_read_only=on`; no SQL from the experiment loop touches PostgreSQL.
- **Private data:** local dumps are ignored and must never be committed.
- **Harness gaming:** hashes and off-limits rules protect frozen evaluation files.
- **Cost:** teacher labels are generated once; ordinary iterations are local and deterministic.

## 6. Validation / testing

- Prove the PostgreSQL session reports `transaction_read_only=on` before export.
- Confirm source row counts and embedding dimensions.
- Verify local dump files are ignored by Git.
- Validate teacher JSON schema, evidence quotes, confidence, and allowed horizon values.
- Hash frozen data, labels, split, evaluator, and metric configuration.
- Run the baseline twice and require identical metrics.
- Run candidate contract checks before accepting each result.
- Confirm `git diff` contains no production-file changes and no local article content.

## 7. Step-by-step execution checklist

- [x] Confirm clean source checkout and create the dedicated autoresearch worktree.
- [x] Confirm PostgreSQL read-only access and inspect available labels/embeddings.
- [x] Define hypotheses, exclusions, metric, and leakage controls.
- [x] Add exporter and create ignored local dataset (2,520 rows; PostgreSQL read-only confirmed).
- [x] Generate and validate 400 frozen semantic-horizon labels (380 usable).
- [x] Add fixed evaluator, candidate baseline, checks, and autoresearch files.
- [x] Commit the reproducible setup on the autoresearch branch (`6650c62`).
- [x] Verify the deterministic starting candidate (`rps=0.21905157`, repeated twice).
- [ ] Initialize autoresearch from interactive Pi and record that candidate as run 1.
- [ ] Start the autonomous 30-iteration segment and preserve experiment learnings.
- [ ] Review the best retained approach and update this plan with results.

## 8. Open questions / assumptions

- Assumption: ordered horizon buckets are a better product target than one exact expiry timestamp because article text rarely supports exact dates.
- Assumption: a fixed strong-model teacher with evidence and abstention is sufficient to start automated comparison; human review can later replace or audit the frozen test labels.
- Assumption: cached EmbeddingGemma vectors are reusable because exporter provenance fixes the same encoder/text-preparation configuration.
- Assumption: the first loop optimizes article-only prediction. Newer-context invalidation is evaluated after this baseline because it requires a separate story-retrieval corpus and metric.
