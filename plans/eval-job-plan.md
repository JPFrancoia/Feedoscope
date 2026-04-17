# Plan: Model Evaluation Job

## Goal

Create a weekly eval job that measures the accuracy of both the **relevance model**
and the **urgency model**. Each model is retrained with a holdout set for evaluation
purposes only. The eval models are discarded after metrics are computed -- production
models (trained on 100% of the data) are NOT affected.

## Design Decisions

1. **Relevance eval** reuses the existing `VALIDATION_SIZE` mechanism in
   `llm_learn.py`. Setting `VALIDATION_SIZE > 0` causes `main()` to hold out the
   first N articles (by ID order) from training, then evaluate against them. The
   eval script just needs to set this value and call the existing code path.
   `VALIDATION_SIZE` becomes an env var so it can be configured without code changes.

2. **Urgency eval** trains on ALL tagged articles (read + unread) with the same
   smart class-balancing as production, but holds out some **read** tagged articles
   for evaluation. Evaluation is only against read articles because their tags are
   verified by the user. Training still includes unread articles (same as production).

3. **Separate eval functions** -- the eval script has its own `eval_relevance()`
   and `eval_urgency()` functions rather than calling the existing `main()` functions.
   This avoids the risk of `find_latest_model(clean_old_models=True)` deleting
   production models, and gives full control over model path and cleanup.

4. **Eval models saved to temp directories** under `models/eval_*`, cleaned up
   after metrics are computed. Uses a prefix that doesn't match any production model
   prefix, so `find_latest_model()` will never find or delete them.

5. **Metrics logged, not stored in DB**.

6. **`VALIDATION_SIZE` as an env var** in `config.py`, defaulting to 0 (no eval).
   The eval K8s CronJob sets it to a non-zero value. This also means `make eval`
   can control it via env var.

## Architecture

```
eval_models.py
    |
    +-- eval_relevance(device)
    |     1. Fetch good articles (read, vote>=0) excluding first VALIDATION_SIZE
    |     2. Fetch bad articles (vote=-1) excluding first VALIDATION_SIZE
    |     3. Class-balance (same logic as llm_learn.main())
    |     4. Train on temp path via llm_learn.train_model()
    |        (internally does 80/20 train/test split for training-time metrics)
    |     5. Fetch held-out good + bad articles (first VALIDATION_SIZE of each)
    |     6. Run inference on held-out set with the trained model
    |     7. Compute & log: accuracy, precision, recall, F1, ROC AUC, AP, log loss
    |     8. Delete the eval model
    |
    +-- eval_urgency(device)
          1. Fetch ALL tagged articles (read + unread)
          2. Separate read articles, hold out VALIDATION_SIZE of them
          3. Train on remaining data (rest of read + all unread)
             with same smart class-balancing as production
          4. Train on temp path via llm_learn_urgency.train_model()
          5. Run inference on held-out read articles
          6. Compute & log: accuracy, precision, recall, F1, ROC AUC, AP, log loss
          7. Delete the eval model
```

## Metrics Reported (both models)

| Metric              | What it measures                                          |
| ------------------- | --------------------------------------------------------- |
| `accuracy`          | % of correct predictions at 0.5 threshold                |
| `precision`         | Of predicted positives, how many are actually positive    |
| `recall`            | Of actual positives, how many were predicted              |
| `f1`                | Harmonic mean of precision and recall                     |
| `roc_auc`           | Area under ROC curve (ranking quality)                    |
| `average_precision` | Area under precision-recall curve (better for imbalanced) |
| `log_loss`          | Calibration quality of probability estimates              |

Additionally, dataset statistics will be logged: total articles, class distribution,
number of articles in train vs eval.

## Implementation Plan

### New files (2)

| File                                             | Purpose                                      |
| ------------------------------------------------ | -------------------------------------------- |
| `feedoscope/eval_models.py`                      | Eval script: eval both models, log metrics   |
| `base/feedoscope-eval-job.yaml` (in infra repo)  | K8s CronJob manifest                         |

### Modified files (3)

| File                                             | Change                                       |
| ------------------------------------------------ | -------------------------------------------- |
| `feedoscope/config.py`                           | Add `VALIDATION_SIZE` env var (default 0)    |
| `Makefile`                                       | Add `eval` target                            |
| `base/kustomization.yaml` (in infra repo)        | Add eval job to resources list               |

### Detailed implementation

#### 1. `feedoscope/config.py` change

Add one line:

```python
VALIDATION_SIZE = int(os.getenv("VALIDATION_SIZE", "0"))
```

`llm_learn.py` already has `VALIDATION_SIZE = 0` as a module constant (line 55).
The eval script will import from `config` instead. The existing `llm_learn.py`
constant stays unchanged (production training always uses 0).

#### 2. `feedoscope/eval_models.py`

```python
"""Weekly evaluation of relevance and urgency models.

For each model, trains on a subset of the data with a holdout set,
runs inference on the holdout, logs metrics, then discards the eval model.
Production models (trained on 100% of data) are NOT affected.
"""

EVAL_RELEVANCE_PREFIX = "eval_relevance"
EVAL_URGENCY_PREFIX = "eval_urgency"


async def eval_relevance(device: torch.device) -> None:
    """Evaluate the relevance model accuracy.

    Uses the existing VALIDATION_SIZE mechanism: the SQL queries for training
    data (get_read_articles_training, get_published_articles) skip the first
    VALIDATION_SIZE articles by ID order. The companion queries
    (get_sample_good, get_sample_not_good) return exactly those held-out
    articles.

    Steps:
        1. Fetch training data with VALIDATION_SIZE articles excluded
        2. Class-balance (same logic as llm_learn.main)
        3. Train on temp path using llm_learn.train_model()
        4. Fetch the held-out VALIDATION_SIZE articles
        5. Run inference on held-out articles
        6. Compute and log metrics
        7. Delete temp model directory
    """


async def eval_urgency(device: torch.device) -> None:
    """Evaluate the urgency model accuracy.

    Trains on ALL tagged articles (read + unread) with the same smart
    class-balancing as production, but holds out some read tagged articles
    for evaluation. Only read articles have verified labels (the user has
    confirmed or corrected the urgency tag), so they form the eval set.

    Steps:
        1. Fetch all tagged articles (read + unread)
        2. Separate read articles, hold out VALIDATION_SIZE
        3. Train on remaining data (rest of read + all unread)
           with smart class-balancing (read-priority)
        4. Run inference on held-out read articles
        5. Compute and log metrics
        6. Delete temp model directory
    """


def compute_and_log_metrics(
    model_name: str,
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
) -> None:
    """Compute classification metrics and log them."""


async def main() -> None:
    """Run evaluation for both models sequentially."""
```

**Key implementation details:**

- **Relevance eval** closely mirrors `llm_learn.main()` lines 230-353:
  - Calls `dr.get_read_articles_training(validation_size=config.VALIDATION_SIZE)`
    and `dr.get_published_articles(validation_size=config.VALIDATION_SIZE)` to get
    training data with holdout excluded.
  - Applies the same class-balancing (protect excellent articles, downsample rest).
  - Calls `llm_learn.train_model()` with a temp path like
    `models/eval_relevance_{date}`.
  - Fetches holdout via `dr.get_sample_good(validation_size=config.VALIDATION_SIZE)`
    and `dr.get_sample_not_good(validation_size=config.VALIDATION_SIZE)`.
  - Runs inference on the holdout (same pattern as `llm_learn.py` lines 305-353).
  - Logs metrics, then `shutil.rmtree()` the temp model directory.

- **Urgency eval** is custom since there's no existing holdout mechanism:
  - Calls `dr.get_articles_with_simplified_time_sensitivity()` to get all tagged
    articles.
  - Separates read articles from unread articles.
  - Holds out the first `VALIDATION_SIZE` read articles (sorted by ID, deterministic).
  - Passes the remaining data to a modified class-balancing flow:
    - The remaining read articles + all unread articles form the training pool
    - Same smart class-balancing as `llm_learn_urgency.py` (lines 218-276):
      read articles prioritized, unread downsampled.
  - Calls `llm_learn_urgency.train_model()` with temp path `models/eval_urgency_{date}`.
  - Runs inference on the held-out read articles using the eval model.
  - Logs metrics, deletes temp directory.

- **Model path isolation:** Using prefixes `eval_relevance` and `eval_urgency`
  ensures `find_latest_model()` (which searches by model name prefix) will never
  match these eval models. Production models use `answerdotai-ModernBERT-base_*`
  and `urgency-ModernBERT-base_*`.

- **Cleanup:** `shutil.rmtree()` is called in a `finally` block to ensure the
  eval model is always deleted, even if metrics computation fails.

- **Edge case:** If `VALIDATION_SIZE` is 0 or if there aren't enough read articles
  for urgency eval, the eval logs a warning and skips that model's evaluation.

#### 3. `Makefile` addition

```makefile
eval:
	LOGGING_CONFIG=dev_logging.conf VALIDATION_SIZE=100 uv run python -m feedoscope.eval_models
```

The `VALIDATION_SIZE=100` default can be overridden: `VALIDATION_SIZE=200 make eval`.

#### 4. K8s CronJob (`base/feedoscope-eval-job.yaml`)

Same pattern as existing feedoscope CronJobs:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: feedoscope-eval-job
spec:
  schedule: "0 3 * * 0"   # Weekly on Sunday at 03:00
  jobTemplate:
    spec:
      backoffLimit: 2
      template:
        metadata:
          labels:
            app: feedoscope-eval
        spec:
          nodeSelector:
            kubernetes.io/hostname: djipey-server
          containers:
            - name: feedoscope-eval
              image: 192.168.0.13:32000/feedoscope
              imagePullPolicy: IfNotPresent
              command:
                - /bin/sh
                - -c
                - python -m feedoscope.eval_models
              volumeMounts:
                - name: models-volume
                  mountPath: /app/models
              resources:
                limits:
                  nvidia.com/gpu: 1
              env:
                - name: VALIDATION_SIZE
                  value: "100"
                - name: EXCELLENT_WEIGHT
                  value: "10"
                - name: POSTGRES_PASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: miniflux
                      key: POSTGRES_PASSWORD
                - name: DATABASE_URL
                  value: postgresql://miniflux:$(POSTGRES_PASSWORD)@miniflux-db:5432/miniflux?sslmode=disable
          restartPolicy: Never
          volumes:
            - name: models-volume
              persistentVolumeClaim:
                claimName: models-pvc
```

**Schedule:** Sunday 03:00. Avoids conflict with:
- Relevance training: daily at 02:00 (finishes well before 03:00)
- Urgency training: Monday 04:00
- Inference: 06:00-23:00 daily

#### 5. `base/kustomization.yaml` update

Add `feedoscope-eval-job.yaml` to the resources list.

### Files summary

| Action | File                              | Repo       |
| ------ | --------------------------------- | ---------- |
| Create | `feedoscope/eval_models.py`       | feedoscope |
| Modify | `feedoscope/config.py`            | feedoscope |
| Modify | `Makefile`                        | feedoscope |
| Create | `base/feedoscope-eval-job.yaml`   | infra      |
| Modify | `base/kustomization.yaml`         | infra      |

**Total: 2 new files, 3 modified files across 2 repos.**

## Execution Order

1. Add `VALIDATION_SIZE` to `config.py`
2. Implement `feedoscope/eval_models.py`
3. Add `eval` target to `Makefile`
4. Test locally with `make eval`
5. Create K8s manifest + update kustomization
6. Deploy with `make pkg` (feedoscope repo) and `make run` (infra repo)
