# Urgency Autoresearch Results

## Snapshot

- `artifacts/urgency_autoresearch/urgency_snapshot_20260420T095942Z`
- exported from read articles tagged `0-urgency` / `1-urgency` only

## Run Log

### 2026-04-20 — Run 1 — KEEP

- config: `answerdotai/ModernBERT-base`, `classifier_type=transformer`, `max_length=512`, `text_prep_mode=title_head`, `train_balance_mode=full`
- average_precision: `0.8707539292`
- roc_auc: `0.7390845070`
- log_loss: `0.6378960659`
- accuracy: `0.6040100251`
- precision: `0.8620689655`
- recall: `0.5281690141`
- f1: `0.6550218341`
- brier_score: `0.2236617651`
- peak_vram_gb: `4.8390879631`
- train_seconds: `111.1519731370`
- total_seconds: `144.0991962420`
- takeaway: bootstrap control established the first kept urgency baseline.

### 2026-04-20 — Run 2 — DISCARD

- config: `answerdotai/ModernBERT-base`, `classifier_type=transformer`, `max_length=512`, `text_prep_mode=single_blob`, `train_balance_mode=full`
- average_precision: `0.8695675423`
- roc_auc: `0.7434017146`
- log_loss: `0.6801753610`
- accuracy: `0.5363408521`
- precision: `0.8960000000`
- recall: `0.3943661972`
- f1: `0.5476772616`
- brier_score: `0.2441232192`
- peak_vram_gb: `4.8433485031`
- train_seconds: `141.7875018090`
- total_seconds: `194.2105380010`
- takeaway: `single_blob` lost to `title_head` on AP, log loss, recall, and runtime.

### 2026-04-20 — Run 3 — DISCARD

- config: `distilbert-base-uncased`, `classifier_type=transformer`, `max_length=512`, `text_prep_mode=title_head`, `train_balance_mode=full`
- average_precision: `0.8651599302`
- roc_auc: `0.7426668708`
- log_loss: `0.6329111125`
- accuracy: `0.6441102757`
- precision: `0.8380952381`
- recall: `0.6197183099`
- f1: `0.7125506073`
- brier_score: `0.2208441072`
- peak_vram_gb: `2.0092988014`
- train_seconds: `57.0956205410`
- total_seconds: `105.3120600280`
- takeaway: DistilBERT was much cheaper and slightly better calibrated, but it lost on the primary AP metric versus the ModernBERT control.

## Current Best Kept Run

- Run 1: `ModernBERT-base` + `title_head` + transformer @ 512
- best average_precision: `0.8707539292`

## Next Required Run

1. `google/embeddinggemma-300m @ 2048`, `title_head`, `embedding_linear`
