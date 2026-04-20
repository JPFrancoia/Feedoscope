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

### 2026-04-20 — Run 4 — KEEP

- config: `google/embeddinggemma-300m`, `classifier_type=embedding_linear`, `max_length=2048`, `text_prep_mode=title_head`, `train_balance_mode=full`, `embed_pooling=mean`, `linear_c=1.0`
- average_precision: `0.8725124745`
- roc_auc: `0.7579914268`
- log_loss: `0.5931367270`
- accuracy: `0.6766917293`
- precision: `0.8506787330`
- recall: `0.6619718310`
- f1: `0.7445544554`
- brier_score: `0.2039524877`
- peak_vram_gb: `3.2217602730`
- train_seconds: `226.4302724800`
- total_seconds: `274.8645930420`

### 2026-04-20 — Run 5 — DISCARD

- config: `BAAI/bge-m3`, `classifier_type=embedding_linear`, `max_length=2048`, `text_prep_mode=title_head`, `train_balance_mode=full`, `embed_pooling=mean`, `linear_c=1.0`
- average_precision: `0.8707938576`
- roc_auc: `0.7388854868`
- log_loss: `0.6070791415`
- accuracy: `0.6566416040`
- precision: `0.8550724638`
- recall: `0.6232394366`
- f1: `0.7209775967`
- brier_score: `0.2105787899`
- peak_vram_gb: `3.0217294693`
- train_seconds: `437.0712764640`
- total_seconds: `476.1181968600`

## Current Best Kept Run

- Run 4: `embeddinggemma-300m` + `title_head` + embedding_linear @ 2048
- best average_precision: `0.8725124745`

## Next Required Run

1. `Alibaba-NLP/gte-large-en-v1.5 @ 2048`, `title_head`, `embedding_linear`
