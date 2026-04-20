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

### 2026-04-20 — Run 6 — KEEP
- config: `Alibaba-NLP/gte-large-en-v1.5`, `classifier_type=embedding_linear`, `max_length=2048`, `text_prep_mode=title_head`, `train_balance_mode=full`, `embed_pooling=cls`, `linear_c=1.0`
- average_precision: `0.8744714509`
- roc_auc: `0.7560012247`
- log_loss: `0.5917234395`
- accuracy: `0.6741854637`
- precision: `0.8468468468`
- recall: `0.6619718310`
- f1: `0.7430830040`
- brier_score: `0.2029436774`
- peak_vram_gb: `3.0421280861`
- train_seconds: `485.2117850150`
- total_seconds: `521.6519080250`

### 2026-04-20 — Run 7 — DISCARD
- config: `Alibaba-NLP/gte-large-en-v1.5`, `classifier_type=embedding_linear`, `max_length=2048`, `text_prep_mode=title_head`, `train_balance_mode=full`, `embed_pooling=cls`, `linear_c=0.5`
- average_precision: `0.8732599267`
- roc_auc: `0.7558481323`
- log_loss: `0.5989961726`
- accuracy: `0.6641604010`
- precision: `0.8472222222`
- recall: `0.6443661972`
- f1: `0.7320000000`
- brier_score: `0.2059881910`
- peak_vram_gb: `3.0421280861`
- train_seconds: `485.2422876510`
- total_seconds: `516.1920871330`

### 2026-04-20 — Run 8 — DISCARD
- config: `Alibaba-NLP/gte-large-en-v1.5`, `classifier_type=embedding_linear`, `max_length=2048`, `text_prep_mode=title_head`, `train_balance_mode=full`, `embed_pooling=cls`, `linear_c=2.0`
- average_precision: `0.8688901832`
- roc_auc: `0.7491120637`
- log_loss: `0.5879036305`
- accuracy: `0.7092731830`
- precision: `0.8559322034`
- recall: `0.7112676056`
- f1: `0.7769230769`
- brier_score: `0.2012125330`
- peak_vram_gb: `3.0421280861`
- train_seconds: `483.4328041710`
- total_seconds: `502.8032532570`

### 2026-04-20 — Run 9 — DISCARD
- config: `google/embeddinggemma-300m`, `classifier_type=embedding_linear`, `max_length=2048`, `text_prep_mode=title_head`, `train_balance_mode=full`, `embed_pooling=mean`, `linear_c=2.0`
- average_precision: `0.8708880257`
- roc_auc: `0.7567666871`
- log_loss: `0.5827402005`
- accuracy: `0.6992481203`
- precision: `0.8474576271`
- recall: `0.7042253521`
- f1: `0.7692307692`
- brier_score: `0.1993891233`
- peak_vram_gb: `3.2199978828`
- train_seconds: `222.7938109310`
- total_seconds: `243.8446040470`

## Current Best Kept Run

- Run 6: `gte-large-en-v1.5` + `title_head` + embedding_linear @ 2048 + `pooling=cls` + `LINEAR_C=1.0`
- best average_precision: `0.8744714509`

## Best-So-Far Takeaways

- `title_head` beat `single_blob` for urgency on the transformer baseline.
- Frozen embedding baselines beat transformer fine-tuning on AP on this frozen read-tagged snapshot.
- `gte-large-en-v1.5` is currently best on AP, but `embeddinggemma-300m` remains much faster.
- Moving `gte-large-en-v1.5` away from `LINEAR_C=1.0` in either direction hurt AP.
- Increasing `LINEAR_C` on `embeddinggemma-300m` improved several thresholded / calibration metrics but hurt AP.

## Next Candidate Run

1. test `google/embeddinggemma-300m` with `LINEAR_C=0.5` to complete a symmetric quick regularization check on the faster embedding baseline
