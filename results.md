🔬 68 runs 15 kept │ ★ metric: 0.96 #2 9💥 │ conf: 253.3×                                                                                                                                                                                                                             ctrl+x expand • ctrl+shift+x ful
~/projects/Feedoscope (pi_research) ↑969k ↓63k R22M $8.784 (sub) 60.8%/272k (auto)

## Other Machine Runs

### validate/relevance-embeddinggemma-prod

- Command: `VALIDATION_SIZE=100 make train`
- Training rows: `6312` good, `1005` bad
- Training time: `482.17s`
- Total time: `500.22s`
- Precision: `0.85`
- Recall: `0.99`
- F1: `0.92`
- ROC AUC: `0.97`
- Average Precision: `0.96`
- Log Loss: `0.32`
- Peak VRAM: `2.17 GB`
- Note: emitted a tokenizer warning about a `4039 > 2048` pre-tokenization path in
  `title_head`, but the run completed successfully.

### validate/relevance-ettin68m-prod

- Command: `VALIDATION_SIZE=100 make train`
- Training rows: `6312` good, `1005` bad
- Training time: `418.31s`
- Total time: `422.43s`
- Precision: `0.83`
- Recall: `1.00`
- F1: `0.90`
- ROC AUC: `0.96`
- Average Precision: `0.94`
- Log Loss: `0.43`

### validate/master-balanced-validation

- Command: `VALIDATION_SIZE=100 make train`
- Training rows: `1003` good, `1003` bad
- Training time before crash: `322.13s`
- Status: crashed during validation
- Root cause: model-family cleanup ran before validation finished and deleted the
  freshly trained local model directory. Validation then tried to reload the local
  tokenizer from a path that no longer existed, which `transformers` treated as a
  Hugging Face repo id and failed with a 404.

### validate/master-balanced-validation (rerun after cleanup timing fix)

- Command: `VALIDATION_SIZE=100 make train`
- Training rows: `1003` good, `1003` bad
- Training time: `311.09s`
- Total time: `317.79s`
- Precision: `0.91`
- Recall: `0.83`
- F1: `0.87`
- ROC AUC: `0.95`
- Average Precision: `0.95`
- Log Loss: `0.31`
- Note: validation succeeded, but the family-wide post-validation cleanup still
  deleted the freshly trained model because same-family directories are sorted by
  name, not true training timestamp. That cleanup should be disabled for these
  comparison branches.

### validate/master-full-validation

- Command: `VALIDATION_SIZE=100 make train`
- Training rows: `2481` good, `1003` bad
- Training time: `522.46s`
- Total time: `529.16s`
- Precision: `0.89`
- Recall: `0.94`
- F1: `0.91`
- ROC AUC: `0.96`
- Average Precision: `0.95`
- Log Loss: `0.27`
- Note: this run still shows the old post-validation family cleanup warning, so the
  other machine likely ran an older branch commit before the no-cleanup follow-up was
  pulled.

### validate/master-full-validation (3-year window rerun)

- Command: `VALIDATION_SIZE=100 make train`
- Training rows: `6312` good, `1005` bad
- Training time: `1076.15s`
- Total time: `1083.17s`
- Precision: `0.85`
- Recall: `1.00`
- F1: `0.92`
- ROC AUC: `0.96`
- Average Precision: `0.94`
- Log Loss: `0.30`
