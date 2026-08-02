# New Embedding Model Evaluation Results

**Status:** In progress — 2026-08-02

## Selected-C artifact evaluations — 2026-08-02

Five saved artifacts completed logistic-regression evaluation. Each result used the approved 7,300-row training set and fixed 400-row holdout. The holdout contains exactly 200 positive rows and 200 negative rows.

`C` selection used only training rows. It used five shuffled stratified folds, seed `42`, and mean average precision. Each selected head then ran once against the fixed holdout.

| Model | Selected C | CV AP ± SD | Holdout AP | Log loss | Accuracy | Precision | Recall | F1 | AP difference vs. Harrier, 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Harrier 0.6B | 100 | 0.989213 ± 0.002933 | 0.960270 | 0.313446 | 0.900000 | 0.838983 | 0.990000 | 0.908257 | reference |
| EmbeddingGemma, unprompted | 10 | 0.987821 ± 0.002472 | 0.957613 | 0.336616 | 0.855000 | 0.784000 | 0.980000 | 0.871111 | -0.002657, [-0.029175, 0.020364] |
| EmbeddingGemma, classification prompt | 10 | 0.988868 ± 0.002383 | **0.963076** | **0.298297** | 0.887500 | 0.827004 | 0.980000 | 0.897025 | +0.002806, [-0.017529, 0.021173] |
| Jina v5 Nano Classification | 100 | 0.976381 ± 0.004164 | 0.897077 | 0.495211 | 0.792500 | 0.714286 | 0.975000 | 0.824524 | -0.063193, [-0.103941, -0.019998] |
| Nemotron 3 Embed 1B | 5 | 0.989041 ± 0.001717 | 0.958361 | 0.322338 | 0.862500 | 0.793522 | 0.980000 | 0.876957 | -0.001909, [-0.027902, 0.020079] |

The paired bootstrap uses seed `42` and 10,000 samples. The EmbeddingGemma prompt produced the highest point AP. Its interval crosses zero. The run reused embeddings on a host without GPU support. It does not satisfy the RTX 3060 confirmation rule.

Qwen3 4B remains pending. Its full embedding artifact must exist before its selected-C evaluation.

## Summary

Harrier 0.6B remains the relevance quality winner in the historical fixed-`C=5` comparison. It achieved the best average precision, log loss, accuracy, precision, recall, and F1.

Nemotron 1B matched EmbeddingGemma on average precision within the `0.001` tie range. It improved calibration and ROC AUC, but it was slower and used more memory. Qwen3 0.6B improved several threshold metrics but reduced average precision. Both Bekko variants were much faster, but their quality was lower.

Keep EmbeddingGemma in production until Harrier passes equivalent validation for every shared Feedoscope head.

## Embedding artifact persistence — 2026-08-02

The benchmark now stores one compressed NPZ artifact for each successful contract. Each artifact includes both embedding matrices, labels, article IDs, snapshot identity, the full model contract, and runtime metadata. Later runs validate the artifact before model loading and then reuse it.

Persisted controls:

| Model | Artifact | SHA-256 | Reproduced AP | Reproduced log loss | Peak VRAM |
|---|---|---|---:|---:|---:|
| Harrier 0.6B | `harrier-0.6b-relevance_new_models_20260802T122707Z.npz` | `e65bc16a9c7a26b894e6f473f6b4f4ffe2dd5d5e4bba4f405d9950dbf57affa3` | `0.963152` | `0.314114` | `1.385 GB` |
| Nemotron 3 Embed 1B | `nemotron-3-embed-1b-relevance_new_models_20260802T122707Z.npz` | `ef3a41d754ebf757b52afbe8cc11a6c60600ba4b1b0ee40f56b741f2532e9bcb` | `0.958361` | `0.322338` | `2.587 GB` |

The NAS-to-local transfer hash matched for both artifacts. A Harrier rerun reused its artifact, skipped embedding inference, and finished in `1.95 s` with `0.0 s` encoding time.

EmbeddingGemma did not run on the NAS. The host lacks a local snapshot and Hugging Face returned `401 Unauthorized` for the gated revision. Add a valid Hugging Face login or copy the approved local model snapshot before this control and the prompted contract can run.

## Benchmark contract

The benchmark used one frozen three-year relevance snapshot:

- snapshot ID: `relevance_new_models_20260802T122707Z`
- total rows: `7,700`
- train rows: `7,300`
- evaluation rows: `400`
- positive train rows: `6,142`
- negative train rows: `1,158`
- positive evaluation rows: `200`
- negative evaluation rows: `200`
- split: random fixed holdout per label with seed `42`
- train and evaluation article ID overlap: `0`

Snapshot SHA-256 values:

- `train.parquet`: `6486031c953ed4f7ddd3f741bb8c12190d918a132c7fa5bea423e2ce3f5bec64`
- `eval.parquet`: `2112fb6b69ddeebdb450f43270612776ac959354469939aa2edc6f64755ea24c`
- `metadata.json`: `e84e29354ad6522ee551365093bfc9939d836c2b7012b314fae5f9a062708372`

Every raw result records this snapshot ID and these three hashes. Every raw result therefore identifies the same 7,300 train articles and 400 evaluation articles.

All runs used:

- unweighted articles.
- `title_head` text.
- a 2,048-token total limit.
- normalized dense vectors.
- logistic regression with `C=5` and seed `42`.
- BF16 model inference.
- batch size `4`.

The earlier 80/20 snapshot results are superseded. The first weighted runs are invalid because production uses a separate super-important model.

## Runtime

All runs used this hardware and base runtime:

- GPU: NVIDIA GeForce RTX 3060, 12 GB
- PyTorch: `2.8.0+cu128`
- CUDA: `12.8`

EmbeddingGemma, Jina, and Harrier used Transformers `4.57.0`. Qwen3, Bekko, and Nemotron used an isolated Transformers `5.12.0` overlay.

The encoding time excludes model download and load time. The total time includes those one-time operations.

## Results

| Model | Average precision | Log loss | ROC AUC | Accuracy | F1 | Encode time | Total time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EmbeddingGemma-300M | 0.958549 | 0.345841 | 0.969075 | 0.852500 | 0.869180 | 276.9 s | 332.8 s | 0.768 GB |
| Jina v5 Small Classification | 0.945775 | 0.411481 | 0.951750 | 0.817500 | 0.841649 | 634.2 s | 687.9 s | 1.408 GB |
| Harrier 0.6B | **0.963152** | **0.314114** | 0.967675 | **0.887500** | **0.897959** | 643.6 s | 697.9 s | 1.408 GB |
| Qwen3 Embedding 0.6B | 0.954963 | 0.337313 | 0.962025 | 0.865000 | 0.877828 | 644.4 s | 705.7 s | 1.408 GB |
| Bekko a8m | 0.925955 | 0.453771 | 0.941550 | 0.807500 | 0.835118 | **35.1 s** | **97.1 s** | **0.376 GB** |
| Bekko a25m | 0.941744 | 0.437147 | 0.944975 | 0.827500 | 0.850972 | 71.3 s | 136.9 s | 0.408 GB |
| Nemotron 3 Embed 1B | 0.958361 | 0.322338 | **0.971025** | 0.862500 | 0.876957 | 766.7 s | 836.3 s | 2.610 GB |

Additional metrics:

| Model | Precision | Recall |
|---|---:|---:|
| EmbeddingGemma-300M | 0.780876 | 0.980000 |
| Jina v5 Small Classification | 0.743295 | 0.970000 |
| Harrier 0.6B | **0.821577** | **0.990000** |
| Qwen3 Embedding 0.6B | 0.801653 | 0.970000 |
| Bekko a8m | 0.730337 | 0.975000 |
| Bekko a25m | 0.749049 | 0.985000 |
| Nemotron 3 Embed 1B | 0.793522 | 0.980000 |

## Model contracts

### EmbeddingGemma control

- model: `google/embeddinggemma-300m`
- contract revision: `57c266a740f537b4dc058e1b0cda161fd15afa75`
- source: local production snapshot from the models PVC
- weight SHA-256: `cbf5a78393b6a033e0b8a63a57549964f7ed5c6fbeb4ba0694214f36123f2fd2`
- verification: the weight hash matches the Hugging Face file metadata for the contract revision
- prefix: none
- pooling: mean
- vector width: `768`
- article token budget: `2,048`

### Jina

- model: `jinaai/jina-embeddings-v5-text-small-classification`
- revision: `4447914a9b5b2fb00db3ce0884602b47a08f9458`
- prefix: `Document: `
- pooling: last token
- vector width: `1,024`
- article token budget: `2,045`

### Harrier

- model: `microsoft/harrier-oss-v1-0.6b`
- revision: `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee`
- instruction: `Classify RSS articles according to the reader's relevance preferences`
- pooling: last token
- vector width: `1,024`
- article token budget: `2,030`

### Qwen3 Embedding

- model: `Qwen/Qwen3-Embedding-0.6B`
- revision: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- instruction: `Classify RSS articles according to the reader's relevance preferences`
- pooling: last token
- attention: SDPA
- vector width: `1,024`
- article token budget: `2,031`

### Bekko

- model: `hotchpotch/bekko-embedding-v1-a8m`
- revision: `b24cde5de82214ada4c01f173b137c78160b13c6`
- active parameters: `7,671,168`
- total parameters: `105,975,168`
- prefix: none
- pooling: mean
- attention: SDPA
- vector width: `384`
- article token budget: `2,048`

### Bekko a25m

- model: `hotchpotch/bekko-embedding-v1-a25m`
- revision: `e0f3136db1b823ccbc67c4bea7d29f295516535b`
- active parameters: `24,930,432`
- total parameters: `123,234,432`
- prefix: none
- pooling: mean
- attention: SDPA
- vector width: `384`
- article token budget: `2,048`

### Nemotron

- model: `nvidia/Nemotron-3-Embed-1B-BF16`
- revision: `a5e0f804b9e90a1ca6784ecbf6e41595774fc834`
- prefix: `passage: `
- pooling: mean
- attention: SDPA
- vector width: `2,048`
- article token budget: `2,044`

## Comparison with EmbeddingGemma

| Model | AP change | Log-loss change | Accuracy change | F1 change | Encode-time ratio | VRAM ratio |
|---|---:|---:|---:|---:|---:|---:|
| Jina | -0.012774 | +0.065640 | -0.035000 | -0.027531 | 2.29x | 1.83x |
| Harrier | +0.004603 | -0.031727 | +0.035000 | +0.028780 | 2.32x | 1.83x |
| Qwen3 | -0.003586 | -0.008528 | +0.012500 | +0.008648 | 2.33x | 1.83x |
| Bekko a8m | -0.032594 | +0.107930 | -0.045000 | -0.034062 | 0.13x | 0.49x |
| Bekko a25m | -0.016805 | +0.091306 | -0.025000 | -0.018208 | 0.26x | 0.53x |
| Nemotron | -0.000188 | -0.023503 | +0.010000 | +0.007778 | 2.77x | 3.40x |

Harrier improved the primary metric and most secondary metrics. Qwen3 and Nemotron improved calibration and threshold metrics but did not improve average precision.

Bekko a8m encoded the snapshot 7.9 times faster than EmbeddingGemma. Bekko a25m was 3.9 times faster and improved average precision by `0.015789` over a8m. It remained `0.016805` below EmbeddingGemma.

## Limitations

- The random split groups rows by label, not by duplicate story or source.
- Syndicated copies can occur in both partitions and raise absolute scores.
- The comparison covers relevance only.
- Instruction-aware results depend on the recorded task instruction.
- Total times include one-time model download and load operations.
- This balanced holdout differs from the imbalanced production article distribution.

These limits affect deployment claims. They do not change the same-snapshot comparison between the seven encoders.

## Rollback and production safety

The evaluation read existing Parquet files. It did not connect to PostgreSQL.

The evaluation created no migration, database row, production cache row, production model, or production dependency change. The isolated NAS environment and model cache were deleted after artifact transfer.

Repository changes affect only the benchmark script, tests, plan, report, and ignored artifacts. Standard Git reversion removes those tracked evaluation changes if required.

## Decision

Harrier remains the relevance quality winner. It beat EmbeddingGemma average precision by `0.004603` and Nemotron by `0.004791`.

Nemotron tied EmbeddingGemma average precision within the `0.001` range. It did not tie the best model, so no confirmation run was required.

Qwen3 and both Bekko variants require no further relevance evaluation for these revisions and contracts. Bekko a25m offers the better Bekko quality and remains 3.9 times faster than EmbeddingGemma.

Keep EmbeddingGemma in production until Harrier passes urgency, freshness, and super-important validation. A later encoder switch requires new cache rows and new classifier artifacts.
