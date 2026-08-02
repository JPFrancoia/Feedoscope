# Freshness Encoder Model Results

**Status:** Partial screening complete — 2026-08-02

## Purpose

This experiment compares the completed relevance encoder candidates against the active three-label Freshness task. It uses one frozen chronological holdout and the production classifier contract. It does not write production embeddings, scores, tags, or model artifacts.

## Fixed contract

- Labels: `fresh_d`, `fresh_m`, `fresh_y`.
- Split: every older effective label trains the classifier. The newest 150 labels form the holdout.
- Text: title plus article head, with a 2,048-token total limit.
- Vectors: model-specific pooling followed by L2 normalization.
- Classifier: two cumulative `LogisticRegression` heads.
- Classifier settings: `C=20`, no intercept, weight exponent `0.375`, seed `42`.
- Primary metric: lower RPS.
- Tie-breakers: higher Macro F1, then higher quadratic weighted kappa.

## Snapshot export

Set `DATABASE_URL` to the existing local PostgreSQL connection string. Run this command from the nested Feedoscope repository:

```bash
uv run python -m experiments.freshness_encoder_models export \
  --output artifacts/freshness_encoder_models/snapshot \
  --validation-size 150
```

The exporter sets `default_transaction_read_only=on` before it creates the connection pool. It verifies `transaction_read_only=on` before it reads effective labels.

## NAS run commands

The NAS repository is `/home/djipey/projects/Feedoscope/feedoscope`. Transfer only the experiment module and ignored snapshot:

```bash
rsync -a experiments/freshness_encoder_models.py \
  nas:/home/djipey/projects/Feedoscope/feedoscope/experiments/
rsync -a artifacts/freshness_encoder_models/snapshot/ \
  nas:/home/djipey/projects/Feedoscope/feedoscope/artifacts/freshness_encoder_models/snapshot/
```

Run models sequentially on the single RTX 3060. Parallel model processes compete for VRAM and do not provide a valid runtime comparison.

Run the active EmbeddingGemma control in float32. This matches its active encoder load path:

```bash
ssh nas 'cd /home/djipey/projects/Feedoscope/feedoscope && mkdir -p artifacts/freshness_encoder_models/results artifacts/freshness_encoder_models/embeddings && uv run python -m experiments.freshness_encoder_models run --snapshot artifacts/freshness_encoder_models/snapshot --model embeddinggemma --model-path models/relevance_encoder/google--embeddinggemma-300m --output artifacts/freshness_encoder_models/results/embeddinggemma.json --embeddings-dir artifacts/freshness_encoder_models/embeddings --batch-size 4 --dtype float32'
```

Run the remaining candidates with BF16:

```bash
ssh nas 'cd /home/djipey/projects/Feedoscope/feedoscope && for model in jina-small harrier qwen3-0.6b bekko-a8m bekko-a25m nemotron; do uv run python -m experiments.freshness_encoder_models run --snapshot artifacts/freshness_encoder_models/snapshot --model "$model" --output "artifacts/freshness_encoder_models/results/$model.json" --embeddings-dir artifacts/freshness_encoder_models/embeddings --batch-size 4 --dtype bfloat16 || exit; done'
```

A local model path records `source=local_unverified` and `loaded_revision=null`. Do not treat it as a verified pinned revision without a separate model-file hash check.

Copy the results and reusable experiment embeddings back after all runs:

```bash
rsync -a nas:/home/djipey/projects/Feedoscope/feedoscope/artifacts/freshness_encoder_models/results/ \
  artifacts/freshness_encoder_models/results/
rsync -a nas:/home/djipey/projects/Feedoscope/feedoscope/artifacts/freshness_encoder_models/embeddings/ \
  artifacts/freshness_encoder_models/embeddings/
```

## Results

| Model | RPS ↓ | Macro F1 ↑ | Weighted kappa ↑ | Log-duration MAE ↓ | Long-lived AUC ↑ | Encoding time | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EmbeddingGemma, unprompted | 0.165034 | 0.507387 | 0.423792 | 1.055460 | 0.831147 | reused | reused |
| EmbeddingGemma, classification prompt | 0.157290 | 0.525696 | 0.443821 | 1.037313 | **0.848951** | reused | reused |
| EmbeddingGemma, classification prompt + MLP | 0.191180 | 0.406944 | 0.440895 | 1.148463 | 0.803036 | reused | CPU |
| Harrier 0.6B, relevance instruction | 0.157779 | **0.547082** | **0.540731** | 1.043134 | 0.848201 | reused | reused |
| Harrier 0.6B, Freshness instruction | **0.150239** | 0.542998 | 0.527629 | 1.070556 | 0.846889 | 163.1 s | 1.385 GB |
| Jina v5 Nano Classification | 0.195847 | 0.448671 | 0.332781 | 1.235871 | 0.719078 | reused | reused |
| Nemotron 3 Embed 1B | 0.159809 | 0.605357 | 0.529247 | **1.000616** | 0.841454 | reused | reused |

## Reused-artifact screen

The stored relevance artifacts cover all 1,226 Freshness snapshot article IDs. This screen reused five complete vector artifacts. It used the frozen three-label Freshness snapshot and the active two-head classifier contract.

Harrier and Nemotron both improved the primary RPS over unprompted EmbeddingGemma. The classification-prompted EmbeddingGemma artifact improved RPS, but it remains the same base model.

Harrier now has a dedicated Freshness-instruction artifact. It reduced RPS from `0.157779` with the relevance instruction to `0.150239`. Its vector file is `artifacts/freshness_encoder_models/embeddings/harrier-0.6b-freshness_encoder_models_20260802T172933Z.npz`. Its SHA-256 is `37d76f24f9a7c59d4dbe047b7b6a3d770003879fe0bf8afd2a488cb688e82d05`.

The prompted EmbeddingGemma MLP screen used two cumulative 64-unit `MLPClassifier` heads with `alpha=0.0001`, early stopping, and the active class-weight exponent. It increased RPS from `0.157290` to `0.191180`. Do not tune a larger MLP. The simple logistic heads are clearly better on this holdout.

Jina Nano was worse on every primary Freshness metric. The stored artifacts do not cover Jina Small, Qwen3, or either Bekko model.

## Recommendation

Harrier with the Freshness instruction is the current screening winner on RPS. Keep unprompted EmbeddingGemma in production until Harrier passes an untouched later holdout. Run the remaining candidates only if a broader comparison remains useful.
