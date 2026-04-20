# Urgency Karpathy Loop

This is the urgency equivalent of the `pi_research` relevance loop: export one
frozen local Parquet snapshot from the database, then let the external coding
agent drive repeated experiments from that fixed snapshot.

## Purpose

The loop exists to optimize urgency using only trusted read-tagged articles,
without relying on one-shot Ministral labels.

The first implementation is experiment-only:

- export a frozen urgency snapshot to local Parquet files
- run exactly one experiment configuration at a time from that snapshot
- append results to the loop logs
- let the external agent choose the next run

It does not promote a winner into production yet.

## Files

Root loop files:

- `program.md`: authoritative brief for the external agent
- `autoresearch.md`: current run objective and queue
- `autoresearch.ideas.md`: scratchpad for follow-up ideas
- `autoresearch.sh`: one-experiment runner
- `autoresearch.jsonl`: append-only machine-readable run log
- `results.md`: human-readable results summary

Experiment code:

- `feedoscope/urgency_experiments/export_snapshot.py`
- `feedoscope/urgency_experiments/harness.py`
- `feedoscope/urgency_experiments/metrics.py`
- `feedoscope/urgency_experiments/sql/get_read_articles_with_urgency_tags.sql`

## Frozen Snapshot Workflow

The snapshot exporter pulls only:

- `entries.status = 'read'`
- user tags `0-urgency` or `1-urgency`

The tag value becomes the binary label:

- `0-urgency` -> 0
- `1-urgency` -> 1

Use:

```bash
LOGGING_CONFIG=dev_logging.conf uv run python -m feedoscope.urgency_experiments.export_snapshot
```

This writes a frozen dataset under:

```text
artifacts/urgency_autoresearch/<snapshot_id>/
```

with:

- `snapshot.parquet`
- `train.parquet`
- `eval.parquet`
- `metadata.json`
- `latest_snapshot.txt`

That frozen snapshot is the only dataset the experiment loop should use after
export.

## Experiment Runner

`autoresearch.sh` runs exactly one configuration against the frozen snapshot.

It supports at least:

- transformer fine-tuning experiments
- frozen embedding plus logistic-regression experiments
- `title_head` and `single_blob` text prep
- `full` and `balanced` train modes

The external agent is expected to edit environment variables for each run and
append the outcome to the loop logs, just like in `pi_research`.

## Running The Loop

```bash
LOGGING_CONFIG=dev_logging.conf ./autoresearch.sh
```

## Output Layout

Snapshot export writes under `artifacts/urgency_autoresearch/<snapshot_id>/`.

Each experiment run writes under:

```text
artifacts/urgency_autoresearch/runs/<run_name>/
```

with saved model or classifier artifacts plus `results.json`.

## Current Limits

- The loop uses one frozen train/eval split per exported snapshot.
- It does not yet write the winner into production inference.
- It is intentionally separated from the old fixed local comparison script; the
  external-agent loop is now the primary experiment path.
