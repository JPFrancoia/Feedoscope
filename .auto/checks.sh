#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha256sum --check --quiet .auto/frozen.sha256
uv run --no-group infer python -m py_compile experiments/freshness/candidate.py
if grep -Eq 'psycopg|postgres|pgcli|psql|subprocess|requests|httpx|urllib|test_ids|teacher_labels' experiments/freshness/candidate.py; then
  echo "candidate.py references forbidden data or external services" >&2
  exit 1
fi
unexpected=$(git diff --name-only -- . ':!.auto' ':!experiments/freshness/candidate.py')
if [[ -n "$unexpected" ]]; then
  echo "experiment changed files outside candidate.py:" >&2
  echo "$unexpected" >&2
  exit 1
fi
