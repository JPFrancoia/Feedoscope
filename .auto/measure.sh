#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha256sum --check --quiet .auto/frozen.sha256
uv run --no-group infer python -m experiments.freshness.evaluate
