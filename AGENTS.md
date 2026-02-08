# AGENTS.md - Feedoscope

## Project Overview

Feedoscope is a Python ML pipeline that scores RSS articles from a Miniflux database
by relevance (fine-tuned ModernBERT) and time sensitivity (local Ministral-8B via
llama.cpp). It reads from and writes to a PostgreSQL database (Miniflux schema with
custom extensions). Deployed on Kubernetes via Docker images with CUDA/GPU support.

## Build & Run Commands

Package manager: **uv** (lockfile: `uv.lock`, Python 3.12.11 pinned in `.python-version`)

```bash
# Install all dependency groups (dev + train + infer)
make install          # or: uv sync

# Install without infer group (no llama-cpp-python, faster)
make install_dev      # or: uv sync --no-group infer

# Add a dependency
uv add <package>
```

## Lint / Format / Type-check

```bash
# Type-check (mypy)
make lint             # or: uv run --no-group infer mypy .

# Format (black + isort)
make format           # or: uv run black . && uv run isort .

# Lint (ruff) - configured but not in Makefile yet
uv run ruff check .
uv run ruff check --fix .
```

Run `make format` before committing. The isort profile is set to `black` for
compatibility. No pre-commit hooks are currently active.

## Tests

There are **no tests** in this project currently. No pytest configuration exists.
If adding tests:
- Place them in a `tests/` directory at the project root
- Use pytest: `uv run pytest tests/`
- Run a single test: `uv run pytest tests/test_file.py::test_function -v`
- The `tests` module is already listed in isort's `known_localproject`

## Run Commands (require DATABASE_URL env var)

```bash
make train            # Train the relevance model
make infer            # Run relevance inference only
make time             # Run time sensitivity inference only
make full_infer       # Run full pipeline (relevance + time sensitivity)
```

All run commands set `LOGGING_CONFIG=dev_logging.conf` for colored console output.

## Database Migrations

Uses golang-migrate CLI (not a Python tool). Migration files in `db/migrations/`.

```bash
make up               # Apply next migration
make down             # Roll back one migration
make up_all           # Apply all pending migrations
make down_all         # Roll back all migrations
```

## Docker

```bash
make pkg              # Build, tag, push to local registry (192.168.0.13:32000)
```

Multi-stage Dockerfile with CUDA 12.8 support. No docker-compose.

## Code Style Guidelines

### Python Version

Target Python 3.12+. Use modern syntax: `list[str]` not `List[str]`,
`dict[str, Any]` not `Dict[str, Any]`, `str | None` union syntax.

### Formatting

- **black** defaults: 88-char line length, double quotes
- **isort** with `profile = "black"`, custom section ordering:
  `FUTURE > STDLIB > THIRDPARTY > FIRSTPARTY > LOCALPROJECT > LOCALFOLDER`
- Local packages in isort: `tests`, `custom_logging`, `feedoscope`

### Import Ordering

```python
# 1. Standard library
import asyncio
import logging
import os

# 2. Third-party
import numpy as np
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification

# 3. Local project
from custom_logging import init_logging
from feedoscope import config
from feedoscope.entities import Article
```

### Naming Conventions

- `snake_case` for functions, variables, parameters, modules
- `PascalCase` for classes (Pydantic models, enums, callbacks)
- `UPPER_SNAKE_CASE` for module-level constants
- File names are always `snake_case.py`

### Type Annotations

Fully annotate all function signatures including return types:

```python
async def get_articles(number_of_days: int = 14) -> list[Article]:
    ...

def find_latest_model(model_name: str, clean_old_models: bool = True) -> str:
    ...
```

- Use `typing.Literal` for constrained values: `Literal[1, 2, 3, 4, 5]`
- Use `typing.LiteralString` + `cast()` for SQL query safety
- Use `Optional[X]` or `X | None` for nullable fields
- Add `# type: ignore[import]` for untyped third-party packages (`cleantext`,
  `datasets`, `llama_cpp`)

### Docstrings

Google style with `Args:`, `Returns:`, `Raises:` sections. Required for public
functions; short/obvious helper functions may omit them:

```python
def find_latest_model(model_name: str, clean_old_models: bool = True) -> str:
    """Find the latest saved model to use for inference.

    Args:
        model_name: family of model to use
        clean_old_models: if True, delete all older models

    Returns:
        The path to the latest model directory.

    Raises:
        FileNotFoundError: if no trained models are found
    """
```

### Error Handling

- `assert` for configuration validation at module level (`config.py`)
- `RuntimeError` for critical infrastructure failures (e.g., no GPU)
- `FileNotFoundError` for missing required resources
- `try/except` with specific exceptions; use `continue` in loops to skip failures
- Log errors with `logger.error()` or `logger.exception()` before raising/continuing

### Logging

- Module-level logger: `logger = logging.getLogger(__name__)` in every module
- Use f-strings in log messages (not lazy `%` formatting)
- Two configs: `logging.conf` (production, JSON) and `dev_logging.conf` (dev, colored)
- Custom `init_logging()` from `custom_logging` package

### Architecture Patterns

- **Functional style**: logic lives in module-level async functions, not service classes
- **Classes only for data**: Pydantic `BaseModel` subclasses, enums, and framework
  callbacks
- **Async throughout**: all DB operations and entry points are `async`;
  `asyncio.run(main())` at `__main__` blocks
- **Raw SQL**: queries stored in `.sql` files under `feedoscope/data_registry/sql/`,
  loaded via `importlib.resources.files()` with `@lru_cache`
- **psycopg3 async**: module-level `AsyncConnectionPool` singleton, accessed via
  `async with global_pool.connection() as conn`
- **Parameterized queries**: use psycopg's `%(param)s` named parameter syntax
- **Config via env vars**: all configuration read from environment in `config.py`

### Key Directories

```
feedoscope/                  # Main application package
  main.py                    # Orchestrator: relevance + time sensitivity
  config.py                  # Environment-based configuration
  entities.py                # Pydantic data models
  utils.py                   # Text cleaning utilities
  llm_learn.py               # Model training (fine-tuning ModernBERT)
  llm_infer.py               # Relevance inference
  infer_time_sensitivity.py  # Time sensitivity via local Llama GGUF
  data_registry/
    data_registry.py         # All DB access functions
    sql/                     # Raw .sql query files
custom_logging/              # Logging setup package
db/migrations/               # golang-migrate SQL migrations
models/                      # (gitignored) trained model checkpoints + GGUF files
embeddings/                  # Pre-computed numpy embeddings
```
