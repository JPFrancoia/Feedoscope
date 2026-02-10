FROM python:3.12-slim AS builder

# Install build dependencies (libpq-dev needed for psycopg compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /bin/

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Disable UV cache to avoid storing package downloads
ENV UV_NO_CACHE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install dependencies (exclude dev and infer groups — llama-cpp-python is only used locally for distillation)
RUN uv sync --locked --no-install-project --no-editable --no-group dev --no-group infer

# Copy the project into the intermediate image
ADD . /app

# Sync the project (exclude dev and infer groups)
RUN uv sync --locked --no-editable --no-group dev --no-group infer

FROM python:3.12-slim AS runtime

# Install runtime dependencies (libpq for psycopg, postgresql-client for migrations)
# CUDA runtime libs (cudart, cublas, etc.) are bundled via PyTorch's nvidia-* pip packages
RUN apt-get update && apt-get install -y \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Place executables in the environment at the front of the path
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy the relevant folders so that the application can run
COPY logging.conf Makefile ./
COPY custom_logging ./custom_logging
COPY feedoscope ./feedoscope
