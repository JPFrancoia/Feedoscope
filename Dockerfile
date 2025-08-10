FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /bin/

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --locked --no-install-project --no-editable --no-group dev

# Copy the project into the intermediate image
ADD . /app

# Sync the project
RUN uv sync --locked --no-editable --no-group dev

FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y libpq-dev postgresql-client build-essential

# Add NVIDIA's CUDA repo
RUN apt-get update && apt-get install -y wget gnupg ca-certificates && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-runtime-12-8
    # rm -rf /var/lib/apt/lists/*

# Place executables in the environment at the front of the path
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy the relevant folders so that the application can run
COPY logging.conf ./
COPY custom_logging ./custom_logging
COPY feedoscope ./feedoscope
