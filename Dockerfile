FROM python:3.12-slim AS builder

# Install build dependencies required for Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /bin/

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Disable UV cache to avoid storing package downloads
ENV UV_NO_CACHE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install dependencies (exclude only dev group)
RUN uv sync --locked --no-install-project --no-editable --no-group dev

# Copy the project into the intermediate image
ADD . /app

# Sync the project (exclude only dev group)
RUN uv sync --locked --no-editable --no-group dev

FROM python:3.12-slim AS runtime

# Install runtime dependencies and CUDA in a single layer
# The CUDA libraries are necessary to compile llama-cpp-python with GPU support.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    wget \
    gnupg \
    ca-certificates \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-runtime-12-8 \
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
