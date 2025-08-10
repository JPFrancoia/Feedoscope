FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS builder

# Install Python 3.12 and uv
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

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

FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS runtime

# Install Python 3.12 and PostgreSQL client
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Place executables in the environment at the front of the path
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /app

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy the relevant folders so that the application can run
COPY logging.conf ./
COPY custom_logging ./custom_logging
COPY feedoscope ./feedoscope
