FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /bin/

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --locked --no-install-project --no-editable --no-group dev --no-group train

# Copy the project into the intermediate image
ADD . /app

# Sync the project
RUN uv sync --locked --no-editable --no-group dev --no-group train

FROM python:3.12-slim AS runtime

# Place executables in the environment at the front of the path
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy the relevant folders so that the application can run
COPY logging.conf ./
COPY custom_logging ./custom_logging
COPY feedoscope ./feedoscope

# Run the application
CMD ["python", "-m", "feedoscope.pu_learn"]
