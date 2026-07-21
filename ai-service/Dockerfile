# ────────────────────────────────────────────────────────────
# Stage 1: Install dependencies
# ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS deps

# Install build tools required by native extensions (e.g., hnswlib needs g++).
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast, reproducible dependency resolution.
# Pin to a specific version so builds are deterministic.
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the lock file, project manifest, and readme first so dependency
# installation is cached when source files change.
# NOTE: README.md is required because pyproject.toml references it and
# hatchling validates its existence during dependency resolution.
COPY pyproject.toml uv.lock README.md ./

# Install production dependencies into the project venv.
# --no-dev  → skip dev-only packages (pytest, ruff, etc.)
# --frozen  → respect the lock file exactly (no silent upgrades)
RUN uv sync --no-dev --frozen

# ────────────────────────────────────────────────────────────
# Stage 2: Run
# ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy the venv created by uv sync from the deps stage.
COPY --from=deps /app/.venv /app/.venv

# Ensure the venv's bin and site-packages are used at runtime.
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Copy application source code.
COPY app/ ./app/

# Run as a non-root user for security.
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

EXPOSE 8000

# Uvicorn with:
#   --workers 1      → single worker; asyncio event loop is already concurrent
#   --no-access-log  → structured logs come from the app; avoid double-logging
CMD ["uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--no-access-log"]