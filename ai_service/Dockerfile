# ────────────────────────────────────────────────────────────
# Stage 1: Install dependencies
# ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS deps

# Install uv for fast, reproducible dependency resolution.
# Pin to a specific version so builds are deterministic.
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the lock file and project manifest first so dependency
# installation is cached when source files change.
COPY pyproject.toml uv.lock ./

# Install production dependencies into the system Python site-packages.
# --no-dev  → skip dev-only packages (pytest, ruff, etc.)
# --frozen  → respect the lock file exactly (no silent upgrades)
# --system  → install into /usr/local/lib rather than a venv
RUN uv sync --no-dev --frozen --system

# ────────────────────────────────────────────────────────────
# Stage 2: Run
# ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from the deps stage.
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

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
