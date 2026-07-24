FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv

WORKDIR /app

# Run as a non-root user for security.
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown appuser:appgroup /app
USER appuser

# Set HOME and cache directories to the app directory which appuser owns
ENV HOME=/app
ENV UV_CACHE_DIR=/app/.uv-cache

# Copy manifests and install dependencies as appuser
COPY --chown=appuser:appgroup pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen

# Copy app code
COPY --chown=appuser:appgroup app/ ./app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PORT="8080"
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]
