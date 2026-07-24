# ── Stage 1: Build React frontend ────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ── Stage 2: Python backend + built frontend ────────────────
FROM python:3.13-slim AS runtime
WORKDIR /code

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install backend dependencies
COPY backend/pyproject.toml backend/uv.lock backend/.python-version ./
RUN uv pip compile pyproject.toml -o requirements.txt && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy backend source
COPY backend/app ./app
COPY backend/engine ./engine
COPY backend/main.py ./app

ENV STATIC_DIR=/code/static
ENV ENVIRONMENT=production

# Copy built frontend dist from stage 1
COPY --from=frontend-build /app/frontend/dist ./static

EXPOSE 8000

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
