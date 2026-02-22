# Dear AI – Mental Health Companion Backend

A conversational AI backend for a mental health companion, built with **FastAPI**, **Vertex AI (Gemini)**, and a **Graph RAG** pipeline backed by FalkorDB.

## Architecture

The system uses a **two-layer concurrent response** model over WebSocket:

```md
User Message
     │
     ├──► Layer 1 – Immediate  (gemini-2.0-flash-lite, <2 s)
     │         └──► { "layer": "immediate", "content": "...", "final": false }
     │
     └──► Layer 2 – Graph RAG  (retrieval + reasoning)
               └──► { "layer": "rag", "content": "...", "final": true }
```

| Layer | Model | Purpose | Cached |
|-------|-------|---------|--------|
| Immediate | `gemini-2.0-flash-lite` | Fast empathetic acknowledgement | No |
| Graph RAG | Full pipeline via FalkorDB | Retrieval-augmented response | Yes (Redis) |

Both layers run **concurrently** via `asyncio.create_task` and support **cancellation** when the user sends a new message or disconnects.

### Caching Strategy

Graph RAG responses are cached in Redis to avoid redundant pipeline executions:

- **Key**: `rag_cache:<sha256(user_message)>`
- **TTL**: Configurable via `RAG_CACHE_TTL_SECONDS` (default: 1 hour)
- **Miss**: Runs `run_graph_rag()` → stores result in Redis
- **Hit**: Returns cached response immediately
- **Failure**: Redis unavailability is non-fatal — falls through to live RAG

The cache wraps the RAG pipeline externally; no internal RAG logic is modified.

## Project Structure

```md
app/
├── api/v1/              # FastAPI route handlers
│   ├── auth.py          # Authentication endpoints
│   ├── chat.py          # Text, voice, and WebSocket chat
│   ├── conversations.py # Conversation CRUD
│   └── users.py         # User management
├── core/
│   ├── config.py        # Pydantic settings (env-driven)
│   ├── database.py      # Async SQLAlchemy engine + session
│   ├── dependencies.py  # FastAPI dependency injection
│   └── redis.py         # Async Redis client singleton
├── models/              # SQLAlchemy ORM models
├── repositories/        # Data access layer
├── services/
│   ├── auth/            # Authentication & JWT
│   ├── cache/
│   │   └── rag_cache.py # Redis caching wrapper for Graph RAG
│   ├── chat/            # Chat orchestration + handlers
│   ├── context/
│   │   ├── graph_rag.py # Graph RAG pipeline (FalkorDB)
│   │   └── summary.py   # Context summarisation
│   ├── emotion/         # Hume.ai emotion detection
│   ├── guardrails/      # Input/output safety filters
│   ├── llm/             # LLM abstraction (Vertex AI)
│   ├── speech/          # STT & TTS providers
│   └── users/           # User service layer
└── migrations/          # Alembic migration scripts
```

## Prerequisites

- **Python** ≥ 3.11
- **Docker** & **Docker Compose** (for Postgres, Redis, FalkorDB)
- **Google Cloud** project with Vertex AI API enabled
- **Hume.ai** API key for STT, TTS, and emotion detection

## Getting Started

### 1. Clone & install

```bash
git clone <repo-url> && cd dearai-backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your own keys and connection strings
```

Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Async PostgreSQL connection string | `postgresql+asyncpg://postgres:postgres@localhost:5432/mental_health_companion` |
| `REDIS_URL` | Redis for RAG caching | `redis://localhost:6380` |
| `RAG_CACHE_TTL_SECONDS` | Cache lifetime in seconds | `3600` |
| `VERTEX_PROJECT` | GCP project ID for Vertex AI | — |
| `VERTEX_LOCATION` | GCP region | `us-central1` |
| `LLM_MODEL` | Default Gemini model | `gemini-2.0-flash` |
| `HUME_API_KEY` | Hume.ai API key (STT + TTS + emotion) | — |
| `HUME_SECRET_KEY` | Hume.ai secret key | — |
| `JWT_SECRET_KEY` | JWT signing secret | *change in production* |
| `FALKOR_HOST` / `FALKOR_PORT` | FalkorDB connection | `localhost:6379` |

### 3. Start infrastructure

```bash
docker compose up -d
```

This starts:

| Service | Container | Host Port |
|---------|-----------|-----------|
| PostgreSQL (pgvector) | `dear-ai-postgres` | 5432 |
| Redis (RAG cache) | `dear-ai-redis` | 6380 |
| FalkorDB (graph DB) | `dear-ai-falkordb` | 6379 |

### 4. Run database migrations

```bash
alembic upgrade head
```

### 5. Start the server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. OpenAPI docs at `/docs`.

## WebSocket Protocol

Connect to `ws://localhost:8000/chat/ws` and send JSON messages:

```json
{ "content": "I've been feeling anxious lately" }
```

The server responds with two messages per turn:

```json
{ "layer": "immediate", "content": "I hear you...", "final": false }
```

```json
{ "layer": "rag", "content": "Based on what I know...", "final": true }
```

Sending a new message while a previous response is still processing will **cancel** the in-flight tasks and start fresh.

## Development

### Run tests

```bash
pytest
```

### Lint & format

```bash
ruff check .
ruff format .
```

### Type checking

```bash
mypy app/
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web framework | FastAPI |
| LLM | Google Vertex AI (Gemini) |
| Graph database | FalkorDB |
| Relational database | PostgreSQL + pgvector |
| Cache | Redis |
| ORM | SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| Auth | JWT (python-jose) |
| Speech | Hume.ai (STT + TTS) |
| Emotion | Hume.ai |
| Safety | Guardrails AI |

## License

Private – all rights reserved.
