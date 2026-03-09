# Dear AI -- Mental Health Companion Backend

A conversational AI backend for a mental health companion, built with **FastAPI**, **Vertex AI (Gemini)**, **Graph RAG** (FalkorDB), and **Sarvam AI** for multilingual voice support.

## Architecture

### Two-Layer Concurrent Response (WebSocket)

Every user message triggers two concurrent pipelines over a single WebSocket connection:

```md
User Message
     |
     +---> Layer 1 -- Immediate  (gemini-2.0-flash-lite, <2s)
     |         +---> {"type": "layer1", "content": "...", "conversation_id": "..."}
     |
     +---> Graph RAG Retrieval   (FalkorDB -- runs in parallel with Layer 1)
                |
                +---> Layer 2 -- Continuation (gemini-2.0-flash + RAG context)
                          +---> {"type": "layer2", "content": "...", "conversation_id": "..."}
```

| Layer | Model | Purpose | Latency |
|-------|-------|---------|---------|
| Layer 1 (Immediate) | `gemini-2.0-flash-lite` | Fast empathetic acknowledgement | <2s |
| Graph RAG Retrieval | FalkorDB Cypher queries | Fetch user context + mental health KB | ~1-3s |
| Layer 2 (Continuation) | `gemini-2.0-flash` | Context-rich response continuing Layer 1 | ~2-4s |

Both layers run **concurrently** via `asyncio.create_task`. Sending a new message while a previous response is in-flight **cancels** pending tasks and starts fresh.

### Graph RAG Pipeline

The Graph RAG pipeline is a `StateGraph`-based state machine:

```md
router --> extract_entities --> graph_retrieval --> llm_generation --> graph_update
  |                                                                       |
  +-- (direct path, no graph needed) --> llm_generation ---------> graph_update
```

- **Router**: LLM classifies whether the message needs graph context (falls back to keyword heuristics)
- **Entity extraction**: LLM extracts people, emotions, conditions, topics as structured JSON
- **Graph retrieval**: Cypher queries against FalkorDB for coping strategies, therapy techniques, user context
- **Graph update**: Stores new entities/relationships back into the user's context graph (fire-and-forget)

### Knowledge Graph (FalkorDB)

Two graph layers seeded at startup:

- **Mental health KB**: 8 conditions, 10 coping strategies (with steps), 3 therapy techniques (CBT/DBT/ACT), 3 crisis resources (988 Lifeline, Vandrevala Foundation, iCall) -- all connected via `HELPED_BY` and `TREATED_BY` relationships
- **User context graph**: Per-user nodes for mentioned people, discussed topics, and expressed emotions -- built up over conversations

### Caching (Redis)

Graph RAG responses can be cached via the `rag_cache` module:

- **Key**: `rag_cache:<sha256(user_id + user_message)>`
- **TTL**: Configurable via `RAG_CACHE_TTL_SECONDS` (default: 1 hour)
- **Failure**: Redis unavailability is non-fatal -- falls through to live RAG

### Voice Pipeline

Supports multilingual voice conversations via Sarvam AI:

- **STT**: Sarvam Saaras v3 -- 22 Indian languages + English, automatic language detection
- **TTS**: Sarvam Bulbul v3 -- natural-sounding speech synthesis
- **Fallback**: Hume AI STT/TTS available as alternative providers

### Summary Generation

Conversation summaries are auto-generated after every N messages (configurable via `SUMMARY_INTERVAL`). Summaries are LLM-powered and stored in the database, then injected into Layer 1 prompts to maintain long-term context without sending full history.

## Project Structure

```md
app/
+-- api/v1/                  # FastAPI route handlers
|   +-- auth.py              # Registration, login, token refresh
|   +-- chat.py              # REST chat + WebSocket (two-layer)
|   +-- conversations.py     # Conversation CRUD
|   +-- users.py             # User profile management
+-- core/
|   +-- config.py            # Pydantic settings (env-driven)
|   +-- database.py          # Async SQLAlchemy engine + session
|   +-- dependencies.py      # FastAPI DI (CurrentUser, DbSession)
|   +-- redis.py             # Async Redis client singleton
+-- models/                  # SQLAlchemy ORM models
|   +-- user.py              # User
|   +-- auth.py              # RefreshToken
|   +-- conversation.py      # Conversation, Message (pgvector), Summary
+-- services/
|   +-- auth/                # JWT + bcrypt auth
|   +-- cache/
|   |   +-- rag_cache.py     # Redis caching wrapper for Graph RAG
|   +-- chat/
|   |   +-- handlers/        # Text + voice chat handlers
|   |   +-- schemas.py       # Request/response models
|   |   +-- service.py       # Chat orchestration
|   +-- context/
|   |   +-- graph_rag.py     # Graph RAG pipeline (StateGraph)
|   |   +-- graph_schema.py  # FalkorDB indexes + KB seed data
|   |   +-- summary.py       # Summary context provider
|   |   +-- summary_generator.py  # Auto summary generation
|   +-- emotion/             # Hume AI emotion detection
|   +-- guardrails/          # Input/output safety (length, injection, crisis)
|   +-- llm/                 # LLM abstraction (Vertex AI / Gemini)
|   +-- speech/
|   |   +-- stt/             # STT providers (Sarvam, Hume)
|   |   +-- tts/             # TTS providers (Sarvam, Hume)
|   +-- users/               # User service layer
+-- migrations/              # Alembic migration scripts
```

## Prerequisites

- **Python** >= 3.11
- **uv** (recommended) or pip
- **Docker** & **Docker Compose** (for Postgres, Redis, FalkorDB)
- **Google Cloud** project with Vertex AI API enabled
- **Sarvam AI** API key for STT/TTS

## Getting Started

### 1. Clone & install

```bash
git clone <repo-url> && cd dearai-backend
uv sync
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys and connection strings
```

Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Async PostgreSQL connection string | `postgresql+asyncpg://postgres:postgres@localhost:5432/dearai` |
| `REDIS_URL` | Redis for RAG caching | `redis://localhost:6380` |
| `RAG_CACHE_TTL_SECONDS` | Cache lifetime in seconds | `3600` |
| `VERTEX_PROJECT` | GCP project ID for Vertex AI | -- |
| `VERTEX_LOCATION` | GCP region | `us-central1` |
| `LLM_MODEL` | Gemini model for Layer 2 / RAG | `gemini-2.0-flash` |
| `SARVAM_API_KEY` | Sarvam AI API key (STT + TTS) | -- |
| `HUME_API_KEY` | Hume AI API key (emotion + fallback STT/TTS) | -- |
| `JWT_SECRET_KEY` | JWT signing secret | *change in production* |
| `FALKOR_HOST` / `FALKOR_PORT` | FalkorDB connection | `localhost:6379` |
| `SUMMARY_INTERVAL` | Generate summary every N messages | `10` |

### 3. Start infrastructure

```bash
make up
```

This starts:

| Service | Container | Host Port |
|---------|-----------|-----------|
| PostgreSQL (pgvector) | `dear-ai-postgres` | 5432 |
| Redis (RAG cache) | `dear-ai-redis` | 6380 |
| FalkorDB (graph DB) | `dear-ai-falkordb` | 6379 |

### 4. Run database migrations

```bash
make migrate
```

### 5. Start the dev server

```bash
make dev
```

The API will be available at `http://localhost:8000`. OpenAPI docs at `/docs`.

## Makefile Targets

```md
make install     Install all dependencies (including dev) via uv
make dev         Start dev server with hot-reload
make run         Start production server
make up          Start Postgres, Redis, FalkorDB containers
make down        Stop all containers
make up-all      Start all containers including the app
make logs        Tail docker compose logs
make migrate     Run Alembic migrations to head
make migration   Auto-generate a new migration (usage: make migration msg="add foo")
make migrate-down  Downgrade one migration revision
make test        Run test suite
make test-cov    Run tests with coverage report
make lint        Run linter (ruff check)
make format      Auto-format code (ruff)
make typecheck   Run mypy type checking
make check       Run lint + typecheck + tests
make clean       Remove build artifacts and caches
```

## WebSocket Protocol

Connect to `ws://localhost:8000/chat/ws?token=<jwt>`.

Alternatively, connect without a query param and send an auth message first:

```json
{"type": "auth", "token": "<jwt>"}
```

Server responds:

```json
{"type": "auth_ok", "user_id": "..."}
```

### Text Messages

Send:

```json
{"type": "message", "conversation_id": "<uuid>", "content": "I've been feeling anxious"}
```

Receive (two messages per turn):

```json
{"type": "layer1", "content": "I hear you...", "conversation_id": "..."}
```

```json
{"type": "layer2", "content": "Anxiety can feel overwhelming...", "conversation_id": "..."}
```

### Voice Messages

Send:

```json
{"type": "voice", "conversation_id": "<uuid>", "audio": "<base64-encoded-audio>"}
```

Receive:

```json
{"type": "transcription", "content": "...", "language": "en-IN", "conversation_id": "..."}
```

```json
{"type": "layer1", "content": "...", "conversation_id": "..."}
```

```json
{"type": "layer2", "content": "...", "conversation_id": "..."}
```

```json
{"type": "audio", "audio": "<base64-wav>", "conversation_id": "..."}
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register a new user |
| POST | `/auth/login` | Login and receive tokens |
| POST | `/auth/refresh` | Refresh access token |
| GET | `/users/me` | Get current user profile |
| GET | `/conversations` | List conversations |
| POST | `/conversations` | Create a conversation |
| GET | `/conversations/{id}` | Get conversation details |
| DELETE | `/conversations/{id}` | Delete a conversation |
| POST | `/chat/text` | Send text message (REST) |
| POST | `/chat/voice` | Send voice message, get text response |
| POST | `/chat/voice/audio` | Send voice message, get audio response |
| WS | `/chat/ws` | WebSocket two-layer chat |
| GET | `/health` | Health check |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web framework | FastAPI + fastapi-cli |
| LLM | Google Vertex AI (Gemini 2.0 Flash / Flash Lite) |
| Graph database | FalkorDB |
| Relational database | PostgreSQL + pgvector |
| Cache | Redis |
| ORM | SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| Auth | JWT (python-jose + bcrypt) |
| Speech (STT) | Sarvam AI Saaras v3 |
| Speech (TTS) | Sarvam AI Bulbul v3 |
| Emotion detection | Hume AI |
| Input/output safety | Custom guardrails (length, injection, crisis detection) |
| Structured logging | structlog |
| Package management | uv |

## License

Private -- all rights reserved.
