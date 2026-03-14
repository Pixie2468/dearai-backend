# Dear AI -- Mental Health Companion

Comprehensive technical documentation for the Dear AI backend and frontend.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Two-Layer Concurrent Response System](#two-layer-concurrent-response-system)
6. [Database Schema](#database-schema)
7. [API Reference](#api-reference)
8. [WebSocket Protocol](#websocket-protocol)
9. [Authentication Flow](#authentication-flow)
10. [Graph RAG Pipeline](#graph-rag-pipeline)
11. [Voice Pipeline](#voice-pipeline)
12. [Guardrails System](#guardrails-system)
13. [Caching Strategy](#caching-strategy)
14. [Frontend (Streamlit)](#frontend-streamlit)
15. [Configuration & Environment Variables](#configuration--environment-variables)
16. [Infrastructure (Docker)](#infrastructure-docker)
17. [Development Setup](#development-setup)
18. [Database Migrations](#database-migrations)
19. [Known Limitations & Technical Debt](#known-limitations--technical-debt)

---

## Project Overview

Dear AI is a mental health companion that provides empathetic, context-aware conversational support. It features:

- **Two-layer concurrent responses**: Every user message triggers two parallel LLM pipelines -- a fast empathetic acknowledgement and a deeper, context-rich therapeutic response.
- **Graph RAG**: Long-term memory and context retrieval using a knowledge graph (FalkorDB) combined with retrieval-augmented generation.
- **Multilingual voice support**: Speech-to-text and text-to-speech via Sarvam AI (primary) and Hume AI (fallback), supporting 22+ Indian languages.
- **Crisis detection**: Input/output guardrails that detect crisis situations and append emergency resources.
- **Two conversation modes**: "Friend" mode for casual supportive chat, and "Therapy" mode for structured CBT/DBT/ACT-based interventions.

---

## Architecture

```
                                     +------------------+
                                     |   Streamlit      |
                                     |   Frontend       |
                                     +--------+---------+
                                              |
                               REST + WebSocket (HTTP/WS)
                                              |
                                     +--------v---------+
                                     |   FastAPI         |
                                     |   Backend         |
                                     +--------+---------+
                                              |
                   +-----------+--------------+--------------+-----------+
                   |           |              |              |           |
            +------v---+ +----v-----+ +------v------+ +----v----+ +---v------+
            |PostgreSQL| |  Redis   | | FalkorDB    | |Vertex AI| |Sarvam/   |
            |+ pgvector| |  Cache   | | Graph DB    | | Gemini  | |Hume AI   |
            +----------+ +----------+ +-------------+ +---------+ +----------+
```

### Request Flow (Text Chat via WebSocket)

1. Client sends a message over WebSocket
2. Backend validates input via guardrails
3. Two concurrent tasks are launched:
   - **Layer 1** (fast): `gemini-2.0-flash-lite` generates a 1-2 sentence acknowledgement
   - **Layer 2** (deep): Graph RAG retrieves context from FalkorDB, then `gemini-2.0-flash` generates a detailed response
4. Each layer streams chunks back to the client as they're generated
5. After Layer 2 completes, extracted entities are stored back into the graph (fire-and-forget)
6. Both responses are persisted as messages in PostgreSQL

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python >= 3.11 |
| Package Manager | uv |
| Web Framework | FastAPI (async) |
| Database | PostgreSQL 16 + pgvector |
| Cache | Redis 7 |
| Graph Database | FalkorDB |
| ORM | SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| LLM | Google Vertex AI (Gemini 2.0 Flash / Flash Lite) |
| Speech-to-Text | Sarvam AI (primary), Hume AI (fallback) |
| Text-to-Speech | Sarvam AI (primary), Hume AI (fallback) |
| Emotion Detection | Hume AI (prosody analysis) |
| Frontend | Streamlit |
| Auth | JWT (access + refresh tokens) with bcrypt |
| Logging | structlog |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
dearai-backend/
|-- app/                          # Main application package
|   |-- __init__.py
|   |-- main.py                   # FastAPI application factory & lifespan
|   |-- core/
|   |   |-- config.py             # Pydantic Settings (env vars)
|   |   |-- database.py           # Async SQLAlchemy engine & session
|   |   |-- security.py           # JWT creation, password hashing (bcrypt)
|   |   |-- dependencies.py       # FastAPI dependency injection (get_db, get_current_user)
|   |-- models/
|   |   |-- base.py               # Declarative base
|   |   |-- user.py               # User model
|   |   |-- conversation.py       # Conversation, Message, Summary models
|   |   |-- auth.py               # RefreshToken model
|   |-- services/
|   |   |-- auth/                 # Registration, login, token lifecycle
|   |   |   |-- router.py
|   |   |   |-- service.py
|   |   |   |-- schemas.py
|   |   |-- users/                # User profile CRUD
|   |   |   |-- router.py
|   |   |   |-- service.py
|   |   |   |-- schemas.py
|   |   |-- conversations/        # Conversation & message management
|   |   |   |-- router.py
|   |   |   |-- service.py
|   |   |   |-- schemas.py
|   |   |-- chat/                 # Chat orchestration (text + voice + WS)
|   |   |   |-- router.py         # REST endpoints + WebSocket handler
|   |   |   |-- schemas.py
|   |   |   |-- handlers/
|   |   |       |-- text.py       # Text chat handler
|   |   |       |-- voice.py      # Voice chat handler
|   |   |-- llm/                  # LLM abstraction layer
|   |   |   |-- base.py           # BaseLLM abstract class
|   |   |   |-- vertex.py         # Vertex AI / Gemini implementation
|   |   |   |-- factory.py        # LLMFactory
|   |   |   |-- prompts.py        # System prompts for friend/therapy modes
|   |   |-- context/              # Graph RAG + conversation summaries
|   |   |   |-- graph_rag.py      # LangGraph state machine for RAG
|   |   |   |-- knowledge_base.py # Seeds the FalkorDB knowledge graph
|   |   |   |-- provider.py       # Context provider interface
|   |   |   |-- summary.py        # Periodic conversation summarization
|   |   |-- cache/                # Redis RAG response caching
|   |   |   |-- service.py
|   |   |-- speech/
|   |   |   |-- stt/              # Speech-to-text providers
|   |   |   |   |-- base.py
|   |   |   |   |-- sarvam.py
|   |   |   |   |-- hume.py
|   |   |   |-- tts/              # Text-to-speech providers
|   |   |       |-- base.py
|   |   |       |-- sarvam.py
|   |   |       |-- hume.py
|   |   |-- emotion/              # Emotion detection (Hume prosody)
|   |   |   |-- service.py
|   |   |-- guardrails/           # Input/output validation & crisis detection
|   |       |-- service.py
|-- frontend/
|   |-- app.py                    # Streamlit frontend application
|   |-- requirements.txt          # Frontend-specific dependencies
|-- migrations/                   # Alembic database migrations
|   |-- versions/
|   |-- env.py
|-- docker-compose.yml
|-- Dockerfile
|-- pyproject.toml
|-- alembic.ini
```

---

## Two-Layer Concurrent Response System

The core innovation of Dear AI is the two-layer response architecture. When a user sends a message, two LLM pipelines run concurrently:

### Layer 1 -- Fast Acknowledgement

| Property | Value |
|----------|-------|
| Model | `gemini-2.0-flash-lite` |
| Target Latency | < 2 seconds |
| Purpose | Immediate empathetic acknowledgement |
| Output | 1-2 sentences |
| Context | Minimal (recent messages only) |

### Layer 2 -- Deep Response

| Property | Value |
|----------|-------|
| Model | `gemini-2.0-flash` |
| Target Latency | 2-4 seconds |
| Purpose | Detailed, therapeutic response |
| Output | Multi-paragraph, structured |
| Context | Full Graph RAG context (entities, coping strategies, history) |

### Cancellation Behavior

Sending a new message while previous tasks are in-flight **cancels** them via `asyncio.Task.cancel()`. This ensures the UI stays responsive and avoids stale responses.

### System Prompts

Two conversation modes with distinct personality prompts:

- **Friend mode**: Casual, warm, supportive tone. Validates feelings, shares relatable perspectives.
- **Therapy mode**: Structured, evidence-based. Uses CBT, DBT, and ACT techniques. Asks reflective questions.

---

## Database Schema

### Entity Relationships

```
User (1) ----> (N) Conversation (1) ----> (N) Message
  |                     |
  |                     +----> (N) Summary
  +----> (N) RefreshToken
```

### User

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | UUID | PK, server-default |
| `full_name` | String(100) | NOT NULL |
| `email` | String(255) | UNIQUE, indexed |
| `password_hash` | String(255) | NOT NULL |
| `gender` | Enum(male, female, other, prefer_not_to_say) | nullable |
| `age` | Integer | nullable |
| `created_at` | DateTime (UTC) | auto |

### RefreshToken

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | UUID | PK |
| `user_id` | UUID | FK -> users.id (CASCADE) |
| `jti` | String(36) | UNIQUE, indexed |
| `token_hash` | String(64) | SHA-256 hash of actual token |
| `expires_at` | DateTime (UTC) | |
| `is_revoked` | Boolean | default False |
| `created_at` | DateTime | auto |
| `revoked_at` | DateTime | nullable |

### Conversation

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | UUID | PK |
| `user_id` | UUID | FK -> users.id (CASCADE) |
| `title` | String(255) | default "New Conversation" |
| `type` | Enum(friend, therapy) | default "friend" |
| `created_at` | DateTime (UTC) | auto |
| `updated_at` | DateTime (UTC) | auto |

### Message

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | UUID | PK |
| `conversation_id` | UUID | FK -> conversations.id (CASCADE), indexed |
| `role` | Enum(user, assistant, system) | NOT NULL |
| `content` | Text | NOT NULL |
| `type` | Enum(text, voice) | default "text" |
| `audio_url` | String(500) | nullable |
| `msg_metadata` | JSONB | nullable (emotion data, etc.) |
| `text_hash` | String(64) | indexed, for dedup/cache |
| `embedding` | Vector(1536) | pgvector, for semantic search |
| `created_at` | DateTime (UTC) | indexed |

### Summary

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | UUID | PK |
| `conversation_id` | UUID | FK -> conversations.id (CASCADE) |
| `content` | Text | NOT NULL |
| `last_message_id` | UUID | FK -> messages.id |
| `created_at` | DateTime (UTC) | auto |

---

## API Reference

### Authentication (`/auth`)

#### `POST /auth/register`
Create a new user account.

**Request Body:**
```json
{
  "full_name": "Jane Doe",
  "email": "jane@example.com",
  "password": "securepassword"
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "full_name": "Jane Doe",
  "email": "jane@example.com",
  "gender": null,
  "age": null,
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### `POST /auth/login`
Authenticate with email and password.

**Request Body:**
```json
{
  "email": "jane@example.com",
  "password": "securepassword"
}
```

**Response (200):**
```json
{
  "access_token": "<jwt>",
  "refresh_token": "<opaque-token>",
  "token_type": "bearer"
}
```

#### `POST /auth/refresh`
Refresh token rotation -- issues new token pair, revokes old refresh token.

**Request Body:**
```json
{
  "refresh_token": "<current-refresh-token>"
}
```

**Response (200):**
```json
{
  "access_token": "<new-jwt>",
  "refresh_token": "<new-refresh-token>",
  "token_type": "bearer"
}
```

#### `POST /auth/logout`
Revoke the provided refresh token.

**Auth Required:** Yes

**Request Body:**
```json
{
  "refresh_token": "<refresh-token>"
}
```

**Response (200):**
```json
{
  "message": "Logged out successfully"
}
```

---

### Users (`/users`)

#### `GET /users/me`
Get the current user's profile.

**Auth Required:** Yes

**Response (200):**
```json
{
  "id": "uuid",
  "full_name": "Jane Doe",
  "email": "jane@example.com",
  "gender": "female",
  "age": 28,
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### `PATCH /users/me`
Update the current user's profile.

**Auth Required:** Yes

**Request Body (all fields optional):**
```json
{
  "full_name": "Jane Smith",
  "gender": "female",
  "age": 29
}
```

---

### Conversations (`/conversations`)

#### `POST /conversations`
Create a new conversation.

**Auth Required:** Yes

**Request Body:**
```json
{
  "title": "Evening check-in",
  "type": "friend"
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "title": "Evening check-in",
  "type": "friend",
  "user_id": "uuid",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

#### `GET /conversations`
List the current user's conversations.

**Auth Required:** Yes

**Query Parameters:**
- `skip` (int, default 0): Pagination offset
- `limit` (int, default 20): Page size

**Response (200):**
```json
{
  "conversations": [
    {
      "id": "uuid",
      "title": "Evening check-in",
      "type": "friend",
      "created_at": "...",
      "updated_at": "..."
    }
  ]
}
```

#### `GET /conversations/{id}`
Get a single conversation with all messages.

**Auth Required:** Yes

**Response (200):**
```json
{
  "id": "uuid",
  "title": "Evening check-in",
  "type": "friend",
  "messages": [
    {
      "id": "uuid",
      "role": "user",
      "content": "I'm feeling anxious today",
      "type": "text",
      "created_at": "..."
    },
    {
      "id": "uuid",
      "role": "assistant",
      "content": "I hear you...",
      "type": "text",
      "created_at": "..."
    }
  ]
}
```

#### `PATCH /conversations/{id}`
Update a conversation's title.

**Auth Required:** Yes

**Request Body:**
```json
{
  "title": "Updated title"
}
```

#### `DELETE /conversations/{id}`
Delete a conversation and all its messages.

**Auth Required:** Yes

**Response (204):** No content

---

### Chat (`/chat`)

#### `POST /chat/text`
Send a text message and receive a response (REST, single layer).

**Auth Required:** Yes

**Request Body:**
```json
{
  "conversation_id": "uuid",
  "content": "I've been feeling overwhelmed at work"
}
```

**Response (200):**
```json
{
  "message_id": "uuid",
  "content": "I understand how stressful...",
  "emotion": "empathy",
  "is_crisis": false
}
```

#### `POST /chat/voice`
Send voice audio and receive a text response.

**Auth Required:** Yes

**Request:** Multipart form data
- `conversation_id` (string): Conversation UUID
- `audio` (file): Audio file (WAV, MP3, etc.)

**Response (200):**
```json
{
  "message_id": "uuid",
  "content": "...",
  "audio_url": null,
  "emotion": "neutral",
  "is_crisis": false
}
```

#### `POST /chat/voice/audio`
Send voice audio and receive an audio response.

**Auth Required:** Yes

**Request:** Same as `/chat/voice`

**Response (200):** Audio file (WAV or MP3)

---

### Health

#### `GET /health`
Health check endpoint.

**Response (200):**
```json
{
  "status": "healthy"
}
```

---

## WebSocket Protocol

### Connection

```
ws://<host>/chat/ws?token=<jwt>&conversation_id=<uuid>
```

Or authenticate via the first message (recommended for security):
```json
{"type": "auth", "token": "<jwt>"}
```

### Server Response on Connection
```json
{"type": "auth_ok"}
```

### Client -> Server Messages

**Text message:**
```json
{
  "type": "message",
  "conversation_id": "uuid",
  "content": "I'm feeling anxious today"
}
```

**Voice message:**
```json
{
  "type": "voice",
  "conversation_id": "uuid",
  "audio": "<base64-encoded-audio>",
  "audio_format": "wav",
  "language_code": "en-IN",
  "response_format": "text"
}
```

### Server -> Client Messages

**Layer 1 (fast acknowledgement):**
```json
{"type": "layer1_chunk", "content": "partial text..."}
{"type": "layer1", "content": "I hear you, that sounds really tough."}
```

**Layer 2 (deep response):**
```json
{"type": "layer2_chunk", "content": "partial text..."}
{"type": "layer2", "content": "It sounds like you're experiencing...", "audio_data": "base64...", "audio_format": "wav"}
```

**Transcription (voice only):**
```json
{"type": "transcription", "content": "transcribed text", "language": "en"}
```

**Audio response (voice only):**
```json
{"type": "audio", "audio": "base64-encoded-audio-data"}
```

**Error:**
```json
{"type": "error", "content": "Error description"}
```

### Cancellation

Sending a new message while previous tasks are in-flight cancels all pending tasks for the previous message.

---

## Authentication Flow

```
Registration:
  Client -> POST /auth/register -> Server creates user with bcrypt-hashed password

Login:
  Client -> POST /auth/login -> Server verifies password
         <- 200: { access_token (30min JWT), refresh_token (7-day opaque) }

Authenticated Request:
  Client -> Any endpoint with Authorization: Bearer <access_token>
         <- 401 if expired

Token Refresh:
  Client -> POST /auth/refresh { refresh_token }
         <- Server: revokes old refresh token, issues new pair
         <- 200: { new access_token, new refresh_token }

Logout:
  Client -> POST /auth/logout { refresh_token }
         <- Server: revokes the refresh token
```

### Security Details

- **Password hashing**: SHA-256 pre-hash (to handle bcrypt's 72-byte limit) followed by bcrypt
- **Refresh token storage**: Only the SHA-256 hash is stored in the database
- **Token rotation**: Every refresh request invalidates the old token and issues a new pair
- **JTI tracking**: Each refresh token has a unique JWT ID (`jti`) for revocation

---

## Graph RAG Pipeline

The Graph RAG pipeline is implemented as a LangGraph `StateGraph` state machine. It enriches LLM responses with context from the user's conversation history stored in a knowledge graph.

### Pipeline Flow

```
START -> router -> [needs_context?]
                      |-- YES -> extract_entities -> graph_retrieval -> llm_generation -> graph_update -> END
                      |-- NO  -> llm_generation -> END
```

### Nodes

1. **Router**: LLM classifies whether the message needs graph context. Falls back to keyword heuristics (emotional words, personal pronouns, therapy terms) if LLM classification fails.

2. **Extract Entities**: LLM extracts structured entities from the user's message:
   - People (name, relationship)
   - Emotions (emotion, intensity 1-10)
   - Conditions (type, severity)
   - Topics (topic, context)

3. **Graph Retrieval**: Runs Cypher queries against FalkorDB to retrieve:
   - Coping strategies matching detected conditions
   - Therapy techniques for conditions
   - User's previously mentioned people
   - Past conversation topics
   - Recent emotional patterns

4. **LLM Generation**: Gemini generates a response with the retrieved context injected into the system prompt.

5. **Graph Update**: Fire-and-forget storage of new entities and relationships back into the user's context graph in FalkorDB.

### Seeded Knowledge Base

The knowledge graph is seeded at startup with:

| Type | Count | Examples |
|------|-------|---------|
| Conditions | 8 | Anxiety, Depression, Stress, Loneliness, Grief, Anger, Low Self-Esteem, Sleep Issues |
| Coping Strategies | 10 | Deep Breathing, Progressive Muscle Relaxation, Mindfulness Meditation, Journaling, Physical Exercise, etc. |
| Therapy Techniques | 3 | CBT, DBT, ACT |
| Crisis Resources | 3 | 988 Suicide & Crisis Lifeline, Crisis Text Line, IASP |

All entities are seeded via `MERGE` (idempotent -- safe to run multiple times).

---

## Voice Pipeline

```
Audio Input -> STT (Sarvam/Hume) -> Text -> Chat Logic (Gemini + Graph RAG) -> Text -> TTS (Sarvam/Hume) -> Audio Output
                                      |
                                 Emotion Detection (Hume prosody) [optional]
```

### Speech-to-Text (STT)

| Provider | API | Languages | Notes |
|----------|-----|-----------|-------|
| Sarvam AI (primary) | REST | 22+ Indian languages + English | Real-time, translate mode |
| Hume AI (fallback) | Batch API | English + others | Higher latency (batch processing) |

### Text-to-Speech (TTS)

| Provider | Output Format | Notes |
|----------|--------------|-------|
| Sarvam AI (primary) | WAV | Multiple Indian language voices |
| Hume AI (fallback) | MP3 | Emotionally expressive voices |

### Emotion Detection

Hume AI's prosody analysis can detect emotions from audio input. Detected emotions are stored in message metadata (`msg_metadata` JSONB field).

---

## Guardrails System

### Input Validation

| Check | Action |
|-------|--------|
| Empty/whitespace | Reject with error |
| Length > 5000 chars | Reject with error |
| Injection patterns | Reject (regex detection for prompt injection) |
| Crisis keywords | Flag message, append crisis resources to response |

### Output Validation

| Check | Action |
|-------|--------|
| Harmful advice patterns | Filter/remove (e.g., "stop taking medication") |
| Crisis flag from input | Append crisis resources to response |

### Crisis Resources

When crisis keywords are detected, these resources are appended:

- **988 Suicide & Crisis Lifeline**: Call or text 988
- **Crisis Text Line**: Text HOME to 741741
- **Emergency Services**: 911

---

## Caching Strategy

- **What is cached**: Graph RAG responses (graph retrieval + LLM generation results)
- **Key format**: `rag_cache:<SHA256(user_id + message_text)>`
- **Storage**: Redis, JSON serialized
- **TTL**: Configurable via `RAG_CACHE_TTL` (default 3600 seconds / 1 hour)
- **Graceful degradation**: Cache failures are logged but do not break the request flow

---

## Frontend (Streamlit)

The frontend is a Streamlit application located in `frontend/app.py`. It provides:

- **Login/Registration**: Email + password authentication
- **Conversation management**: Create, rename, delete conversations with friend/therapy type selection
- **Text chat**: WebSocket-based two-layer responses with REST fallback
- **Voice chat**: Audio recording (via `audio-recorder-streamlit`) or file upload, with playback of audio responses
- **Profile editing**: Update name, gender, age

### Running the Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Environment Variables (Frontend)

| Variable | Description | Default |
|----------|-------------|---------|
| `DEARAI_API_BASE` | Backend REST API URL | `http://localhost:8000` |
| `DEARAI_WS_BASE` | Backend WebSocket URL | `ws://localhost:8000` |

---

## Configuration & Environment Variables

All backend configuration is managed via Pydantic Settings in `app/core/config.py`. Values are loaded from environment variables or a `.env` file.

### Required Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (e.g., `postgresql+asyncpg://user:pass@localhost:5432/dearai`) |
| `JWT_SECRET_KEY` | Secret key for signing JWT access tokens |
| `GCP_PROJECT_ID` | Google Cloud project ID (for Vertex AI) |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | `"Dear AI"` |
| `APP_VERSION` | Version string | `"0.1.0"` |
| `DEBUG` | Enable debug mode | `False` |
| `LOG_LEVEL` | Logging level | `"INFO"` |
| `REDIS_URL` | Redis connection string | `"redis://localhost:6380"` |
| `JWT_ALGORITHM` | JWT signing algorithm | `"HS256"` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token TTL | `30` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token TTL | `7` |
| `GCP_LOCATION` | GCP region | `"us-central1"` |
| `GEMINI_MODEL` | Primary Gemini model | `"gemini-2.0-flash"` |
| `GEMINI_FLASH_LITE_MODEL` | Fast Gemini model (Layer 1) | `"gemini-2.0-flash-lite"` |
| `LLM_PROVIDER` | LLM provider | `"vertex"` |
| `FALKORDB_HOST` | FalkorDB host | `"localhost"` |
| `FALKORDB_PORT` | FalkorDB port | `6379` |
| `FALKORDB_PASSWORD` | FalkorDB password | `None` |
| `SARVAM_API_KEY` | Sarvam AI API key | `None` |
| `SARVAM_STT_URL` | Sarvam STT endpoint | `"https://api.sarvam.ai/speech-to-text-translate"` |
| `SARVAM_TTS_URL` | Sarvam TTS endpoint | `"https://api.sarvam.ai/text-to-speech"` |
| `HUME_API_KEY` | Hume AI API key | `None` |
| `HUME_STT_URL` | Hume STT endpoint | (Hume batch API default) |
| `HUME_TTS_URL` | Hume TTS endpoint | `"https://api.hume.ai/v0/tts"` |
| `RAG_CACHE_TTL` | RAG cache TTL in seconds | `3600` |
| `SUMMARY_INTERVAL` | Messages between auto-summaries | `10` |
| `ALLOWED_ORIGINS` | CORS allowed origins (list) | `["http://localhost:3000", ...]` |

---

## Infrastructure (Docker)

### Docker Compose Services

| Service | Image | Host Port | Container Port |
|---------|-------|-----------|----------------|
| `db` | `pgvector/pgvector:pg16` | 5432 | 5432 |
| `redis` | `redis:7-alpine` | 6380 | 6379 |
| `falkordb` | `falkordb/falkordb:latest` | 6379 | 6379 |
| `app` | Built from Dockerfile | 8000 | 8000 |

### Running with Docker Compose

```bash
# Start all services
docker compose up -d

# Start only infrastructure (db, redis, falkordb)
docker compose up -d db redis falkordb

# View logs
docker compose logs -f app
```

### Dockerfile

The Dockerfile uses a multi-stage build with `uv` for fast dependency installation:

1. **Base stage**: Python 3.11 slim + uv
2. **Dependencies stage**: Install Python dependencies via uv
3. **Runtime stage**: Copy app code, expose port 8000, run via FastAPI CLI

---

## Development Setup

### Prerequisites

- Python >= 3.11
- uv (package manager)
- Docker + Docker Compose (for infrastructure services)
- Google Cloud credentials (for Vertex AI)

### Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd dearai-backend

# 2. Start infrastructure services
docker compose up -d db redis falkordb

# 3. Install dependencies
uv sync

# 4. Create .env file (see Configuration section)
cp .env.example .env  # Edit with your values

# 5. Run database migrations
uv run alembic upgrade head

# 6. Start the backend
uv run fastapi dev app/main.py --port 8000

# 7. Start the frontend (in a separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Development Tools

```bash
# Install dev dependencies
uv sync --extra dev

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy app/

# Run tests
uv run pytest
```

---

## Database Migrations

Migrations are managed with Alembic. The configuration is in `alembic.ini` and `migrations/env.py`.

```bash
# Apply all migrations
uv run alembic upgrade head

# Create a new migration
uv run alembic revision --autogenerate -m "description"

# Rollback one migration
uv run alembic downgrade -1

# View migration history
uv run alembic history
```

---

## Known Limitations & Technical Debt

1. **No test suite**: The `tests/` directory referenced in `pyproject.toml` does not exist. No automated tests are currently present.

2. **GraphRAGContextProvider stub**: The `get_context()` method in `provider.py` always returns an empty string. The actual RAG runs per-message inside the WebSocket handler, making this class a no-op.

3. **Fire-and-forget graph update passes empty data**: In the WebSocket handler, `_fire_graph_update` receives `extracted_entities={}` instead of the actual extracted entities from the RAG pipeline.

4. **Unused dependency**: `guardrails-ai` is listed in `pyproject.toml` but the guardrails implementation is entirely custom -- the library is never imported.

5. **Deprecated model shims**: Files like `services/users/models.py`, `services/conversations/models.py`, and `repositories/database.py` exist only as re-export shims for backward compatibility. These could be removed.

6. **Frontend is monolithic**: The entire Streamlit frontend is a single ~1600-line file. It could benefit from being split into modules (api client, auth, pages, styles).

7. **CSS embedded in Python**: 300+ lines of CSS are embedded as a Python string constant. This could be extracted to a separate `.css` file.

8. **Hume STT uses batch API**: The Hume speech-to-text implementation uses a batch API endpoint, which introduces higher latency that may not be suitable for real-time voice conversations.

9. **Session management in WebSocket**: The WebSocket handler creates a new database session per operation instead of reusing one session for the connection lifetime.

10. **License inconsistency**: The `pyproject.toml` does not declare a license, but `mental_health_companion.egg-info/PKG-INFO` references an older project name, suggesting a rename occurred.
