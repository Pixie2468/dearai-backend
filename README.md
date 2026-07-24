# Dear AI — Backend Monorepo

A mental health companion backend consisting of four cooperating services:

| Service | Language | Role |
|---------|----------|------|
| [`gateway/`](./gateway/) | Go | Public-facing API gateway — OIDC auth, PASETO minting, Reverse proxy |
| [`ai-service/`](./ai-service/) | Python 3.13 | Internal AI service — GraphRAG pipeline, LLM streaming, WS |
| [`chat-service/`](./chat-service/) | Python 3.13 | Internal Chat service — PostgreSQL storage for user chat history |
| [`diary-service/`](./diary-service/) | Python 3.13 | Internal Diary service — PostgreSQL storage for user diaries |

---

## Architecture

```text
                         ┌─────────────────────────────────┐
Client (browser / app)   │  Go Gateway  :8080              │
  ──[OIDC JWT Bearer]──► │  • Verify JWT (OIDC discovery)  │
                         │  • Mint PASETO (15 s, internal) │
                         │  • Strip Authorization header    │
                         │  • Add X-Internal-Auth header   │
                         │  • Reverse-proxy /chat  ──────► │──► AI Service :8000
                         │  • Reverse-proxy /api/chat ───► │──► Chat Service :8000
                         │  • Reverse-proxy /api/diary ──► │──► Diary Service :8000
                         └─────────────────────────────────┘
                                                              │  • Verify PASETO
                                                              │  • Upgrade to WebSocket
                                                              │  • GraphRAG (FalkorDB)
                                                              │  • Save to Chat Service (HTTP)
                                                              └──► FalkorDB :6379
                                                              └──► PostgreSQL :5432 (chat & diary)
```

### Request lifecycle

1. Client connects to `ws://gateway:8080/chat` with an OIDC `Bearer` JWT.
2. Gateway verifies the JWT (signature, `email_verified`, `sub`/`email` claims).
3. Gateway mints a short-lived PASETO V4-local token carrying the user's immutable OIDC `sub`.
4. Gateway strips `Authorization` and injects `X-Internal-Auth: <paseto>`, then proxies.
5. AI Service verifies the PASETO (`iss`, `aud`, `sub`, `exp`) and extracts `user_id`.
6. AI Service accepts the WebSocket, then for every message:
   - Cancels any in-flight task (interrupt support).
   - Runs `DearAIGraphService` — ingests the message into the user's personal FalkorDB graph, retrieves context.
   - Streams the LLM response chunk-by-chunk back to the client.
   - Sends HTTP POST to `chat-service` to persist the user and AI messages in PostgreSQL.

### WebSocket message format

**Client → Server**

```json
{ "content": "I've been feeling anxious lately" }
```

**Server → Client** (two phases per message)

```json
{ "layer": "immediate", "content": "Thanks for sharing — give me a moment to think.", "final": false }
{ "layer": "rag",       "content": "<streamed LLM chunk>",                              "final": false }
{ "layer": "rag",       "content": "",                                                   "final": true  }
```

Sending a new message while a response is in-flight **cancels** the active task immediately.

---

## Repository Layout

```
dearai-backend/
├── gateway/          # Go API gateway
│   ├── cmd/          # main.go entrypoint
│   └── internal/
│       ├── auth/     # OIDC verifier + PASETO manager
│       ├── config/   # Env-driven config with validation
│       ├── middleware/# RequireAuth middleware
│       ├── proxy/    # WebSocket-aware reverse proxy
│       ├── server/   # Router wiring
│       └── utils/    # Token extraction, JSON helpers
│
├── ai-service/       # Python FastAPI AI backend (WebSocket, GraphRAG, STT, TTS)
│   └── app/
│       ├── auth/     # PASETO verification (internal only)
│       ├── schemas/  # FalkorDB graph schema
│       ├── services/ # Orchestration and API integrations
│       └── utils/
│
├── chat-service/     # Python FastAPI Chat Storage backend
│   └── app/          # SQLAlchemy + PostgreSQL integration
│
├── diary-service/    # Python FastAPI Diary Storage backend
│   └── app/          # SQLAlchemy + PostgreSQL integration
│
├── docker-compose.yml  # Full stack setup
```

---

## Quick Start

### Prerequisites

- **Docker** & **Docker Compose**
- **Go** ≥ 1.26 (gateway local dev)
- **Python** ≥ 3.13 + [uv](https://github.com/astral-sh/uv) (microservices local dev)
- An **OIDC provider** (e.g. Firebase Auth, Auth0, Clerk) — gateway needs `OIDC_ISSUER` + `OIDC_CLIENT_ID`
- A **Gemini API key** or **Vertex AI** project (ai_service)

### 1. Start infrastructure

```bash
docker compose up -d postgres falkordb
```

This starts **FalkorDB** (`:6379`) for the personal knowledge graph and **PostgreSQL** (`:5432`) for chat/diary storage.

### 2. Configure environment

```bash
# Gateway
cp gateway/.env.example gateway/.env

# AI Service
cp ai-service/.env.example ai-service/.env

# Chat Service & Diary Service
cp chat-service/.env.example chat-service/.env
cp diary-service/.env.example diary-service/.env
```

All internal services share the same `PASETO_SYMMETRIC_KEY` — generate one with:

```bash
openssl rand -hex 32
```

### 3. Run the gateway

```bash
cd gateway
go run ./cmd/main.go
```

### 4. Run the microservices

```bash
cd ai-service
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# (In separate terminals)
cd chat-service && uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
cd diary-service && uv run uvicorn app.main:app --host 0.0.0.0 --port 8002
```

### 5. Connect

```bash
# Example using wscat (needs a valid OIDC JWT)
wscat -c "ws://localhost:8080/chat?token=<your-jwt>"

# Standard HTTP routes
curl -H "Authorization: Bearer <your-jwt>" http://localhost:8080/api/chat/chats
```

---

## Shared Environment Variable

| Variable | Used by | Description |
|----------|---------|-------------|
| `PASETO_SYMMETRIC_KEY` | All Services | 64-char hex string (32 bytes). **Must match exactly in all services.** |

See individual microservice directories for `.env.example` configurations.

---

## Running Full Stack with Docker

```bash
# Build and start everything (gateway + ai-service + chat + diary + DBs)
docker compose up --build -d

# Verify all containers are healthy
docker compose ps

# Gateway:         http://localhost:8080   (public — only externally exposed service)
# Microservices:   internal only           (reachable via gateway reverse proxy)
# Databases:       localhost:6379, 5432    (exposed for local dev tooling only)
```

> **Note:** The AI service is **not** port-mapped to the host. It is only reachable
> through the gateway's reverse proxy on the internal Docker network.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Gateway | Go 1.26, `go-oidc/v3`, `go-paseto`, `net/http` std reverse proxy |
| AI Service | Python 3.13, FastAPI, `pyseto`, `google-genai`, `graphrag-sdk` |
| Persistence | Python 3.13, FastAPI, `pyseto`, `SQLAlchemy`, `psycopg2` |
| Databases | PostgreSQL (chats & diaries), FalkorDB (knowledge graph) |
| LLM | Google Gemini (via API key or Vertex AI) |
| Auth (external) | OIDC — any compliant provider |
| Auth (internal) | PASETO V4-local symmetric tokens |

---
