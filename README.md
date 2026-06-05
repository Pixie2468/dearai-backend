# Dear AI — Backend Monorepo

A mental health companion backend consisting of two cooperating services:

| Service | Language | Role |
|---------|----------|------|
| [`gateway/`](./gateway/) | Go | Public-facing API gateway — OIDC auth, PASETO minting, WebSocket reverse proxy |
| [`ai_service/`](./ai_service/) | Python 3.11 | Internal AI service — GraphRAG pipeline, LLM streaming, conversation state |

---

## Architecture

```
                         ┌─────────────────────────────────┐
Client (browser / app)   │  Go Gateway  :8080               │
  ──[OIDC JWT Bearer]──► │  • Verify JWT (OIDC discovery)   │
                         │  • Mint PASETO (15 s, internal)  │
                         │  • Strip Authorization header     │
                         │  • Add X-Internal-Auth header    │
                         │  • Reverse-proxy /chat  ──────►  │──► AI Service :8000
                         └─────────────────────────────────┘
                                                              │  • Verify PASETO
                                                              │  • Upgrade to WebSocket
                                                              │  • GraphRAG (FalkorDB)
                                                              │  • Stream LLM response
                                                              └──► FalkorDB :6379
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
├── ai_service/       # Python FastAPI AI backend
│   └── app/
│       ├── auth/     # PASETO verification (internal only)
│       ├── schemas/  # FalkorDB graph schema
│       ├── services/
│       │   ├── context/  # GraphRAG orchestration
│       │   ├── graph/    # FalkorDB ingest + retrieval
│       │   ├── llm/      # Gemini streaming + prompt builder
│       │   └── guardrails/ # (planned)
│       └── utils/    # LLM + client setup
│
├── docker-compose.yml  # Infrastructure: FalkorDB
└── .github/workflows/  # CI/CD
```

---

## Quick Start

### Prerequisites

- **Docker** & **Docker Compose**
- **Go** ≥ 1.22 (gateway local dev)
- **Python** ≥ 3.11 + [uv](https://github.com/astral-sh/uv) (ai_service local dev)
- An **OIDC provider** (e.g. Auth0, Google, Clerk) — gateway needs `ISSUER_URL` + `AUDIENCE_CLIENT_ID`
- A **Gemini API key** or **Vertex AI** project (ai_service)

### 1. Start infrastructure

```bash
docker compose up -d
```

This starts FalkorDB (`:6379`) for the personal knowledge graph.

### 2. Configure environment

```bash
# Gateway
cp gateway/.env.example gateway/.env

# AI Service
cp ai_service/.env.example ai_service/.env
```

Both services share the same `PASETO_SYMMETRIC_KEY` — generate one with:

```bash
openssl rand -hex 32
```

### 3. Run the gateway

```bash
cd gateway
go run ./cmd/main.go
```

### 4. Run the AI service

```bash
cd ai_service
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Connect

```bash
# Example using websocat (needs a valid OIDC JWT)
websocat -H "Authorization: Bearer <your-jwt>" ws://localhost:8080/chat
```

---

## Shared Environment Variable

| Variable | Used by | Description |
|----------|---------|-------------|
| `PASETO_SYMMETRIC_KEY` | Gateway + AI Service | 64-char hex string (32 bytes). **Must match exactly in both services.** |

See [`gateway/README.md`](./gateway/README.md) and [`ai_service/README.md`](./ai_service/README.md) for full per-service variable references.

---

## Running Full Stack with Docker

```bash
# Build and start everything
docker compose up --build

# Gateway:    http://localhost:8080
# AI Service: http://localhost:8000  (internal, accessed through gateway only)
# FalkorDB:   localhost:6379
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Gateway | Go 1.26, `go-oidc/v3`, `go-paseto`, `net/http` std reverse proxy |
| AI Service | Python 3.11, FastAPI, `pyseto`, `google-genai`, `graphrag-sdk` |
| Knowledge Graph | FalkorDB (Redis-compatible graph DB) |
| LLM | Google Gemini (via API key or Vertex AI) |
| Auth (external) | OIDC — any compliant provider |
| Auth (internal) | PASETO V4-local symmetric tokens |

---

## License

Private — all rights reserved.
