# Dear AI — Backend Monorepo

A mental health companion backend consisting of cooperative services. This monorepo manages all internal and external-facing components for the Dear AI ecosystem.

| Service | Language | Role |
|---------|----------|------|
| [`gateway/`](./gateway/README.md) | Go | Public-facing API gateway — OIDC auth, PASETO minting, Reverse proxy |
| [`ai-service/`](./ai-service/README.md) | Python 3.13 | Internal AI service — GraphRAG pipeline, LLM streaming, WS |
| [`chat-service/`](./chat-service/README.md) | Python 3.13 | Internal Chat service — PostgreSQL storage for user chat history (with auto-cleanup) |
| [`diary-service/`](./diary-service/README.md) | Python 3.13 | Internal Diary service — PostgreSQL storage for user diaries |
| [`agent-service/`](./agent-service/README.md) | Python 3.11 | Internal Agent service — LangGraph agent for chat summarization |
| [`diary-agent/`](./diary-agent/README.md) | Python 3.13 | Diary generation pipeline using LangGraph (standalone/development engine) |

> **Note:** For specific details about each service, including internal architecture, WebSocket protocols, schema structures, and environment variable requirements, please refer to the linked `README.md` inside each service's directory.

---

## Architecture

```text
                         ┌─────────────────────────────────┐
Client (browser / app)   │  Go Gateway  :8080              │
  ──[OIDC JWT Bearer]──► │  • Verify JWT (OIDC discovery)  │
                         │  • Mint PASETO (90 s, internal) │
                         │  • Strip Authorization header    │
                         │  • Add X-Internal-Auth header   │
                         │  • Reverse-proxy /chat  ──────► │──► AI Service :8000
                         │  • Reverse-proxy /api/sessions ─► │──► Chat Service :8001
                         │  • Reverse-proxy /api/chats ────► │
                         │  • Reverse-proxy /api/diary ──► │──► Diary Service :8002
                         │  • Reverse-proxy /api/agent ──► │──► Agent Service :8003
                         └─────────────────────────────────┘
                                                              │  • Verify PASETO
                                                              │  • Upgrade to WebSocket
                                                              │  • GraphRAG (FalkorDB)
                                                              │  • Save to Chat Service (HTTP)
                                                              └──► FalkorDB :6379
                                                              └──► PostgreSQL :5432 (chat & diary)
```

---

## Repository Layout

```
dearai-backend/
├── gateway/          # Go API gateway (handles Auth & routing)
├── ai-service/       # Python FastAPI AI backend (WebSocket, GraphRAG, STT, TTS)
├── chat-service/     # Python FastAPI Chat Storage backend (PostgreSQL)
├── diary-service/    # Python FastAPI Diary Storage backend (PostgreSQL)
├── agent-service/    # Python FastAPI Agent backend (Diary summarization)
├── diary-agent/      # Python LangGraph standalone engine for Diary intelligence
├── docker-compose.yml  # Full stack setup
```

For frontend integration instructions (how to connect to the backend), please see [`frontend_integration.md`](./frontend_integration.md).

For cloud deployment instructions to Google Cloud Platform, please see [`gcp.md`](./gcp.md).

---

## Quick Start

### Prerequisites

- **Docker** & **Docker Compose**
- **Go** ≥ 1.26 (gateway local dev)
- **Python** ≥ 3.13 + [uv](https://github.com/astral-sh/uv) (microservices local dev)
- An **OIDC provider** (e.g. Firebase Auth, Auth0, Clerk) — gateway needs `OIDC_ISSUER` + `OIDC_CLIENT_ID`
- A **Gemini API key** or **Vertex AI** project (ai_service)

### Running Full Stack with Docker

The easiest way to run the entire backend is via Docker Compose:

```bash
# Build and start everything (gateway + ai-service + chat + diary + agent + DBs)
docker compose up --build -d

# Verify all containers are healthy
docker compose ps
```

The gateway will be accessible at `http://localhost:8080`. The internal microservices and databases are not directly exposed to the host machine for security reasons, except where explicitly configured in `docker-compose.yml` for local tooling (like Postgres on `5432` or FalkorDB on `6379`).

### Local Development Setup

To run services natively on your machine:

1. **Start infrastructure** (Databases):
   ```bash
   docker compose up -d postgres falkordb
   ```

2. **Configure environments**:
   Copy `.env.example` to `.env` in each service directory and fill in the values.
   Generate a shared `PASETO_SYMMETRIC_KEY` (32 bytes hex) used across all services:
   ```bash
   openssl rand -hex 32
   ```

3. **Run services individually**:
   Refer to the Quick Start sections inside each individual service's README.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Gateway | Go 1.26, `go-oidc/v3`, `go-paseto`, `net/http` std reverse proxy |
| AI Service | Python 3.13, FastAPI, `pyseto`, `google-genai`, `graphrag-sdk` |
| Persistence | Python 3.13, FastAPI, `SQLAlchemy`, `psycopg2` |
| Agents | Python, LangGraph, FastAPI |
| Databases | PostgreSQL (chats & diaries), FalkorDB (knowledge graph) |
| LLM | Google Gemini (via API key or Vertex AI) |
| Auth (external) | OIDC — any compliant provider |
| Auth (internal) | PASETO V4-local symmetric tokens |
