# Dear AI — AI Service

Internal FastAPI service that handles the WebSocket chat session. It sits **behind the gateway** and is never directly reachable by clients. Authentication is handled by verifying the PASETO token injected by the gateway — no OIDC logic lives here.

---

## Responsibilities

| Concern | Detail |
|---------|--------|
| **PASETO verification** | Reads `X-Internal-Auth` header, decrypts and validates the V4-local token (`iss`, `aud`, `sub`, `exp`). |
| **WebSocket management** | Accepts the connection after auth, drives the per-message loop. |
| **Interrupt handling** | Cancels any in-flight `asyncio.Task` when a new message arrives or the client disconnects. |
| **GraphRAG pipeline** | Ingests each user message into a personal FalkorDB knowledge graph, retrieves relevant context. |
| **LLM streaming** | Streams Gemini responses chunk-by-chunk back to the client via the WebSocket. |

---

## WebSocket Protocol

**Endpoint:** `ws://<gateway>/chat` (accessed through the gateway, not directly)

**Client → Service**

```json
{ "content": "I've been feeling really overwhelmed lately" }
```

**Service → Client** (per message turn)

```json
{ "layer": "immediate", "content": "Thanks for sharing — give me a moment to think.", "final": false }
{ "layer": "rag",       "content": "<streamed LLM token>",                              "final": false }
{ "layer": "rag",       "content": "",                                                   "final": true  }
```

Sending a new message while the previous response is still streaming **immediately cancels** the in-flight task before starting the new one.

**Error responses**

```json
{ "error": "invalid_json" }
{ "error": "missing_content" }
```

---

## Auth Flow

```
Gateway injects:   X-Internal-Auth: v4.local.<encrypted-payload>
                          │
                   verify_websocket_handshake()
                          │
                   pyseto.decode() → check iss, aud, exp, sub
                          │
                   user_id = payload["sub"]   ← immutable OIDC sub
                          │
                   websocket.accept()
```

If the token is missing, expired, or has invalid claims, the WebSocket is closed with **`1008 Policy Violation`** before the connection is accepted.

---

## Graph Pipeline

Each user has their own named graph in FalkorDB: `graph_{user_id}`.

For every message:

1. **Ingest** — the raw message text is ingested into the graph as structured entities (mood, people, topics, sessions) via LiteLLM + the graph schema. 
2. **Finalize** — the transaction is committed.
3. **Retrieve** — relevant context is queried back from the graph.
4. **Prompt** — the context is injected into the system prompt alongside the user message.
5. **Stream** — Gemini generates and streams the response.

### Graph Schema

| Entity | Description |
|--------|-------------|
| `User` | The human chatting with the bot |
| `Mood` | Emotional state (e.g. Happy, Anxious) |
| `Person` | People in the user's life |
| `Topic` | Subjects, hobbies, events |
| `Session` | A specific chat session (date-stamped) |

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PASETO_SYMMETRIC_KEY` | **yes** | — | 64-char hex (32 bytes). Must match the gateway's key. |
| `PASETO_ISSUER` | no | `dear-ai-gateway` | Expected `iss` claim in incoming tokens |
| `PASETO_AUDIENCE` | no | `dear-ai-python-backend` | Expected `aud` claim in incoming tokens |
| `FALKORDB_HOST` | no | `localhost` | FalkorDB hostname |
| `FALKORDB_PORT` | no | `6379` | FalkorDB port |
| `FALKORDB_PASSWORD` | no | _(empty)_ | FalkorDB password |
| `GEMINI_API_KEY` | no* | — | Gemini API key (required if not using Vertex AI) |
| `VERTEX_MODEL_ID` | no* | — | Vertex AI model ID (e.g. `gemini-2.5-flash`) |
| `GOOGLE_CLOUD_PROJECT` | no* | — | GCP project ID (required for Vertex AI) |
| `GOOGLE_CLOUD_LOCATION` | no* | — | GCP region (required for Vertex AI) |
| `EMBEDDING_MODEL` | no | `text-embedding-004` | Embedding model for GraphRAG |

> **LLM selection:** If `VERTEX_MODEL_ID`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` are all set, Vertex AI is used. Otherwise, the service falls back to the Gemini Developer API using `GEMINI_API_KEY`.

### Example `.env`

```bash
PASETO_SYMMETRIC_KEY=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

FALKORDB_HOST=falkordb
FALKORDB_PORT=6379

GEMINI_API_KEY=your-api-key
EMBEDDING_MODEL=text-embedding-004
```

---

## Project Structure

```md
ai_service/
└── app/
    ├── main.py              # FastAPI app, WebSocket handler, interrupt loop
    ├── auth/
    │   ├── paseto.py        # PASETO decryption + claim validation (iss/aud/sub/exp)
    │   └── dependencies.py  # verify_websocket_handshake() — closes socket on failure
    ├── schemas/
    │   └── graph_schema.py  # FalkorDB GraphSchema (entities + relations)
    ├── services/
    │   ├── context/
    │   │   └── graphrag.py  # DearAIGraphService — async context manager, pipeline orchestration
    │   ├── graph/
    │   │   ├── generation.py # rag.ingest() + rag.finalize()
    │   │   └── retrieval.py  # rag.retrieve() → context string
    │   ├── llm/
    │   │   ├── generate_output.py  # stream_response() — Gemini async streaming
    │   │   └── prompt_manager.py   # build_system_prompt() — injects graph context
    │   └── guardrails/      # (planned)
    └── utils/
        ├── setup_client.py  # Google GenAI client (API key or Vertex AI)
        └── llm_setup.py     # LiteLLM + embedder for GraphRAG
```

---

## Running Locally

```bash
# Install dependencies
uv sync

# Copy and fill in the required variables
cp .env.example .env

# Start the service
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `GET http://localhost:8000/health`

---

## Docker

```bash
# Build
docker build -t dearai-ai-service .

# Run
docker run --rm -p 8000:8000 --env-file .env dearai-ai-service
```

---

## Linting & Type Checking

```bash
# Lint + format
uv run ruff check .
uv run ruff format .

# Type check
uv run ty check
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | ASGI web framework + WebSocket support |
| `uvicorn` | ASGI server |
| `pyseto` | PASETO V4-local token decryption |
| `google-genai` | Gemini API client (streaming) |
| `graphrag-sdk` | GraphRAG orchestration over FalkorDB |
| `litellm` | LLM abstraction for GraphRAG ingestion |
| `falkordb` | FalkorDB Python client |
