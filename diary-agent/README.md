# DiaryFlow

AI-powered Diary Intelligence Engine that takes a raw chat history and decomposes it into optimized multi-entry arcs for 90-second diary format (TikTok/Reels/Shorts). Analyzes emotional progression, reflection depth, emotional_peak strength, and generates structured optimization suggestions.

## Architecture

```md
frontend/         (React SPA — Vite + Tailwind)
       |                        or
frontend/app.py   (Streamlit UI — legacy/prototyping)
       |
       | POST /diary-intelligence/analyze/stream  (SSE)
       v
backend/main.py   (FastAPI + CORS + optional SPA serving)
       |
       v
engine/graph.py   (LangGraph pipeline — 10 nodes, 2 conditional loops)
       |
       v
PostgreSQL 17     (analysis run persistence, JSONB payloads)
```

In production, the backend serves the built React frontend as a static SPA (via the `STATIC_DIR` env var). In development, the React dev server proxies API calls to the backend.

### LangGraph Pipeline (A0-A8 + Optimizer)

```md
START -> A0 (input_classifier)
              |
              v
         A1 (story_expander) <-----------+
              |                           | (fail: score < 8)
              v                           |
         A2 (story_validator) --[should_retry_story]
              | (pass: score >= 8)
              v
    +-> A3 (entry_planner) <---------------------------+
    |         |                                          |
    |         v                                          |
    |    A4 (entry_scripter)                           |
    |         |                                          |
    |         +--> A5 (emotional_arc_scorer) --+          |
    |         +--> A6 (emotional_peak_scorer) ----+          |
    |                                          v          |
    |                          A7 (reflection_depth) -------+
    |                                          |          |
    |                                          v          |
    |                          A8 (final_validator) -[should_replan]
    |                                          | (pass)   | (fail)
    |                                          v          |
    |                                    optimizer -> END |
    |                                                     |
    +-----------------------------------------------------+
```

- **A5 and A6** run in parallel (fan-out from A4, fan-in to A7).
- **A1 <-> A2 loop**: retries story expansion up to `max_revisions` times.
- **A3 -> A8 loop**: retries the full script pipeline up to `max_revisions` times.
- **Optimizer** is advisory-only (no feedback loop), runs after A8 passes.
- All scripts use **third-person narrative voiceover style** (no direct dialogue).

All LLM calls use `langchain-google-genai` (`ChatGoogleGenerativeAI`) with `.with_structured_output()` to produce strict Pydantic models.

## Prerequisites

- Python >= 3.13
- Node.js >= 20 (for React frontend)
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for PostgreSQL)
- Google Cloud credentials (ADC or service account) with access to Gemini models

## Setup

### 1. Start PostgreSQL

```bash
make docker-up
```

This starts a `pgvector/pgvector:pg17` container on port 5432 with database `vplayer`.

### 2. Configure environment

Copy the sample and fill in your values:

```bash
cp backend/.env.sample backend/.env
```

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vplayer
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
VERTEX_GENERATION_MODEL=gemini-2.5-flash
GOOGLE_GENAI_USE_VERTEXAI=true
```

Optional variables:

| Variable                   | Description                                           |
|----------------------------|-------------------------------------------------------|
| `GCP_SERVICE_ACCOUNT_JSON` | Inline JSON service account credentials (alternative to ADC) |
| `STATIC_DIR`               | Path to built React frontend (enables SPA serving)    |

### 3. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

Or set `GCP_SERVICE_ACCOUNT_JSON` in `.env` for service account authentication.

### 4. Install dependencies

```bash
make backend-install
```

### 5. Run database migration

```bash
make backend-migrate
```

### 6. Start the API

**Development (auto-reload):**

```bash
make backend-dev
```

**Production:**

```bash
make backend-run
```

The API starts at `http://localhost:8000`. OpenAPI docs at `http://localhost:8000/docs`.

### 7. Start the frontend

**React frontend (recommended):**

```bash
cd frontend
npm install
npm run dev
```

Or use the Makefile:

```bash
make frontend-dev     # Vite dev server with API proxy
make frontend-build   # Production build to frontend/dist/
```

The Vite dev server proxies `/diary-intelligence` and `/health` requests to `http://localhost:8000`.

**Streamlit frontend (legacy/prototyping):**

```bash
make frontend-install
make frontend-run
```

The Streamlit UI connects to `http://localhost:8000` by default (configurable in the sidebar).

## Deployment

### Docker

A multi-stage Dockerfile builds both the React frontend and Python backend into a single image:

```bash
make docker-build          # Builds story-flow image
docker run -p 8000:8000 \
  -e DATABASE_URL=... \
  -e GOOGLE_CLOUD_PROJECT=... \
  story-flow
```

Stage 1 builds the React frontend with Node 20, Stage 2 installs Python dependencies and copies the built frontend into `/code/static`. The `STATIC_DIR` env var is set automatically.

### Fly.io

The project includes Fly.io deployment configuration:

- **`fly.toml`** — app config (region: `bom`, 1 shared CPU, 1 GB RAM)
- **`.github/workflows/fly-deploy.yml`** — auto-deploys on push to `main` via GitHub Actions

Set the `FLY_API_TOKEN` secret in your GitHub repository settings.

## API

### `GET /health`

Liveness probe. Returns `{"status": "ok"}`.

### `POST /diary-intelligence/analyze`

Runs the full LangGraph pipeline synchronously and returns structured results. This endpoint blocks until completion (1-3 minutes typical).

**Request:**

```json
{
  "chat_history": "A broke food-delivery rider discovers that one customer is leaving clues to a missing sister case.",
  "genre": "thriller",
  "target_audience": "18-30 mobile-first viewers",
  "tone": "tense",
  "entry_count_preference": 6,
  "max_revisions": 2
}
```

| Field                      | Type   | Default                     | Description                                  |
|----------------------------|--------|-----------------------------|----------------------------------------------|
| `chat_history`               | string | (required)                  | The raw chat history to analyze                |
| `genre`                    | string | `""`                        | Genre hint (e.g. thriller)                   |
| `target_audience`          | string | `"18-30 mobile-first viewers"` | Target audience description               |
| `tone`                     | string | `""`                        | Desired tone (e.g. tense)                    |
| `entry_count_preference` | int    | `6`                         | Number of entrys (5-8)                     |
| `max_revisions`            | int    | `2`                         | Max revision loops for both story and pipeline (1-5) |

**Response:**

Returns a JSON object with:

- `run_id` — UUID for this analysis run
- `chat_history` — echoed back
- `revisions_completed` — number of pipeline revision passes completed
- `entry_planner` — structured entry plan (entrys with outlines, emotional arc notes, emotional_peak ideas, reflection hooks)
- `entry_scripts` — full narrative voiceover scripts per entry with scene directions and continuity notes
- `emotional_arc` — per-entry emotion beats, coherence score, tension curve
- `reflection_analysis` — per-entry reflection scores, risk zones, suggested fixes
- `emotional_peak_analysis` — per-entry emotional_peak scores with curiosity/stakes/emotional breakdown
- `optimization_report` — quality scores, top priorities, per-entry suggestions
- `created_at` — timestamp

Each analysis run is persisted to the `analysis_runs` table in PostgreSQL.

### `POST /diary-intelligence/analyze/stream`

Streaming variant that emits [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) (SSE) as each LangGraph node starts and completes. Same request body as `/analyze`. Both the React and Streamlit frontends use this endpoint.

**SSE event types:**

| Event       | When                             | Payload                                              |
|-------------|----------------------------------|------------------------------------------------------|
| `progress`  | A graph node starts or ends      | `{"node": "<name>", "status": "started\|completed"}` |
| `thinking`  | LLM emits a reasoning/thinking chunk | `{"node": "<name>", "text": "<thinking content>"}` |
| `complete`  | Pipeline finished                | Full `AnalyzeResponse` JSON                          |
| `error`     | Pipeline failed                  | `{"detail": "<error message>"}`                      |

> **Note:** The `thinking` event requires `include_thoughts=True` in the LLM configuration (`engine/llm.py`). This is currently disabled by default.

**Example SSE stream:**

```
event: progress
data: {"node": "input_classifier", "status": "started"}

event: progress
data: {"node": "input_classifier", "status": "completed"}

event: progress
data: {"node": "story_expander", "status": "started"}

event: progress
data: {"node": "story_expander", "status": "completed"}

event: progress
data: {"node": "story_validator", "status": "started"}

event: progress
data: {"node": "story_validator", "status": "completed"}

event: progress
data: {"node": "entry_planner", "status": "started"}

...

event: progress
data: {"node": "emotional_arc_scorer", "status": "started"}

event: progress
data: {"node": "emotional_peak_strength_scorer", "status": "started"}

...

event: complete
data: {"run_id": "...", "entry_planner": {...}, "entry_scripts": {...}, ...}
```

### Interactive docs

OpenAPI docs are available at `http://localhost:8000/docs` when the server is running.

## Project Structure

```
Story-Flow/
├── Makefile                          # Build and run targets
├── Dockerfile                        # Multi-stage build (React + Python)
├── fly.toml                          # Fly.io deployment config
├── LICENSE
├── README.md
├── documentation.md
│
├── .github/
│   └── workflows/
│       └── fly-deploy.yml            # CI/CD: auto-deploy to Fly.io on push to main
│
├── backend/
│   ├── main.py                       # FastAPI entry point (+ SPA serving)
│   ├── pyproject.toml                # Python dependencies (uv)
│   ├── uv.lock                       # Dependency lockfile
│   ├── docker-compose.yml            # PostgreSQL (pgvector) container
│   ├── alembic.ini                   # Alembic migration config
│   ├── .env.sample                   # Environment variable template
│   ├── .python-version               # Python 3.13
│   ├── graph.png                     # Visual diagram of the pipeline
│   │
│   ├── app/                          # Application layer
│   │   ├── config.py                 # Pydantic settings (env vars)
│   │   ├── db.py                     # SQLAlchemy engine & sessions
│   │   ├── models.py                 # ORM models (AnalysisRun)
│   │   ├── schemas.py                # API request/response schemas
│   │   └── routes/
│   │       └── analyze.py            # /diary-intelligence/* endpoints
│   │
│   ├── engine/                       # AI pipeline layer
│   │   ├── graph.py                  # LangGraph graph builder (10 nodes, 2 loops)
│   │   ├── llm.py                    # LLM factory (Gemini via Vertex AI)
│   │   ├── prompts.py                # All prompt templates (726 lines)
│   │   ├── state.py                  # Pipeline state & Pydantic models
│   │   ├── context/                  # Literary reference texts for story expansion
│   │   │   ├── The_Yellow_Wallpaper.txt
│   │   │   └── A_Scandal_in_Bohemia.txt
│   │   └── nodes/                    # Individual pipeline nodes
│   │       ├── input_classifier.py   # A0: Input classification + A2: Story validator
│   │       ├── story_expander.py     # A1: Idea to narrative expansion
│   │       ├── entry_planner.py    # A3: Entry outline planning
│   │       ├── entry_scripter.py   # A4: Script generation
│   │       ├── emotional_arc_scorer.py       # A5: Emotional arc analysis
│   │       ├── emotional_peak_strength_scorer.py # A6: Emotional Peak scoring
│   │       ├── reflection_depth_analyzer.py    # A7: Retention risk prediction
│   │       ├── final_validator.py    # A8: Quality gate
│   │       └── optimizer.py          # Advisory optimization suggestions
│   │
│   └── alembic/                      # Database migrations
│       └── versions/
│           ├── 0001_create_analysis_runs.py
│           └── 0002_rebuild_analysis_runs.py
│
└── frontend/
    ├── app.py                        # Streamlit UI (SSE streaming)
    ├── requirements.txt              # Streamlit dependencies
    ├── package.json                  # React app dependencies
    ├── vite.config.js                # Vite config (API proxy)
    ├── tailwind.config.js            # Tailwind CSS config
    ├── postcss.config.js             # PostCSS config
    ├── index.html                    # React SPA entry point
    └── src/                          # React source code
        ├── App.jsx
        ├── main.jsx
        ├── index.css
        ├── mockData.js
        ├── components/               # Reusable UI components
        ├── hooks/                    # Custom React hooks
        ├── pages/                    # Page-level components
        └── utils/                    # Utility functions
```
