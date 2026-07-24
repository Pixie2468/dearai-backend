# Agent Service

The `agent-service` is a standalone LangGraph-based microservice for `dearai-backend`. It handles processing the user's conversation to extract an emotional summary and converting it into a diary entry.

## Setup

For instructions on how to configure the Vertex AI model and authentication, please refer to Section 9 of the `gcp.md` documentation in the project root.

## Usage

The agent exposes a single internal endpoint:
`POST /summarize-to-diary`

**Headers required**:
- `X-Internal-Auth`: Your PASETO internal token.

**Flow**:
1. The endpoint is hit.
2. It fetches up to 50 recent messages from the `chat-service`.
3. The LangGraph engine analyzes the messages, summarizes the emotional state, and formats it as a diary entry.
4. The generated entry is POSTed to the `diary-service`.

## Local Development

If you prefer to run it locally without Docker:
```bash
uv pip install -e .
uvicorn app.main:app --reload
```
