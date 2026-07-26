"""FastAPI entrypoint for Agent Service."""

import logging
import os
import httpx

from dotenv import load_dotenv
load_dotenv()

CHAT_SERVICE_URL = os.getenv("CHAT_SERVICE_URL", "http://chat_service:8000")
DIARY_SERVICE_URL = os.getenv("DIARY_SERVICE_URL", "http://diary_service:8000")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.auth import verify_internal_token
from app.engine.graph import build_graph

logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Service")

@app.get("/health")
async def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}

@app.post("/summarize-to-diary")
async def summarize_to_diary(request: Request) -> dict:
    """Takes the user's conversation, generates an emotional summary, and adds it to the diary."""
    token = request.headers.get("x-internal-auth")
    if not token:
        return JSONResponse(content={"error": "missing_auth"}, status_code=401)
        
    user_id = verify_internal_token(token)
    if not user_id:
        return JSONResponse(content={"error": "invalid_auth"}, status_code=401)
        
    # 1. Fetch latest diary to get threshold
    after_timestamp = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"X-Internal-Auth": token}
            diary_resp = await client.get(
                f"{DIARY_SERVICE_URL}/diary",
                headers=headers,
                params={"limit": 1}
            )
            if diary_resp.status_code != 200:
                return JSONResponse(content={"error": f"Diary service error: {diary_resp.status_code} - {diary_resp.text}"}, status_code=500)
            diary_resp.raise_for_status()
            diaries = diary_resp.json()
            if diaries:
                after_timestamp = diaries[0].get("created_at")
    except Exception as exc:
        logger.exception("Failed to fetch latest diary: %s", exc)
        return JSONResponse(content={"error": f"failed_to_fetch_diary: {str(exc)}"}, status_code=500)

    # 2. Fetch chats from chat-service
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"X-Internal-Auth": token}
            params = {"limit": 100}
            if after_timestamp:
                params["after"] = after_timestamp
                
            chat_resp = await client.get(
                f"{CHAT_SERVICE_URL}/chats",
                headers=headers,
                params=params
            )
            chat_resp.raise_for_status()
            chats = chat_resp.json()
    except Exception as exc:
        logger.exception("Failed to fetch chats: %s", exc)
        return JSONResponse(content={"error": f"failed_to_fetch_chats: {str(exc)}"}, status_code=500)
        
    if not chats:
        return {"message": "No new chats to summarize", "entry": None}
        
    formatted_chats = [
        {"role": c.get("role", "user"), "content": c.get("content", "")} 
        for c in chats
    ]
    
    # 2. Invoke Diary Agent
    try:
        graph = build_graph()
        result = graph.invoke({"chats": formatted_chats})
    except Exception as exc:
        logger.exception("Diary agent failed: %s", exc)
        return JSONResponse(content={"error": f"agent_failed: {str(exc)}"}, status_code=500)
        
    title = result.get("diary_title", "My Diary Entry")
    content = result.get("diary_content", "")
    
    if not content:
        return JSONResponse(content={"error": "empty_summary"}, status_code=500)
        
    # 3. Post to diary-service
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"X-Internal-Auth": token}
            diary_resp = await client.post(
                f"{DIARY_SERVICE_URL}/diary",
                json={"title": title, "content": content},
                headers=headers
            )
            diary_resp.raise_for_status()
            entry = diary_resp.json()
    except Exception as exc:
        logger.exception("Failed to post diary: %s", exc)
        return JSONResponse(content={"error": "failed_to_post_diary"}, status_code=500)
        
    return {"message": "Diary entry created successfully", "entry": entry}
