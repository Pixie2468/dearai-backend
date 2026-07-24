"""LLM factory for the Diary Agent."""

import os
import json
from google.auth import default as google_auth_default
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI

def _get_credentials():
    _SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials: Credentials | None = None
    
    gcp_service_account_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if gcp_service_account_json:
        info = json.loads(gcp_service_account_json)
        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=_SCOPES,
        )
    else:
        try:
            credentials, _ = google_auth_default(scopes=_SCOPES)
        except Exception:
            credentials = None
            
    return credentials

def get_model(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Return a ChatGoogleGenerativeAI instance configured for Vertex AI or Gemini."""
    
    project_id = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    model_name = os.getenv("VERTEX_GENERATION_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if project_id:
        credentials = _get_credentials()
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            credentials=credentials,
            project=project_id,
            location=location,
        )
    else:
        # Fallback to pure API key
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )
