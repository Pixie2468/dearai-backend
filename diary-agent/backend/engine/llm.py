"""LLM factory for the Episodic Intelligence Engine.

Uses langchain-google-genai with Vertex AI backend via ADC or service account.

Environment variables (via backend/.env):
    GOOGLE_CLOUD_PROJECT:     GCP project ID (required)
    GOOGLE_CLOUD_LOCATION:    GCP region (default: us-central1)
    VERTEX_GENERATION_MODEL:  Gemini model name (default: gemini-2.5-flash)
"""

from __future__ import annotations

import json

from google.auth import default as google_auth_default
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings

_DEFAULT_MODEL = settings.vertex_generation_model
_DEFAULT_LOCATION = settings.google_cloud_location


def _get_credentials():
    """Resolve Google Cloud credentials.

    Checks for an explicit service account key file via
    GOOGLE_APPLICATION_CREDENTIALS, otherwise falls back to
    Application Default Credentials (ADC).
    """

    _SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials: Credentials | None = None

    if settings.gcp_service_account_json:
        info = json.loads(settings.gcp_service_account_json)
        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=_SCOPES,
        )
    else:
        credentials, _ = google_auth_default(scopes=_SCOPES)

    return credentials


def get_model(
    model_name: str | None = None,
    temperature: float = 0,
) -> ChatGoogleGenerativeAI:
    """Return a ChatGoogleGenerativeAI instance configured for Vertex AI.

    Args:
        model_name: Gemini model identifier. If None, uses
            VERTEX_GENERATION_MODEL from settings, then falls back to default.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        A configured ChatGoogleGenerativeAI chat model backed by Vertex AI.
    """
    resolved_model = model_name or _DEFAULT_MODEL

    project_id = settings.google_cloud_project
    if not project_id:
        raise EnvironmentError(
            "GOOGLE_CLOUD_PROJECT env var is required. "
            "Set it to your GCP project ID and run: gcloud auth application-default login"
        )

    project_location = settings.google_cloud_location or _DEFAULT_LOCATION
    credentials = _get_credentials()

    return ChatGoogleGenerativeAI(
        model=resolved_model,
        temperature=temperature,
        credentials=credentials,
        project=project_id,
        location=project_location,
        # include_thoughts=True,
    )
