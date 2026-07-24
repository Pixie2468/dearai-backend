"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration backed by .env / environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/vplayer"

    # Vertex AI / Gemini
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"
    vertex_generation_model: str = "gemini-2.5-flash"
    google_genai_use_vertex: str = ""
    gcp_service_account_json: str = ""

    static_dir: str = ""

    @property
    def sync_database_url(self) -> str:
        """Return a psycopg2-compatible URL for SQLAlchemy / Alembic."""
        url = self.database_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url


settings = Settings()
