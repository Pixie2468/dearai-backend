"""Tests for app.core.config."""

from pydantic_settings import BaseSettings
from app.core.config import Settings


class TestSettingsNoEnv(Settings):
    model_config = Settings.model_config.copy()
    model_config["env_file"] = None  # don't read .env during tests


class TestSettings:
    def test_defaults(self):
        s = TestSettingsNoEnv()
        assert s.app_name == "Mental Health Companion"
        assert s.debug is False
        assert s.jwt_algorithm == "HS256"
        assert s.access_token_expire_minutes == 30
        assert s.refresh_token_expire_days == 7
        assert s.llm_provider == "vertex"
        assert s.tts_provider == "hume"
        assert s.stt_provider == "google"
        assert s.llm_model == "gemini-2.0-flash"
        assert s.vertex_location == "us-central1"
        assert s.guardrails_enabled is True

    def test_database_url_default(self):
        s = TestSettingsNoEnv()
        assert "postgresql+asyncpg" in s.database_url
