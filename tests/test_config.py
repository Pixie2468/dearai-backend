"""Tests for app.core.config."""

from app.core.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings(
            _env_file=None,  # don't read .env during tests
        )
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
        s = Settings(_env_file=None)
        assert "postgresql+asyncpg" in s.database_url
