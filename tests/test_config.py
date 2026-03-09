"""Tests for app.core.config."""

from pydantic_settings import BaseSettings
from app.core.config import Settings


class TestSettingsNoEnv(Settings):
    model_config = Settings.model_config.copy()
    model_config["env_file"] = None  # don't read .env during tests


class TestSettings:
    def test_defaults(self):
        s = TestSettingsNoEnv()
        assert s.app_name == "Dear AI"
        assert s.debug is False
        assert s.jwt_algorithm == "HS256"
        assert s.access_token_expire_minutes == 30
        assert s.refresh_token_expire_days == 7
        assert s.llm_provider == "vertex"
        assert s.tts_provider == "sarvam"
        assert s.stt_provider == "sarvam"
        assert s.llm_model == "gemini-2.0-flash"
        assert s.vertex_location == "us-central1"
        assert s.guardrails_enabled is True

    def test_sarvam_defaults(self):
        s = TestSettingsNoEnv()
        assert s.sarvam_stt_model == "saaras:v3"
        assert s.sarvam_stt_mode == "transcribe"
        assert s.sarvam_tts_model == "bulbul:v3"
        assert s.sarvam_tts_speaker == "shubh"
        assert s.sarvam_tts_language == "en-IN"

    def test_falkordb_defaults(self):
        s = TestSettingsNoEnv()
        assert s.falkor_host == "localhost"
        assert s.falkor_port == 6379
        assert s.falkor_graph == "dear_ai"

    def test_summary_interval_default(self):
        s = TestSettingsNoEnv()
        assert s.summary_interval == 10

    def test_database_url_default(self):
        s = TestSettingsNoEnv()
        assert "postgresql+asyncpg" in s.database_url
