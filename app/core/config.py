from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "Dear AI"
    debug: bool = False

    # Database
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/mental_health_companion"
    )

    # JWT
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 30

    # LLM – Vertex AI / Gemini
    llm_provider: str = "vertex"
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    llm_model: str = "Dear-AI_1.1"

    # Hume.ai (STT + TTS + Emotion Detection)
    stt_provider: str = "hume"
    tts_provider: str = "hume"
    hume_api_key: str = ""
    hume_secret_key: str = ""
    hume_tts_voice: str = "Kora"

    # SarvamAI
    sarvam_api_key: str = ""

    # Redis Cache
    redis_url: str = "redis://localhost:6380"
    rag_cache_ttl_seconds: int = 3600  # 1 hour

    # Guardrails
    guardrails_enabled: bool = True


settings = Settings()
