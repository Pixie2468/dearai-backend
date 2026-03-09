from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_name: str = "Dear AI"
    debug: bool = False
    environment: str = "development"

    # Database
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/dearai"
    )

    # JWT
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # LLM -- Vertex AI / Gemini
    llm_provider: str = "vertex"
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    llm_model: str = "gemini-2.0-flash"

    # Hume.ai (Emotion Detection -- kept as fallback for STT/TTS)
    hume_api_key: str = ""
    hume_secret_key: str = ""
    hume_tts_voice: str = "Kora"

    # Sarvam AI (STT + TTS)
    stt_provider: str = "sarvam"
    tts_provider: str = "sarvam"
    sarvam_api_key: str = ""
    sarvam_stt_model: str = "saaras:v3"
    sarvam_stt_mode: str = "transcribe"
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "shubh"
    sarvam_tts_language: str = "en-IN"

    # FalkorDB (Graph RAG)
    falkor_host: str = "localhost"
    falkor_port: int = 6379
    falkor_graph: str = "dear_ai"

    # Redis Cache
    redis_url: str = "redis://localhost:6380"
    rag_cache_ttl_seconds: int = 3600  # 1 hour

    # Guardrails
    guardrails_enabled: bool = True

    # Summary generation
    summary_interval: int = 10  # Generate summary every N messages


settings = Settings()
