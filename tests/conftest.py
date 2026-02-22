"""
Shared fixtures for the test suite.

Provides:
  - A FastAPI TestClient with overridden dependencies (no real DB)
  - Mock database sessions
  - Authenticated user helpers
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings


# ---------------------------------------------------------------------------
# Settings override – use test-friendly defaults
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _override_settings(monkeypatch):
    """Ensure guardrails are disabled and test secrets are used."""
    monkeypatch.setattr("app.core.config.settings.guardrails_enabled", False)
    monkeypatch.setattr("app.core.config.settings.jwt_secret_key", "test-jwt-secret")
    monkeypatch.setattr("app.core.config.settings.jwt_algorithm", "HS256")
    monkeypatch.setattr("app.core.config.settings.access_token_expire_minutes", 30)
    monkeypatch.setattr("app.core.config.settings.refresh_token_expire_days", 7)


# ---------------------------------------------------------------------------
# Fake user for dependency injection
# ---------------------------------------------------------------------------
def make_fake_user(**overrides):
    defaults = {
        "id": uuid.uuid4(),
        "full_name": "Test User",
        "email": "test@example.com",
        "password_hash": "hashed",
        "gender": None,
        "age": None,
        "created_at": datetime.now(timezone.utc),
        "conversations": [],
        "refresh_tokens": [],
    }
    defaults.update(overrides)
    user = MagicMock()
    for k, v in defaults.items():
        setattr(user, k, v)
    return user


@pytest.fixture()
def fake_user():
    return make_fake_user()


# ---------------------------------------------------------------------------
# Mock AsyncSession
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_db():
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# FastAPI TestClient with dependency overrides
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(fake_user, mock_db):
    from app.core.database import get_db
    from app.core.dependencies import get_current_user
    from app.main import app

    async def override_get_db():
        yield mock_db

    async def override_get_current_user():
        return fake_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture()
def unauthed_client(mock_db):
    """Client without an authenticated user (for auth endpoints)."""
    from app.core.database import get_db
    from app.main import app

    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()
