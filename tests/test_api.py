"""Tests for API endpoints: health, auth, users, conversations, chat."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =========================================================================
# Health endpoint
# =========================================================================


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


# =========================================================================
# Auth endpoints — patch at the router module where functions are imported
# =========================================================================


class TestAuthRegister:
    def test_register_success(self, unauthed_client, mock_db):
        fake_user_id = uuid.uuid4()

        with patch("app.api.v1.auth.register_user", new_callable=AsyncMock) as mock_reg:
            mock_user = MagicMock()
            mock_user.id = fake_user_id
            mock_user.full_name = "Alice Smith"
            mock_user.email = "alice@example.com"
            mock_user.gender = None
            mock_user.age = None
            mock_user.created_at = datetime.now(timezone.utc)
            mock_reg.return_value = mock_user

            response = unauthed_client.post(
                "/auth/register",
                json={
                    "full_name": "Alice Smith",
                    "email": "alice@example.com",
                    "password": "strongpass123",
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["full_name"] == "Alice Smith"
        assert data["email"] == "alice@example.com"

    def test_register_duplicate_email(self, unauthed_client, mock_db):
        from fastapi import HTTPException

        with patch("app.api.v1.auth.register_user", new_callable=AsyncMock) as mock_reg:
            mock_reg.side_effect = HTTPException(
                status_code=400, detail="Email already registered"
            )

            response = unauthed_client.post(
                "/auth/register",
                json={
                    "full_name": "Alice",
                    "email": "alice@example.com",
                    "password": "pass",
                },
            )

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]


class TestAuthLogin:
    def test_login_success(self, unauthed_client, mock_db):
        user_id = uuid.uuid4()
        fake_user = MagicMock()
        fake_user.id = user_id

        with (
            patch(
                "app.api.v1.auth.authenticate_user", new_callable=AsyncMock
            ) as mock_auth,
            patch(
                "app.api.v1.auth.create_tokens_with_session", new_callable=AsyncMock
            ) as mock_tokens,
        ):
            mock_auth.return_value = fake_user
            mock_tokens.return_value = MagicMock(
                access_token="access-tok",
                refresh_token="refresh-tok",
                token_type="bearer",
            )

            response = unauthed_client.post(
                "/auth/login",
                json={
                    "email": "alice@example.com",
                    "password": "pass123",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "access-tok"
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self, unauthed_client, mock_db):
        from fastapi import HTTPException

        with patch(
            "app.api.v1.auth.authenticate_user", new_callable=AsyncMock
        ) as mock_auth:
            mock_auth.side_effect = HTTPException(
                status_code=401, detail="Incorrect email or password"
            )

            response = unauthed_client.post(
                "/auth/login",
                json={
                    "email": "a@b.com",
                    "password": "wrong",
                },
            )

        assert response.status_code == 401


class TestAuthLogout:
    def test_logout_success(self, unauthed_client, mock_db):
        with patch(
            "app.api.v1.auth.logout_user", new_callable=AsyncMock
        ) as mock_logout:
            mock_logout.return_value = True

            response = unauthed_client.post(
                "/auth/logout",
                json={
                    "refresh_token": "some-token",
                },
            )

        assert response.status_code == 200
        assert "logged out" in response.json()["message"].lower()


class TestAuthRefresh:
    def test_refresh_success(self, unauthed_client, mock_db):
        with patch(
            "app.api.v1.auth.refresh_access_token", new_callable=AsyncMock
        ) as mock_refresh:
            mock_refresh.return_value = MagicMock(
                access_token="new-access",
                refresh_token="new-refresh",
                token_type="bearer",
            )

            response = unauthed_client.post(
                "/auth/refresh",
                json={
                    "refresh_token": "old-token",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "new-access"

    def test_refresh_invalid(self, unauthed_client, mock_db):
        from fastapi import HTTPException

        with patch(
            "app.api.v1.auth.refresh_access_token", new_callable=AsyncMock
        ) as mock_refresh:
            mock_refresh.side_effect = HTTPException(
                status_code=401, detail="Invalid or expired refresh token"
            )

            response = unauthed_client.post(
                "/auth/refresh",
                json={
                    "refresh_token": "bad-token",
                },
            )

        assert response.status_code == 401


# =========================================================================
# Users endpoints
# =========================================================================


class TestUsersEndpoint:
    def test_get_me(self, client, fake_user):
        response = client.get("/users/me")
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == fake_user.email

    def test_update_me(self, client, fake_user, mock_db):
        updated_user = MagicMock()
        updated_user.id = fake_user.id
        updated_user.full_name = "Updated Name"
        updated_user.email = fake_user.email
        updated_user.gender = None
        updated_user.age = 30
        updated_user.created_at = fake_user.created_at

        with patch(
            "app.api.v1.users.update_user", new_callable=AsyncMock
        ) as mock_update:
            mock_update.return_value = updated_user

            response = client.patch(
                "/users/me",
                json={
                    "full_name": "Updated Name",
                    "age": 30,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Name"


# =========================================================================
# Conversations endpoints
# =========================================================================


class TestConversationsEndpoint:
    def _make_conv_mock(self, **overrides):
        now = datetime.now(timezone.utc)
        defaults = {
            "id": uuid.uuid4(),
            "title": "Chat",
            "type": "friend",
            "created_at": now,
            "updated_at": now,
        }
        defaults.update(overrides)
        m = MagicMock()
        for k, v in defaults.items():
            setattr(m, k, v)
        return m

    def test_create_conversation(self, client, fake_user, mock_db):
        with patch(
            "app.api.v1.conversations.create_conversation",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = self._make_conv_mock(title="Test Chat")

            response = client.post(
                "/conversations",
                json={
                    "title": "Test Chat",
                    "type": "friend",
                },
            )

        assert response.status_code == 201
        assert response.json()["title"] == "Test Chat"

    def test_list_conversations(self, client, fake_user, mock_db):
        with patch(
            "app.api.v1.conversations.get_conversations",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = ([], 0)

            response = client.get("/conversations")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["conversations"] == []

    def test_get_conversation(self, client, fake_user, mock_db):
        conv_id = uuid.uuid4()

        with patch(
            "app.api.v1.conversations.get_conversation_by_id",
            new_callable=AsyncMock,
        ) as mock_get:
            conv = self._make_conv_mock(id=conv_id, title="Session")
            conv.messages = []
            mock_get.return_value = conv

            response = client.get(f"/conversations/{conv_id}")

        assert response.status_code == 200
        assert response.json()["title"] == "Session"

    def test_get_conversation_not_found(self, client, fake_user, mock_db):
        from fastapi import HTTPException

        with patch(
            "app.api.v1.conversations.get_conversation_by_id",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.side_effect = HTTPException(
                status_code=404, detail="Conversation not found"
            )

            response = client.get(f"/conversations/{uuid.uuid4()}")

        assert response.status_code == 404

    def test_delete_conversation(self, client, fake_user, mock_db):
        with patch(
            "app.api.v1.conversations.delete_conversation",
            new_callable=AsyncMock,
        ) as mock_del:
            mock_del.return_value = None

            response = client.delete(f"/conversations/{uuid.uuid4()}")

        assert response.status_code == 204

    def test_update_conversation(self, client, fake_user, mock_db):
        conv_id = uuid.uuid4()

        with patch(
            "app.api.v1.conversations.update_conversation",
            new_callable=AsyncMock,
        ) as mock_upd:
            mock_upd.return_value = self._make_conv_mock(id=conv_id, title="Renamed")

            response = client.patch(
                f"/conversations/{conv_id}",
                json={
                    "title": "Renamed",
                },
            )

        assert response.status_code == 200
        assert response.json()["title"] == "Renamed"

    def test_list_with_pagination(self, client, fake_user, mock_db):
        with patch(
            "app.api.v1.conversations.get_conversations",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = ([], 0)

            response = client.get("/conversations?skip=10&limit=5")

        assert response.status_code == 200


# =========================================================================
# Chat endpoints
# =========================================================================


class TestChatTextEndpoint:
    def test_text_chat(self, client, fake_user, mock_db):
        msg_id = uuid.uuid4()
        conv_id = uuid.uuid4()

        with patch(
            "app.api.v1.chat.process_text_chat",
            new_callable=AsyncMock,
        ) as mock_chat:
            from app.services.chat.schemas import TextChatResponse

            mock_chat.return_value = TextChatResponse(
                message_id=msg_id,
                content="I hear you. How are you feeling?",
                is_crisis=False,
            )

            response = client.post(
                "/chat/text",
                json={
                    "conversation_id": str(conv_id),
                    "content": "I had a rough day.",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "I hear you. How are you feeling?"
        assert data["is_crisis"] is False

    def test_text_chat_crisis(self, client, fake_user, mock_db):
        msg_id = uuid.uuid4()
        conv_id = uuid.uuid4()

        with patch(
            "app.api.v1.chat.process_text_chat",
            new_callable=AsyncMock,
        ) as mock_chat:
            from app.services.chat.schemas import TextChatResponse

            mock_chat.return_value = TextChatResponse(
                message_id=msg_id,
                content="I'm here for you. Please reach out to 988.",
                is_crisis=True,
            )

            response = client.post(
                "/chat/text",
                json={
                    "conversation_id": str(conv_id),
                    "content": "I want to end my life",
                },
            )

        assert response.status_code == 200
        assert response.json()["is_crisis"] is True
