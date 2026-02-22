"""Tests for Pydantic schemas across the project."""

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError


# --- Auth Schemas ---


class TestAuthSchemas:
    def test_token_response(self):
        from app.services.auth.schemas import TokenResponse

        t = TokenResponse(access_token="a", refresh_token="r")
        assert t.token_type == "bearer"

    def test_login_request_valid(self):
        from app.services.auth.schemas import LoginRequest

        r = LoginRequest(email="user@example.com", password="pass")
        assert r.email == "user@example.com"

    def test_login_request_invalid_email(self):
        from app.services.auth.schemas import LoginRequest

        with pytest.raises(ValidationError):
            LoginRequest(email="not-an-email", password="pass")

    def test_register_request(self):
        from app.services.auth.schemas import RegisterRequest

        r = RegisterRequest(full_name="Test", email="a@b.com", password="pw")
        assert r.full_name == "Test"

    def test_logout_response_default(self):
        from app.services.auth.schemas import LogoutResponse

        r = LogoutResponse()
        assert "logged out" in r.message.lower()


# --- User Schemas ---


class TestUserSchemas:
    def test_gender_enum(self):
        from app.services.users.schemas import Gender

        assert Gender.male.value == "male"
        assert Gender.prefer_not_to_say.value == "prefer_not_to_say"

    def test_user_update_empty(self):
        from app.services.users.schemas import UserUpdate

        u = UserUpdate()
        data = u.model_dump(exclude_unset=True)
        assert data == {}

    def test_user_update_partial(self):
        from app.services.users.schemas import UserUpdate

        u = UserUpdate(full_name="New Name")
        data = u.model_dump(exclude_unset=True)
        assert data == {"full_name": "New Name"}

    def test_user_response_from_attributes(self):
        from app.services.users.schemas import UserResponse
        from unittest.mock import MagicMock

        obj = MagicMock()
        obj.id = uuid.uuid4()
        obj.full_name = "Test"
        obj.email = "t@e.com"
        obj.gender = None
        obj.age = 25
        obj.created_at = datetime.now(timezone.utc)
        resp = UserResponse.model_validate(obj)
        assert resp.full_name == "Test"
        assert resp.age == 25


# --- Conversation Schemas ---


class TestConversationSchemas:
    def test_conversation_create_defaults(self):
        from app.services.conversations.schemas import (
            ConversationCreate,
            ConversationType,
        )

        c = ConversationCreate()
        assert c.title is None
        assert c.type == ConversationType.friend

    def test_conversation_create_therapy(self):
        from app.services.conversations.schemas import (
            ConversationCreate,
            ConversationType,
        )

        c = ConversationCreate(title="Session 1", type=ConversationType.therapy)
        assert c.type == ConversationType.therapy

    def test_conversation_update(self):
        from app.services.conversations.schemas import ConversationUpdate

        u = ConversationUpdate(title="Renamed")
        assert u.title == "Renamed"

    def test_conversation_response_from_attributes(self):
        from app.services.conversations.schemas import ConversationResponse
        from unittest.mock import MagicMock

        obj = MagicMock()
        obj.id = uuid.uuid4()
        obj.title = "Chat"
        obj.type = "friend"
        obj.created_at = datetime.now(timezone.utc)
        obj.updated_at = datetime.now(timezone.utc)
        resp = ConversationResponse.model_validate(obj)
        assert resp.title == "Chat"

    def test_conversation_list_response(self):
        from app.services.conversations.schemas import ConversationListResponse

        r = ConversationListResponse(conversations=[], total=0)
        assert r.total == 0

    def test_message_response_maps_msg_metadata(self):
        from app.services.conversations.schemas import MessageResponse
        from unittest.mock import MagicMock

        obj = MagicMock()
        obj.id = uuid.uuid4()
        obj.role = "user"
        obj.content = "hello"
        obj.type = "text"
        obj.audio_url = None
        obj.msg_metadata = {"emotion": "happy"}
        obj.created_at = datetime.now(timezone.utc)
        resp = MessageResponse.model_validate(obj)
        assert resp.metadata == {"emotion": "happy"}


# --- Emotion Schemas ---


class TestEmotionSchemas:
    def test_emotion_score(self):
        from app.services.emotion.schemas import EmotionScore

        s = EmotionScore(emotion="joy", score=0.95)
        assert s.emotion == "joy"

    def test_emotion_result(self):
        from app.services.emotion.schemas import EmotionResult, EmotionScore

        r = EmotionResult(
            emotions=[EmotionScore(emotion="joy", score=0.9)],
            dominant_emotion="joy",
            confidence=0.9,
        )
        assert r.dominant_emotion == "joy"


# --- Chat Schemas ---


class TestChatSchemas:
    def test_text_chat_request(self):
        from app.services.chat.schemas import TextChatRequest

        r = TextChatRequest(
            conversation_id=uuid.uuid4(),
            content="Hello",
        )
        assert r.content == "Hello"

    def test_text_chat_response(self):
        from app.services.chat.schemas import TextChatResponse

        r = TextChatResponse(
            message_id=uuid.uuid4(),
            content="Hi there",
        )
        assert r.is_crisis is False

    def test_voice_chat_response(self):
        from app.services.chat.schemas import VoiceChatResponse

        r = VoiceChatResponse(
            message_id=uuid.uuid4(),
            content="Hello",
        )
        assert r.audio_url is None
        assert r.emotion is None
