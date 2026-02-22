"""Tests for app.services.auth.utils – pure functions, no DB needed."""

import uuid
from datetime import UTC, datetime

from jose import jwt

from app.services.auth.utils import (
    _prehash_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_refresh_token_expiry,
    hash_password,
    hash_token,
    verify_password,
)


# --- Password hashing ---


class TestPasswordHashing:
    def test_hash_password_returns_string(self):
        h = hash_password("secret123")
        assert isinstance(h, str)
        assert len(h) > 0

    def test_hash_password_not_plaintext(self):
        h = hash_password("secret123")
        assert h != "secret123"

    def test_verify_password_correct(self):
        h = hash_password("mypassword")
        assert verify_password("mypassword", h) is True

    def test_verify_password_wrong(self):
        h = hash_password("mypassword")
        assert verify_password("wrongpassword", h) is False

    def test_different_passwords_different_hashes(self):
        h1 = hash_password("password1")
        h2 = hash_password("password2")
        assert h1 != h2

    def test_prehash_returns_bytes(self):
        result = _prehash_password("hello")
        assert isinstance(result, bytes)


# --- Token hashing ---


class TestTokenHashing:
    def test_hash_token_deterministic(self):
        assert hash_token("abc") == hash_token("abc")

    def test_hash_token_different_inputs(self):
        assert hash_token("abc") != hash_token("xyz")

    def test_hash_token_length(self):
        # SHA-256 hex digest = 64 chars
        assert len(hash_token("anything")) == 64


# --- JWT tokens ---


class TestAccessToken:
    def test_create_access_token_is_valid_jwt(self):
        user_id = str(uuid.uuid4())
        token = create_access_token(user_id)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_access_token_payload(self):
        user_id = str(uuid.uuid4())
        token = create_access_token(user_id)
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == user_id
        assert payload["type"] == "access"

    def test_access_token_has_expiry(self):
        token = create_access_token(str(uuid.uuid4()))
        payload = decode_token(token)
        assert payload is not None
        assert "exp" in payload


class TestRefreshToken:
    def test_create_refresh_token_returns_tuple(self):
        result = create_refresh_token(str(uuid.uuid4()))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_refresh_token_has_jti(self):
        user_id = str(uuid.uuid4())
        token, jti = create_refresh_token(user_id)
        payload = decode_token(token)
        assert payload is not None
        assert payload["jti"] == jti
        assert payload["type"] == "refresh"

    def test_refresh_token_jti_is_uuid(self):
        _, jti = create_refresh_token(str(uuid.uuid4()))
        uuid.UUID(jti)  # should not raise


class TestDecodeToken:
    def test_decode_valid_token(self):
        token = create_access_token(str(uuid.uuid4()))
        assert decode_token(token) is not None

    def test_decode_invalid_token(self):
        assert decode_token("not.a.jwt") is None

    def test_decode_empty_string(self):
        assert decode_token("") is None


class TestGetRefreshTokenExpiry:
    def test_expiry_in_future(self):
        expiry = get_refresh_token_expiry()
        assert expiry > datetime.now(UTC)
