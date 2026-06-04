"""PASETO token verification for internal gateway auth."""

import json
import os

from pyseto import Key, decode
from pyseto.exceptions import DecryptError, VerifyError


def _load_paseto_key() -> Key:
    key_hex = os.getenv("PASETO_SYMMETRIC_KEY")
    if not key_hex:
        raise RuntimeError("PASETO_SYMMETRIC_KEY is required")

    try:
        return Key.new(version=4, purpose="local", key=bytes.fromhex(key_hex))
    except ValueError as exc:
        raise RuntimeError("PASETO_SYMMETRIC_KEY must be a valid 32-byte hex string") from exc


PASETO_KEY = _load_paseto_key()
EXPECTED_ISSUER = os.getenv("PASETO_ISSUER", "dear-ai-gateway")
EXPECTED_AUDIENCE = os.getenv("PASETO_AUDIENCE", "dear-ai-python-backend")


def _coerce_subject(subject: str | bytes) -> str:
    if isinstance(subject, bytes):
        return subject.decode("utf-8")
    return str(subject)


def verify_internal_token(token_string: str) -> str | None:
    """Verify PASETO token and return the user id if valid."""
    try:
        decoded = decode(PASETO_KEY, token_string)

        if isinstance(decoded.payload, dict):
            payload = decoded.payload
        else:
            payload = json.loads(decoded.payload)

        if payload.get("iss") != EXPECTED_ISSUER:
            raise ValueError("Invalid issuer")

        if payload.get("aud") != EXPECTED_AUDIENCE:
            raise ValueError("Invalid audience")

        user_id = payload.get("sub")
        if not user_id:
            raise ValueError("Missing subject (user_id)")

        return _coerce_subject(user_id)

    except (VerifyError, DecryptError, ValueError) as exc:
        print(f"Token verification failed: {exc}")
        return None
