import os
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pyseto import decode, Key

API_KEY_NAME = "X-Internal-Auth"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def _load_paseto_key() -> Key:
    key_hex = os.getenv("PASETO_SYMMETRIC_KEY")
    if not key_hex:
        raise RuntimeError("PASETO_SYMMETRIC_KEY is required")
    try:
        key_bytes = bytes.fromhex(key_hex)
        return Key.new(version=4, purpose="local", key=key_bytes)
    except Exception as exc:
        raise RuntimeError("PASETO_SYMMETRIC_KEY must be a valid 32-byte hex string") from exc

PASETO_KEY = _load_paseto_key()
EXPECTED_ISSUER = os.getenv("PASETO_ISSUER", "dear-ai-gateway")
EXPECTED_AUDIENCE = os.getenv("PASETO_AUDIENCE", "dear-ai-python-backend")

def verify_internal_token(token_string: str = Security(api_key_header)) -> str:
    if not token_string:
        raise HTTPException(status_code=401, detail="Missing X-Internal-Auth header")
    try:
        decoded = decode(PASETO_KEY, token_string)
        payload = decoded.payload
        if payload.get("iss") != EXPECTED_ISSUER:
            raise ValueError(f"Invalid issuer: {payload.get('iss')}")
        if payload.get("aud") != EXPECTED_AUDIENCE:
            raise ValueError(f"Invalid audience: {payload.get('aud')}")
        sub = payload.get("sub")
        if not sub:
            raise ValueError("Missing 'sub' claim")
        return str(sub)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid PASETO token: {str(e)}")
