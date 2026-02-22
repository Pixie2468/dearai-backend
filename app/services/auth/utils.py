import hashlib
import uuid
from datetime import UTC, datetime, timedelta

import bcrypt
from jose import jwt

from app.core.config import settings


def _prehash_password(password: str) -> bytes:
    """
    Pre-hash password with SHA-256 before bcrypt.

    bcrypt has a 72-byte limit. By pre-hashing with SHA-256, we:
    1. Allow passwords of any length
    2. Always produce a fixed 64-character hex string (well under 72 bytes)
    3. Maintain security (SHA-256 is a secure hash function)

    This technique is used by Dropbox and other security-conscious services.
    """
    return hashlib.sha256(password.encode()).hexdigest().encode()


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 pre-hash + bcrypt."""
    return bcrypt.hashpw(_prehash_password(password), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(_prehash_password(plain_password), hashed_password.encode())


def hash_token(token: str) -> str:
    """Create SHA-256 hash of token for secure storage in database."""
    return hashlib.sha256(token.encode()).hexdigest()


def create_access_token(user_id: str) -> str:
    """Create a JWT access token with expiration."""
    expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.now(UTC),
        "type": "access",
    }
    return jwt.encode(
        payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def create_refresh_token(user_id: str) -> tuple[str, str]:
    """
    Create a JWT refresh token with a unique jti.

    Returns:
        tuple[str, str]: (token, jti) - The encoded token and its unique identifier
    """
    jti = str(uuid.uuid4())
    expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.now(UTC),
        "type": "refresh",
        "jti": jti,
    }
    token = jwt.encode(
        payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )
    return token, jti


def get_refresh_token_expiry() -> datetime:
    """Get the expiration datetime for a new refresh token."""
    return datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)


def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token. Returns None if invalid."""
    try:
        return jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except Exception:
        return None
