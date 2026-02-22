from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RefreshToken, User
from app.services.auth.schemas import RegisterRequest, TokenResponse
from app.services.auth.utils import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_refresh_token_expiry,
    hash_password,
    hash_token,
    verify_password,
)
from app.services.users.service import get_user_by_email, get_user_by_id


async def register_user(db: AsyncSession, data: RegisterRequest) -> User:
    """Register a new user with email validation."""
    if await get_user_by_email(db, data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    user = User(
        full_name=data.full_name,
        email=data.email,
        password_hash=hash_password(data.password),
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User:
    """Authenticate user by email and password."""
    user = await get_user_by_email(db, email)
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    return user


async def create_user_session(
    db: AsyncSession, user_id: UUID, refresh_token: str, jti: str
) -> RefreshToken:
    """
    Store a new refresh token in the database.

    Args:
        db: Database session
        user_id: The user's UUID
        refresh_token: The raw refresh token (will be hashed before storage)
        jti: The unique token identifier from the JWT

    Returns:
        The created RefreshToken record
    """
    token_record = RefreshToken(
        user_id=user_id,
        jti=jti,
        token_hash=hash_token(refresh_token),
        expires_at=get_refresh_token_expiry(),
        is_revoked=False,
    )
    db.add(token_record)
    await db.flush()
    return token_record


async def create_tokens_with_session(db: AsyncSession, user_id: UUID) -> TokenResponse:
    """
    Create access and refresh tokens, storing the refresh token in the database.

    This is the main function to call when issuing new tokens (login, refresh).
    It creates both tokens and stores the refresh token for session tracking.
    """
    access_token = create_access_token(str(user_id))
    refresh_token, jti = create_refresh_token(str(user_id))

    # Store the refresh token in the database
    await create_user_session(db, user_id, refresh_token, jti)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


async def validate_refresh_token(db: AsyncSession, token: str) -> RefreshToken | None:
    """
    Validate a refresh token against the database.

    Checks:
    1. Token can be decoded
    2. Token type is "refresh"
    3. Token has a jti claim
    4. Token exists in database
    5. Token is not revoked
    6. Token is not expired
    7. Token hash matches (extra security)

    Returns:
        The RefreshToken record if valid, None otherwise
    """
    payload = decode_token(token)
    if not payload:
        return None

    if payload.get("type") != "refresh":
        return None

    jti = payload.get("jti")
    if not jti:
        return None

    # Find the token in the database
    stmt = select(RefreshToken).where(
        RefreshToken.jti == jti,
        RefreshToken.is_revoked == False,  # noqa: E712
        RefreshToken.expires_at > datetime.now(UTC),
    )
    result = await db.execute(stmt)
    token_record = result.scalar_one_or_none()

    if not token_record:
        return None

    # Verify the hash matches (prevents token tampering)
    if token_record.token_hash != hash_token(token):
        return None

    return token_record


async def revoke_token_by_jti(db: AsyncSession, jti: str) -> bool:
    """
    Revoke a specific refresh token by its jti.

    Returns:
        True if a token was revoked, False if no matching token found
    """
    stmt = (
        update(RefreshToken)
        .where(RefreshToken.jti == jti, RefreshToken.is_revoked == False)  # noqa: E712
        .values(is_revoked=True, revoked_at=datetime.now(UTC))
    )
    result = await db.execute(stmt)
    return cast(CursorResult, result).rowcount > 0


async def revoke_all_user_tokens(db: AsyncSession, user_id: UUID) -> int:
    """
    Revoke all refresh tokens for a user (e.g., on password change).

    Returns:
        Number of tokens revoked
    """
    stmt = (
        update(RefreshToken)
        .where(
            RefreshToken.user_id == user_id,
            RefreshToken.is_revoked == False,  # noqa: E712
        )
        .values(is_revoked=True, revoked_at=datetime.now(UTC))
    )
    result = await db.execute(stmt)
    return cast(CursorResult, result).rowcount


async def refresh_access_token(db: AsyncSession, refresh_token: str) -> TokenResponse:
    """
    Refresh an access token using a valid refresh token.

    Implements refresh token rotation:
    1. Validate the incoming refresh token
    2. Revoke the old refresh token
    3. Issue new access and refresh tokens

    This limits the window of opportunity for token theft.
    """
    # Validate the refresh token
    token_record = await validate_refresh_token(db, refresh_token)
    if not token_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # Verify the user still exists
    user = await get_user_by_id(db, token_record.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )

    # Revoke the old refresh token (rotation)
    await revoke_token_by_jti(db, token_record.jti)

    # Create new tokens with a new session
    return await create_tokens_with_session(db, user.id)


async def logout_user(db: AsyncSession, refresh_token: str) -> bool:
    """
    Logout a user by revoking their refresh token.

    Returns:
        True if logout was successful, False if token was invalid
    """
    # Validate the token first
    token_record = await validate_refresh_token(db, refresh_token)
    if not token_record:
        # Token is already invalid or expired - still return success
        # (user wanted to logout, and they effectively are logged out)
        return True

    # Revoke the token
    await revoke_token_by_jti(db, token_record.jti)
    return True
