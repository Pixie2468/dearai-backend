from fastapi import APIRouter

from app.core.dependencies import DbSession
from app.services.auth.schemas import (
    LoginRequest,
    LogoutRequest,
    LogoutResponse,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)
from app.services.auth.service import (
    authenticate_user,
    create_tokens_with_session,
    logout_user,
    refresh_access_token,
    register_user,
)
from app.services.users.schemas import UserResponse

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(db: DbSession, data: RegisterRequest):
    """
    Register a new user.

    Creates a new user account with the provided information.
    Does not automatically log in the user - call /login after registration.
    """
    return await register_user(db, data)


@router.post("/login", response_model=TokenResponse)
async def login(db: DbSession, data: LoginRequest):
    """
    Authenticate user and return access and refresh tokens.

    The refresh token is stored in the database for session management.
    Use the refresh token to obtain new access tokens when they expire.
    """
    user = await authenticate_user(db, data.email, data.password)
    return await create_tokens_with_session(db, user.id)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(db: DbSession, data: RefreshRequest):
    """
    Refresh access token using a valid refresh token.

    Implements refresh token rotation:
    - The old refresh token is revoked
    - A new refresh token is issued

    This limits the window of opportunity for token theft.
    """
    return await refresh_access_token(db, data.refresh_token)


@router.post("/logout", response_model=LogoutResponse)
async def logout(db: DbSession, data: LogoutRequest):
    """
    Logout user by revoking their refresh token.

    After logout, the refresh token can no longer be used to obtain new access tokens.
    The access token will remain valid until it expires (typically 30 minutes).
    """
    await logout_user(db, data.refresh_token)
    return LogoutResponse()
