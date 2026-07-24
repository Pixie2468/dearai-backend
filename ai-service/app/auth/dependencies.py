"""WebSocket auth helpers for internal gateway tokens."""

from fastapi import WebSocket, status

from app.auth.paseto import verify_internal_token


async def verify_websocket_handshake(websocket: WebSocket) -> tuple[str, str] | None:
    """Validate the internal PASETO token during the WS handshake.

    Returns a tuple of (user_id, token) when valid, otherwise closes the socket and returns None.
    """
    # Extract the header injected by the Go Gateway.
    token = websocket.headers.get("x-internal-auth")

    if not token:
        # 1008 Policy Violation is the standard WS code for auth failures
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Missing internal auth header"
        )
        return None

    # Verify the cryptographic signature and claims.
    user_id = verify_internal_token(token)

    if not user_id:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Invalid internal auth token"
        )
        return None

    return (user_id, token)
