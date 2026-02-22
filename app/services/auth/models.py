"""
DEPRECATED – models have moved to app/models/auth.py

This shim re-exports the RefreshToken model for backward compatibility.
All new code should import from app.models directly.
"""

from app.models.auth import RefreshToken

__all__ = ["RefreshToken"]

