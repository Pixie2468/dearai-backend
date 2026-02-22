"""
DEPRECATED – models have moved to app/models/user.py

This shim re-exports the User model for backward compatibility.
All new code should import from app.models directly.
"""

from app.models.user import User

__all__ = ["User"]

