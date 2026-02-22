"""Tests for app.services.users.service (with mocked DB)."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.users.service import get_user_by_email, get_user_by_id, update_user
from app.services.users.schemas import UserUpdate


class TestGetUserById:
    @pytest.mark.asyncio
    async def test_returns_user(self, mock_db):
        fake = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake
        mock_db.execute.return_value = mock_result

        user = await get_user_by_id(mock_db, uuid.uuid4())
        assert user is fake

    @pytest.mark.asyncio
    async def test_returns_none(self, mock_db):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        user = await get_user_by_id(mock_db, uuid.uuid4())
        assert user is None


class TestGetUserByEmail:
    @pytest.mark.asyncio
    async def test_returns_user(self, mock_db):
        fake = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake
        mock_db.execute.return_value = mock_result

        user = await get_user_by_email(mock_db, "a@b.com")
        assert user is fake


class TestUpdateUser:
    @pytest.mark.asyncio
    async def test_update_with_data(self, mock_db):
        fake = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake
        mock_db.execute.return_value = mock_result

        result = await update_user(mock_db, uuid.uuid4(), UserUpdate(full_name="New"))
        assert result is fake
        # execute should have been called twice: once for update, once for get
        assert mock_db.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_update_no_data(self, mock_db):
        fake = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake
        mock_db.execute.return_value = mock_result

        result = await update_user(mock_db, uuid.uuid4(), UserUpdate())
        assert result is fake
        # Only the get_user_by_id call
        assert mock_db.execute.call_count == 1
