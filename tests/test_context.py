"""Tests for the summary context provider."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.context.summary import SummaryContextProvider


class TestSummaryContextProvider:
    @pytest.fixture
    def provider(self):
        return SummaryContextProvider()

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_summary(self, provider, mock_db):
        # Mock the execute result to return no summary
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        context = await provider.get_context(mock_db, uuid.uuid4(), uuid.uuid4())
        assert context == ""

    @pytest.mark.asyncio
    async def test_returns_summary_content(self, provider, mock_db):
        mock_summary = MagicMock()
        mock_summary.content = "User discussed anxiety about work."

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_summary
        mock_db.execute.return_value = mock_result

        context = await provider.get_context(mock_db, uuid.uuid4(), uuid.uuid4())
        assert "User discussed anxiety about work." in context
        assert "Previous conversation summary" in context
