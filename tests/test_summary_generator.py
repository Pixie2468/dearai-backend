"""Tests for app.services.context.summary_generator."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_message(role: str = "user", content: str = "hello", ts=None):
    """Create a lightweight mock Message object."""
    msg = MagicMock()
    msg.id = uuid.uuid4()
    msg.role = role
    msg.content = content
    msg.conv_id = uuid.uuid4()
    msg.created_at = ts or MagicMock()
    return msg


# ---------------------------------------------------------------------------
# _count_messages_since_summary
# ---------------------------------------------------------------------------


class TestCountMessagesSinceSummary:
    @pytest.mark.asyncio
    async def test_no_prior_summary_counts_all_messages(self):
        """When no summary exists, all messages should be counted."""
        db = AsyncMock()
        conv_id = uuid.uuid4()

        # First query: last summary → None
        summary_result = MagicMock()
        summary_result.scalar_one_or_none.return_value = None

        # Second query: count of messages → 15
        count_result = MagicMock()
        count_result.scalar.return_value = 15

        db.execute = AsyncMock(side_effect=[summary_result, count_result])

        from app.services.context.summary_generator import _count_messages_since_summary

        count, last_msg_id = await _count_messages_since_summary(db, conv_id)
        assert count == 15
        assert last_msg_id is None

    @pytest.mark.asyncio
    async def test_with_prior_summary_counts_newer_messages(self):
        """When a prior summary exists, only messages after it should be counted."""
        db = AsyncMock()
        conv_id = uuid.uuid4()
        anchor_msg_id = uuid.uuid4()

        # First query: last summary anchor → anchor_msg_id
        summary_result = MagicMock()
        summary_result.scalar_one_or_none.return_value = anchor_msg_id

        # Second query: anchor message created_at
        anchor_ts_result = MagicMock()
        anchor_ts_result.scalar_one_or_none.return_value = MagicMock()  # some timestamp

        # Third query: count of messages after anchor → 5
        count_result = MagicMock()
        count_result.scalar.return_value = 5

        db.execute = AsyncMock(
            side_effect=[summary_result, anchor_ts_result, count_result]
        )

        from app.services.context.summary_generator import _count_messages_since_summary

        count, last_msg_id = await _count_messages_since_summary(db, conv_id)
        assert count == 5
        assert last_msg_id == anchor_msg_id


# ---------------------------------------------------------------------------
# should_generate_summary
# ---------------------------------------------------------------------------


class TestShouldGenerateSummary:
    @pytest.mark.asyncio
    async def test_returns_true_when_threshold_met(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.summary_interval", 10)

        with patch(
            "app.services.context.summary_generator._count_messages_since_summary",
            new_callable=AsyncMock,
            return_value=(10, None),
        ):
            from app.services.context.summary_generator import should_generate_summary

            result = await should_generate_summary(AsyncMock(), uuid.uuid4())
            assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_below_threshold(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.summary_interval", 10)

        with patch(
            "app.services.context.summary_generator._count_messages_since_summary",
            new_callable=AsyncMock,
            return_value=(5, None),
        ):
            from app.services.context.summary_generator import should_generate_summary

            result = await should_generate_summary(AsyncMock(), uuid.uuid4())
            assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_above_threshold(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.summary_interval", 10)

        with patch(
            "app.services.context.summary_generator._count_messages_since_summary",
            new_callable=AsyncMock,
            return_value=(20, None),
        ):
            from app.services.context.summary_generator import should_generate_summary

            result = await should_generate_summary(AsyncMock(), uuid.uuid4())
            assert result is True


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    @pytest.mark.asyncio
    async def test_skips_when_below_threshold(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.summary_interval", 10)

        with patch(
            "app.services.context.summary_generator._count_messages_since_summary",
            new_callable=AsyncMock,
            return_value=(3, None),
        ):
            from app.services.context.summary_generator import generate_summary

            result = await generate_summary(AsyncMock(), uuid.uuid4())
            assert result is None

    @pytest.mark.asyncio
    async def test_generates_summary_when_threshold_met(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.summary_interval", 10)

        db = AsyncMock()
        conv_id = uuid.uuid4()

        messages = [_make_message("user", "I feel anxious"), _make_message("assistant", "Tell me more")]

        # Mock _count_messages_since_summary
        with patch(
            "app.services.context.summary_generator._count_messages_since_summary",
            new_callable=AsyncMock,
            return_value=(12, None),
        ):
            # Mock db.execute for fetching messages and previous summary
            messages_result = MagicMock()
            messages_result.scalars.return_value.all.return_value = messages

            prev_summary_result = MagicMock()
            prev_summary_result.scalar_one_or_none.return_value = None

            db.execute = AsyncMock(
                side_effect=[messages_result, prev_summary_result]
            )

            # Mock LLM
            mock_llm = MagicMock()
            mock_llm.chat = AsyncMock(return_value="The user expressed anxiety about work.")

            with (
                patch(
                    "app.services.context.summary_generator.get_llm",
                    return_value=mock_llm,
                ),
            ):
                from app.services.context.summary_generator import generate_summary

                result = await generate_summary(db, conv_id)

                # The function should have added a Summary and flushed
                assert db.add.called
                assert db.flush.called

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_failure(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.summary_interval", 10)

        db = AsyncMock()
        conv_id = uuid.uuid4()

        messages = [_make_message("user", "hello")]

        with patch(
            "app.services.context.summary_generator._count_messages_since_summary",
            new_callable=AsyncMock,
            return_value=(12, None),
        ):
            messages_result = MagicMock()
            messages_result.scalars.return_value.all.return_value = messages

            prev_summary_result = MagicMock()
            prev_summary_result.scalar_one_or_none.return_value = None

            db.execute = AsyncMock(
                side_effect=[messages_result, prev_summary_result]
            )

            # LLM that raises
            with patch(
                "app.services.context.summary_generator.get_llm",
                side_effect=Exception("LLM down"),
            ):
                from app.services.context.summary_generator import generate_summary

                result = await generate_summary(db, conv_id)
                assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_messages(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.summary_interval", 10)

        db = AsyncMock()
        conv_id = uuid.uuid4()

        with patch(
            "app.services.context.summary_generator._count_messages_since_summary",
            new_callable=AsyncMock,
            return_value=(12, None),
        ):
            # No messages found
            messages_result = MagicMock()
            messages_result.scalars.return_value.all.return_value = []

            db.execute = AsyncMock(return_value=messages_result)

            from app.services.context.summary_generator import generate_summary

            result = await generate_summary(db, conv_id)
            assert result is None


# ---------------------------------------------------------------------------
# maybe_generate_summary
# ---------------------------------------------------------------------------


class TestMaybeGenerateSummary:
    @pytest.mark.asyncio
    async def test_delegates_to_generate_when_should(self):
        with (
            patch(
                "app.services.context.summary_generator.should_generate_summary",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "app.services.context.summary_generator.generate_summary",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_gen,
        ):
            from app.services.context.summary_generator import maybe_generate_summary

            result = await maybe_generate_summary(AsyncMock(), uuid.uuid4())
            mock_gen.assert_awaited_once()
            assert result is not None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_needed(self):
        with (
            patch(
                "app.services.context.summary_generator.should_generate_summary",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "app.services.context.summary_generator.generate_summary",
                new_callable=AsyncMock,
            ) as mock_gen,
        ):
            from app.services.context.summary_generator import maybe_generate_summary

            result = await maybe_generate_summary(AsyncMock(), uuid.uuid4())
            mock_gen.assert_not_awaited()
            assert result is None
