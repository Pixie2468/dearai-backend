"""Tests for app.services.cache.rag_cache."""

import hashlib
import json

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestBuildCacheKey:
    """Test the deterministic cache key generation."""

    def test_basic_key(self):
        from app.services.cache.rag_cache import _build_cache_key

        key = _build_cache_key("hello", "user1")
        raw = "user1:hello"
        expected = f"rag_cache:{hashlib.sha256(raw.encode()).hexdigest()}"
        assert key == expected

    def test_strips_whitespace(self):
        from app.services.cache.rag_cache import _build_cache_key

        k1 = _build_cache_key("  hello  ", "u1")
        k2 = _build_cache_key("hello", "u1")
        assert k1 == k2

    def test_empty_user_id(self):
        from app.services.cache.rag_cache import _build_cache_key

        key = _build_cache_key("hello")
        raw = ":hello"
        expected = f"rag_cache:{hashlib.sha256(raw.encode()).hexdigest()}"
        assert key == expected

    def test_different_messages_different_keys(self):
        from app.services.cache.rag_cache import _build_cache_key

        k1 = _build_cache_key("hello", "u1")
        k2 = _build_cache_key("world", "u1")
        assert k1 != k2

    def test_different_users_different_keys(self):
        from app.services.cache.rag_cache import _build_cache_key

        k1 = _build_cache_key("hello", "u1")
        k2 = _build_cache_key("hello", "u2")
        assert k1 != k2


class TestGetCachedRagResponse:
    """Test the Redis-backed caching wrapper around run_graph_rag."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_value(self):
        """When Redis holds a cached value, it should be returned directly."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps("cached response"))

        with (
            patch("app.services.cache.rag_cache.get_client", return_value=mock_redis),
            patch("app.services.cache.rag_cache.run_graph_rag") as mock_rag,
        ):
            from app.services.cache.rag_cache import get_cached_rag_response

            result = await get_cached_rag_response("hello", user_id="u1")
            assert result == "cached response"
            mock_rag.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_runs_pipeline_and_stores(self):
        """When Redis has no cached value, run_graph_rag is called and result is stored."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        with (
            patch("app.services.cache.rag_cache.get_client", return_value=mock_redis),
            patch(
                "app.services.cache.rag_cache.run_graph_rag",
                new_callable=AsyncMock,
                return_value="fresh response",
            ) as mock_rag,
        ):
            from app.services.cache.rag_cache import get_cached_rag_response

            result = await get_cached_rag_response(
                "hello", user_id="u1", conversation_history=[]
            )
            assert result == "fresh response"
            mock_rag.assert_awaited_once()
            mock_redis.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_redis_unavailable_falls_through(self):
        """When Redis client is None, the pipeline runs without caching."""
        with (
            patch("app.services.cache.rag_cache.get_client", return_value=None),
            patch(
                "app.services.cache.rag_cache.run_graph_rag",
                new_callable=AsyncMock,
                return_value="live response",
            ) as mock_rag,
        ):
            from app.services.cache.rag_cache import get_cached_rag_response

            result = await get_cached_rag_response("hello")
            assert result == "live response"
            mock_rag.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_redis_get_error_falls_through(self):
        """When Redis GET raises an exception, the pipeline runs as cache miss."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=ConnectionError("redis down"))
        mock_redis.set = AsyncMock()

        with (
            patch("app.services.cache.rag_cache.get_client", return_value=mock_redis),
            patch(
                "app.services.cache.rag_cache.run_graph_rag",
                new_callable=AsyncMock,
                return_value="fallback response",
            ),
        ):
            from app.services.cache.rag_cache import get_cached_rag_response

            result = await get_cached_rag_response("hello", user_id="u1")
            assert result == "fallback response"

    @pytest.mark.asyncio
    async def test_redis_set_error_does_not_break(self):
        """When Redis SET fails, the result is still returned (best-effort caching)."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock(side_effect=ConnectionError("redis down"))

        with (
            patch("app.services.cache.rag_cache.get_client", return_value=mock_redis),
            patch(
                "app.services.cache.rag_cache.run_graph_rag",
                new_callable=AsyncMock,
                return_value="good response",
            ),
        ):
            from app.services.cache.rag_cache import get_cached_rag_response

            result = await get_cached_rag_response("hello")
            assert result == "good response"
