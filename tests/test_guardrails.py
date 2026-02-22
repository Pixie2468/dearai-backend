"""Tests for guardrails (input & output)."""

import pytest


class TestInputGuardrails:
    """Test the InputGuardrails class with guardrails disabled (test default)."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.guardrails_enabled", False)
        from app.services.guardrails.input import InputGuardrails

        self.guardrails = InputGuardrails()

    @pytest.mark.asyncio
    async def test_validate_passes_when_disabled(self):
        is_valid, error = await self.guardrails.validate("anything here")
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_crisis_keyword_suicide(self):
        result = await self.guardrails.check_crisis_keywords("I want to commit suicide")
        assert result is True

    @pytest.mark.asyncio
    async def test_crisis_keyword_kill_myself(self):
        result = await self.guardrails.check_crisis_keywords("I want to kill myself")
        assert result is True

    @pytest.mark.asyncio
    async def test_crisis_keyword_self_harm(self):
        result = await self.guardrails.check_crisis_keywords(
            "I've been thinking about self-harm"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_no_crisis_keyword(self):
        result = await self.guardrails.check_crisis_keywords("I had a great day today")
        assert result is False

    @pytest.mark.asyncio
    async def test_crisis_case_insensitive(self):
        result = await self.guardrails.check_crisis_keywords("I WANT TO DIE")
        assert result is True


class TestOutputGuardrails:
    """Test the OutputGuardrails class."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.guardrails_enabled", True)
        from app.services.guardrails.output import OutputGuardrails

        self.guardrails = OutputGuardrails()

    @pytest.mark.asyncio
    async def test_validate_clean_text(self):
        is_valid, error = await self.guardrails.validate("You are doing great!")
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_harmful_medication(self):
        is_valid, error = await self.guardrails.validate(
            "You should stop taking your medication"
        )
        assert is_valid is False
        assert error is not None

    @pytest.mark.asyncio
    async def test_validate_harmful_secret(self):
        is_valid, error = await self.guardrails.validate(
            "Keep this a secret from everyone"
        )
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_disabled(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.guardrails_enabled", False)
        from app.services.guardrails.output import OutputGuardrails

        g = OutputGuardrails()
        is_valid, _ = await g.validate("you should stop taking your medication")
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_add_crisis_resources_when_crisis(self):
        result = await self.guardrails.add_crisis_resources(
            "I hear you.", is_crisis=True
        )
        assert "988" in result
        assert "741741" in result

    @pytest.mark.asyncio
    async def test_no_crisis_resources_when_not_crisis(self):
        result = await self.guardrails.add_crisis_resources(
            "I hear you.", is_crisis=False
        )
        assert result == "I hear you."
