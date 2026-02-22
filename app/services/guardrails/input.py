"""
Input guardrails for content moderation.

Uses guardrails-ai library for input validation and safety checks.
"""

from guardrails import Guard

# from guardrails.hub import ToxicLanguage
from app.core.config import settings


class InputGuardrails:
    """Validates and moderates user input before sending to LLM."""

    def __init__(self):
        self.enabled = settings.guardrails_enabled
        if self.enabled:
            self.guard = Guard()

    async def validate(self, text: str) -> tuple[bool, str | None]:
        """
        Validate user input.

        Returns:
            tuple: (is_valid, error_message or None)
        """
        if not self.enabled:
            return True, None

        try:
            self.guard.validate(text)
            return True, None
        except Exception as e:
            return False, str(e)

    async def check_crisis_keywords(self, text: str) -> bool:
        """Check for crisis-related keywords that may need immediate attention."""
        crisis_keywords = [
            "suicide",
            "kill myself",
            "end my life",
            "want to die",
            "self-harm",
            "hurt myself",
            "no reason to live",
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in crisis_keywords)


input_guardrails = InputGuardrails()
