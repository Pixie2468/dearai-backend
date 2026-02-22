"""
Output guardrails for LLM response filtering.

Ensures LLM responses are safe and appropriate for mental health context.
"""

from app.core.config import settings


class OutputGuardrails:
    """Filters and validates LLM output before sending to user."""

    def __init__(self):
        self.enabled = settings.guardrails_enabled

    async def validate(self, text: str) -> tuple[bool, str | None]:
        """
        Validate LLM output.

        Returns:
            tuple: (is_valid, error_message or None)
        """
        if not self.enabled:
            return True, None

        # Check for harmful advice patterns
        harmful_patterns = [
            "you should stop taking your medication",
            "don't tell anyone",
            "keep this a secret",
        ]

        text_lower = text.lower()
        for pattern in harmful_patterns:
            if pattern in text_lower:
                return False, f"Response contained potentially harmful content"

        return True, None

    async def add_crisis_resources(self, text: str, is_crisis: bool) -> str:
        """Append crisis resources if needed."""
        if not is_crisis:
            return text

        crisis_message = (
            "\n\n---\n"
            "If you're in crisis or need immediate support, please reach out:\n"
            "- National Suicide Prevention Lifeline: 988\n"
            "- Crisis Text Line: Text HOME to 741741\n"
            "- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
        )
        return text + crisis_message


output_guardrails = OutputGuardrails()
