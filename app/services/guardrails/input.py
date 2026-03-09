"""
Input guardrails for content moderation.

Performs input validation and safety checks. When ``guardrails_enabled``
is True, the validate method rejects excessively long or empty input.
Crisis keyword detection always runs regardless of the enabled flag.
"""

import re

from app.core.config import settings


# Maximum allowed message length (characters)
_MAX_INPUT_LENGTH = 5000

# Patterns that indicate prompt-injection or jailbreak attempts
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+(instructions|prompts)", re.I),
    re.compile(r"you\s+are\s+now\s+(DAN|in\s+developer\s+mode)", re.I),
    re.compile(r"disregard\s+(your|all)\s+(rules|guidelines|instructions)", re.I),
]


class InputGuardrails:
    """Validates and moderates user input before sending to LLM."""

    def __init__(self) -> None:
        self.enabled = settings.guardrails_enabled

    async def validate(self, text: str) -> tuple[bool, str | None]:
        """Validate user input.

        Checks:
        - Non-empty after stripping whitespace
        - Within maximum length
        - No prompt-injection patterns

        Returns:
            (is_valid, error_message or None)
        """
        if not self.enabled:
            return True, None

        stripped = text.strip()
        if not stripped:
            return False, "Message cannot be empty."

        if len(stripped) > _MAX_INPUT_LENGTH:
            return False, (
                f"Message exceeds maximum length of {_MAX_INPUT_LENGTH} characters."
            )

        for pattern in _INJECTION_PATTERNS:
            if pattern.search(stripped):
                return False, "Message contains disallowed content."

        return True, None

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
