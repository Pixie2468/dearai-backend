"""Regex-based guardrail checks for user input."""

import re


def check_query_safety(user_query: str) -> dict:
    """Evaluate a user query against regex patterns for unsafe content.

    Returns a dict with keys: is_safe (bool), reason (str).
    """
    pattern_injection = re.compile(
        r"\b(ignore previous instructions|forget everything|system prompt|bypass rules|you are now|DAN)\b"
        r"|^\s*(/|sudo |exec )",
        re.IGNORECASE,
    )

    pattern_code = re.compile(
        r"(<script.*?>|javascript:|os\.system|subprocess|eval\()"
        r"|\b(DROP TABLE|TRUNCATE TABLE|DELETE FROM|SELECT \* FROM)\b",
        re.IGNORECASE,
    )

    pattern_harm = re.compile(
        r"\b(kill myself|suicide|end my life|cut myself|slit my|shoot myself)\b",
        re.IGNORECASE,
    )

    pattern_toxicity = re.compile(r"\b(hate speech word 1|hate speech word 2)\b", re.IGNORECASE)

    if pattern_injection.search(user_query):
        return {"is_safe": False, "reason": "prompt_injection_or_command_detected"}

    if pattern_code.search(user_query):
        return {"is_safe": False, "reason": "script_or_code_execution_detected"}

    if pattern_harm.search(user_query):
        return {"is_safe": False, "reason": "self_harm_or_violence_detected"}

    if pattern_toxicity.search(user_query):
        return {"is_safe": False, "reason": "toxicity_detected"}

    return {"is_safe": True, "reason": "passed_regex_checks"}
