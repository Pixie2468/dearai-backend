import re

# A basic list of patterns that indicate immediate danger, self-harm, suicide, or severe emergency.
# Using word boundaries (\b) to avoid matching substrings in unrelated words where possible,
# though safety regexes should be broad enough to catch intent.
EMERGENCY_PATTERNS = [
    r"\b(suicide|kill(ing)?\s+myself|end(ing)?\s+my\s+life|want\s+to\s+die)\b",
    r"\b(self-harm|cut(ting)?\s+myself|hurt(ing)?\s+myself)\b",
    r"\b(overdose|swallow(ing)?\s+pills)\b",
    r"\b(shoot(ing)?\s+myself|hang(ing)?\s+myself|jump(ing)?\s+off)\b",
    r"\b(don'?t\s+want\s+to\s+live|can'?t\s+take\s+it\s+anymore|better\s+off\s+dead)\b",
]

# Compile patterns for performance
COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in EMERGENCY_PATTERNS]

def check_safety(content: str) -> bool:
    """
    Checks if the user content is safe.
    Returns False if any emergency/self-harm patterns are detected, True otherwise.
    """
    if not content:
        return True
        
    for pattern in COMPILED_PATTERNS:
        if pattern.search(content):
            return False
            
    return True

# A basic list of patterns that indicate irrelevant topics like programming, coding, or unrelated tasks.
IRRELEVANT_PATTERNS = [
    r"\b(write code|code in|javascript|python|c\+\+|java|html|css|sql|script|debug|compile)\b",
    r"\b(how to code|write a script|function for|array|variable|database|api|json)\b",
    r"\b(build an app|create a website|program a|developer)\b",
]

COMPILED_IRRELEVANT = [re.compile(pattern, re.IGNORECASE) for pattern in IRRELEVANT_PATTERNS]

def check_relevance(content: str) -> bool:
    """
    Checks if the user content is relevant for a friendly chatbot.
    Returns False if irrelevant topics like coding are detected, True otherwise.
    """
    if not content:
        return True
        
    for pattern in COMPILED_IRRELEVANT:
        if pattern.search(content):
            return False
            
    return True
