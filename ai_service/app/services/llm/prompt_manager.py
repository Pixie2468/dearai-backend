"""Prompt construction for the LLM."""


def build_system_prompt(graph_context: str) -> str:
    """Create the system prompt with optional graph context appended."""
    base_instructions = (
        "You are Dear AI, an empathetic, highly conversational companion. "
        "Your goal is to provide thoughtful, supportive, and natural responses. "
        "Do not act like a sterile assistant; be warm and engaging.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Use the provided USER CONTEXT to personalize your response.\n"
        "- Subtly weave this knowledge into the conversation. Do NOT explicitly say "
        "'Based on your context' or 'I see in my database'. Act as if you naturally remember it.\n"
        "- If the context is empty, simply respond naturally to the user's query."
    )

    if graph_context and "No prior context found" not in graph_context:
        context_block = (
            "\n\n--- BACKGROUND USER CONTEXT ---\n"
            f"{graph_context}\n"
            "-------------------------------\n"
        )
    else:
        context_block = (
            "\n\n--- BACKGROUND USER CONTEXT ---\n"
            "No previous context available.\n"
            "-------------------------------\n"
        )

    return base_instructions + context_block
