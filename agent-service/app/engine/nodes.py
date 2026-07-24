"""Nodes for the Diary Agent."""

import json
from langchain_core.messages import HumanMessage
from .state import DiaryAgentState
from .prompts import SUMMARIZER_PROMPT, FORMATTER_PROMPT
from .llm import get_model

def summarizer_node(state: DiaryAgentState) -> dict:
    """Analyze chat history and generate an emotional summary."""
    chats = state.get("chats", [])
    if not chats:
        return {"emotional_summary": "No conversation to summarize."}
        
    chat_text = "\n".join([f"{c['role'].capitalize()}: {c['content']}" for c in chats])
    
    prompt = SUMMARIZER_PROMPT.format(chat_history=chat_text)
    
    llm = get_model(temperature=0.3)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"emotional_summary": str(response.content).strip()}

def diary_formatter_node(state: DiaryAgentState) -> dict:
    """Format the emotional summary into a diary entry (title and content)."""
    summary = state.get("emotional_summary", "")
    
    prompt = FORMATTER_PROMPT.format(summary=summary)
    
    llm = get_model(temperature=0.7)
    
    # We enforce JSON output by passing response_format if model supports it, 
    # but for simplicity we'll just extract JSON from the text response.
    # Alternatively, we can use a tool call or prompt engineering. The prompt explicitly asks for JSON.
    response = llm.invoke([HumanMessage(content=prompt)])
    content = str(response.content).strip()
    
    # Simple JSON extraction
    title = "My Diary Entry"
    diary_content = summary
    
    try:
        # Try to find JSON block
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(content)
        if "title" in data and "content" in data:
            title = data["title"]
            diary_content = data["content"]
    except Exception as e:
        # Fallback if JSON parsing fails
        print(f"Failed to parse diary formatter output as JSON: {e}")
        diary_content = content
        
    return {
        "diary_title": title,
        "diary_content": diary_content
    }
