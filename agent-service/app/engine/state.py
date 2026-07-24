from typing import TypedDict

class ChatMessage(TypedDict):
    role: str
    content: str

class DiaryAgentState(TypedDict, total=False):
    # Input
    chats: list[ChatMessage]
    
    # Intermediary
    emotional_summary: str
    
    # Output
    diary_title: str
    diary_content: str
