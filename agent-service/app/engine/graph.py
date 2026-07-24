"""Graph builder for the Diary Agent."""

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from .state import DiaryAgentState
from .nodes import summarizer_node, diary_formatter_node

def build_graph() -> CompiledStateGraph:
    """Construct and compile the Diary Agent graph."""
    builder = StateGraph(DiaryAgentState)
    
    # Add nodes
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("diary_formatter", diary_formatter_node)
    
    # Add edges
    builder.add_edge(START, "summarizer")
    builder.add_edge("summarizer", "diary_formatter")
    builder.add_edge("diary_formatter", END)
    
    # Compile
    graph = builder.compile()
    
    return graph
