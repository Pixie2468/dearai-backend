"""
Graph RAG pipeline for Layer 2 responses.

Uses a StateGraph execution engine that:
1. Extracts entities from the user message via LLM
2. Queries FalkorDB to find matching nodes and walk the graph
3. Generates the final response using the fine-tuned Vertex model
   with injected graph context

If FalkorDB is unavailable or no relevant nodes are found, the pipeline
gracefully falls back to generating a response without graph context.
"""

import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict
from uuid import UUID

from app.core import falkordb as falkordb_client
from app.services.context.base import BaseContextProvider
from app.services.llm.base import LLMMessage
from app.services.llm.factory import get_layer1_llm, get_layer2_llm
from app.services.llm.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    LAYER2_BASE_PROMPT,
    LAYER2_CONTEXT_TEMPLATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GraphRAG")


# ---------------------------------------------------------------------------
# State & Type Definitions
# ---------------------------------------------------------------------------


@dataclass
class ChatState:
    session_id: str
    user_message: str
    conversation_history: list[LLMMessage] = field(default_factory=list)
    extracted_entities: dict[str, Any] = field(default_factory=dict)
    graph_context: list[str] = field(default_factory=list)
    llm_response: str | None = None
    should_stop: bool = False
    next_node: str | None = None


NodeFunc = Callable[[ChatState], Awaitable[ChatState]]
ConditionFunc = Callable[[ChatState], str | None]


# ---------------------------------------------------------------------------
# Graph Architecture
# ---------------------------------------------------------------------------


class StateGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, NodeFunc] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, ConditionFunc] = {}

    def add_node(self, name: str, func: NodeFunc) -> None:
        self.nodes[name] = func

    def add_edge(self, from_node: str, to_node: str) -> None:
        self.edges[from_node] = to_node

    def add_conditional_edge(self, from_node: str, condition: ConditionFunc) -> None:
        self.conditional_edges[from_node] = condition

    async def execute(self, start_node: str, state: ChatState) -> ChatState:
        current_node_name = start_node
        logger.info("--- Starting Graph RAG Execution ---")

        while current_node_name and not state.should_stop:
            logger.info("Executing: '%s'", current_node_name)
            node_func = self.nodes.get(current_node_name)

            if not node_func:
                raise ValueError(f"Node '{current_node_name}' not found.")

            state = await node_func(state)

            if state.should_stop:
                break

            # Routing Priority: Explicit state override -> Conditional -> Standard
            next_node_name = None
            if state.next_node:
                next_node_name = state.next_node
                state.next_node = None
            elif current_node_name in self.conditional_edges:
                next_node_name = self.conditional_edges[current_node_name](state)
            elif current_node_name in self.edges:
                next_node_name = self.edges[current_node_name]

            current_node_name = next_node_name

        logger.info("--- Execution Complete ---")
        return state


# ---------------------------------------------------------------------------
# Graph RAG Nodes
# ---------------------------------------------------------------------------


async def extract_entities_node(state: ChatState) -> ChatState:
    """Use the fast LLM to extract searchable entities from the user message."""
    logger.info("  [Extractor] Extracting entities via LLM...")
    try:
        llm = get_layer1_llm()
        prompt = ENTITY_EXTRACTION_PROMPT.format(user_message=state.user_message)
        messages = [LLMMessage(role="user", content=prompt)]
        raw = await llm.chat(messages)

        # Strip markdown fences if the model wraps the JSON
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove ```json ... ``` wrapper
            lines = cleaned.split("\n")
            cleaned = "\n".join(line for line in lines if not line.strip().startswith("```"))

        entities = json.loads(cleaned)
        state.extracted_entities = entities
        logger.info("  [Extractor] Found entities: %s", entities)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("  [Extractor] Entity extraction failed: %s", exc)
        state.extracted_entities = {"persons": [], "topics": [], "keywords": []}

    return state


async def graph_retrieval_node(state: ChatState) -> ChatState:
    """Query FalkorDB using extracted entities and walk the graph for context."""
    graph = falkordb_client.get_graph()
    if graph is None:
        logger.warning("  [FalkorDB] Not connected – skipping graph retrieval.")
        return state

    entities = state.extracted_entities
    persons = entities.get("persons", [])
    topics = entities.get("topics", [])
    keywords = entities.get("keywords", [])

    all_search_terms = persons + topics + keywords

    if not all_search_terms:
        logger.info("  [FalkorDB] No entities to search for.")
        return state

    # Search for matching nodes by name/title across all node types,
    # then walk one hop of relationships to gather context.
    for term in all_search_terms[:5]:  # Limit to 5 terms to avoid overload
        try:
            # Case-insensitive search across any node with a 'name' property
            cypher = """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($term)
            OPTIONAL MATCH (n)-[r]->(related)
            RETURN
                labels(n)[0] AS node_type,
                n.name AS node_name,
                type(r) AS rel_type,
                labels(related)[0] AS related_type,
                related.name AS related_name
            LIMIT 10
            """
            result = await graph.query(cypher, params={"term": term})

            for row in result.result_set:
                node_type = row[0] or "Node"
                node_name = row[1] or "unknown"
                rel_type = row[2]
                related_type = row[3]
                related_name = row[4]

                if rel_type and related_name:
                    ctx = (
                        f"{node_type} '{node_name}' "
                        f"--[{rel_type}]--> "
                        f"{related_type} '{related_name}'"
                    )
                else:
                    ctx = f"{node_type} '{node_name}' exists in the knowledge graph."

                if ctx not in state.graph_context:
                    state.graph_context.append(ctx)
                    logger.info("  [Context Found] %s", ctx)

        except Exception as exc:
            logger.error("  [FalkorDB Error] Query failed for '%s': %s", term, exc)

    return state


async def llm_generation_node(state: ChatState) -> ChatState:
    """Generate the final response using the fine-tuned model with graph context."""
    logger.info("  [LLM] Generating Layer 2 response...")

    llm = get_layer2_llm()

    # Build system prompt with or without graph context
    if state.graph_context:
        context_block = "\n".join(f"- {c}" for c in state.graph_context)
        system_prompt = LAYER2_CONTEXT_TEMPLATE.format(
            base_prompt=LAYER2_BASE_PROMPT,
            graph_context=context_block,
        )
    else:
        system_prompt = LAYER2_BASE_PROMPT

    # Include conversation history if available
    messages = list(state.conversation_history)
    messages.append(LLMMessage(role="user", content=state.user_message))

    state.llm_response = await llm.chat(messages, system_prompt)
    state.should_stop = True
    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_workflow() -> StateGraph:
    """Build and return the Graph RAG workflow."""
    workflow = StateGraph()
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("graph_retrieval", graph_retrieval_node)
    workflow.add_node("llm_generation", llm_generation_node)

    workflow.add_edge("extract_entities", "graph_retrieval")
    workflow.add_edge("graph_retrieval", "llm_generation")

    return workflow


async def run_graph_rag(
    user_message: str,
    conversation_history: list[LLMMessage] | None = None,
) -> str:
    """Run the Graph RAG pipeline and return the final LLM response string.

    This is the non-streaming variant, used by the cache layer.
    """
    state = ChatState(
        session_id="api",
        user_message=user_message,
        conversation_history=conversation_history or [],
    )

    workflow = _build_workflow()
    final_state = await workflow.execute(start_node="extract_entities", state=state)
    return final_state.llm_response or ""


async def run_graph_rag_stream(
    user_message: str,
    conversation_history: list[LLMMessage] | None = None,
) -> AsyncGenerator[str, None]:
    """Run entity extraction + graph retrieval, then stream the LLM response.

    Yields text chunks from the fine-tuned model.
    """
    history = conversation_history or []

    # Step 1 & 2: Extract entities and retrieve graph context
    state = ChatState(
        session_id="api",
        user_message=user_message,
        conversation_history=history,
    )

    # Run extraction + retrieval (non-streaming part)
    state = await extract_entities_node(state)
    state = await graph_retrieval_node(state)

    # Step 3: Stream the LLM generation
    llm = get_layer2_llm()

    if state.graph_context:
        context_block = "\n".join(f"- {c}" for c in state.graph_context)
        system_prompt = LAYER2_CONTEXT_TEMPLATE.format(
            base_prompt=LAYER2_BASE_PROMPT,
            graph_context=context_block,
        )
    else:
        system_prompt = LAYER2_BASE_PROMPT

    messages = list(history)
    messages.append(LLMMessage(role="user", content=user_message))

    async for chunk in llm.chat_stream(messages, system_prompt):
        yield chunk


# ---------------------------------------------------------------------------
# Context Provider (for REST endpoints that use the provider interface)
# ---------------------------------------------------------------------------


class GraphRAGContextProvider(BaseContextProvider):
    """Context provider backed by the Graph RAG pipeline."""

    async def get_context(self, db, conversation_id: UUID, user_id: UUID) -> str:
        # Graph RAG context is message-driven; return empty when no
        # specific query is available (context is fetched per-message
        # via run_graph_rag in the WebSocket handler).
        return ""


graph_rag_context_provider = GraphRAGContextProvider()
