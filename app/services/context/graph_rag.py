"""
Graph RAG pipeline for Dear AI.

Implements a StateGraph-based pipeline that:
1. Routes queries to determine if graph context is needed
2. Extracts entities using the LLM (people, topics, emotions, conditions)
3. Retrieves relevant context from FalkorDB (user graph + mental health KB)
4. Generates a context-rich response via Vertex AI
5. Updates the graph with new entities/relationships from the conversation

The pipeline is used by:
- ``run_graph_rag()`` -- called by the WebSocket RAG layer
- ``GraphRAGContextProvider`` -- implements the BaseContextProvider interface
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict
from uuid import UUID

from app.services.context.base import BaseContextProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared FalkorDB connection (set during app startup via ``init_falkordb``)
# ---------------------------------------------------------------------------
_falkor_graph = None


def set_falkor_graph(graph) -> None:
    """Called once at app startup to inject the FalkorDB graph handle."""
    global _falkor_graph
    _falkor_graph = graph


def get_falkor_graph():
    """Return the shared FalkorDB graph handle."""
    return _falkor_graph


# ---------------------------------------------------------------------------
# State & Type Definitions
# ---------------------------------------------------------------------------


@dataclass
class ChatState:
    """Mutable state bag threaded through every node in the graph."""

    session_id: str
    user_id: str
    user_message: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    extracted_entities: dict[str, Any] = field(default_factory=dict)
    graph_context: list[str] = field(default_factory=list)
    llm_response: str | None = None
    should_stop: bool = False
    next_node: str | None = None


NodeFunc = Callable[[ChatState], Awaitable[ChatState]]
ConditionFunc = Callable[[ChatState], str | None]


# ---------------------------------------------------------------------------
# StateGraph engine (kept from original -- well designed)
# ---------------------------------------------------------------------------


class StateGraph:
    """Lightweight async state-machine for orchestrating the RAG pipeline."""

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
        current = start_node
        logger.info("--- Starting Graph RAG Execution ---")

        while current and not state.should_stop:
            logger.info("Executing node: '%s'", current)
            node_func = self.nodes.get(current)
            if not node_func:
                raise ValueError(f"Node '{current}' not found in graph.")

            state = await node_func(state)

            if state.should_stop:
                break

            # Priority: explicit override -> conditional -> static edge
            nxt = None
            if state.next_node:
                nxt = state.next_node
                state.next_node = None
            elif current in self.conditional_edges:
                nxt = self.conditional_edges[current](state)
            elif current in self.edges:
                nxt = self.edges[current]

            current = nxt

        logger.info("--- Graph RAG Execution Complete ---")
        return state


# ---------------------------------------------------------------------------
# Node implementations (production)
# ---------------------------------------------------------------------------

# Lazy import helper so Vertex AI is not loaded at module level.
_llm_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        from app.services.llm import get_llm

        _llm_instance = get_llm()
    return _llm_instance


async def router_node(state: ChatState) -> ChatState:
    """Classify whether the user message needs graph context.

    Uses a lightweight LLM call to decide routing. Falls back to
    keyword heuristics if the LLM is unavailable.
    """
    try:
        llm = _get_llm()
        from app.services.llm import LLMMessage

        classification_prompt = (
            "You are a router for a mental health companion chatbot.\n"
            "Classify the following user message into ONE of these categories:\n"
            "- GRAPH_NEEDED: The message references specific people, past events, "
            "relationships, asks about coping strategies, therapeutic techniques, "
            "or mentions mental health conditions.\n"
            "- DIRECT: The message is general conversation, greetings, or simple "
            "emotional expression that doesn't need knowledge graph context.\n\n"
            f"User message: {state.user_message}\n\n"
            "Reply with ONLY the category name: GRAPH_NEEDED or DIRECT"
        )
        result = await llm.chat(
            [LLMMessage(role="user", content=classification_prompt)]
        )
        needs_graph = "GRAPH_NEEDED" in result.upper()
    except Exception:
        logger.warning("Router LLM call failed, using keyword heuristics.")
        keywords = {
            "anxious", "anxiety", "depressed", "depression", "stress",
            "stressed", "lonely", "grief", "angry", "anger", "sleep",
            "insomnia", "cope", "coping", "help", "strategy", "technique",
            "therapy", "therapist", "breathing", "meditat", "exercise",
            "journal", "friend", "family", "relationship", "remember",
            "last time", "you told me", "we talked about", "who",
            "self-esteem", "confident", "worthless", "hopeless", "scared",
        }
        msg_lower = state.user_message.lower()
        needs_graph = any(kw in msg_lower for kw in keywords)

    if needs_graph:
        logger.info("[Router] Graph context needed -> extract_entities")
        state.next_node = "extract_entities"
    else:
        logger.info("[Router] Direct response -> llm_generation")
        state.next_node = "llm_generation"
    return state


async def extract_entities_node(state: ChatState) -> ChatState:
    """Use the LLM to extract structured entities from the user message."""
    try:
        llm = _get_llm()
        from app.services.llm import LLMMessage

        extraction_prompt = (
            "Extract entities from the following user message for a mental health "
            "companion chatbot. Return a JSON object with these keys (use empty "
            "lists if none found):\n"
            '- "people": list of person names mentioned\n'
            '- "emotions": list of emotions expressed (e.g. "anxious", "sad")\n'
            '- "conditions": list of mental health conditions referenced '
            '(e.g. "Anxiety", "Depression", "Stress", "Loneliness", "Grief", '
            '"Anger Management", "Sleep Issues", "Low Self-Esteem")\n'
            '- "topics": list of topics discussed (e.g. "work", "relationship")\n'
            '- "needs_strategies": boolean, true if user is asking for help or '
            "coping strategies\n\n"
            f"User message: {state.user_message}\n\n"
            "Return ONLY valid JSON, no markdown fences."
        )
        result = await llm.chat(
            [LLMMessage(role="user", content=extraction_prompt)]
        )

        # Parse the JSON response
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
        entities = json.loads(cleaned)
        state.extracted_entities = entities
        logger.info("[Extractor] Entities: %s", entities)
    except Exception:
        logger.warning("Entity extraction failed, proceeding with empty entities.")
        state.extracted_entities = {
            "people": [],
            "emotions": [],
            "conditions": [],
            "topics": [],
            "needs_strategies": False,
        }
    return state


async def graph_retrieval_node(state: ChatState) -> ChatState:
    """Query FalkorDB for relevant context based on extracted entities."""
    graph = get_falkor_graph()
    if graph is None:
        logger.warning("[FalkorDB] No graph connection available, skipping retrieval.")
        return state

    entities = state.extracted_entities
    user_id = state.user_id

    try:
        # 1. Retrieve coping strategies for mentioned conditions
        conditions = entities.get("conditions", [])
        if conditions or entities.get("needs_strategies"):
            # If user needs strategies but no specific condition, check emotions
            if not conditions and entities.get("emotions"):
                emotion_to_condition = {
                    "anxious": "Anxiety", "worried": "Anxiety", "nervous": "Anxiety",
                    "sad": "Depression", "hopeless": "Depression", "empty": "Depression",
                    "stressed": "Stress", "overwhelmed": "Stress", "burnt out": "Stress",
                    "lonely": "Loneliness", "isolated": "Loneliness", "alone": "Loneliness",
                    "grief": "Grief", "loss": "Grief", "mourning": "Grief",
                    "angry": "Anger Management", "frustrated": "Anger Management",
                    "can't sleep": "Sleep Issues", "insomnia": "Sleep Issues",
                    "worthless": "Low Self-Esteem", "not good enough": "Low Self-Esteem",
                }
                for emo in entities.get("emotions", []):
                    mapped = emotion_to_condition.get(emo.lower())
                    if mapped and mapped not in conditions:
                        conditions.append(mapped)

            for condition_name in conditions:
                cypher = """
                MATCH (c:Condition {name: $name})-[:HELPED_BY]->(cs:CopingStrategy)
                RETURN cs.name AS strategy, cs.description AS description,
                       cs.steps AS steps
                """
                try:
                    result = await graph.query(
                        cypher, params={"name": condition_name}
                    )
                    for row in result.result_set:
                        ctx = f"Coping strategy for {condition_name}: {row[0]}"
                        if row[1]:
                            ctx += f" - {row[1]}"
                        if row[2]:
                            ctx += f" Steps: {row[2]}"
                        state.graph_context.append(ctx)
                except Exception:
                    logger.warning(
                        "Failed to query strategies for %s", condition_name
                    )

                # Also fetch therapy techniques
                cypher_therapy = """
                MATCH (c:Condition {name: $name})-[:TREATED_BY]->(tt:TherapyTechnique)
                RETURN tt.name AS technique, tt.description AS description
                """
                try:
                    result = await graph.query(
                        cypher_therapy, params={"name": condition_name}
                    )
                    for row in result.result_set:
                        state.graph_context.append(
                            f"Relevant therapy for {condition_name}: "
                            f"{row[0]} - {row[1]}"
                        )
                except Exception:
                    logger.warning(
                        "Failed to query techniques for %s", condition_name
                    )

        # 2. Retrieve user's personal context (past topics, people, emotions)
        if user_id:
            # People the user has mentioned before
            people = entities.get("people", [])
            if people:
                for person_name in people:
                    cypher_person = """
                    MATCH (u:User {user_id: $uid})-[:MENTIONED]->(p:Person {name: $name})
                    OPTIONAL MATCH (p)-[r]->(related)
                    RETURN p.name AS person, type(r) AS rel_type,
                           labels(related)[0] AS related_label,
                           related.name AS related_name,
                           p.context AS context
                    LIMIT 10
                    """
                    try:
                        result = await graph.query(
                            cypher_person,
                            params={"uid": user_id, "name": person_name},
                        )
                        for row in result.result_set:
                            ctx = f"Previously mentioned person: {row[0]}"
                            if row[4]:  # context field
                                ctx += f" (context: {row[4]})"
                            if row[1] and row[3]:
                                ctx += f" [{row[1]} -> {row[3]}]"
                            state.graph_context.append(ctx)
                    except Exception:
                        logger.warning(
                            "Failed to query person context for %s", person_name
                        )

            # Past topics discussed by this user
            cypher_topics = """
            MATCH (u:User {user_id: $uid})-[:DISCUSSED]->(t:Topic)
            RETURN t.name AS topic, t.last_mentioned AS last_mentioned
            ORDER BY t.last_mentioned DESC
            LIMIT 5
            """
            try:
                result = await graph.query(
                    cypher_topics, params={"uid": user_id}
                )
                for row in result.result_set:
                    state.graph_context.append(
                        f"User has previously discussed: {row[0]}"
                    )
            except Exception:
                logger.debug("Failed to query user topics.")

            # Recent emotions
            cypher_emotions = """
            MATCH (u:User {user_id: $uid})-[:FEELS]->(e:Emotion)
            RETURN e.name AS emotion, e.intensity AS intensity,
                   e.last_recorded AS last_recorded
            ORDER BY e.last_recorded DESC
            LIMIT 5
            """
            try:
                result = await graph.query(
                    cypher_emotions, params={"uid": user_id}
                )
                for row in result.result_set:
                    ctx = f"User has recently expressed: {row[0]}"
                    if row[1]:
                        ctx += f" (intensity: {row[1]})"
                    state.graph_context.append(ctx)
            except Exception:
                logger.debug("Failed to query user emotions.")

    except Exception:
        logger.exception("[FalkorDB] Graph retrieval error.")

    logger.info(
        "[Graph Retrieval] Found %d context items.", len(state.graph_context)
    )
    return state


async def llm_generation_node(state: ChatState) -> ChatState:
    """Generate the final response using graph context + conversation history."""
    try:
        llm = _get_llm()
        from app.services.llm import LLMMessage

        system_prompt = (
            "You are a compassionate mental health companion. Your role is to:\n"
            "- Listen actively and empathetically\n"
            "- Provide emotional support without judgment\n"
            "- Help users explore their feelings\n"
            "- Encourage healthy coping strategies\n"
            "- Suggest professional help when appropriate\n\n"
            "Important guidelines:\n"
            "- Never provide medical diagnoses or prescribe medication\n"
            "- Always take crisis situations seriously\n"
            "- Use warm, understanding language\n"
            "- You are a supportive companion, not a replacement for "
            "professional mental health care.\n"
        )

        if state.graph_context:
            context_block = "\n".join(f"- {c}" for c in state.graph_context)
            system_prompt += (
                f"\n\nRelevant context from knowledge graph:\n{context_block}\n\n"
                "Use this context naturally in your response. Do not list "
                "strategies mechanically -- weave them into an empathetic reply. "
                "If coping strategies are relevant, suggest 1-2 gently."
            )

        # Build message history
        messages = []
        for msg in state.conversation_history[-10:]:
            messages.append(LLMMessage(role=msg["role"], content=msg["content"]))
        messages.append(LLMMessage(role="user", content=state.user_message))

        state.llm_response = await llm.chat(messages, system_prompt=system_prompt)
    except Exception:
        logger.exception("[LLM] Generation failed.")
        state.llm_response = (
            "I hear you, and I want you to know that what you're feeling matters. "
            "I'm having a moment of difficulty processing right now, but I'm here "
            "for you. Could you tell me a bit more about what's on your mind?"
        )

    state.should_stop = True
    return state


async def graph_update_node(state: ChatState) -> ChatState:
    """After generating a response, update the user's context graph.

    Stores new entities (people, topics, emotions) as nodes linked to the user.
    This is fire-and-forget; failures are logged but don't affect the response.
    """
    graph = get_falkor_graph()
    if graph is None or not state.user_id:
        return state

    entities = state.extracted_entities
    user_id = state.user_id

    try:
        # Ensure User node exists
        await graph.query(
            "MERGE (u:User {user_id: $uid})",
            params={"uid": user_id},
        )

        # Store mentioned people
        for person_name in entities.get("people", []):
            await graph.query(
                "MATCH (u:User {user_id: $uid}) "
                "MERGE (p:Person {name: $name}) "
                "MERGE (u)-[:MENTIONED]->(p)",
                params={"uid": user_id, "name": person_name},
            )

        # Store emotions
        for emotion in entities.get("emotions", []):
            await graph.query(
                "MATCH (u:User {user_id: $uid}) "
                "MERGE (e:Emotion {name: $name}) "
                "SET e.last_recorded = timestamp() "
                "MERGE (u)-[:FEELS]->(e)",
                params={"uid": user_id, "name": emotion},
            )

        # Store topics
        for topic in entities.get("topics", []):
            await graph.query(
                "MATCH (u:User {user_id: $uid}) "
                "MERGE (t:Topic {name: $name}) "
                "SET t.last_mentioned = timestamp() "
                "MERGE (u)-[:DISCUSSED]->(t)",
                params={"uid": user_id, "name": topic},
            )

        logger.info("[Graph Update] User context updated for %s", user_id)
    except Exception:
        logger.warning("[Graph Update] Failed to update user context.", exc_info=True)

    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_graph_rag(
    user_message: str,
    user_id: str = "",
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Run the full Graph RAG pipeline and return the LLM response.

    Args:
        user_message: The current user message.
        user_id: UUID string of the authenticated user (for personal context).
        conversation_history: Recent messages as ``[{"role": ..., "content": ...}]``.

    Returns:
        The generated response string.
    """
    state = ChatState(
        session_id="api",
        user_id=user_id,
        user_message=user_message,
        conversation_history=conversation_history or [],
    )

    workflow = StateGraph()
    workflow.add_node("router", router_node)
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("graph_retrieval", graph_retrieval_node)
    workflow.add_node("llm_generation", llm_generation_node)
    workflow.add_node("graph_update", graph_update_node)

    # Wiring
    workflow.add_conditional_edge("router", lambda s: s.next_node)
    workflow.add_edge("extract_entities", "graph_retrieval")
    workflow.add_edge("graph_retrieval", "llm_generation")
    # graph_update runs after llm_generation but doesn't produce a new response
    workflow.add_edge("llm_generation", "graph_update")

    # llm_generation sets should_stop=True, but graph_update should still run.
    # Override: reset should_stop after llm_generation so graph_update executes.
    original_llm_gen = llm_generation_node

    async def _llm_then_continue(s: ChatState) -> ChatState:
        s = await original_llm_gen(s)
        s.should_stop = False  # allow graph_update to run
        return s

    workflow.nodes["llm_generation"] = _llm_then_continue

    # For the direct path (no entities), still generate + update
    workflow.add_edge("llm_generation", "graph_update")

    final_state = await workflow.execute(start_node="router", state=state)
    return final_state.llm_response or ""


class GraphRAGContextProvider(BaseContextProvider):
    """Context provider backed by the Graph RAG pipeline.

    The Graph RAG context is message-driven (fetched per-message via
    ``run_graph_rag`` in the WebSocket handler), so this provider returns
    empty string when called generically. It exists to satisfy the
    ``BaseContextProvider`` interface for the context module exports.
    """

    async def get_context(self, db, conversation_id: UUID, user_id: UUID) -> str:
        return ""


graph_rag_context_provider = GraphRAGContextProvider()
