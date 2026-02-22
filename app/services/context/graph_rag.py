import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, Any
from uuid import UUID

# FalkorDB Async Imports
from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool

from app.services.context.base import BaseContextProvider

# Configure trace logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GraphRAG")


# State & Type Definitions
@dataclass
class ChatState:
    session_id: str
    user_message: str
    extracted_entities: dict[str, Any] = field(default_factory=dict)
    graph_context: list[str] = field(default_factory=list)
    llm_response: str | None = None
    should_stop: bool = False
    next_node: str | None = None


NodeFunc = Callable[[ChatState], Awaitable[ChatState]]
ConditionFunc = Callable[[ChatState], str | None]


# Graph Architecture
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
        logger.info(f"\n--- Starting Graph RAG Execution ---")

        while current_node_name and not state.should_stop:
            logger.info(f"Executing: '{current_node_name}'")
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


# Graph RAG Nodes


async def router_node(state: ChatState) -> ChatState:
    """Decides if we need Graph context based on the query."""
    if "alice" in state.user_message.lower() or "who" in state.user_message.lower():
        logger.info("  [Router] Graph search required. Routing to extraction.")
        state.next_node = "extract_entities"
    else:
        logger.info("  [Router] General query. Routing to LLM directly.")
        state.next_node = "llm_generation"
    return state


async def extract_entities_node(state: ChatState) -> ChatState:
    """Simulates an LLM structured output call to extract Cypher parameters."""
    logger.info("  [Extractor] Extracting parameters for Cypher query...")
    # Simulated entity extraction
    if "alice" in state.user_message.lower():
        state.extracted_entities["person_name"] = "Alice"
    return state


def make_graph_retrieval_node(db_graph) -> NodeFunc:
    """
    Dependency Injection pattern: Injects the FalkorDB connection into the node
    while maintaining the strict `state -> state` signature expected by the Graph.
    """

    async def graph_retrieval_node(state: ChatState) -> ChatState:
        person_name = state.extracted_entities.get("person_name")
        if not person_name:
            return state

        logger.info(f"  [FalkorDB] Executing custom Cypher for: {person_name}")

        cypher_query = """
        MATCH (p:Person {name: $name})-[:KNOWS]->(friend:Person)
        OPTIONAL MATCH (friend)-[:HAS_EXPERTISE]->(skill:Skill)
        RETURN friend.name AS friend_name, collect(skill.name) AS skills
        """
        params = {"name": person_name}

        try:
            # Execute async query against FalkorDB
            result = await db_graph.query(cypher_query, params=params)

            # Format results for the LLM context window
            for row in result.result_set:
                friend_name = row[0]
                skills = ", ".join(row[1]) if row[1] else "no specific skills"
                context_str = (
                    f"{person_name} knows {friend_name}, who is skilled in {skills}."
                )
                state.graph_context.append(context_str)
                logger.info(f"  [Context Found] {context_str}")

        except Exception as e:
            logger.error(f"  [FalkorDB Error] Failed to execute Cypher: {e}")

        return state

    return graph_retrieval_node


async def llm_generation_node(state: ChatState) -> ChatState:
    """Constructs the final response, injecting graph context if available."""
    logger.info("  [LLM] Generating final response...")

    if state.graph_context:
        context_block = "\n".join(state.graph_context)
        state.llm_response = (
            f"Based on the Knowledge Graph context provided:\n"
            f"{context_block}\n\n"
            f"(Simulated LLM Synthesis: I can see that Alice has connections with valuable skills!)"
        )
    else:
        state.llm_response = (
            "I am a helpful AI. I don't need the graph for this question!"
        )

    state.should_stop = True
    return state


# Execution & Database Setup


async def seed_database(kg_graph):
    """Utility to safely seed the DB so the example runs perfectly."""
    try:
        await kg_graph.query(
            """
        MERGE (a:Person {name: 'Alice'})
        MERGE (b:Person {name: 'Bob'})
        MERGE (c:Person {name: 'Charlie'})
        MERGE (s1:Skill {name: 'Python'})
        MERGE (s2:Skill {name: 'Graph Databases'})
        
        MERGE (a)-[:KNOWS]->(b)
        MERGE (a)-[:KNOWS]->(c)
        MERGE (b)-[:HAS_EXPERTISE]->(s1)
        MERGE (b)-[:HAS_EXPERTISE]->(s2)
        """
        )
    except Exception as e:
        logger.error(f"Failed to seed data. Is Docker running? Error: {e}")


async def main():
    # Setup FalkorDB Async Connection Pool
    pool = BlockingConnectionPool(host="localhost", port=6379, decode_responses=True)
    falkor_db = FalkorDB(connection_pool=pool)
    kg_graph = falkor_db.select_graph("rag_graph")

    await seed_database(kg_graph)

    # Initialize Workflow
    workflow = StateGraph()
    workflow.add_node("router", router_node)
    workflow.add_node("extract_entities", extract_entities_node)

    # Inject DB here using the closure function
    workflow.add_node("graph_retrieval", make_graph_retrieval_node(kg_graph))
    workflow.add_node("llm_generation", llm_generation_node)

    # Wire edges
    workflow.add_conditional_edge("router", lambda s: s.next_node)
    workflow.add_edge("extract_entities", "graph_retrieval")
    workflow.add_edge("graph_retrieval", "llm_generation")

    # Run Scenario
    state = ChatState(
        session_id="rag_session_01",
        user_message="Who does Alice know and what are their skills?",
    )

    final_state = await workflow.execute(start_node="router", state=state)
    print(f"\nFinal LLM Output:\n{final_state.llm_response}")

    # Cleanup Database Pool
    await pool.aclose()


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# Public API used by chat handlers and context __init__
# ---------------------------------------------------------------------------


async def run_graph_rag(user_message: str) -> str:
    """Run the Graph RAG pipeline on a user message and return the LLM response."""
    state = ChatState(session_id="api", user_message=user_message)

    workflow = StateGraph()
    workflow.add_node("router", router_node)
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("llm_generation", llm_generation_node)

    workflow.add_conditional_edge("router", lambda s: s.next_node)
    workflow.add_edge("extract_entities", "llm_generation")

    final_state = await workflow.execute(start_node="router", state=state)
    return final_state.llm_response or ""


class GraphRAGContextProvider(BaseContextProvider):
    """Context provider backed by the Graph RAG pipeline."""

    async def get_context(self, db, conversation_id: UUID, user_id: UUID) -> str:
        # Graph RAG context is message-driven; return empty when no
        # specific query is available (context is fetched per-message
        # via run_graph_rag in the WebSocket handler).
        return ""


graph_rag_context_provider = GraphRAGContextProvider()
