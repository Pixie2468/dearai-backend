"""Graph retrieval helpers."""

import logging

from graphrag_sdk import GraphRAG

logger = logging.getLogger(__name__)


async def get_graph_context(rag: GraphRAG, user_query: str) -> str:
    """Retrieve relevant context from the graph for a user query."""
    retrieval_result = await rag.retrieve(user_query)

    if not retrieval_result or not retrieval_result.items:
        logger.info("No prior context found in graph for query: '%s'", user_query)
        return "No prior context found."

    return str(retrieval_result)
