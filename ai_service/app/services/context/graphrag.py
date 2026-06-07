"""GraphRAG orchestration for Dear AI context retrieval."""

import logging
import os
import uuid

from graphrag_sdk import ConnectionConfig, GraphRAG

from app.schemas.graph_schema import create_graph_schema
from app.services.graph.generation import process_user_query
from app.services.graph.retrieval import get_graph_context
from app.utils.llm_setup import setup_llm

logger = logging.getLogger(__name__)


class DearAIGraphService:
    """Manages GraphRAG lifecycle and operations for a single user."""

    def __init__(self, user_id: str):
        self.llm, self.embedder = setup_llm()

        self.connection = ConnectionConfig(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", 6379)),
            graph_name=f"graph_{user_id}",
            password=os.getenv("FALKORDB_PASSWORD", ""),
        )

        self.rag: GraphRAG | None = None

    async def __aenter__(self):
        """Initialize the GraphRAG instance for async usage."""
        self.rag = GraphRAG(
            connection=self.connection,
            llm=self.llm,
            embedder=self.embedder,
            embedding_dimension=768,
            schema=create_graph_schema(),
        )
        await self.rag.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure FalkorDB connections are safely closed."""
        if self.rag:
            await self.rag.__aexit__(exc_type, exc_val, exc_tb)

    async def execute_graph_pipeline(self, user_query: str) -> str:
        """Ingest new text, finalize, and retrieve relevant graph context."""
        interaction_id = f"interaction_{uuid.uuid4().hex[:8]}"

        if self.rag is None:
            raise RuntimeError(
                "GraphRAG instance is not initialized. Use 'async with DearAIGraphService(...)' before calling this method."
            )

        await process_user_query(self.rag, user_query, interaction_id)

        return await get_graph_context(self.rag, user_query)
