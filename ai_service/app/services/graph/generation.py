"""Graph ingestion helpers."""

from graphrag_sdk import GraphRAG, IngestionResult


async def process_user_query(
    rag: GraphRAG, user_query: str, document_id: str
) -> IngestionResult:
    """Ingest user text into the graph and finalize the transaction."""
    ingest_result = await rag.ingest(text=user_query, document_id=document_id)

    await rag.finalize()

    return ingest_result
