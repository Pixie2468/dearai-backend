import asyncio
from graphrag_sdk import ConnectionConfig, GraphRAG
from app.utils.llm_setup import setup_llm
from app.schemas.graph_schema import create_graph_schema

async def main():
    llm, embedder = setup_llm()
    connection = ConnectionConfig(
        host="localhost",
        port=6379,
        graph_name="graph_test_123",
        password=""
    )
    rag = GraphRAG(
        connection=connection,
        llm=llm,
        embedder=embedder,
        embedding_dimension=768,
        schema=create_graph_schema(),
    )
    await rag.__aenter__()
    print("Ingesting...")
    res = await rag.ingest(text="User: I am sad because I didn't get the promotion\nAI: Oh no, I am so sorry to hear that.", document_id="test_1")
    print(f"Ingest Result: {res}")
    await rag.finalize()
    print("Retrieving...")
    ret = await rag.retrieve("What did I say about promotion?")
    print(f"Retrieve Result: {ret}")
    await rag.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
