import asyncio
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval.search import ProductionVectorSearcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestHybridSearch")


async def test_pure_hybrid_retrieval(user_query: str):
    print("\n" + "=" * 60)
    print(f"HYBRID SEARCH TESTING FOR: '{user_query}'")
    print("=" * 60)

    # Main Hybrid Search Engine initialize hoga
    search_engine = ProductionVectorSearcher(collection_name="enterprise_rag_vector_index")

    try:
        # Mock Dense & Sparse Query Vectors (Aapke embedding dimension ke acc: default 384/1536)
        # Note: Production pipeline me embedding model text ko vector me convert karega
        mock_dense_vector = [0.1] * 384  
        mock_sparse_vector = {"indices": [1, 5, 10], "values": [0.4, 0.7, 0.9]}

        # Hybrid Search Call
        results = await search_engine.hybrid_search(
            dense_query_vector=mock_dense_vector,
            sparse_query_vector=mock_sparse_vector,
            limit=3
        )

        print("\n" + "-" * 50)
        print(f"FOUND {len(results)} HYBRID MATCHES (DENSE + SPARSE + RRF):")
        print("-" * 50)

        if not results:
            print("No matching documents found in collection.")

        for rank, hit in enumerate(results, 1):
            print(f"\n[Result #{rank}] (RRF Score: {hit['score']:.4f})")
            print(f"ID: {hit['id']}")
            print(f"Text Snippet: {hit.get('text', 'N/A')}")
            print(f"Metadata: {hit.get('metadata', {})}")

    except Exception as e:
        logger.error(f"Hybrid Search failed: {e}", exc_info=True)


if __name__ == "__main__":
    query = "What are the rules and policies in the document?"
    asyncio.run(test_pure_hybrid_retrieval(query))