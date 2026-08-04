import sys
import inspect
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure root directory is at the top of sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Subsystem Imports
from llm.query_expander import MultiQueryExpander
from llm.embeddings import ProductionEmbeddingEngine
from retrieval.search import ProductionVectorSearcher
from retrieval.reranker import ProductionDocumentReranker
from retrieval.context_builder import ParentContextAssembler
from llm.generator import ProductionLLMGenerator

logger = logging.getLogger("EnterpriseQueryEngine")

ENTERPRISE_RAG_SYSTEM_PROMPT = """You are an Enterprise AI Knowledge Gateway Assistant.
Your primary task is to answer user queries strictly using the provided VERIFIED KNOWLEDGE CONTEXT.

STRICT EXECUTION RULES:
1. Base all answers strictly on the facts, tables, API endpoints, and metrics in the provided context.
2. Do NOT hallucinate or extrapolate beyond the verified context.
3. If the context partially answers the question, state what is available and clearly identify what details are missing.
4. If NO context matches the question, explicitly state: "Requested information is unavailable in the indexed document repository."
5. Format technical output (APIs, JSON, Code, Tables) clearly using Markdown."""

class EnterpriseQueryEngine:
    def __init__(
        self, 
        collection_name: str = "enterprise_rag_vector_index",
        reranker_provider: str = "local"
    ):
        self.expander = MultiQueryExpander()
        self.searcher = ProductionVectorSearcher(collection_name=collection_name)
        self.reranker = ProductionDocumentReranker(provider=reranker_provider)
        self.llm_generator = ProductionLLMGenerator()
        self.embedding_engine = ProductionEmbeddingEngine()

    async def _generate_real_embedding(self, text: str) -> List[float]:
        """Safely awaits embedding engine coroutines."""
        if hasattr(self.embedding_engine, "get_embedding_async"):
            return await self.embedding_engine.get_embedding_async(text)
        elif hasattr(self.embedding_engine, "get_embedding"):
            res = self.embedding_engine.get_embedding(text)
            if inspect.isawaitable(res):
                return await res
            return res
        elif hasattr(self.embedding_engine, "encode"):
            return self.embedding_engine.encode(text).tolist()
        else:
            res = self.embedding_engine.embed_query(text)
            if inspect.isawaitable(res):
                return await res
            return res

    async def execute_query(
        self, 
        user_query: str, 
        tenant_id: Optional[str] = None,
        search_top_k: int = 25,
        rerank_top_n: int = 8,
        rerank_score_threshold: float = 0.05  # Relaxed slightly to prevent false-negative drops
    ) -> Dict[str, Any]:
        logger.info(f"Initiating Enterprise RAG Pipeline for query: '{user_query}'")

        # STEP 1: Query Expansion
        expanded_queries = await self.expander.expand_query(user_query)
        if not expanded_queries:
            expanded_queries = [user_query]
        elif user_query not in expanded_queries:
            # Ensure the primary user query is at index 0
            expanded_queries.insert(0, user_query)
            
        logger.info(f"Expanded raw query into {len(expanded_queries)} search vectors.")

        # STEP 2: Parallel Search with Global Recall Optimization
        search_tasks = []
        for idx, q_text in enumerate(expanded_queries):
            query_vector = await self._generate_real_embedding(q_text)
            
            # Disable strict file filter restrictions during search to ensure all indexed files participate
            task = self.searcher.hybrid_search(
                dense_query_vector=query_vector,
                raw_query_text=q_text,
                tenant_id=tenant_id,
                limit=search_top_k,
                enforce_file_filter=False 
            )
            search_tasks.append(task)

        query_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

        all_retrieved_hits: List[Dict[str, Any]] = []
        for res in query_results_list:
            if isinstance(res, list):
                all_retrieved_hits.extend(res)
            elif isinstance(res, Exception):
                logger.error(f"Search task execution fault: {res}")

        # STEP 3: Deduplication across search vectors
        unique_hits_dict: Dict[str, Dict[str, Any]] = {}
        for hit in all_retrieved_hits:
            hit_id = hit.get("id") or hit.get("metadata", {}).get("chunk_id") or hit.get("payload", {}).get("chunk_id")
            if hit_id and hit_id not in unique_hits_dict:
                unique_hits_dict[hit_id] = hit

        deduplicated_hits = list(unique_hits_dict.values())
        logger.info(f"Retrieved {len(deduplicated_hits)} unique candidate chunks after search fusion.")

        # STEP 4: Neural Reranking
        reranked_hits = await self.reranker.rerank(
            query=user_query,
            documents=deduplicated_hits,
            top_n=rerank_top_n,
            score_threshold=rerank_score_threshold
        )
        logger.info(f"Neural Cross-Encoder filtered down to {len(reranked_hits)} top chunks.")

        # STEP 5: Parent Context Assembly
        formatted_context = ParentContextAssembler.build_llm_prompt_context(reranked_hits)
        final_context_payload = formatted_context if formatted_context.strip() else "NO RECORD FOUND"

        # STEP 6: System Prompt Formulation (For inspection/logging)
        final_prompt = ENTERPRISE_RAG_SYSTEM_PROMPT + f"\n\nVERIFIED KNOWLEDGE CONTEXT:\n{final_context_payload}\n\nUSER QUESTION:\n{user_query}\n\nANSWER:"

        # STEP 7: Synthesis / Answer Generation (Passing custom Enterprise System Instruction)
        final_answer = await self.llm_generator.generate_answer(
            query=user_query,
            context_chunks=reranked_hits,
            custom_system_instruction=ENTERPRISE_RAG_SYSTEM_PROMPT
        )

        return {
            "query": user_query,
            "expanded_queries": expanded_queries,
            "retrieved_chunks_count": len(deduplicated_hits),
            "reranked_chunks_count": len(reranked_hits),
            "context_payload": final_context_payload,
            "final_prompt": final_prompt,
            "answer": final_answer
        }

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        engine = EnterpriseQueryEngine()
        
        # Yahan apni nayi query dalein aur file SAVE karein (Ctrl + S)
        test_query = "List all components (Ingestion, Embedding, Vector DB, LLM Gen) along with their exact throughput and P99 latency metrics in a Markdown table."
        output = await engine.execute_query(test_query)
        
        print("\n" + "="*60)
        print("ENTERPRISE RAG PIPELINE EXECUTION RESULT")
        print("="*60)
        print(f"User Query      : {output['query']}")
        print(f"Expanded Queries: {output['expanded_queries']}")
        print(f"Retrieved Chunks: {output['retrieved_chunks_count']}")
        print(f"Reranked Chunks : {output['reranked_chunks_count']}")
        print("="*60)
        print("LLM ANSWER:")
        print(output["answer"])
        print("="*60 + "\n")

    asyncio.run(main())