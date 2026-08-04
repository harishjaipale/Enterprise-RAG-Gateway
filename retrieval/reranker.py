import os
import logging
import asyncio
import math
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger("NeuralReranker")


class RerankerError(Exception):
    """Base exception for reranking subsystem operations."""
    pass


class ModelInitializationError(RerankerError):
    """Raised when local or remote reranker models fail to load."""
    pass


class RerankExecutionError(RerankerError):
    """Raised when runtime scoring transformations fail."""
    pass


class ProductionDocumentReranker:
    """
    A production-ready asynchronous document reranking engine.
    Supports both remote cloud APIs (e.g., Cohere) and local open-source Cross-Encoders 
    (e.g., BAAI/bge-reranker-base) with automatic thread pool isolation.
    """
    def __init__(
        self, 
        provider: str = "local", 
        model_name: str = "BAAI/bge-reranker-base", 
        api_key: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        
        self._local_model = None
        self._cohere_client = None

    def _init_local_model(self):
        if self._local_model is None:
            try:
                logger.info(f"Loading local neural Cross-Encoder model: [{self.model_name}]...")
                from sentence_transformers import CrossEncoder
                self._local_model = CrossEncoder(self.model_name)
                logger.info("Local Cross-Encoder model loaded successfully.")
            except Exception as e:
                raise ModelInitializationError(f"Failed to instantiate sentence-transformers model: {str(e)}")

    def _init_cohere_client(self):
        if self._cohere_client is None:
            if not self.api_key:
                raise ModelInitializationError("COHERE_API_KEY is missing.")
            try:
                logger.info("Initializing Cohere Rerank API client...")
                import cohere
                self._cohere_client = cohere.AsyncClient(api_key=self.api_key)
            except Exception as e:
                raise ModelInitializationError(f"Failed to instantiate Cohere endpoint client: {str(e)}")

    def _sigmoid(self, x: float) -> float:
        """Helper to convert raw logits into 0-1 probability scores."""
        return 1 / (1 + math.exp(-x))

    def _extract_text_and_meta(self, doc: Any) -> Dict[str, Any]:
        """
        Normalizes various input document formats (dict, LangChain Document, Qdrant Record)
        into a uniform internal dictionary structure.
        """
        if isinstance(doc, dict):
            # Handles dictionary payloads
            text = doc.get("text") or doc.get("page_content") or doc.get("content", "")
            metadata = doc.get("metadata") or doc.get("payload", {})
            doc_id = doc.get("id") or metadata.get("id", None)
            return {
                "id": doc_id, 
                "text": text, 
                "raw_text": doc.get("raw_text", text),
                "metadata": metadata, 
                "_original_doc": doc
            }
        
        # Handles Objects (e.g. LangChain Document / Qdrant ScoredPoint)
        text = getattr(doc, "page_content", None) or getattr(doc, "text", "")
        if not text and hasattr(doc, "payload"):
            payload = getattr(doc, "payload", {}) or {}
            text = payload.get("text", "") or payload.get("page_content", "")
            metadata = payload.get("metadata", payload)
        else:
            metadata = getattr(doc, "metadata", {})
            
        doc_id = getattr(doc, "id", None)
        return {
            "id": doc_id, 
            "text": text, 
            "raw_text": text,
            "metadata": metadata, 
            "_original_doc": doc
        }

    async def _rerank_local(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._init_local_model()
        
        try:
            pairs = [[query, doc.get("text", "")] for doc in documents]
            
            loop = asyncio.get_running_loop()
            # Batch execution for memory safety on large candidate sets
            scores = await loop.run_in_executor(
                None, 
                lambda: self._local_model.predict(pairs, batch_size=32)
            )
            
            reranked_docs = []
            for idx, raw_score in enumerate(scores):
                doc_payload = documents[idx].copy()
                doc_payload["rerank_score"] = self._sigmoid(float(raw_score))
                reranked_docs.append(doc_payload)
                
            return sorted(reranked_docs, key=lambda x: x["rerank_score"], reverse=True)
            
        except Exception as e:
            raise RerankExecutionError(f"Local Cross-Encoder scoring process failed: {str(e)}")

    async def _rerank_cohere(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._init_cohere_client()
        
        try:
            texts = [doc.get("text", "") for doc in documents]
            
            target_model = "rerank-english-v3.0" if self.model_name == "BAAI/bge-reranker-base" else self.model_name
            
            response = await self._cohere_client.rerank(
                model=target_model,
                query=query,
                documents=texts,
                top_n=len(texts)
            )
            
            reranked_docs = []
            for result in response.results:
                original_idx = result.index
                doc_payload = documents[original_idx].copy()
                doc_payload["rerank_score"] = float(result.relevance_score)
                reranked_docs.append(doc_payload)
                
            return reranked_docs
        except Exception as e:
            raise RerankExecutionError(f"Cohere cloud endpoint processing failed: {str(e)}")

    async def rerank(
        self, 
        query: str, 
        documents: List[Any], 
        top_n: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks retrieved candidate documents using cross-encoders or remote API endpoints.
        Accepts dicts, Objects, or VectorDB Payload points seamlessly.
        """
        if not documents:
            logger.warning("Empty document array passed to reranker.")
            return []

        # Step 1: Normalize all input documents into uniform structure
        normalized_docs = [self._extract_text_and_meta(doc) for doc in documents]

        logger.info(f"Initiating reranking using provider [{self.provider.upper()}] over {len(normalized_docs)} documents.")
        
        # Step 2: Rerank based on provider
        if self.provider == "local":
            sorted_docs = await self._rerank_local(query, normalized_docs)
        elif self.provider == "cohere":
            sorted_docs = await self._rerank_cohere(query, normalized_docs)
        else:
            raise RerankerError(f"Unsupported provider parameter: {self.provider}")

        # Step 3: Score threshold check with fallback safety
        if score_threshold is not None:
            filtered_docs = [doc for doc in sorted_docs if doc.get("rerank_score", 0.0) >= score_threshold]
            
            if filtered_docs:
                sorted_docs = filtered_docs
            else:
                logger.warning(
                    f"All {len(sorted_docs)} documents fell below score_threshold [{score_threshold}]. "
                    f"Bypassing threshold to prevent empty context failure."
                )

        return sorted_docs[:top_n]


if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        reranker = ProductionDocumentReranker(provider="local", model_name="BAAI/bge-reranker-base")

        mock_query = "What is the company compliance policy for data security?"
        mock_hits = [
            {"id": "chunk_01", "text": "Employees must wear ID badges at all times.", "metadata": {"file": "hr.pdf"}},
            {"id": "chunk_02", "text": "Data protection rules enforce strict AES-256 encryption across databases.", "metadata": {"file": "security.pdf"}},
            {"id": "chunk_03", "text": "Cafeteria lunch hours are scheduled strictly from 12 PM to 2 PM daily.", "metadata": {"file": "ops.pdf"}}
        ]

        try:
            results = await reranker.rerank(query=mock_query, documents=mock_hits, top_n=8, score_threshold=0.3)
            print(f"\nReranked Results ({len(results)} chunks returned):")
            for res in results:
                print(f" - [{res['rerank_score']:.4f}] {res['text']}")
        except Exception as e:
            print(f"Error during execution: {e}")

    asyncio.run(main())