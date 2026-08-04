import logging
import asyncio
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client.http import models

# Multi-tenant / Vector connection manager fallback import
try:
    from database.connection import VectorDBConnectionManager, ConnectionPoolError
except ImportError:
    class VectorDBConnectionManager:
        @classmethod
        async def get_instance(cls):
            return cls()
        async def get_qdrant_client(self):
            from config import settings
            from qdrant_client import AsyncQdrantClient
            return AsyncQdrantClient(
                url=settings.qdrant.url, 
                api_key=settings.qdrant.api_key
            )

# System Logger Configuration
logger = logging.getLogger("VectorDBSearcher")

# ----------------------------------------------------------------------
# Custom Retrieval Pipeline Exceptions
# ----------------------------------------------------------------------
class RetrievalError(Exception):
    """Base exception for search cluster operations failures."""
    pass

class EmptyQueryVectorError(RetrievalError):
    """Raised when query vector calculations map to empty vectors."""
    pass

# ----------------------------------------------------------------------
# High-Throughput Hybrid Search Engine Class
# ----------------------------------------------------------------------
class ProductionVectorSearcher:
    """
    Advanced Asynchronous Retrieval Engine executing multi-vector Hybrid Search 
    (Dense Semantic + Sparse Token Matches) and applying Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, collection_name: str, data_dir: str = "data", is_named_vector: bool = True):
        self.collection_name = collection_name
        self.data_dir = Path(data_dir)
        self.is_named_vector = is_named_vector
        # Dynamic indexing of the data directory on initialization
        self.dynamic_file_map = self._build_dynamic_file_map()

    def _build_dynamic_file_map(self) -> Dict[str, set]:
        """
        Scans the 'data/' directory and dynamically maps actual PDF filenames 
        to their core keyword sets. No hardcoding required.
        """
        file_map = {}
        if self.data_dir.exists() and self.data_dir.is_dir():
            for filepath in self.data_dir.glob("*.pdf"):
                filename = filepath.name
                # Clean filename to extract meaningful keywords (e.g., 'diabetes.pdf' -> {'diabetes'})
                clean_name = re.sub(r'[^a-zA-Z0-9]', ' ', filepath.stem).lower()
                # Ignore common stop words in filenames
                stop_words = {"and", "or", "of", "the", "report", "statement", "spec", "pdf"}
                keywords = {word for word in clean_name.split() if word not in stop_words and len(word) > 2}
                file_map[filename] = keywords
            logger.info(f"Dynamically mapped {len(file_map)} files from {self.data_dir} directory.")
        else:
            logger.warning(f"Data directory '{self.data_dir}' not found. Dynamic mapping disabled.")
        return file_map

    def _detect_file_filter(self, query_text: str, min_overlap: int = 2) -> Optional[str]:
        """
        Intelligently matches query text to available files.
        Requires at least `min_overlap` keyword matches (default: 2) to lock search to a specific file.
        Otherwise returns None to allow global vector search across all documents.
        """
        if not query_text or not self.dynamic_file_map:
            return None
        
        q_lower = query_text.lower()
        query_words = set(re.findall(r'\b\w+\b', q_lower))

        best_match_file = None
        max_overlap = 0

        # Dynamically evaluate the query against the generated file map
        for filename, file_keywords in self.dynamic_file_map.items():
            overlap = len(query_words.intersection(file_keywords))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match_file = filename

        # SAFETY VALVE: Only lock to a specific file if keyword match is strong (>= 2)
        if max_overlap >= min_overlap: 
            return best_match_file
            
        return None

    def _apply_reciprocal_rank_fusion(
        self, 
        dense_results: List[Any], 
        sparse_results: List[Any], 
        rrf_constant: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Applies mathematical Reciprocal Rank Fusion (RRF) to merge 
        and re-score hit payloads from independent vector spaces safely.
        """
        rrf_scores: Dict[str, float] = {}
        doc_mapping: Dict[str, Any] = {}

        # 1. Rank & Score Dense Hits
        for rank, hit in enumerate(dense_results):
            doc_id = str(hit.id)
            doc_mapping[doc_id] = hit
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_constant + rank + 1))

        # 2. Rank & Score Sparse Hits
        for rank, hit in enumerate(sparse_results):
            doc_id = str(hit.id)
            if doc_id not in doc_mapping:
                doc_mapping[doc_id] = hit
            # Weighted scaling for sparse hits to prevent score dilution in long queries
            sparse_weight = 0.5 
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (sparse_weight / (rrf_constant + rank + 1))

        # 3. Sort merged mapping based on computed RRF weights
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        final_fused_results = []
        for doc_id, score in sorted_docs:
            source_hit = doc_mapping[doc_id]
            payload = getattr(source_hit, "payload", {}) or {}
            
            final_fused_results.append({
                "id": doc_id,
                "score": score,
                "text": payload.get("text", "") or payload.get("page_content", ""),
                "metadata": payload.get("metadata", payload)
            })

        return final_fused_results

    async def hybrid_search(
        self, 
        dense_query_vector: List[float], 
        sparse_query_vector: Optional[Dict[str, Any]] = None,
        raw_query_text: Optional[str] = None,  
        tenant_id: Optional[str] = None,
        limit: int = 40,                       
        filter_metadata: Optional[Dict[str, Any]] = None,
        rrf_constant: int = 60,
        enforce_file_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes parallelized async search workflows across dense and sparse indexes 
        with multi-tenancy isolation and dynamic metadata pre-filter injections.
        Updated to support qdrant-client >= 1.10.0 (query_points API).
        """
        if not dense_query_vector:
            raise EmptyQueryVectorError("Primary dense query vector cannot be an empty array sequence.")

        try:
            manager = await VectorDBConnectionManager.get_instance()
            client = await manager.get_qdrant_client()

            filter_conditions = []
            
            # Multi-tenancy pre-filter mapping
            if tenant_id:
                filter_conditions.append(
                    models.FieldCondition(
                        key="tenant_id", 
                        match=models.MatchValue(value=tenant_id)
                    )
                )
            
            # Explicit Metadata pre-filters injection
            if filter_metadata:
                for key, val in filter_metadata.items():
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=val)
                        )
                    )
            # Automatic Source File Metadata Filtering
            elif raw_query_text and enforce_file_filter:
                auto_source_file = self._detect_file_filter(raw_query_text)
                if auto_source_file:
                    logger.info(f"Injecting Auto-Detected Source File Filter: [{auto_source_file}]")
                    filter_conditions.append(
                        models.FieldCondition(
                            key="metadata.source_file",
                            match=models.MatchValue(value=auto_source_file)
                        )
                    )
                else:
                    logger.info("No definitive file match detected. Executing Global Search across all documents.")
            else:
                logger.info("Auto-file filter bypassed or raw query absent. Executing Global Search.")

            query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

            # Check Sparse Query Vector payload securely
            has_sparse = (
                sparse_query_vector is not None 
                and isinstance(sparse_query_vector.get("indices"), list) 
                and len(sparse_query_vector.get("indices")) > 0
                and isinstance(sparse_query_vector.get("values"), list)
                and len(sparse_query_vector.get("values")) > 0
            )

            if has_sparse:
                logger.info(f"Triggering Asynchronous Hybrid Retrieval for [{self.collection_name}]")
                oversample_limit = max(limit * 2, 30)  

                dense_task = client.query_points(
                    collection_name=self.collection_name,
                    query=dense_query_vector,
                    using="dense",
                    query_filter=query_filter,
                    limit=oversample_limit
                )
                
                sparse_task = client.query_points(
                    collection_name=self.collection_name,
                    query=models.SparseVector(
                        indices=sparse_query_vector.get("indices", []),
                        values=sparse_query_vector.get("values", [])
                    ),
                    using="sparse",
                    query_filter=query_filter,
                    limit=oversample_limit
                )

                # Concurrent execution
                dense_response, sparse_response = await asyncio.gather(dense_task, sparse_task)
                
                # Extract ScoredPoints list from query_points response object
                dense_res = getattr(dense_response, "points", dense_response)
                sparse_res = getattr(sparse_response, "points", sparse_response)

                return self._apply_reciprocal_rank_fusion(
                    dense_results=dense_res, 
                    sparse_results=sparse_res, 
                    rrf_constant=rrf_constant
                )[:limit]

            else:
                logger.info(f"Executing standard dense vector search for [{self.collection_name}]")
                
                # Use query_points for dense-only retrieval
                dense_response = await client.query_points(
                    collection_name=self.collection_name,
                    query=dense_query_vector,
                    using="dense" if getattr(self, "is_named_vector", True) else None,
                    query_filter=query_filter,
                    limit=limit
                )
                
                dense_res = getattr(dense_response, "points", dense_response)
                
                results = []
                for hit in dense_res:
                    payload = getattr(hit, "payload", {}) or {}
                    results.append({
                        "id": str(hit.id),
                        "score": hit.score,
                        "text": payload.get("text", "") or payload.get("page_content", ""),
                        "metadata": payload.get("metadata", payload)
                    })
                return results

        except Exception as e:
            logger.critical(f"Failed to execute vector search workflow: {str(e)}")
            raise RetrievalError(f"Vector search backend operation aborted: {str(e)}")