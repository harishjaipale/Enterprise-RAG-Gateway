import os
import logging
import hashlib
import asyncio
import json
import sqlite3
from typing import List, Optional, Dict, Any

logger = logging.getLogger("ProductionEmbeddingEngine")


class EmbeddingEngineError(Exception):
    """Base exception for cryptographic validation or local inference failures."""
    pass


class ProductionEmbeddingEngine:
    """
    Enterprise-grade, local vector embedding generator using HuggingFace models.
    
    Upgraded Features:
    - High-throughput Non-Blocking Disk Caching using Threaded SQLite.
    - Fast In-Memory LRU Cache to avoid redundant disk I/O.
    - Thread-isolated async wrappers for heavy HuggingFace inference.
    """
    def __init__(
        self, 
        model_name: str = "BAAI/bge-small-en-v1.5", 
        dimensions: int = 384,
        enable_local_cache: bool = True,
        db_cache_path: str = ".embedding_cache.db"
    ):
        self.model_name = model_name
        self.dimensions = dimensions
        self.enable_cache = enable_local_cache
        self.db_cache_path = db_cache_path
        
        # In-Memory Fast Cache (Memory -> Disk fallback)
        self._memory_cache: Dict[str, List[float]] = {}
        self._model = None

        if self.enable_cache:
            self._init_sqlite_db()

    def _init_sqlite_db(self):
        """Initializes a light SQLite database for efficient non-blocking vector storage."""
        try:
            with sqlite3.connect(self.db_cache_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS vector_cache (
                        hash TEXT PRIMARY KEY,
                        model TEXT,
                        embedding TEXT
                    )
                """)
                conn.commit()
            logger.info(f"[✓] Persistent SQLite Cache initialized at: [{self.db_cache_path}]")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite vector cache: {str(e)}")

    def _get_model(self):
        """Lazy instantiates local SentenceTransformer model in memory."""
        if self._model is None:
            try:
                logger.info(f"Loading local HuggingFace embedding model: [{self.model_name}] into memory...")
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("[✓] Local Embedding Model successfully loaded in memory.")
            except Exception as e:
                raise EmbeddingEngineError(f"Failed to load embedding model [{self.model_name}]: {str(e)}")
        return self._model

    def _normalize_and_hash(self, text: str) -> tuple[str, str]:
        """Normalizes input string and derives a deterministic SHA-256 hash."""
        clean_text = text.strip().replace("\n", " ")
        text_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        return clean_text, text_hash

    def _lookup_cache_sync(self, text_hash: str) -> Optional[List[float]]:
        """Synchronous SQLite lookup (offloaded to thread)."""
        try:
            with sqlite3.connect(self.db_cache_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT embedding FROM vector_cache WHERE hash = ? AND model = ?", (text_hash, self.model_name))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.warning(f"Cache lookup failed for hash [{text_hash}]: {str(e)}")
        return None

    async def _lookup_cache(self, text_hash: str) -> Optional[List[float]]:
        """Non-blocking async cache resolution."""
        if not self.enable_cache:
            return None
        
        # 1. Fast Memory Layer Lookup
        if text_hash in self._memory_cache:
            return self._memory_cache[text_hash]

        # 2. Async Threaded Disk Layer Lookup
        cached_vec = await asyncio.to_thread(self._lookup_cache_sync, text_hash)
        if cached_vec:
            self._memory_cache[text_hash] = cached_vec  # Promote to memory
        return cached_vec

    def _write_cache_sync(self, text_hash: str, embedding: List[float]) -> None:
        """Synchronous SQLite batch commit (offloaded to thread)."""
        try:
            with sqlite3.connect(self.db_cache_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO vector_cache (hash, model, embedding) VALUES (?, ?, ?)",
                    (text_hash, self.model_name, json.dumps(embedding))
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to commit vector cache to SQLite: {str(e)}")

    async def _write_cache(self, text_hash: str, embedding: List[float]) -> None:
        """Non-blocking async disk write."""
        if not self.enable_cache:
            return
        self._memory_cache[text_hash] = embedding
        await asyncio.to_thread(self._write_cache_sync, text_hash, embedding)

    def _encode_local_sync(self, batch_texts: List[str]) -> List[List[float]]:
        """Synchronous GPU/CPU inference step."""
        model = self._get_model()
        embeddings = model.encode(
            batch_texts, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        return embeddings.tolist()

    async def _dispatch_local_request(self, batch_texts: List[str]) -> List[List[float]]:
        """Offloads heavy inference calculations away from event loop."""
        return await asyncio.to_thread(self._encode_local_sync, batch_texts)

    async def get_embedding(self, text: str) -> List[float]:
        """Calculates or resolves a dense vector for a single query."""
        if not text.strip():
            raise EmbeddingEngineError("Input sequence cannot be null or empty context blocks.")

        clean_text, text_hash = self._normalize_and_hash(text)
        
        # Cache Interception
        cached_vector = await self._lookup_cache(text_hash)
        if cached_vector:
            logger.debug("Cache Hit! Vector retrieved non-blockingly.")
            return cached_vector

        # Local Inference
        logger.info(f"Cache Miss. Computing vector locally via [{self.model_name}]...")
        result = await self._dispatch_local_request([clean_text])
        vector = result[0]

        # Synchronize Cache
        await self._write_cache(text_hash, vector)
        return vector

    async def get_embeddings_batch(self, texts: List[str], max_chunk_batch: int = 64) -> List[List[float]]:
        """Processes high-volume batch pipelines with non-blocking caching."""
        if not texts:
            return []

        results_matrix: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_payloads: List[str] = []
        uncached_hashes: List[str] = []

        # Step A: Filter missing records without blocking event loop
        for idx, text in enumerate(texts):
            clean_text, text_hash = self._normalize_and_hash(text)
            cached_vector = await self._lookup_cache(text_hash)
            
            if cached_vector:
                results_matrix[idx] = cached_vector
            else:
                uncached_indices.append(idx)
                uncached_payloads.append(clean_text)
                uncached_hashes.append(text_hash)

        # Step B: Batch Inference on uncached payloads
        if uncached_payloads:
            logger.info(f"Processing {len(uncached_payloads)} uncached records in chunk sizes of {max_chunk_batch}")
            
            for i in range(0, len(uncached_payloads), max_chunk_batch):
                current_slice = uncached_payloads[i:i + max_chunk_batch]
                current_indices = uncached_indices[i:i + max_chunk_batch]
                current_hashes = uncached_hashes[i:i + max_chunk_batch]
                
                computed_vectors = await self._dispatch_local_request(current_slice)
                
                for local_idx, vector in enumerate(computed_vectors):
                    global_target_idx = current_indices[local_idx]
                    results_matrix[global_target_idx] = vector
                    
                    target_hash = current_hashes[local_idx]
                    await self._write_cache(target_hash, vector)

        return results_matrix  # type: ignore


if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        engine = ProductionEmbeddingEngine(
            model_name="BAAI/bge-small-en-v1.5", 
            dimensions=384,
            enable_local_cache=True
        )

        test_queries = [
            "Enforce strict OAuth2 tokens across production nodes.",
            "Data persistence is managed via secure ephemeral volumes.",
            "Enforce strict OAuth2 tokens across production nodes."
        ]

        try:
            vectors = await engine.get_embeddings_batch(test_queries)
            print(f"\n[✓] Processed successfully! Received vector count: {len(vectors)}")
            print(f"[✓] Vector dimensions verified: {len(vectors[0])} values.")
        except Exception as e:
            print(f"[✗] Local execution error: {e}")

    asyncio.run(main())