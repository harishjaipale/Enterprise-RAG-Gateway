import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff

# Ingestion domain dataclasses mapping references
try:
    from ingestion.chunker import ChildChunk
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class ChildChunk:
        id: str
        text: str
        metadata: Dict[str, Any]

try:
    from database.connection import VectorDBConnectionManager, ConnectionPoolError
except ImportError:
    # Dummy placeholder for standalone testing
    class VectorDBConnectionManager:
        @classmethod
        async def get_instance(cls):
            return cls()
        async def get_qdrant_client(self):
            from qdrant_client import AsyncQdrantClient
            return AsyncQdrantClient(url="http://localhost:6333")

# Setup system logger
logger = logging.getLogger("VectorDBIndexer")


# ----------------------------------------------------------------------
# Custom Indexing Pipeline Exceptions
# ----------------------------------------------------------------------
class IndexerError(Exception):
    """Base exception for cluster indexing failures."""
    pass


class CollectionCreationFailed(IndexerError):
    """Raised when schema validation initialization fails on the host cluster."""
    pass


class BatchUpsertFailed(IndexerError):
    """Raised when bulk stream pushes face processing pipeline aborts."""
    pass


class ChunkValidationError(IndexerError):
    """Raised when generated chunks break system boundary limits."""
    pass


# ----------------------------------------------------------------------
# Production Vector Indexer Class
# ----------------------------------------------------------------------
class ProductionVectorIndexer:
    """
    A unified, high-performance asynchronous indexing engine for building schemas, 
    structuring HNSW parameters, and executing concurrent batch payloads.
    """
    def __init__(self, collection_name: str, vector_dimension: int = 1536):
        """
        Initializes the indexer instance.
        Default dimension 1536 maps to OpenAI text-embedding-3-small standard.
        """
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension

    def _format_to_valid_uuid(self, chunk_id: str) -> str:
        """
        Converts arbitrary string IDs into deterministic UUIDv5 strings
        required for Qdrant storage compatibility.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    async def create_optimized_collection(
        self, 
        recreate: bool = False,
        m_param: int = 16, 
        ef_construct: int = 200
    ) -> None:
        """
        Creates or validates an optimized collection schema using enterprise 
        HNSW indexing and Sparse Vector allocation hooks for Hybrid Search.
        """
        try:
            manager = await VectorDBConnectionManager.get_instance()
            client = await manager.get_qdrant_client()
            
            collections_response = await client.get_collections()
            existing_collections = [col.name for col in collections_response.collections]
            
            # Handle collection drop/recreation explicitly
            if self.collection_name in existing_collections:
                if recreate:
                    logger.info(f"Recreate flag active. Deleting existing collection: [{self.collection_name}]")
                    await client.delete_collection(collection_name=self.collection_name)
                else:
                    logger.info(f"Collection [{self.collection_name}] already operational. Skipping schema creation.")
                    return

            logger.info(f"Configuring enterprise schema bounds for collection: [{self.collection_name}]")
            
            # Enterprise HNSW Configuration (High Accuracy + Memory Preservation)
            hnsw_config = HnswConfigDiff(
                m=m_param,
                ef_construct=ef_construct,
                full_scan_threshold=10000,
                on_disk=True  # Optimizes RAM by storing vector graph indexes on disk
            )

            # Modern Qdrant Sparse Vector Configuration
            sparse_config = None
            if hasattr(models, 'SparseVectorParams'):
                sparse_config = {
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True
                        )
                    )
                }

            await client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.vector_dimension,
                        distance=Distance.COSINE,
                        hnsw_config=hnsw_config
                    )
                },
                sparse_vectors_config=sparse_config
            )
            
            # Create Keyword Index on source_file for sub-millisecond filtering
            await client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.source_file",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            logger.info(f"Successfully deployed optimized collection schema: [{self.collection_name}]")
            
        except Exception as e:
            logger.error(f"Failed to compile collection structure: {str(e)}")
            raise CollectionCreationFailed(f"Cluster rejected collection configs: {str(e)}")

    async def batch_upsert_chunks(
        self, 
        chunks: List[ChildChunk], 
        dense_embeddings: List[List[float]],
        sparse_embeddings: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 256
    ) -> None:
        """
        Executes multi-threaded chunk updates using automated batch window mechanics.
        """
        if not chunks:
            logger.warning("Empty chunk matrix data passed to indexer. Operation skipped.")
            return

        if len(chunks) != len(dense_embeddings):
            raise ChunkValidationError("Dimension mismatch: Chunks length must match embeddings length.")

        try:
            manager = await VectorDBConnectionManager.get_instance()
            client = await manager.get_qdrant_client()
            
            total_records = len(chunks)
            logger.info(f"Initiating bulk indexing execution stream: Total [{total_records}] items...")

            for i in range(0, total_records, batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_dense = dense_embeddings[i : i + batch_size]
                
                points = []
                for idx, chunk in enumerate(batch_chunks):
                    payload = {
                        "text": chunk.text,
                        "metadata": chunk.metadata
                    }
                    
                    vector_record: Dict[str, Any] = {
                        "dense": batch_dense[idx]
                    }
                    
                    if sparse_embeddings and len(sparse_embeddings) > (i + idx):
                        vector_record["sparse"] = sparse_embeddings[i + idx]

                    # Convert raw string ID to deterministic Qdrant UUID
                    point_id = self._format_to_valid_uuid(chunk.id)

                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=vector_record,
                            payload=payload
                        )
                    )

                logger.info(f"Pushing processing batch slice [{i} to {min(i + batch_size, total_records)}]...")
                
                await client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=points
                )

            logger.info(f"Upsert pipeline successfully committed [{total_records}] vector points.")

        except Exception as e:
            logger.critical(f"Fatal transaction error pushing vectors to database: {str(e)}")
            raise BatchUpsertFailed(f"Database rejected transaction: {str(e)}")


# ----------------------------------------------------------------------
# Local Driver Test Module
# ----------------------------------------------------------------------
if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        print("\n" + "="*50)
        print("RUNNING PERSISTENT VECTOR DATA INDEXER PIPELINE")
        print("="*50)

        indexer = ProductionVectorIndexer(
            collection_name="enterprise_knowledge_base", 
            vector_dimension=4
        )

        try:
            await indexer.create_optimized_collection(recreate=True)

            mock_chunks = [
                ChildChunk(id="doc_01_C_0", text="Compliance rules clause 1a.", metadata={"source_file": "policy.pdf"}),
                ChildChunk(id="doc_01_C_1", text="HR onboarding guidelines data.", metadata={"source_file": "hr.pdf"})
            ]
            
            mock_dense_embeddings = [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8]
            ]

            await indexer.batch_upsert_chunks(
                chunks=mock_chunks,
                dense_embeddings=mock_dense_embeddings,
                batch_size=2
            )
            print("="*50 + "\n")

        except Exception as e:
            print(f"[.] Database pipeline bypass: Local cluster verification failed (Is Qdrant container running?). Detail: {e}")

    asyncio.run(main())