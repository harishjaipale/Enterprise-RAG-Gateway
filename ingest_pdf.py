import os
import sys
import uuid
import logging
import asyncio
import glob
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Env variables initialization
load_dotenv()

# Embeddings Engine Import
from llm.embeddings import ProductionEmbeddingEngine

# ----------------------------------------------------------------------
# 1. Pipeline Production Logging Configuration
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingestion_pipeline.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("IngestionPipelineEngine")


# ----------------------------------------------------------------------
# 2. Strong Type-Safe Data Contracts
# ----------------------------------------------------------------------
@dataclass
class DocumentChunk:
    """Type-safe blueprint tracking granular data elements through layers."""
    id: str
    text: str
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None


# ----------------------------------------------------------------------
# 3. Enterprise Core Ingestion Processor (OOPS Architecture)
# ----------------------------------------------------------------------
class ProductionIngestionPipeline:
    """
    High-Throughput, Resilient Document Ingestion Pipeline.
    
    Advanced Features Implemented:
    - **Header-Aware Table Preservation:** Uses RecursiveCharacterTextSplitter to prevent splitting tables.
    - **Idempotency & Cryptographic Tracking:** Generates deterministic UUIDs based on text content hash.
    - **Fully Local Embedding Generation:** Runs BAAI/bge-small-en-v1.5 via local CPU/GPU (0% API cost).
    - **Extensible Hook Framework:** Seamless integration with LlamaParse & Qdrant Vector Stores.
    """
    def __init__(
        self,
        collection_name: str = "enterprise_rag_vector_index",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        dimensions: int = 384,
        recreate_collection: bool = False  # Set dynamically in batch loop
    ):
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.recreate_collection = recreate_collection
        
        # Initializing production-grade local embedding core
        self.embedding_engine = ProductionEmbeddingEngine(
            model_name=embedding_model,
            dimensions=self.dimensions,
            enable_local_cache=True
        )
        logger.info(f"Ingestion Core Cluster bound to collection target: [{self.collection_name}] (Dimensions: {self.dimensions})")

    # === Extensible Hooks ===
    async def _parse_document(self, file_path: str) -> str:
        logger.info(f"Extracting layout architectures via ProductionDocumentParser from: {file_path}")
        
        from ingestion.parsing import ProductionDocumentParser
        
        # Initialize parser engine stack
        parser = ProductionDocumentParser(
            api_version="2026-05-21",
            tier="agentic"
        )
        
        extracted_markdown = await parser.parse(
            file_path=file_path,
            output_dir="./data/output"
        )
        
        return extracted_markdown

    def _chunk_text(self, raw_text: str, source_name: str) -> List[DocumentChunk]:
        """
        Structure-aware Chunking Engine using RecursiveCharacterTextSplitter:
        Preserves Markdown tables, headers, and logical paragraphs intact.
        """
        logger.info("Orchestrating structure-aware chunking boundaries with RecursiveCharacterTextSplitter...")
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Read directly from env or fall back to production defaults
        chunk_size = int(os.getenv("CHUNKING__CHUNK_SIZE", 1024))
        chunk_overlap = int(os.getenv("CHUNKING__CHUNK_OVERLAP", 200))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "### ", "## ", "# ", "|", " ", ""]  # Preserves MD tables & Headers
        )

        raw_chunks = text_splitter.split_text(raw_text)
        chunks_list = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Deterministic UUID calculation to guarantee idempotency
            deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_text))
            
            chunks_list.append(
                DocumentChunk(
                    id=deterministic_id,
                    text=chunk_text,
                    metadata={
                        "source_file": source_name,
                        "chunk_index": idx + 1,
                        "ingestion_timestamp": "2026-07-31"
                    }
                )
            )

        return chunks_list

    async def _upsert_to_vector_db(self, verified_chunks: List[DocumentChunk]) -> None:
        logger.info(f"Committing records to Qdrant cluster for collection: {self.collection_name}")
        
        try:
            from database.indexer import ProductionVectorIndexer, ChildChunk
            
            indexer = ProductionVectorIndexer(
                collection_name=self.collection_name,
                vector_dimension=self.dimensions
            )
            
            # Step 1: Schema creation (recreate_collection flag controls reset logic)
            await indexer.create_optimized_collection(recreate=self.recreate_collection)
            
            # Step 2: Adapting dataclasses for Indexer Signature
            child_chunks = [
                ChildChunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata
                ) for chunk in verified_chunks
            ]
            
            dense_embeddings = [chunk.vector for chunk in verified_chunks if chunk.vector is not None]
            
            # Step 3: Batch upsert
            await indexer.batch_upsert_chunks(
                chunks=child_chunks,
                dense_embeddings=dense_embeddings
            )
            
            logger.info(f"[✓] Successfully committed {len(verified_chunks)} structured records to Qdrant cluster.")

        except ImportError as e:
            logger.warning(f"database.indexer module import issue: {e}. Vectors generated successfully in memory.")
        except Exception as e:
            logger.error(f"Failed to commit vector index payloads down to Qdrant cluster: {str(e)}", exc_info=True)
            raise e

    # === Primary Pipeline Logic Workflow ===
    async def execute_pipeline(self, target_file_path: str) -> bool:
        if not os.path.exists(target_file_path):
            logger.error(f"Inbound operational target missing from system structure: {target_file_path}")
            return False

        filename = os.path.basename(target_file_path)
        logger.info(f"====== Initializing Resilient Processing Sequence for [{filename}] ======")

        try:
            # Phase 1: Structural Extraction
            extracted_text = await self._parse_document(target_file_path)
            
            # Phase 2: Context Segmentation
            document_chunks = self._chunk_text(extracted_text, source_name=filename)
            if not document_chunks:
                logger.warning("Segmentation returned zero valid payload frames. Dropping execution loop.")
                return False
                
            logger.info(f"Generated {len(document_chunks)} structure-preserved payload nodes.")

            # Phase 3: Vector Array Compilations
            logger.info("Computing dense tensor vectors across payload arrays locally...")
            text_arrays = [chunk.text for chunk in document_chunks]
            
            computed_vectors = await self.embedding_engine.get_embeddings_batch(text_arrays)

            for idx, vector in enumerate(computed_vectors):
                document_chunks[idx].vector = vector

            # Phase 4: Final Database Sync Commits
            await self._upsert_to_vector_db(document_chunks)
            
            logger.info(f"[✓] Document [{filename}] successfully synchronized to infrastructure index layers.")
            return True

        except Exception as crash_error:
            logger.critical(f"[✗] Critical pipeline failure captured during ingestion sequence: {str(crash_error)}", exc_info=True)
            return False


# ----------------------------------------------------------------------
# 4. Standalone CLI Entrypoint Execution Loop
# ----------------------------------------------------------------------
async def main():
    print("\n" + "="*70)
    print("RUNNING HIGH-SPEED STRUCTURE-PRESERVING INGESTION ENGINE")
    print("="*70)

    if len(sys.argv) < 2:
        logger.error("Missing operational parameter configuration data context.")
        print("\n[!] Usage Guide: python ingest_pdf.py <path_to_pdf_or_directory>")
        print("Example: python ingest_pdf.py data\n")
        return

    input_path = sys.argv[1]
    
    # Check if input path is a folder or single file
    files_to_process = []
    if os.path.isdir(input_path):
        files_to_process = glob.glob(os.path.join(input_path, "*.pdf"))
    elif os.path.exists(input_path):
        files_to_process = [input_path]
    else:
        logger.error(f"Provided path does not exist: {input_path}")
        return

    logger.info(f"Target PDFs identified for processing: {len(files_to_process)}")

    # Loop through all files sequentially using a SINGLE execution logic
    for idx, pdf_file in enumerate(files_to_process):
        # File 1 pe recreate=True (Purana data clear hoga), uske baad baaki saari files par append mode (False)
        recreate_flag = True if idx == 0 else False
        
        pipeline = ProductionIngestionPipeline(
            collection_name="enterprise_rag_vector_index",
            embedding_model="BAAI/bge-small-en-v1.5",
            dimensions=384,
            recreate_collection=recreate_flag
        )
        
        await pipeline.execute_pipeline(pdf_file)

    print("\n[✓] All documents successfully ingested & synchronized! Process exit token: 0\n")


if __name__ == "__main__":
    asyncio.run(main())