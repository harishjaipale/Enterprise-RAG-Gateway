import os
import hashlib
import asyncio
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Logger setup
logger = logging.getLogger("AsyncChunker")


# ----------------------------------------------------------------------
# Domain Models (Data Encapsulation)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ChildChunk:
    """Immutable data model representing a granular child chunk for vector search."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParentChunk:
    """Immutable data model representing a larger contextual parent chunk."""
    id: str
    text: str
    child_chunks: List[ChildChunk]


# ----------------------------------------------------------------------
# Custom Production Exceptions
# ----------------------------------------------------------------------
class ChunkerError(Exception):
    """Base exception for chunking subsystem."""
    pass


class FileReadError(ChunkerError):
    """Raised when the ingestion pipeline fails to read source content."""
    pass


class ChunkValidationError(ChunkerError):
    """Raised when generated chunks break system boundary limits."""
    pass


# ----------------------------------------------------------------------
# Asynchronous Production Chunker Class
# ----------------------------------------------------------------------
class ProductionParentChildChunker:
    """
    Enterprise-grade asynchronous Parent-Child document chunking engine.
    Ensures structural line-boundary preservation (Markdown Tables, Lists),
    memory-optimized metadata payloading, and high-throughput concurrent batch ingestion.
    """
    def __init__(
        self,
        parent_size: int = 2500,        # Expanded to preserve large table blocks
        parent_overlap: int = 300,
        child_size: int = 600,          # Expanded for table rows + header retention
        child_overlap: int = 100,
        concurrency_limit: int = 10,
        store_parent_text_in_child: bool = False  # Set False to prevent DB bloat
    ):
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.store_parent_text_in_child = store_parent_text_in_child
        
        self.semaphore = asyncio.Semaphore(concurrency_limit)

        if self.child_size >= self.parent_size:
            raise ChunkValidationError("Child chunk size must be strictly smaller than Parent chunk size!")
        if self.child_overlap >= self.child_size:
            raise ChunkValidationError("Overlap parameter cannot exceed or equal chunk size boundary!")

    def _infer_document_topic(self, filename: str) -> str:
        """Helper method to infer topic context for synthetic header injection."""
        file_lower = filename.lower()
        if any(k in file_lower for k in ["architecture", "api", "security", "spec"]):
            return "System Architecture, Security Authentication & API Specifications"
        elif any(k in file_lower for k in ["q4", "financial", "performance", "statement"]):
            return "Q4 Financial Performance, Revenues, Budget Allocation & Margins"
        elif any(k in file_lower for k in ["clinical", "trial", "report", "medical"]):
            return "Medical Clinical Trial Metrics, Safety & Endpoints"
        return "Enterprise Reference Context"

    def _split_text_structure_aware(self, text: str, target_size: int, overlap: int) -> List[str]:
        """
        Splits text by preserving line breaks (\n) and Markdown table boundaries.
        Prevents breaking Markdown tables, sentences, or word tokens mid-row.
        """
        if not text or not text.strip():
            return []

        # Split into structural blocks (paragraphs or table rows) while preserving line breaks
        blocks = text.splitlines(keepends=True)
        if not blocks:
            return []

        chunks = []
        current_chunk_blocks = []
        current_length = 0

        for block in blocks:
            block_len = len(block)
            
            # Single block exceeds limit, force line-aware word split
            if block_len > target_size:
                if current_chunk_blocks:
                    chunks.append("".join(current_chunk_blocks).strip())
                    current_chunk_blocks = []
                    current_length = 0
                
                # Split huge block by space preserving boundaries
                words = block.split(' ')
                sub_chunk = []
                sub_len = 0
                for w in words:
                    if sub_len + len(w) + 1 > target_size:
                        if sub_chunk:
                            chunks.append(" ".join(sub_chunk).strip())
                        sub_chunk = [w]
                        sub_len = len(w)
                    else:
                        sub_chunk.append(w)
                        sub_len += len(w) + 1
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk).strip())
                continue

            # Standard block accumulation
            if current_length + block_len > target_size:
                merged_text = "".join(current_chunk_blocks).strip()
                if merged_text:
                    chunks.append(merged_text)
                
                # Calculate overlap retention
                overlap_blocks = []
                overlap_len = 0
                for prev_block in reversed(current_chunk_blocks):
                    if overlap_len + len(prev_block) <= overlap:
                        overlap_blocks.insert(0, prev_block)
                        overlap_len += len(prev_block)
                    else:
                        break
                
                current_chunk_blocks = overlap_blocks + [block]
                current_length = sum(len(b) for b in current_chunk_blocks)
            else:
                current_chunk_blocks.append(block)
                current_length += block_len

        if current_chunk_blocks:
            final_text = "".join(current_chunk_blocks).strip()
            if final_text:
                chunks.append(final_text)

        return chunks

    def _generate_chunk_id(self, source_name: str, prefix: str, index: int, content: str) -> str:
        """Generates deterministic unique MD5-backed chunk ID to prevent database duplicate collisions."""
        hash_digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
        return f"{source_name}_{prefix}_{index}_{hash_digest}"

    async def _process_single_file(self, file_path: Path) -> List[ParentChunk]:
        """Processes a single file path asynchronously with isolated bounded context."""
        async with self.semaphore:
            logger.info(f"Starting ingestion chunking for file: {file_path.name}")
            
            try:
                loop = asyncio.get_running_loop()
                def read_file():
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read()

                content = await loop.run_in_executor(None, read_file)
            except Exception as e:
                logger.error(f"File system failure reading {file_path.name}: {str(e)}")
                raise FileReadError(f"Failed to access text stream: {str(e)}")

            if not content.strip():
                logger.warning(f"File {file_path.name} is empty. Skipping compilation.")
                return []

            # Determine Document Topic Header
            doc_topic = self._infer_document_topic(file_path.name)

            # 1. Parent Chunk Generation (Structure Preserved)
            parent_texts = self._split_text_structure_aware(content, self.parent_size, self.parent_overlap)
            compiled_parents: List[ParentChunk] = []

            for p_idx, p_text in enumerate(parent_texts):
                parent_id = self._generate_chunk_id(file_path.stem, "P", p_idx, p_text)

                # 2. Child Chunk Generation within Parent Context
                child_texts = self._split_text_structure_aware(p_text, self.child_size, self.child_overlap)
                compiled_children: List[ChildChunk] = []

                for c_idx, raw_c_text in enumerate(child_texts):
                    child_id = self._generate_chunk_id(parent_id, "C", c_idx, raw_c_text)

                    # --- SYNTHETIC HEADER INJECTION (Critical for Retrieval) ---
                    enriched_child_text = (
                        f"[DOCUMENT: {file_path.name} | TOPIC: {doc_topic}]\n"
                        f"{raw_c_text}"
                    )

                    # Lean, Memory-Optimized Metadata Envelope
                    metadata = {
                        "source_file": file_path.name,
                        "doc_topic": doc_topic,
                        "parent_chunk_id": parent_id,
                        "character_count": len(enriched_child_text)
                    }

                    if self.store_parent_text_in_child:
                        metadata["parent_chunk_text"] = p_text

                    compiled_children.append(
                        ChildChunk(id=child_id, text=enriched_child_text, metadata=metadata)
                    )

                compiled_parents.append(
                    ParentChunk(id=parent_id, text=p_text, child_chunks=compiled_children)
                )

            logger.info(
                f"[✓] Chunking finished for {file_path.name}: "
                f"Created {len(compiled_parents)} Parents & {sum(len(p.child_chunks) for p in compiled_parents)} Children."
            )
            return compiled_parents

    async def chunk_batch(self, file_paths: List[str]) -> List[ParentChunk]:
        """Entry point for executing parallel asynchronous batch chunking."""
        tasks = []
        for path in file_paths:
            p_obj = Path(path).resolve()
            if not p_obj.exists():
                logger.warning(f"File missing, skipping: {p_obj}")
                continue
            tasks.append(self._process_single_file(p_obj))

        if not tasks:
            logger.error("No valid file targets identified for chunking.")
            return []

        logger.info(f"Spawning {len(tasks)} concurrent async chunking streams...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        flat_parents: List[ParentChunk] = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Execution error in batch sequence: {res}")
            else:
                flat_parents.extend(res)

        return flat_parents


# ----------------------------------------------------------------------
# Local Driver Test Module
# ----------------------------------------------------------------------
if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        chunker = ProductionParentChildChunker(
            parent_size=2000, 
            parent_overlap=200,
            child_size=500,
            child_overlap=50,
            concurrency_limit=5
        )

        test_dir = Path("./ingestion_test_temp")
        test_dir.mkdir(exist_ok=True)
        
        file_targets = []
        for i in range(2):
            file_path = test_dir / f"sample_doc_{i}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(
                    f"| Parameter | Week 24 | Week 52 |\n"
                    f"| AST (U/L) | 420 ± 65 | 35 ± 6 |\n"
                    f"| ALT (U/L) | 310 ± 50 | 40 ± 8 |\n\n" * 10
                )
            file_targets.append(str(file_path))

        try:
            processed_data = await chunker.chunk_batch(file_targets)
            logger.info(f"Test batch execution complete: {len(processed_data)} parent chunks created.")
        finally:
            for path in file_targets:
                if os.path.exists(path):
                    os.remove(path)
            if test_dir.exists():
                test_dir.rmdir()

    asyncio.run(main())