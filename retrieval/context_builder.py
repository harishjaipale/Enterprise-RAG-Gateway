import logging
from typing import List, Dict, Any

logger = logging.getLogger("ContextAssembler")

class ParentContextAssembler:
    """
    Groups retrieved child chunks, removes redundancy, and injects parent-level 
    or fully formatted context into the final prompt payload for the LLM.
    """
    @staticmethod
    def build_llm_prompt_context(reranked_results: List[Dict[str, Any]]) -> str:
        """
        Takes reranked hits and compiles a clean, structured context string.
        """
        if not reranked_results:
            return "NO RELEVANT CONTEXT FOUND IN VECTOR STORE."

        grouped_by_doc: Dict[str, List[str]] = {}

        for hit in reranked_results:
            metadata = hit.get("metadata", {})
            source_file = metadata.get("source_file", "Unknown Document")
            doc_topic = metadata.get("doc_topic", "General Context")
            
            # Prefer parent_chunk_text if available, else raw text
            text_content = metadata.get("parent_chunk_text") or hit.get("text", "")

            doc_key = f"{source_file} ({doc_topic})"
            if doc_key not in grouped_by_doc:
                grouped_by_doc[doc_key] = []

            if text_content not in grouped_by_doc[doc_key]:
                grouped_by_doc[doc_key].append(text_content)

        # Build clean string layout
        formatted_sections = []
        for doc_info, text_blocks in grouped_by_doc.items():
            combined_blocks = "\n\n".join(text_blocks)
            section = (
                f"SOURCE DOCUMENT: [{doc_info}]\n"
                f"--------------------------------------------------\n"
                f"{combined_blocks}\n"
                f"--------------------------------------------------"
            )
            formatted_sections.append(section)

        logger.info(f"Successfully assembled context from {len(grouped_by_doc)} unique source files.")
        return "\n\n".join(formatted_sections)