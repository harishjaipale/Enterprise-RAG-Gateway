import logging
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# ----------------------------------------------------------------------
# 1. System Logging Configurations
# ----------------------------------------------------------------------
logger = logging.getLogger("RAGDataSchemas")


# ----------------------------------------------------------------------
# 2. Base Metadata Reference Block (Data Encapsulation Architecture)
# ----------------------------------------------------------------------
class CitationSource(BaseModel):
    """Encapsulates secure tracing parameters mapping grounding context data frames."""
    
    # Strict API Schema Config
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_file: str = Field(
        ..., 
        min_length=1,
        description="The clean isolated filename from which this factual node was derived."
    )
    page_number: Optional[int] = Field(
        None, 
        description="Dynamic reference page index bounds tracking location within target document."
    )
    exact_quote: Optional[str] = Field(
        None, 
        description="The pristine textual sentence anchor extracted inside vector search structures."
    )

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: Optional[int]) -> Optional[int]:
        """Ensures page numbers are valid positive integers."""
        if value is not None and value < 1:
            logger.warning(f"Invalid page number index [{value}] received. Normalizing to None.")
            return None
        return value


# ----------------------------------------------------------------------
# 3. Enterprise Production Grade RAG Output Models
# ----------------------------------------------------------------------
class RAGStructuredResponse(BaseModel):
    """
    Standard Enterprise Response Blueprint.
    Forces the LLM to output highly organized structural JSON payloads,
    eliminating raw string post-parsing anomalies.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    answer: str = Field(
        ..., 
        min_length=1,
        description="The comprehensive contextual answer generated strictly under grounding matrix constraints."
    )
    confidence_score: float = Field(
        ..., 
        description="Calculated neural alignment indicator scaling mathematically between 0.00 (Zero evidence) and 1.00 (Fully Grounded)."
    )
    citations: List[CitationSource] = Field(
        default_factory=list,
        description="Array containing tracked source document footprints used to formulate the response."
    )
    identified_missing_info: Optional[List[str]] = Field(
        default=None,
        description="List capturing explicit query requirements which weren't verifiable inside grounding data fragments."
    )

    # ------------------------------------------------------------------
    # Advanced Pydantic Validation Handlers
    # ------------------------------------------------------------------
    @field_validator("confidence_score")
    @classmethod
    def validate_confidence_range(cls, value: float) -> float:
        """Enforces runtime boundary restrictions across confidence score outputs."""
        if not (0.0 <= value <= 1.0):
            clamped = max(0.0, min(value, 1.0))
            logger.warning(f"Out-of-bounds confidence validation score: {value}. Auto-clamping to {clamped}.")
            return clamped
        return value

    @field_validator("identified_missing_info")
    @classmethod
    def sanitize_missing_info(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        """Filters out empty strings from missing info lists."""
        if value:
            cleaned = [item.strip() for item in value if item and item.strip()]
            return cleaned if cleaned else None
        return None


# ----------------------------------------------------------------------
# 4. Agentic Classification Schema (Multi-Tenant Routing)
# ----------------------------------------------------------------------
class QueryIntentClassifier(BaseModel):
    """
    A standalone modular schema to dynamically route user incoming queries 
    into dedicated system workflows.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    primary_intent: Literal["knowledge_retrieval", "general_chit_chat", "escalation", "unsupported"] = Field(
        ..., 
        description="Intent classification category for strict pipeline execution routing."
    )
    extracted_keywords: List[str] = Field(
        default_factory=list,
        description="Clean parsed semantic keyword tokens captured across processing layers."
    )
    requires_rag_lookup: bool = Field(
        ...,
        description="Flag indicating if infrastructure resources should trigger vector database search paths."
    )

    @model_validator(mode="after")
    def validate_intent_consistency(self) -> "QueryIntentClassifier":
        """
        Self-healing validator: Synchronizes `requires_rag_lookup` boolean 
        with the primary intent to prevent workflow routing conflicts.
        """
        if self.primary_intent == "knowledge_retrieval" and not self.requires_rag_lookup:
            logger.warning("Contradiction detected: 'knowledge_retrieval' intent requires RAG. Auto-enabling `requires_rag_lookup=True`.")
            self.requires_rag_lookup = True
        elif self.primary_intent in ("general_chit_chat", "unsupported") and self.requires_rag_lookup:
            logger.warning(f"Contradiction detected: Intent '{self.primary_intent}' does not need vector search. Auto-disabling `requires_rag_lookup=False`.")
            self.requires_rag_lookup = False
            
        return self


# ----------------------------------------------------------------------
# 5. Local Validation Engine Suite
# ----------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    print("\n" + "="*50)
    print("RUNNING UPGRADED PYDANTIC SCHEMAS COMPLIANCE SUITE")
    print("="*50)

    # 1. Test RAG Response Payload
    mock_payload = {
        "answer": "AES-256 database configurations are enabled globally across all operations.",
        "confidence_score": 1.25,  # Intentional out-of-range value to evaluate auto-clamp
        "citations": [
            {"source_file": "security_policy.pdf", "page_number": 12, "exact_quote": "Enforce AES-256 across storage databases."}
        ],
        "identified_missing_info": ["   ", "Details about key rotation schedule"] # Tests empty string sanitization
    }

    try:
        validated_object = RAGStructuredResponse(**mock_payload)
        
        print("[✓] Response Model Parsing Completed.")
        print(f"    - Corrected Confidence Score : {validated_object.confidence_score} (Clamped to 1.0)")
        print(f"    - Primary Source Verified    : {validated_object.citations[0].source_file}")
        print(f"    - Cleaned Missing Info       : {validated_object.identified_missing_info}")
        print(f"    - Serialized JSON Payload:\n{validated_object.model_dump_json(indent=2)}")
        print("="*50)
        
    except Exception as e:
        print(f"[✗] Schema Validation tracking drop encountered: {e}")

    # 2. Test Agentic Intent Classifier Self-Healing Logic
    mock_classifier = {
        "primary_intent": "knowledge_retrieval",
        "extracted_keywords": ["encryption", "database"],
        "requires_rag_lookup": False  # Contradiction: intent says retrieval, flag says False
    }

    try:
        classifier_obj = QueryIntentClassifier(**mock_classifier)
        print("\n[✓] Intent Classifier Self-Healing Verification:")
        print(f"    - Primary Intent             : {classifier_obj.primary_intent}")
        print(f"    - Corrected RAG Lookup Flag  : {classifier_obj.requires_rag_lookup} (Auto-fixed to True)")
        print("="*50 + "\n")
    except Exception as e:
        print(f"[✗] Classifier Schema Validation failed: {e}")


if __name__ == "__main__":
    main()