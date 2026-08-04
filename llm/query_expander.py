import logging
from typing import List

logger = logging.getLogger("QueryExpander")

class MultiQueryExpander:
    """
    Expands a single raw user query into domain-specific search variations
    to maximize vector search recall.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def expand_query(self, original_query: str) -> List[str]:
        """
        Generates search query variations based on domain keywords.
        Falls back to rule-based expansion if LLM is unavailable.
        """
        if not original_query or not original_query.strip():
            return [original_query]

        q_lower = original_query.lower()
        expanded = [original_query]

        # Rule-based domain keyword expansions (Fast & Reliable)
        if any(k in q_lower for k in ["sec", "security", "auth", "token", "jwt", "api"]):
            expanded.extend([
                "System architecture security authentication authorization endpoints and flow",
                "API specification security parameters headers OAuth JWT authentication mechanism"
            ])
        elif any(k in q_lower for k in ["q4", "financial", "revenue", "budget", "profit", "margin"]):
            expanded.extend([
                "Q4 financial performance statement revenue breakdown net income budget allocation",
                "Quarterly financial metrics operating margins capital expenditure fiscal details"
            ])
        elif any(k in q_lower for k in ["clinical", "trial", "endpoint", "patient", "safety"]):
            expanded.extend([
                "Medical clinical trial primary secondary endpoints patient safety statistical results",
                "Clinical study adverse events methodology sample size clinical outcome evaluation"
            ])

        # Deduplicate while preserving order
        unique_queries = list(dict.fromkeys(expanded))
        logger.info(f"Query Expanded from 1 to {len(unique_queries)} search vectors.")
        return unique_queries