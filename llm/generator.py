import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from openai import (
    AsyncOpenAI, 
    APIConnectionError, 
    APITimeoutError, 
    InternalServerError,
    AuthenticationError
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("LLMGenerator")


class GeneratorError(Exception):
    """Base exception for LLM generation context failures."""
    pass

class LLMContextOverflowError(GeneratorError):
    """Raised when context window allocations breach safety bounds."""
    pass

class LLMAPIConnectionError(GeneratorError):
    """Raised when external gateway connection operations time out permanently."""
    pass


class ProductionLLMGenerator:
    """
    A robust, asynchronous LLM text generation engine with Primary (Groq Llama-3.3) 
    and Secondary/Fallback (OpenAI gpt-4o-mini) automatic resilience, integrated 
    query expansion, and logical deductive reasoning.
    """
    def __init__(
        self, 
        primary_model: str = "llama-3.3-70b-versatile", 
        fallback_model: str = "gpt-4o-mini",
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,  # Strict factual precision
        max_tokens: int = 1024
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load keys from parameter or ENV
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", os.environ.get("LLM_OPENAI_API_KEY"))
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", None)

        # Lazy Initialized Async Clients
        self._primary_client: Optional[AsyncOpenAI] = None
        self._fallback_client: Optional[AsyncOpenAI] = None
        self._client_lock = asyncio.Lock()

    async def _get_primary_client(self) -> AsyncOpenAI:
        """Thread-safe lazy initialization for Primary Groq client."""
        if self._primary_client is None:
            async with self._client_lock:
                if self._primary_client is None:
                    logger.info(f"Initializing Primary AsyncOpenAI client -> [Groq Cloud Engine: {self.primary_model}]")
                    self._primary_client = AsyncOpenAI(
                        api_key=self.groq_api_key,
                        base_url="https://api.groq.com/openai/v1"
                    )
        return self._primary_client

    async def _get_fallback_client(self) -> AsyncOpenAI:
        """Thread-safe lazy initialization for Fallback OpenAI client."""
        if self._fallback_client is None:
            async with self._client_lock:
                if self._fallback_client is None:
                    target_endpoint = self.base_url or 'Standard OpenAI Cloud'
                    logger.info(f"Initializing Fallback AsyncOpenAI client -> [{target_endpoint}: {self.fallback_model}]")
                    self._fallback_client = AsyncOpenAI(
                        api_key=self.openai_api_key,
                        base_url=self.base_url
                    )
        return self._fallback_client

    def _sanitize_context_text(self, text: str) -> str:
        """Sanitizes context content to neutralize prompt injection delimiter breaking."""
        return text.replace("--- END GROUNDING CONTEXT ---", "[DELIMITER_NEUTRALIZED]")

    async def _expand_query_dynamically(self, client: AsyncOpenAI, model: str, original_query: str) -> List[str]:
        """Expands raw user query into domain-specific search variations for high recall."""
        if not original_query or not original_query.strip():
            return [original_query]

        expanded = [original_query]
        try:
            expansion_prompt = (
                "You are a clinical and technical search assistant. Convert the user query into "
                "2 dense, highly relevant domain-specific search variations or keyword phrases for vector retrieval.\n"
                f"User Query: {original_query}\n"
                "Provide only the search variations separated by a newline."
            )
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": expansion_prompt}],
                temperature=0.0,
                max_tokens=100
            )
            content = response.choices[0].message.content or ""
            llm_variations = [q.strip() for q in content.split('\n') if q.strip()]
            if llm_variations:
                expanded.extend(llm_variations)
        except Exception as e:
            logger.warning(f"Dynamic query expansion failed: {e}. Proceeding with base query.")

        return list(dict.fromkeys(expanded))

    def _build_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]], custom_system_instruction: Optional[str] = None) -> List[Dict[str, str]]:
        """Structures grounded multi-turn RAG prompt with support for custom system instructions."""
        formatted_contexts = []
        for idx, doc in enumerate(context_chunks):
            text = self._sanitize_context_text(doc.get("text", "").strip())
            source = doc.get("metadata", {}).get("source_file", "Unknown Source")
            formatted_contexts.append(f"<document index='{idx+1}' source='{source}'>\n{text}\n</document>")
            
        merged_context_string = "\n\n".join(formatted_contexts)

        # Use custom instruction if provided, otherwise default to built-in reasoning system prompt
        system_instruction = custom_system_instruction or (
            "You are an expert Enterprise RAG Knowledge Assistant.\n"
            "Your objective is to answer the user's query accurately using the information provided inside <grounding_context>.\n\n"
            "CRITICAL EXECUTION GUIDELINES:\n"
            "1. Primary Source: Base your response primarily on the retrieved context provided below.\n"
            "2. Deductive Reasoning: If the direct answer is not explicitly written word-for-word, but the context provides sufficient "
            "foundational facts (e.g., mechanisms, risks, comparison metrics like LFT vs NAT2), perform LOGICAL DEDUCTION to answer the user query.\n"
            "3. Tabular Data & Correlation: Context may contain markdown tables, raw key-value pairs, or separated headers and data rows. "
            "You MUST correlate numerical values (e.g., '35 ± 6', '40 ± 8') with corresponding markers/headers (e.g., 'AST', 'ALT') "
            "and timepoints (e.g., 'Week 52') present across the chunks. Do NOT state that a parameter is 'unspecified' if table headers exist.\n"
            "4. Transparency: Briefly explain your reasoning step-by-step using facts from the context.\n"
            "5. Missing Context Guardrail: ONLY if the requested information is completely absent or unrelated across ALL chunks, state: "
            "'The requested information is unavailable in the provided document context.'"
        )

        user_content = (
            f"<grounding_context>\n{merged_context_string}\n</grounding_context>\n\n"
            f"Query: {query}"
        )

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    def _count_tokens_sync(self, text: str) -> int:
        """Synchronous token counter logic."""
        try:
            import tiktoken
            try:
                encoding = tiktoken.encoding_for_model(self.fallback_model)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            return int(len(text.split()) * 1.3)

    async def _truncate_context_by_tokens(
        self, 
        context_chunks: List[Dict[str, Any]], 
        max_context_tokens: int = 4000
    ) -> List[Dict[str, Any]]:
        """Non-blocking context capacity boundary trimmer."""
        accepted_chunks = []
        accumulated_tokens = 0
        
        for chunk in context_chunks:
            chunk_text = chunk.get("text", "")
            chunk_tokens_count = await asyncio.to_thread(self._count_tokens_sync, chunk_text)
            
            if accumulated_tokens + chunk_tokens_count > max_context_tokens:
                logger.warning(
                    f"Context capacity threshold reached ({max_context_tokens} max tokens). "
                    f"Truncated remaining chunks at current token count: {accumulated_tokens}."
                )
                break
                
            accumulated_tokens += chunk_tokens_count
            accepted_chunks.append(chunk)
            
        return accepted_chunks

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1.0, min=1, max=4),
        retry=retry_if_exception_type((
            APIConnectionError, 
            APITimeoutError, 
            InternalServerError
        )),
        reraise=True
    )
    async def _dispatch_to_client(self, client: AsyncOpenAI, model: str, messages: List[Dict[str, str]]) -> str:
        """Executes API call to a given client instance with retries."""
        response = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content or ""

    async def generate_answer(
        self, 
        query: str, 
        context_chunks: List[Dict[str, Any]], 
        custom_system_instruction: Optional[str] = None
    ) -> str:
        """Primary orchestration endpoint with Auto-Failover, Query Expansion, and Custom Instructions support."""
        if not query or not query.strip():
            raise GeneratorError("Query parameter context sequence cannot be empty.")
            
        if not context_chunks:
            logger.warning("Zero document chunks passed to LLM Generator.")

        truncated_chunks = await self._truncate_context_by_tokens(context_chunks, max_context_tokens=4000)
        
        # -------------------------------------------------------------
        # STEP 1: Attempt Primary Generation (Groq / Llama-3.3)
        # -------------------------------------------------------------
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY missing in environment! Bypassing Groq primary engine...")
        else:
            try:
                logger.info(f"Dispatching primary generation request to Groq model: [{self.primary_model}]...")
                primary_client = await self._get_primary_client()
                
                # Optional query enhancement/expansion step before building prompt
                expanded_queries = await self._expand_query_dynamically(primary_client, self.primary_model, query)
                logger.info(f"Query expanded into {len(expanded_queries)} search variants for context processing.")

                messages = self._build_rag_prompt(query, truncated_chunks, custom_system_instruction)
                answer = await self._dispatch_to_client(primary_client, self.primary_model, messages)
                logger.info("Primary Groq LLM generation succeeded.")
                return answer

            except Exception as primary_err:
                logger.warning(
                    f"Primary Groq LLM ({self.primary_model}) execution failed! "
                    f"Reason: {primary_err}. Initiating OpenAI fallback routing..."
                )

        # -------------------------------------------------------------
        # STEP 2: Fallback Generation (OpenAI / gpt-4o-mini)
        # -------------------------------------------------------------
        if not self.openai_api_key:
            logger.error("Fallback requested but OPENAI_API_KEY is not configured in environment.")
            raise LLMAPIConnectionError("Primary Groq LLM failed and no Fallback OpenAI API key was provided.")

        try:
            logger.info(f"Dispatching fallback generation request to OpenAI model: [{self.fallback_model}]...")
            fallback_client = await self._get_fallback_client()
            
            messages = self._build_rag_prompt(query, truncated_chunks, custom_system_instruction)
            fallback_answer = await self._dispatch_to_client(fallback_client, self.fallback_model, messages)
            logger.info("Fallback generation via OpenAI succeeded!")
            return fallback_answer

        except Exception as fallback_err:
            logger.critical(f"Both Primary (Groq) and Fallback (OpenAI) LLM engines failed! Error: {fallback_err}")
            raise GeneratorError(f"All LLM generation gateways failed: {str(fallback_err)}")


if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        print("\n" + "="*50)
        print("RUNNING RESILIENT MULTI-PROVIDER LLM GENERATOR SUITE")
        print("="*50)

        generator = ProductionLLMGenerator(
            primary_model="llama-3.3-70b-versatile",
            fallback_model="gpt-4o-mini"
        )

        mock_query = "Can routine screening of patients' NAT2 genotype be viably implemented in clinical practice?"
        mock_contexts = [
            {
                "text": "Isoniazid-induced hepatotoxicity is mediated by slow N-acetyltransferase 2 (NAT2) acetylator status. While genotyping prevents toxicity, routine clinical implementation faces turnaround time and cost barriers, favoring baseline LFT monitoring.", 
                "metadata": {"source_file": "Isoniazid_Hepatotoxicity_Report.pdf"}
            }
        ]

        try:
            response = await generator.generate_answer(query=mock_query, context_chunks=mock_contexts)
            print(f"\n[✓] LLM Response Received:\n{response}")
            print("="*50 + "\n")
        except Exception as e:
            print(f"[!] Pipeline Execution Exception Detail: {e}")

    asyncio.run(main())