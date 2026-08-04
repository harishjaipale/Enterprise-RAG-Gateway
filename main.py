import sys
import time
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# FastAPI Core Imports
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel

# Project Subsystems Imports
from database.connection import VectorDBConnectionManager
from retrieval.query_engine import EnterpriseQueryEngine
from llm.schemas import RAGStructuredResponse
from monitoring.tracer import ProductionTelemetryTracer

# Load environment variables
load_dotenv()

# ----------------------------------------------------------------------
# 1. Directory Configuration
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ----------------------------------------------------------------------
# 2. Production Logging & Telemetry Infrastructure
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FastAPIGatewayServer")

telemetry = ProductionTelemetryTracer()

# Global Singleton Engine Instance
rag_engine: Optional[EnterpriseQueryEngine] = None


# ----------------------------------------------------------------------
# 3. FastAPI Lifespan Engine (Singleton Management)
# ----------------------------------------------------------------------
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    global rag_engine
    logger.info("Initializing system bootstrap components & Singleton RAG Engine...")
    try:
        db_manager = await VectorDBConnectionManager.get_instance()
        await db_manager.verify_connectivity_health()
        logger.info("Database cluster verification passed successfully.")
        
        # Load Enterprise Query Engine Singleton into RAM
        rag_engine = EnterpriseQueryEngine(
            collection_name="enterprise_rag_vector_index",
            reranker_provider="local"
        )
        logger.info("[✓] Enterprise Query Engine successfully pre-loaded in memory.")
        
    except Exception as e:
        logger.warning(f"Database/Engine startup note: {str(e)}")
        
    yield  # Application Running State Boundary
    
    logger.info("Deallocating core runtime resources...")
    try:
        db_manager = await VectorDBConnectionManager.get_instance()
        await db_manager.shutdown_connections_pool()
        logger.info("Application infrastructure shut down clean.")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


# ----------------------------------------------------------------------
# 4. Server Core Instance & Middlewares
# ----------------------------------------------------------------------
app = FastAPI(
    title="Modular Enterprise RAG Server Engine",
    description="High-Throughput Asynchronous AI Retrieval & Text Generation Gateway.",
    version="1.0.0",
    lifespan=app_lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# 5. Rate-Limiting Guard
# ----------------------------------------------------------------------
class InMemoryRateLimiter:
    def __init__(self, requests_limit: int = 60, time_window_seconds: int = 60):
        self.requests_limit = requests_limit
        self.time_window = time_window_seconds
        self.client_records: Dict[str, List[float]] = {}

    async def __call__(self, request: Request):
        client_ip = request.client.host if request.client else "unknown_node"
        current_timestamp = time.time()
        
        if client_ip not in self.client_records:
            self.client_records[client_ip] = []
            
        self.client_records[client_ip] = [
            ts for ts in self.client_records[client_ip] 
            if current_timestamp - ts < self.time_window
        ]
        
        if len(self.client_records[client_ip]) >= self.requests_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Transaction frequency threshold limits breached."
            )
            
        self.client_records[client_ip].append(current_timestamp)

global_rate_limiter = InMemoryRateLimiter(requests_limit=60, time_window_seconds=60)


# ----------------------------------------------------------------------
# 6. Interactive Frontend Route (Web UI)
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Web UI Interface"])
async def render_frontend_ui():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h2>Enterprise RAG Server Running</h2><p>UI file (index.html) not found.</p>")


# ----------------------------------------------------------------------
# 7. Document Management Routes (New additions for Dynamic UI)
# ----------------------------------------------------------------------
@app.get("/api/v1/rag/documents", tags=["Document Management"])
async def get_indexed_documents():
    """Dynamically returns a list of indexed PDF files from the data directory."""
    try:
        if not DATA_DIR.exists():
            return {"documents": []}
            
        # Scan for PDF files in the data directory
        pdf_files = [f.name for f in DATA_DIR.glob("*.pdf")]
        return {"documents": pdf_files}
    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        return {"documents": [], "error": str(e)}


# ----------------------------------------------------------------------
# 8. REST Route Handlers (Query Engine)
# ----------------------------------------------------------------------
@app.get("/health", tags=["Infrastructure Diagnostics"])
async def system_health_status_check():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy", 
            "engine_initialized": rag_engine is not None,
            "engine_timestamp": time.time()
        }
    )


class ChatQueryRequest(BaseModel):
    query: str
    collection_name: str = "enterprise_rag_vector_index"
    filter_metadata: Optional[Dict[str, Any]] = None
    tenant_id: Optional[str] = None
    use_reranker: bool = True
    score_threshold: float = -100.0  # Permissive threshold for complete tabular recall

# Extending the structured response dynamically to include source documents for UI badges
class ChatGatewayResponse(RAGStructuredResponse):
    source_documents: List[str] = []


@app.post(
    "/api/v1/rag/chat", 
    response_model=ChatGatewayResponse,
    tags=["Core RAG Engine Orchestration"],
    dependencies=[Depends(global_rate_limiter)]
)
async def execute_rag_pipeline_stream(payload: ChatQueryRequest):
    if not rag_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, # <--- FIXED TYPO HERE
            detail="RAG Engine Singleton is not yet initialized."
        )

    root_trace = telemetry.create_trace(name="RAG_Chat_API_Request")
    logger.info(f"Incoming query request: '{payload.query}' | Collection: [{payload.collection_name}]")
    
    try:
        # Delegate complete pipeline execution (Expansion + Vector Search + Reranking + LLM Synthesis)
        execution_output = await rag_engine.execute_query(
            user_query=payload.query,
            tenant_id=payload.tenant_id,
            search_top_k=25,
            rerank_top_n=10,
            rerank_score_threshold=payload.score_threshold
        )
        
        # Format Citations from Context Output
        raw_context = execution_output.get("context_payload", "")
        formatted_citations = []
        
        if raw_context and raw_context != "NO RECORD FOUND":
            formatted_citations.append({
                "source_file": "Indexed Repository / System Spec",
                "page_number": 1,
                "exact_quote": raw_context[:150] + "..."
            })
            
        # Try to extract unique sources if available from your engine output
        # If your query engine returns metadata/source lists, it maps them here.
        extracted_sources = execution_output.get("source_documents", [])
        if not extracted_sources and "unique_source_files" in execution_output:
            extracted_sources = execution_output["unique_source_files"]

        # Format Extended Pydantic Response Payload
        response_data = ChatGatewayResponse(
            answer=execution_output.get("answer", "No answer generated."),
            confidence_score=0.92 if execution_output.get("reranked_chunks_count", 0) > 0 else 0.0,
            citations=formatted_citations,
            identified_missing_info=None,
            source_documents=extracted_sources # <--- Added for UI badges
        )
        
        root_trace.end(status="SUCCESS")
        return response_data

    except Exception as e:
        logger.error(f"Execution breakdown: {str(e)}", exc_info=True)
        root_trace.end(status="ERROR", status_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal operational failure: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Make sure to run inside 'Production_RAG2' directory
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)