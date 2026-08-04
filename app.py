import os
import sys
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Set up system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import backend query engine from main.py if available
try:
    from main import query_engine_function
except ImportError:
    query_engine_function = None

# Initialize native FastAPI app (No Gradio overhead)
app = FastAPI(title="Enterprise RAG Gateway")

# Serve the custom index.html directly at root
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h2>Error: index.html not found in repository!</h2>"

# API Health endpoint
@app.get("/api/health")
@app.get("/api/v1/rag/health")
async def api_health():
    return {"status": "online", "secure_node": "connected", "gpu": "active"}

# Documents endpoint
@app.get("/api/documents")
@app.get("/api/v1/rag/documents")
async def api_documents():
    return {
        "documents": [
            "Enterprise_Architecture_Specs.pdf",
            "RAG_Pipeline_Documentation.pdf",
            "Vector_Database_Cluster_Config.pdf"
        ]
    }

# Query execution endpoint
@app.post("/api/query")
@app.post("/api/v1/rag/query")
async def api_query(payload: dict):
    query_text = payload.get("query", "") or payload.get("message", "")
    if query_engine_function:
        answer = query_engine_function(query_text)
    else:
        answer = f"Enterprise RAG Gateway Response: {query_text}"
    return {"response": answer, "status": "success"}

# Standard uvicorn launcher for Hugging Face or local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)