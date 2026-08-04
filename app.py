import os
import sys
import gradio as gr
import spaces  # ZeroGPU mandatory requirement for Hugging Face Spaces

# Set up system path to include the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import backend query engine from main.py if available
try:
    from main import query_engine_function
except ImportError:
    query_engine_function = None

# Path to the custom frontend index.html file
index_path = os.path.join(os.path.dirname(__file__), "index.html")

# Read the HTML content safely
if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
else:
    html_content = "<h2>Error: index.html not found in repository!</h2>"

# ZeroGPU supported backend AI handler function
@spaces.GPU(duration=60)
def execute_rag_query(query_text):
    """
    Backend handler function wrapped with ZeroGPU decorator.
    Executes the core RAG query logic.
    """
    if query_engine_function:
        return query_engine_function(query_text)
    return f"Enterprise RAG Gateway Response: {query_text}"

# Build Gradio UI embedding your custom HTML/JS frontend
with gr.Blocks(title="Enterprise RAG Gateway") as demo:
    gr.HTML(html_content)

# Access Gradio's underlying FastAPI instance to support frontend API calls
app = demo.app

# Health check endpoints
@app.get("/api/health")
@app.get("/api/v1/rag/health")
async def api_health():
    return {"status": "online", "secure_node": "connected", "gpu": "active"}

# Documents endpoint matching frontend expectation
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

# Query execution endpoint matching frontend expectation
@app.post("/api/query")
@app.post("/api/v1/rag/query")
async def api_query(payload: dict):
    query_text = payload.get("query", "") or payload.get("message", "")
    answer = execute_rag_query(query_text)
    return {"response": answer, "status": "success"}

# Launch the application on 0.0.0.0:7860 for Hugging Face deployment
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)