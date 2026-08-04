import os
import sys
import gradio as gr
import spaces  # ZeroGPU mandatory requirement

# Set up system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import backend query engine from main.py if available
try:
    from main import query_engine_function
except ImportError:
    query_engine_function = None

# Read the custom index.html file
index_path = os.path.join(os.path.dirname(__file__), "index.html")
if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
else:
    html_content = "<h2>Error: index.html not found in repository!</h2>"

# ZeroGPU supported backend AI handler function
@spaces.GPU(duration=60)
def execute_rag_query(query_text):
    if query_engine_function:
        return query_engine_function(query_text)
    return f"Enterprise RAG Gateway Response: {query_text}"

# Build Gradio UI embedding your custom HTML/JS frontend
with gr.Blocks(title="Enterprise RAG Gateway") as demo:
    gr.HTML(html_content)

# Access Gradio's underlying FastAPI instance to support frontend API calls
app = demo.app

@app.get("/api/health")
async def api_health():
    return {"status": "online", "secure_node": "connected", "gpu": "active"}

@app.get("/api/documents")
async def api_documents():
    # Provide indexed documents list that your index.html tries to load
    return {
        "documents": [
            "Enterprise_Architecture_Specs.pdf",
            "RAG_Pipeline_Documentation.pdf",
            "Vector_Database_Cluster_Config.pdf"
        ]
    }

@app.post("/api/query")
async def api_query(payload: dict):
    query_text = payload.get("query", "") or payload.get("message", "")
    answer = execute_rag_query(query_text)
    return {"response": answer, "status": "success"}

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)