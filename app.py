import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
import gradio as gr
from fastapi.responses import FileResponse

# Import main FastAPI instance
from main import app as fastapi_app

# HTML UI route serve karein
@fastapi_app.get("/", include_in_schema=False)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Enterprise RAG Gateway Backend API Active"}

# Lightweight Gradio Blocks wrapper
with gr.Blocks(title="Enterprise RAG Gateway") as demo:
    gr.Markdown("# 🚀 Enterprise RAG Gateway Active")
    gr.Markdown("FastAPI backend & Cyberpunk Web UI running successfully.")

# Mount Gradio onto FastAPI root runner
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)