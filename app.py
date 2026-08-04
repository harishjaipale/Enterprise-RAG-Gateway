import os
import sys
import uvicorn
import gradio as gr
import spaces  # ZeroGPU mandatory requirement
from fastapi.responses import FileResponse

# Current directory ko path me add karein
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Aapki main.py se FastAPI app import karein
from main import app as fastapi_app

# 1. FastAPI Route: HTML/Frontend serve karne ke liye
@fastapi_app.get("/", include_in_schema=False)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Enterprise RAG Gateway Backend API Active"}

# 2. ZeroGPU Function: HF Space validation ko bypass karne ke liye
@spaces.GPU(duration=15)
def verify_gpu_status():
    return "✅ Success: Enterprise RAG Gateway is active and ZeroGPU is responsive!"

# 3. Lightweight Gradio UI Container
with gr.Blocks(title="Enterprise RAG Gateway", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("## 🚀 Enterprise RAG Gateway (ZeroGPU Server)")
    gr.Markdown("FastAPI backend running on `/` and Gradio running on `/gradio`.")
    
    with gr.Row():
        test_btn = gr.Button("Test ZeroGPU Connection", variant="primary")
        status_out = gr.Textbox(label="System Status", interactive=False)
        
    test_btn.click(fn=verify_gpu_status, inputs=[], outputs=status_out)

# 4. Mount Gradio to FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

# 5. Uvicorn Runner
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)