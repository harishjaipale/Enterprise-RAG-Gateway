import os
import sys
import spaces  # Import spaces
import uvicorn
import gradio as gr
from fastapi.responses import FileResponse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import app as fastapi_app

@fastapi_app.get("/", include_in_schema=False)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Enterprise RAG Gateway Backend API Active"}

# ZeroGPU required decorator for HF Space when ZeroGPU is active
@spaces.GPU
def dummy_func():
    return "Enterprise RAG Gateway active on GPU"

demo = gr.Interface(
    fn=dummy_func,
    inputs=[],
    outputs="text",
    title="Enterprise RAG Gateway"
)

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)