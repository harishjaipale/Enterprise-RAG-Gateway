import os
import sys
import gradio as gr
import spaces  # ZeroGPU mandatory requirement for Hugging Face Spaces

# Set up system path to include the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Path to the custom frontend index.html file
index_path = os.path.join(os.path.dirname(__file__), "index.html")

# Read the HTML content safely
if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
else:
    html_content = "<h2>Error: index.html not found in repository!</h2>"

# ZeroGPU supported backend handler function
@spaces.GPU(duration=60)
def process_backend_request(data):
    """
    Backend handler function wrapped with ZeroGPU decorator.
    Integrates with your core RAG engine logic if needed.
    """
    return f"Backend processed successfully: {data}"

# Build the Gradio Blocks interface embedding the custom HTML frontend
with gr.Blocks(title="Enterprise RAG Gateway") as demo:
    # Inject the custom HTML/CSS/JS frontend UI directly into Gradio
    gr.HTML(html_content)

# Launch the application on 0.0.0.0:7860 for Hugging Face deployment
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)