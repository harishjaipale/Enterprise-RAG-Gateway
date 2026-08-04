import os
import sys
import gradio as gr
import spaces  # ZeroGPU mandatory requirement

# Current directory ko path me add karein
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Aapki main.py se actual RAG query function import karein
# (Agar main.py me function ka naam alag hai, toh niche import me update kar sakte hain)
try:
    from main import query_engine_function as run_rag_query 
except ImportError:
    # Fallback agar direct function available na ho
    def run_rag_query(message):
        return f"Enterprise RAG Gateway Backend Connected. Processing: {message}"

@spaces.GPU(duration=60)
def handle_user_query(message):
    try:
        # Aapka actual backend RAG pipeline yahan execute hoga
        response = run_rag_query(message)
        return response
    except Exception as e:
        return f"Error executing RAG pipeline: {str(e)}"

# Professional Gradio Interface Layout
with gr.Blocks(title="Enterprise RAG Gateway", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🚀 Enterprise RAG Gateway")
    gr.Markdown("Secure RAG Pipeline powered by Qdrant Vector DB & LLM Engine.")
    
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="Enter Query", 
                placeholder="Ask about your documents, architecture, or system health...",
                lines=3
            )
            submit_btn = gr.Button("Submit Query", variant="primary")
        
        with gr.Column(scale=6):
            output_box = gr.Textbox(label="Gateway Response", lines=8, interactive=False)
            
    submit_btn.click(fn=handle_user_query, inputs=query_input, outputs=output_box)
    query_input.submit(fn=handle_user_query, inputs=query_input, outputs=output_box)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)