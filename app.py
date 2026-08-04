import gradio as gr
import spaces  # Import spaces for ZeroGPU support

@spaces.GPU(duration=30)
def process_query(message):
    return f"Enterprise RAG Gateway Active. Echo: {message}"

with gr.Blocks(title="Enterprise RAG Gateway") as demo:
    gr.Markdown("# 🚀 Enterprise RAG Gateway")
    inp = gr.Textbox(label="Query")
    out = gr.Textbox(label="Response")
    btn = gr.Button("Submit")
    btn.click(fn=process_query, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)