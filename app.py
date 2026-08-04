import gradio as gr

def process_query(message):
    return f"Enterprise RAG Gateway Active. Echo: {message}"

demo = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(label="Query"),
    outputs=gr.Textbox(label="Response"),
    title="Enterprise RAG Gateway"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)