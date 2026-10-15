import time

import gradio as gr

from generation import generate_rag_response

SEARCH_CHOICES = [
    ("🔤 Keyword", "keyword"),
    ("🧠 Semantic", "semantic"),
    ("⚡ Hybrid (recommended)", "hybrid"),
]
MODEL_CHOICES = [
    ("🚀 Groq (cloud)", "groq"),
    ("💻 Ollama (local Deepseek)", "ollama"),
]

CSS = """
.gradio-container {max-width: 1200px !important; margin: auto;}
#header {text-align: center; padding: 8px 0 0 0;}
#answer-box {
    min-height: 420px;
    padding: 16px 20px;
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    background: var(--background-fill-secondary);
}
#status {font-size: 0.9em; opacity: 0.8;}
"""


def on_submit(query, search_type, model_type, top_k, stream):
    """Run the RAG pipeline and yield (answer, status) updates."""
    if not query or not query.strip():
        yield "⚠️ Please enter a question.", "Waiting for input"
        return

    start = time.time()
    yield "_Retrieving relevant information..._", "🔎 Retrieving..."

    try:
        if stream:
            full_response = ""
            last_update = 0.0
            for chunk in generate_rag_response(
                query, search_type, int(top_k), model_type, stream=True
            ):
                full_response += chunk
                # Throttle UI updates (max ~20 per second)
                now = time.time()
                if now - last_update > 0.05:
                    last_update = now
                    yield full_response, "✍️ Generating..."
            yield full_response, f"✅ Done in {time.time() - start:.1f}s"
        else:
            response = generate_rag_response(
                query, search_type, int(top_k), model_type, stream=False
            )
            yield response, f"✅ Done in {time.time() - start:.1f}s"
    except Exception as e:
        yield f"❌ Something went wrong:\n\n`{e}`", "Error"


def clear_all():
    return "", "", "Ready"


with gr.Blocks(
    title="LocalRAG Q&A System",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css=CSS,
) as demo:
    gr.Markdown(
        "# 📚 LocalRAG Q&A System\n"
        "Ask questions about the RAG paper and get answers grounded in retrieved documents.",
        elem_id="header",
    )

    with gr.Row(equal_height=False):
        # ---------------- Left: input + settings ----------------
        with gr.Column(scale=4, min_width=320):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about Retrieval-Augmented Generation...",
                lines=5,
                max_lines=10,
                autofocus=True,
            )

            with gr.Row():
                submit_btn = gr.Button("Generate Answer", variant="primary", scale=3)
                stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
                clear_btn = gr.Button("🗑 Clear", scale=1)

            with gr.Accordion("⚙️ Settings", open=True):
                search_type = gr.Radio(
                    SEARCH_CHOICES,
                    label="Search Method",
                    value="hybrid",
                    info="How relevant chunks are retrieved",
                )
                model_type = gr.Radio(
                    MODEL_CHOICES,
                    label="AI Model",
                    value="groq",
                    info="Model used to generate the answer",
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of chunks (top-k)",
                    info="More chunks = more context, but slower",
                )
                stream_checkbox = gr.Checkbox(
                    label="Stream response",
                    value=True,
                    info="Show the answer as it is being generated",
                )

        # ---------------- Right: answer ----------------
        with gr.Column(scale=6, min_width=400):
            gr.Markdown("### 💬 Answer")
            output = gr.Markdown(
                value="_Your answer will appear here._",
                elem_id="answer-box",
            )
            status = gr.Markdown("Ready", elem_id="status")

    gr.Examples(
        examples=[
            ["How does RAG work?", "hybrid", "groq", 5, True],
            [
                "What are the benefits of RAG compared to fine-tuning?",
                "semantic",
                "groq",
                5,
                True,
            ],
            ["Explain RAG architecture", "hybrid", "ollama", 5, True],
            [
                "What are common challenges in RAG implementations?",
                "keyword",
                "groq",
                5,
                False,
            ],
        ],
        inputs=[query_input, search_type, model_type, top_k, stream_checkbox],
        label="💡 Example questions",
    )

    with gr.Accordion("📘 How to use this system", open=False):
        gr.Markdown(
            """
1. **Enter your question** about Retrieval-Augmented Generation (RAG).
2. Choose a **search method**:
   - **Keyword**: traditional text search
   - **Semantic**: meaning-based search using embeddings
   - **Hybrid**: combines keyword and semantic search
3. Select an **AI model**:
   - **Groq**: fast cloud inference (requires `GROQ_API_KEY`)
   - **Ollama**: local Deepseek model running via Ollama
4. Toggle **streaming** to see the response generated in real time.

The system retrieves relevant information from the indexed RAG paper and generates an answer based on it.
            """
        )

    # ---------------- Events ----------------
    inputs = [query_input, search_type, model_type, top_k, stream_checkbox]
    outputs = [output, status]

    click_event = submit_btn.click(
        on_submit, inputs=inputs, outputs=outputs, show_progress="hidden"
    )
    enter_event = query_input.submit(
        on_submit, inputs=inputs, outputs=outputs, show_progress="hidden"
    )

    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[click_event, enter_event])
    clear_btn.click(clear_all, inputs=None, outputs=[query_input, output, status])

# Launch the app
if __name__ == "__main__":
    demo.queue().launch()