import gradio as gr
from extraction_agent import refine_question, extract_keywords
from retrieval import load_vectorstore
from generate_response import answer_question_with_context

# Load vectorstore once
vectorstore = load_vectorstore()

# Hold pipeline state
last_refined = ""
last_keywords = []
last_chunks = []
last_answer = ""
last_note = ""


def run_rag(query):
    global last_refined, last_keywords, last_chunks, last_answer, last_note

    refined = refine_question(query)
    keywords = extract_keywords(refined)
    chunks = vectorstore.similarity_search_with_score(refined, k=5)
    result = answer_question_with_context(chunks, refined)

    last_refined = refined
    last_keywords = keywords
    last_chunks = chunks
    last_answer = result["answer"]
    last_note = result["note"]

    return last_answer


def show_pipeline():
    return f"""
### 🔍 Refined Question
{last_refined}

### 🔑 Extracted Keywords
{last_keywords}

### 📝 Note
{last_note}
"""


def show_chunks():
    if not last_chunks:
        return "No chunks yet. Ask a question first."

    out = ""
    for doc, score in last_chunks:
        preview = doc.page_content[:350].replace("\n", " ")
        out += f"""
### 🧩 Chunk ID: {doc.metadata.get('id')}
Score: {round(score, 4)}

{preview}...

---
"""
    return out


# ========================================
# UI with NO Chatbot + Large Answer Window
# ========================================

custom_css = """
#big_answer_box {
    height: 800px !important;
    overflow-y: auto !important;
    border: 1px solid #ccc;
    padding: 16px;
    background: #fafafa;
    border-radius: 12px;
    font-size: 17px;
}
"""

with gr.Blocks(css=custom_css, title="Cow Expert Chatbot") as demo:

    gr.Markdown("# 🐄 Cow Expert Chatbot (RAG UI)")
    gr.Markdown("### Large Answer Window • No Chatbot • Best for Long Text")

    with gr.Tabs():

        # --- Chat Tab (No Chatbot Component) ---
        with gr.Tab("Chat"):
            query = gr.Textbox(
                label="Ask a question",
                placeholder="Type your dairy cow question here…"
            )

            answer_md = gr.Markdown(
                "Ask something to see the answer here.",
                elem_id="big_answer_box"
            )

            submit_btn = gr.Button("Submit", variant="primary")

            def update_answer(q):
                ans = run_rag(q)
                return f"### 💬 Final Answer\n\n{ans}"

            submit_btn.click(update_answer, [query], [answer_md])


        # --- Pipeline Tab ---
        with gr.Tab("Pipeline Details"):
            pipeline_box = gr.Markdown("Ask something first.")
            gr.Button("Refresh").click(show_pipeline, None, pipeline_box)

        # --- Chunks Tab ---
        with gr.Tab("Retrieved Chunks"):
            chunk_box = gr.Markdown("Ask something first.")
            gr.Button("Refresh").click(show_chunks, None, chunk_box)


demo.launch()