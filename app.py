"""
PDF RAG Chatbot
---------------
Upload a PDF, then chat with it.

Pipeline (same as the notebook):
  PDF -> PyPDFLoader -> text splitter -> MiniLM embeddings -> FAISS -> Gemini

Each browser session gets its own index and chat memory, so several users
can upload different PDFs at the same time without seeing each other's data.
"""

import os
from functools import lru_cache

import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads GOOGLE_API_KEY from a local .env file if present

# ----------------------------------------------------------------------------
# Settings (override with environment variables if you want)
# ----------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "6"))  # past Q&A pairs sent to Gemini

# ----------------------------------------------------------------------------
# Prompt (your notebook prompt, generalised so it works for ANY uploaded PDF)
# ----------------------------------------------------------------------------
PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert assistant that helps the user understand an uploaded PDF document.

Your only source of truth is the document context below. Answer clearly and accurately.
You also have the previous conversation, which you must use to understand follow-up
questions and references to earlier answers.

CONVERSATION HISTORY:
{chat_history}

DOCUMENT CONTEXT:
{context}

CURRENT USER QUESTION:
{question}


IMPORTANT INSTRUCTIONS:

1. DOCUMENT-GROUNDED ANSWERS
   - Use ONLY information supported by the document context.
   - Do not invent facts, definitions, examples, commands, code, or numbers that the
     document does not support.
   - If the answer cannot be found or reasonably determined from the context, say exactly:
     "I could not find this information in the document."

2. CONVERSATION MEMORY
   - If the user says "it", "this", "that", "the above", "explain more",
     "give another example", "why", "how", or similar, work out what they refer to
     from the previous conversation.
   - Do not repeat the entire previous explanation unnecessarily.

3. CODE AND TECHNICAL CONTENT
   - When the document contains code, commands, formulas or syntax, show them in a
     code block and preserve them exactly.
   - Explain the important parts in simple language.
   - Do not silently modify the user's code or formulas. If you correct something,
     show what changed and why.

4. TEACHING STYLE
   - Start with the direct answer, then explain WHY.
   - Use a simple example only when the document supports one.
   - Prefer beginner-friendly language; go deeper only if the user asks.

5. OUTPUT FORMAT
   Follow the user's requested format. Otherwise use:
   - Definition -> short explanation (+ example if the document has one)
   - Comparison -> Markdown table
   - Steps / procedure -> numbered list
   - Several concepts -> headings and bullet points

6. FORMAT CHANGES
   If the user asks to reformat a previous answer, keep the same information and
   reformat it. Do not retrieve an unrelated topic.

7. DO NOT HALLUCINATE
   - Never make up information to make an answer look complete.
   - Do not present outside knowledge as if it came from the document.

ANSWER:
"""
)


# ----------------------------------------------------------------------------
# Heavy objects are created once and shared by all users
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_llm():
    api_key = os.getenv("AQ.Ab8RN6IefEhFppnlBi3Exv28-YfplheUIAHzVpIFgdOjjkfqvw")
    if not api_key:
        raise gr.Error(
            "GOOGLE_API_KEY is not set. Add it to a .env file "
            "(or as a secret if you deploy to Hugging Face Spaces)."
        )
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=api_key)


SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def extract_text(content) -> str:
    """Gemini may return a str or a list of parts; always return plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def new_session() -> dict:
    """Empty per-user state."""
    return {"vectorstore": None, "filename": None, "memory": []}


# ----------------------------------------------------------------------------
# Step 1: user uploads a PDF -> build the index
# ----------------------------------------------------------------------------
def process_pdf(file_path, session):
    session = new_session()  # a new upload always starts a fresh conversation

    if not file_path:
        return session, "Upload a PDF to begin.", []

    filename = os.path.basename(file_path)

    try:
        pages = PyPDFLoader(file_path).load()
        pages = [p for p in pages if p.page_content.strip()]  # drop empty pages

        if not pages:
            return (
                session,
                "❌ No text could be extracted. This PDF may be scanned images "
                "(it would need OCR first).",
                [],
            )

        chunks = SPLITTER.split_documents(pages)
        session["vectorstore"] = FAISS.from_documents(chunks, get_embeddings())
        session["filename"] = filename

    except gr.Error:
        raise
    except Exception as e:  # corrupted / encrypted PDF etc.
        return session, f"❌ Could not read this PDF: {e}", []

    status = (
        f"✅ **{filename}** is ready — {len(pages)} pages with text, "
        f"{len(chunks)} chunks indexed. Ask your questions below."
    )
    greeting = [
        {
            "role": "assistant",
            "content": f"I've read **{filename}**. What would you like to know about it?",
        }
    ]
    return session, status, greeting


def clear_pdf():
    return new_session(), "Upload a PDF to begin.", []


# ----------------------------------------------------------------------------
# Step 2: chat
# ----------------------------------------------------------------------------
def chat(message, history, session):
    message = (message or "").strip()
    if not message:
        yield "", history, session
        return

    history = list(history or [])
    history.append({"role": "user", "content": message})

    if not session or session.get("vectorstore") is None:
        history.append(
            {"role": "assistant", "content": "📎 Please upload a PDF first."}
        )
        yield "", history, session
        return

    memory = session["memory"]

    # Follow-ups like "what about WHERE?" retrieve badly on their own, so
    # include the previous question in the search query.
    search_query = message
    if memory:
        search_query = f"{memory[-2][1]} {message}"  # memory[-2] = last user turn

    docs = session["vectorstore"].similarity_search(search_query, k=TOP_K)
    context = "\n\n".join(d.page_content for d in docs)
    pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})

    recent = memory[-MEMORY_TURNS * 2 :]
    history_text = "\n".join(f"{role}: {text}" for role, text in recent)

    prompt_text = PROMPT.format(
        context=context, question=message, chat_history=history_text
    )

    # Stream the answer token by token
    history.append({"role": "assistant", "content": ""})
    answer = ""
    try:
        for chunk in get_llm().stream(prompt_text):
            answer += extract_text(chunk.content)
            history[-1]["content"] = answer
            yield "", history, session
    except gr.Error:
        raise
    except Exception as e:
        history[-1]["content"] = f"⚠️ Something went wrong while asking Gemini: {e}"
        yield "", history, session
        return

    answer = answer.strip()
    sources = ", ".join(f"Page {p}" for p in pages)
    history[-1]["content"] = f"{answer}\n\n📄 **Sources:** {sources}"

    memory.append(("User", message))
    memory.append(("Assistant", answer))

    yield "", history, session


def clear_chat(session):
    if session:
        session["memory"] = []
    return []


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
with gr.Blocks(title="PDF Chatbot") as demo:
    gr.Markdown(
        "# 📄 Chat with your PDF\n"
        "Upload a PDF, wait for the ✅ message, then ask questions about it."
    )

    session = gr.State(new_session())

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            pdf_file = gr.File(
                label="Upload PDF", file_types=[".pdf"], type="filepath"
            )
            status = gr.Markdown("Upload a PDF to begin.")
            clear_btn = gr.Button("🗑️ Clear conversation")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=520, show_label=False)
            msg = gr.Textbox(
                placeholder="Ask a question about the document...",
                show_label=False,
                container=True,
            )

    # Upload / remove the PDF
    pdf_file.upload(
        process_pdf, inputs=[pdf_file, session], outputs=[session, status, chatbot]
    )
    pdf_file.clear(clear_pdf, outputs=[session, status, chatbot])

    # Send a message
    msg.submit(chat, inputs=[msg, chatbot, session], outputs=[msg, chatbot, session])

    # Clear only the conversation (keeps the PDF)
    clear_btn.click(clear_chat, inputs=[session], outputs=[chatbot])


if __name__ == "__main__":
    demo.queue().launch()
