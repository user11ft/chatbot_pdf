"""
PDF RAG Chatbot (Streamlit version)
-----------------------------------
Upload a PDF in the sidebar, then chat with it.

Pipeline: PDF -> PyPDFLoader -> text splitter -> MiniLM embeddings -> FAISS -> Gemini
"""

import os
import tempfile

import streamlit as st

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
MEMORY_TURNS = 6  # past Q&A pairs sent to Gemini

PROMPT = """
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


# ----------------------------------------------------------------------------
# Helpers (heavy libraries are imported lazily so the page appears immediately)
# ----------------------------------------------------------------------------
def get_api_key():
    try:
        key = st.secrets.get("AQ.Ab8RN6IefEhFppnlBi3Exv28-YfplheUIAHzVpIFgdOjjkfqvw")
    except Exception:  # no secrets configured
        key = None
    return key or os.getenv("AQ.Ab8RN6IefEhFppnlBi3Exv28-YfplheUIAHzVpIFgdOjjkfqvw")


@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_llm(api_key):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=api_key)


def build_index(pdf_bytes):
    """Read the PDF, split it, and build a FAISS index. Returns (index, pages, chunks)."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        path = f.name
    try:
        pages = PyPDFLoader(path).load()
    finally:
        os.remove(path)

    pages = [p for p in pages if p.page_content.strip()]  # drop empty pages
    if not pages:
        return None, 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    index = FAISS.from_documents(chunks, get_embeddings())
    return index, len(pages), len(chunks)


def extract_text(content):
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


# ----------------------------------------------------------------------------
# Session state (one per browser tab, so users don't see each other's data)
# ----------------------------------------------------------------------------
def reset_state():
    st.session_state.index = None
    st.session_state.file_key = None
    st.session_state.messages = []  # what is shown on screen
    st.session_state.memory = []  # (role, text) pairs sent to Gemini


if "messages" not in st.session_state:
    reset_state()

# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("📄 Chat with your PDF")

api_key = get_api_key()
if not api_key:
    st.error(
        "**GOOGLE_API_KEY is missing.**\n\n"
        "On Streamlit Cloud: open **Manage app → Settings → Secrets** and add:\n\n"
        '`GOOGLE_API_KEY = "your-key-here"`\n\n'
        "Then reboot the app."
    )
    st.stop()

# ---- Sidebar: upload ----
with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded is None:
        if st.session_state.file_key is not None:
            reset_state()  # user removed the file
    else:
        file_key = (uploaded.name, uploaded.size)
        if st.session_state.file_key != file_key:  # a new file
            reset_state()
            with st.spinner(f"Reading {uploaded.name}..."):
                try:
                    index, n_pages, n_chunks = build_index(uploaded.getvalue())
                except Exception as e:
                    index, n_pages, n_chunks = None, 0, 0
                    st.error(f"Could not read this PDF: {e}")

            if index is not None:
                st.session_state.index = index
                st.session_state.file_key = file_key
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": f"I've read **{uploaded.name}**. What would you like to know about it?",
                    }
                ]
                st.session_state.stats = (n_pages, n_chunks)
            elif n_pages == 0 and index is None:
                st.warning(
                    "No text could be extracted. This PDF may be scanned images "
                    "(it would need OCR first)."
                )

        if st.session_state.index is not None:
            n_pages, n_chunks = st.session_state.stats
            st.success(f"✅ Ready: {n_pages} pages, {n_chunks} chunks")

    if st.button("🗑️ Clear conversation", disabled=st.session_state.index is None):
        st.session_state.messages = st.session_state.messages[:1]
        st.session_state.memory = []
        st.rerun()

# ---- Chat history ----
if st.session_state.index is None:
    st.info("👈 Upload a PDF in the sidebar to start chatting.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---- New question ----
question = st.chat_input(
    "Ask a question about the document...",
    disabled=st.session_state.index is None,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    memory = st.session_state.memory

    # Follow-ups like "what about WHERE?" search badly alone, so include the
    # previous question in the search query.
    search_query = question if not memory else f"{memory[-2][1]} {question}"
    docs = st.session_state.index.similarity_search(search_query, k=TOP_K)
    context = "\n\n".join(d.page_content for d in docs)
    pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})

    recent = memory[-MEMORY_TURNS * 2 :]
    history_text = "\n".join(f"{role}: {text}" for role, text in recent)
    prompt_text = PROMPT.format(
        context=context, question=question, chat_history=history_text
    )

    def token_stream():
        for chunk in get_llm(api_key).stream(prompt_text):
            yield extract_text(chunk.content)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(token_stream()).strip()
            sources = ", ".join(f"Page {p}" for p in pages)
            st.markdown(f"📄 **Sources:** {sources}")
        except Exception as e:
            answer = None
            st.error(f"Something went wrong while asking Gemini: {e}")

    if answer:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"{answer}\n\n📄 **Sources:** {sources}",
            }
        )
        memory.append(("User", question))
        memory.append(("Assistant", answer))
