"""
Streamlit Chat Interface for the RAG system.
Run with: python -m streamlit run app.py
"""

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from rag_pipeline import (
    build_rag_chain,
    load_vector_store,
    ask_question,
    RETRIEVAL_K,
    LLMUnavailableError,
)
from embedding_utils import get_embedding_model, EMBEDDING_DIM
from document_processor import chunk_documents

MAX_PDF_SIZE_MB = 50
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Academic Paper RAG",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit chrome and collapse their space entirely */
#MainMenu { visibility: hidden; height: 0; overflow: hidden; }
footer    { visibility: hidden; height: 0; overflow: hidden; }
header    { visibility: hidden; height: 0; overflow: hidden; }

/* ── Main background ── */
.stApp {
    background: #0f1117;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    font-size: 0.82rem;
    color: #8b9ab5;
    line-height: 1.7;
}

/* Sidebar section headers */
.sidebar-section-title {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a6fa5;
    margin: 1.4rem 0 0.6rem 0;
}

/* Paper card in sidebar */
.paper-card {
    background: #1c2333;
    border: 1px solid #252d3d;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.78rem;
    color: #c9d1e0;
    line-height: 1.45;
    transition: border-color 0.2s;
}
.paper-card:hover { border-color: #3d6ad6; }

/* LLM status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-top: 6px;
}
.status-online  { background: #0d2d1a; color: #3ecf8e; border: 1px solid #1a4d2e; }
.status-offline { background: #2d1a0d; color: #f59e0b; border: 1px solid #4d3010; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero p {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 0.4rem;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 2rem;
    padding: 0.8rem;
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    margin: 0 auto 1.5rem;
    max-width: 500px;
}
.stat-item { text-align: center; }
.stat-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #3d6ad6;
}
.stat-label {
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
}

/* ── Source pills ── */
.source-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #1e2535;
}
.source-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a6fa5;
    width: 100%;
    margin-bottom: 2px;
}
.source-pill {
    background: #1c2333;
    border: 1px solid #252d3d;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #8b9ab5;
    white-space: nowrap;
}
.source-pill span {
    color: #3d6ad6;
    font-weight: 500;
    margin-right: 4px;
}

/* ── Info / warning banners ── */
.stAlert {
    background: #1c2333 !important;
    border: 1px solid #252d3d !important;
    border-radius: 10px !important;
    color: #8b9ab5 !important;
}

/* ── Bottom sticky bar that wraps the chat input ── */
[data-testid="stBottom"] {
    background: #0f1117 !important;
    border-top: 1px solid #1e2535;
    padding: 0.6rem 1rem 0.6rem !important;
}
/* Kill the white inner wrapper div that Streamlit inserts inside stBottom */
[data-testid="stBottom"] > div {
    background: #0f1117 !important;
}
/* The chat-input component wrapper (gray by default) */
[data-testid="stBottom"] > div > div {
    background: #161b27 !important;
    border-radius: 12px !important;
    border: 1px solid #252d3d !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    padding-top: 0;
}
[data-testid="stChatInput"] textarea {
    background: #161b27 !important;
    border: none !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    outline: none !important;
}
[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}
/* Focused state: highlight the outer wrapper, not the textarea itself */
[data-testid="stBottom"] > div > div:focus-within {
    border-color: #3d6ad6 !important;
    box-shadow: 0 0 0 2px rgba(61,106,214,0.15) !important;
}

/* ── Main content: bottom padding so messages aren't hidden behind sticky bar ── */
[data-testid="stMainBlockContainer"] {
    padding-bottom: 6rem !important;
}

/* Clear button */
.stButton > button {
    background: #1c2333;
    border: 1px solid #252d3d;
    color: #8b9ab5;
    border-radius: 8px;
    font-size: 0.75rem;
    padding: 4px 14px;
    transition: all 0.2s;
}
.stButton > button:hover {
    border-color: #3d6ad6;
    color: #3d6ad6;
    background: #1c2333;
}

/* Spinner */
[data-testid="stSpinner"] { color: #3d6ad6 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #252d3d; border-radius: 4px; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #161b27;
    border: 1px dashed #252d3d;
    border-radius: 10px;
    padding: 4px;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3d6ad6;
}
[data-testid="stFileDropzoneInstructions"] {
    color: #8b9ab5 !important;
    font-size: 0.78rem !important;
}

/* Progress bar */
.stProgress > div > div {
    background: #3d6ad6 !important;
}

/* Success / error boxes */
.upload-success {
    background: #0d2d1a;
    border: 1px solid #1a4d2e;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.78rem;
    color: #3ecf8e;
    margin-top: 6px;
}
.upload-error {
    background: #2d1010;
    border: 1px solid #4d1a1a;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.78rem;
    color: #f87171;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

RAW_PDFS_DIR = Path(__file__).resolve().parent / "data" / "raw_pdfs"

# ---------------------------------------------------------------------------
# Init RAG (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initializing RAG system...")
def init_rag():
    vs = load_vector_store()
    try:
        chain, retriever = build_rag_chain(vs)
        return vs, chain, retriever, True
    except LLMUnavailableError:
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVAL_K})
        return vs, None, retriever, False

vector_store, chain, retriever, llm_available = init_rag()


def ingest_uploaded_pdf(pdf_path: Path, vs) -> int:
    """Parse, chunk, and add a new PDF to the given vector store.

    Returns the number of chunks added.
    """
    loader = PyMuPDFLoader(str(pdf_path))
    pages = loader.load()
    chunks = chunk_documents(pages)
    vs.add_documents(chunks)
    return len(chunks)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# Live counts (re-read each render so they update after uploads)
pdf_files = sorted(RAW_PDFS_DIR.glob("*.pdf")) if RAW_PDFS_DIR.exists() else []
live_pdf_count = len(pdf_files)
try:
    live_chunk_count = len(vector_store.get(include=[])["ids"])
except Exception:
    live_chunk_count = "?"

with st.sidebar:
    st.markdown("""
        <div style="padding: 1rem 0 0.5rem;">
            <span style="font-size:1.4rem;">📖</span>
            <span style="font-size:1rem; font-weight:600; color:#e2e8f0; margin-left:8px;">
                Paper RAG
            </span>
        </div>
    """, unsafe_allow_html=True)

    # LLM status
    if llm_available:
        st.markdown(
            '<div class="status-badge status-online">&#x25CF; LLM Online</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-offline">&#x25CF; Retrieval Only</div>',
            unsafe_allow_html=True,
        )

    # Documents
    st.markdown('<div class="sidebar-section-title">Indexed Documents</div>', unsafe_allow_html=True)
    if pdf_files:
        for pdf in pdf_files:
            name = pdf.stem
            parts = name.split("_", 1)
            display_name = parts[1] if len(parts) > 1 else name
            safe_name = html.escape(display_name)
            st.markdown(f'<div class="paper-card">{safe_name}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="font-size:0.78rem; color:#4a6fa5; line-height:1.7;
                        background:#161b27; border:1px dashed #252d3d;
                        border-radius:8px; padding:10px 12px;">
                No documents yet.<br>Upload a PDF below to get started.
            </div>
        """, unsafe_allow_html=True)

    # LLM config
    st.markdown('<div class="sidebar-section-title">LLM Backend</div>', unsafe_allow_html=True)
    if not llm_available:
        st.markdown("""
            <div style="font-size:0.78rem; color:#8b9ab5; line-height:1.7;">
                No LLM backend found.<br>
                Options:<br>
                <code style="color:#c9d1e0;">OPENAI_API_KEY=sk-... in .env</code><br>
                or:<br>
                <code style="color:#c9d1e0;">ollama pull llama3</code><br>
                <code style="color:#c9d1e0;">ollama serve</code>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="font-size:0.78rem; color:#8b9ab5;">LLM backend is active and ready.</div>
        """, unsafe_allow_html=True)

    # PDF Upload
    st.markdown('<div class="sidebar-section-title">Upload a PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Drop a PDF to index it",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if uploaded_file.size > MAX_PDF_SIZE_BYTES:
            st.markdown(
                f'<div class="upload-error">File too large. Maximum allowed size is {MAX_PDF_SIZE_MB} MB.</div>',
                unsafe_allow_html=True,
            )
        else:
            dest = RAW_PDFS_DIR / uploaded_file.name
            if dest.exists():
                st.markdown(
                    '<div class="upload-success">Already indexed: this PDF is in the corpus.</div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Parsing & indexing..."):
                    try:
                        RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(uploaded_file.getvalue())
                        n_chunks = ingest_uploaded_pdf(dest, vector_store)
                        safe_fname = html.escape(uploaded_file.name)
                        st.markdown(
                            f'<div class="upload-success">'
                            f'Indexed <b>{safe_fname}</b> &mdash; {n_chunks} chunks added.'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.rerun()
                    except Exception as e:
                        dest.unlink(missing_ok=True)
                        st.markdown(
                            f'<div class="upload-error">Failed: {html.escape(str(e))}</div>',
                            unsafe_allow_html=True,
                        )

    # Stats
    st.markdown('<div class="sidebar-section-title">Corpus Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div style="font-size:0.78rem; color:#8b9ab5; line-height:1.8;">
            Papers indexed: <b style="color:#3d6ad6;">{live_pdf_count}</b><br>
            Vector chunks: <b style="color:#3d6ad6;">{live_chunk_count}</b><br>
            Embedding model: <b style="color:#3d6ad6;">BGE-small-en</b><br>
            Vector DB: <b style="color:#3d6ad6;">ChromaDB</b>
        </div>
    """, unsafe_allow_html=True)

    # Clear chat
    st.markdown('<div class="sidebar-section-title">Session</div>', unsafe_allow_html=True)
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
        <div style="position:absolute; bottom:1.2rem; font-size:0.68rem; color:#2d3748;">
            LangChain · ChromaDB · Streamlit
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main area — Hero / Empty state / Chat
# ---------------------------------------------------------------------------

corpus_is_empty = live_pdf_count == 0

if corpus_is_empty:
    # ── Empty state: prompt user to upload ──────────────────────────────────
    st.markdown("""
        <div class="hero">
            <h1>Academic Paper RAG</h1>
            <p>Upload your own PDF documents and ask questions about them.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="max-width:480px; margin:0 auto 2rem; text-align:center;
                    background:#161b27; border:1px dashed #3d6ad6;
                    border-radius:16px; padding:2.5rem 2rem;">
            <div style="font-size:2.5rem; margin-bottom:0.8rem;">📄</div>
            <div style="font-size:1rem; font-weight:600; color:#e2e8f0;
                        margin-bottom:0.5rem;">No documents indexed yet</div>
            <div style="font-size:0.83rem; color:#64748b; line-height:1.6;">
                Use the <b style="color:#3d6ad6;">Upload a PDF</b> panel in the
                sidebar to add your first document.<br><br>
                Once indexed, you can ask any question about its content
                and the system will retrieve the most relevant passages
                and generate a grounded answer.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.chat_input("Upload a PDF first to start chatting...", disabled=True)

else:
    # ── Has documents: show stats bar + chat ────────────────────────────────
    if not st.session_state.messages:
        st.markdown(f"""
            <div class="hero">
                <h1>Academic Paper RAG</h1>
                <p>Ask questions about your indexed documents. Answers are grounded in source text.</p>
            </div>
            <div class="stats-bar">
                <div class="stat-item">
                    <div class="stat-value">{live_pdf_count}</div>
                    <div class="stat-label">Papers</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{live_chunk_count}</div>
                    <div class="stat-label">Chunks</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{EMBEDDING_DIM}d</div>
                    <div class="stat-label">Embeddings</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">top&#8209;{RETRIEVAL_K}</div>
                    <div class="stat-label">Retrieved</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Chat history — rendered as plain markdown (no raw HTML from stored content)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input & response
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching papers and generating answer..."):
                sources = []

                if llm_available and chain is not None:
                    try:
                        result = ask_question(prompt, chain=chain, retriever=retriever)
                        answer = result["answer"]
                        sources = result["sources"]
                    except Exception as e:
                        answer = (
                            f"An error occurred while generating the answer: {e}\n\n"
                            "The LLM may have become unavailable. Try restarting the app."
                        )
                else:
                    docs = retriever.invoke(prompt)
                    answer_parts = ["**Retrieved Context** *(no LLM available)*\n"]
                    seen: set[str] = set()
                    for i, doc in enumerate(docs, 1):
                        meta = doc.metadata
                        source_file = Path(meta.get("source", "unknown")).name
                        page = meta.get("page", "?")
                        title = meta.get("title", source_file)
                        answer_parts.append(
                            f"**{title}** — p.{page}\n\n"
                            f"{doc.page_content[:400]}…\n"
                        )
                        key = f"{source_file}:p{page}"
                        if key not in seen:
                            seen.add(key)
                            sources.append({"title": title, "page": page, "file": source_file})
                    answer = "\n---\n".join(answer_parts)

                # Render answer as plain markdown — no unsafe HTML from LLM output
                st.markdown(answer)

                # Source pills — metadata values are HTML-escaped before injection
                if sources:
                    pills_html = '<div class="source-container"><div class="source-label">Sources</div>'
                    for s in sources:
                        truncated = s["title"][:55]
                        ellipsis = "…" if len(s["title"]) > 55 else ""
                        safe_title = html.escape(truncated) + ellipsis
                        safe_page = html.escape(str(s["page"]))
                        pills_html += (
                            f'<div class="source-pill">'
                            f'<span>p.{safe_page}</span>{safe_title}'
                            f'</div>'
                        )
                    pills_html += "</div>"
                    st.markdown(pills_html, unsafe_allow_html=True)

        # Persist to history as plain markdown
        source_md = ""
        if sources:
            source_md = "\n\n---\n**Sources:** " + " · ".join(
                f'*{s["title"][:40]}…* p.{s["page"]}' for s in sources
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": answer + source_md}
        )
