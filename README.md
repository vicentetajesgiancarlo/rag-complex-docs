# RAG for Complex Academic Documents

A Retrieval-Augmented Generation (RAG) system that indexes PDF documents and answers questions grounded in the source text. Upload any academic paper via the web UI — or download sample papers from arxiv — and ask questions about them.

**Features**
- 📄 Upload PDFs via the sidebar — indexed into ChromaDB instantly
- 🗑️ Delete individual documents from the index with a single click
- 🔍 Retrieval-only mode — works without any LLM, shows the most relevant passages
- 🤖 LLM answers via **Ollama** (free, local) or **OpenAI**
- ↺ **Reconnect LLM** button — no restart needed if the backend starts after the app

---

## Architecture Overview

```
User Query
    |
    v
[Streamlit Chat UI]
    |
    v
[Retriever] ── similarity search ──> [ChromaDB Vector Store]
    |                                        ^
    |  top-4 chunks (RunnableParallel)       |
    v                                        |
[LLM (OpenAI / Ollama)]              [BAAI/bge-small-en-v1.5 Embeddings]
    |                                        ^
    v                                        |
Generated Answer + Sources            [Document Chunks]
                                             ^
                                             |
                                      [PyMuPDFLoader + RecursiveCharacterTextSplitter]
                                             ^
                                             |
                                      [arxiv PDF Downloads  /  user uploads]
```

1. **Data Acquisition**: Papers are downloaded from arxiv using the `arxiv` library, or uploaded directly via the UI.
2. **Parsing**: PDFs are loaded with `PyMuPDFLoader`, which preserves reading order, equations, and multi-column layouts.
3. **Chunking**: Documents are split into overlapping chunks (1000 chars, 200 overlap) using `RecursiveCharacterTextSplitter`.
4. **Embedding**: Chunks are embedded with `BAAI/bge-small-en-v1.5` and stored in a local ChromaDB database.
5. **Retrieval & Generation**: A `RunnableParallel` call retrieves the top-4 chunks and generates an answer in a single pass — sources displayed always match the context the LLM used.

---

## Requirements

- **Python ≥ 3.9**
- No GPU required — all embedding runs on CPU

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/vicentetajesgiancarlo/rag-complex-docs.git
cd rag-complex-docs

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure an LLM backend (choose one):

# Option A: OpenAI
cp .env.example .env          # then open .env and paste your API key

# Option B: Ollama (free, runs locally)
ollama pull llama3
ollama serve

# The app also works with NO LLM — it displays the most relevant raw passages.

# 4. (Optional) Pre-populate with arxiv sample papers
python src/document_processor.py   # downloads ~8 papers to data/raw_pdfs/
python src/embedding_utils.py      # embeds and indexes them into ChromaDB

# Skip step 4 and upload your own PDFs via the sidebar instead.

# 5. Launch the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **Tip:** If the LLM backend starts *after* the app, click **↺ Reconnect LLM** in the sidebar — no restart needed.

---

## Project Structure

```
rag-complex-docs/
├── data/
│   ├── raw_pdfs/          # Downloaded or user-uploaded PDFs  (gitignored)
│   └── vector_db/         # Persistent ChromaDB storage       (gitignored)
├── src/
│   ├── document_processor.py  # arxiv download, PDF parsing, and chunking
│   ├── embedding_utils.py     # Embedding model & vector DB creation
│   └── rag_pipeline.py        # RAG chain (retriever + LLM)
├── app.py                 # Streamlit chat interface
├── .env.example           # OpenAI key template — copy to .env
├── requirements.txt
├── README.md
└── TECHNICAL_DOCS.md      # In-depth architecture and API reference
```

---

## Design Decisions

### Why PyMuPDF?
Academic papers contain complex layouts: multi-column text, mathematical equations, tables, and embedded figures. PyMuPDF (`fitz`) handles these far better than simpler parsers like `pdfminer` or `PyPDF2`. It preserves reading order and extracts equations as unicode text rather than garbled symbols.

### Why RecursiveCharacterTextSplitter?
Mathematical and hardware architecture texts have deeply nested logical structures (theorem → proof → lemma, pipeline stage → hazard → solution). The recursive splitter respects paragraph and sentence boundaries before falling back to character-level splits, keeping related concepts together within each chunk. The 200-character overlap prevents context loss at boundaries.

### Why BAAI/bge-small-en-v1.5?
Excellent balance of retrieval quality and speed for local CPU inference. Produces normalized embeddings (ideal for cosine similarity) and ranks highly on the MTEB benchmark. The 512-dimensional output keeps ChromaDB storage small.

### Why RunnableParallel for retrieval?
A naïve implementation calls the retriever twice — once inside the chain for context, and once separately to surface sources. `RunnableParallel` issues both in a single invocation so the sources displayed in the UI are guaranteed to be the same documents the LLM actually read.
