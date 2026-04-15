# Technical Documentation — Academic Paper RAG System

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Project Structure](#3-project-structure)
4. [Dependencies](#4-dependencies)
5. [Module: document_processor.py](#5-module-document_processorpy)
6. [Module: embedding_utils.py](#6-module-embedding_utilspy)
7. [Module: rag_pipeline.py](#7-module-rag_pipelinepy)
8. [Module: app.py](#8-module-apppy)
9. [Data Flow: Offline Indexing](#9-data-flow-offline-indexing)
10. [Data Flow: Online Querying](#10-data-flow-online-querying)
11. [Data Flow: User PDF Upload](#11-data-flow-user-pdf-upload)
12. [Key Technical Decisions](#12-key-technical-decisions)
13. [LangChain Internals](#13-langchain-internals)
14. [Vector Database Internals](#14-vector-database-internals)
15. [Embedding Model Internals](#15-embedding-model-internals)
16. [LLM Backend Configuration](#16-llm-backend-configuration)
17. [Chunking Strategy Deep Dive](#17-chunking-strategy-deep-dive)
18. [Metadata Schema](#18-metadata-schema)
19. [Failure Modes & Mitigations](#19-failure-modes--mitigations)
20. [Performance Characteristics](#20-performance-characteristics)
21. [Memory & Disk Usage](#21-memory--disk-usage)
22. [Configuration Reference](#22-configuration-reference)
23. [Scalability & Limitations](#23-scalability--limitations)
24. [How to Extend the System](#24-how-to-extend-the-system)
25. [Security Considerations](#25-security-considerations)
26. [RAG Quality & Evaluation](#26-rag-quality--evaluation)
27. [Comparison with Alternative Approaches](#27-comparison-with-alternative-approaches)
28. [Glossary](#28-glossary)

---

## 1. System Overview

This is a **Retrieval-Augmented Generation (RAG)** system built to query complex academic documents — specifically papers on computer architecture, topology, and differential equations.

RAG is an AI pattern that grounds Large Language Model (LLM) responses in a private document corpus. Rather than relying on the LLM's training data (which has a cutoff date and may not contain your documents), the system:

1. **Indexes** documents into a vector database at setup time.
2. **Retrieves** the most semantically relevant passages for any query at runtime.
3. **Generates** a grounded answer by injecting those passages as context into the LLM prompt.

This prevents hallucination, allows working with private/custom documents, and provides traceable source citations.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      OFFLINE INDEXING                           │
│                                                                 │
│  arxiv API ──► download_arxiv_papers()                          │
│                       │                                         │
│                       ▼                                         │
│              data/raw_pdfs/*.pdf                                │
│                       │                                         │
│                       ▼                                         │
│              PyMuPDFLoader.load()          ← 1 Document/page    │
│                       │                                         │
│                       ▼                                         │
│     RecursiveCharacterTextSplitter         ← chunk_size=1000    │
│              chunk_documents()               chunk_overlap=200   │
│                       │                                         │
│                       ▼                                         │
│         HuggingFaceEmbeddings              ← BAAI/bge-small     │
│          (BAAI/bge-small-en-v1.5)            512-dim vectors     │
│                       │                                         │
│                       ▼                                         │
│           ChromaDB.from_documents()        ← persists to disk   │
│              data/vector_db/                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      ONLINE QUERYING                            │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  BAAI/bge-small-en-v1.5  ── embed query ──►  512-dim vector     │
│                                                    │            │
│                                                    ▼            │
│                                        ChromaDB cosine search   │
│                                          (top-4 chunks)         │
│                                                    │            │
│      ┌─────────────────────────────────────────────┘            │
│      │  4 most relevant Document chunks                         │
│      │                                                          │
│      ▼                                                          │
│  _format_docs() ── joined context string                        │
│      │                                                          │
│      ▼                                                          │
│  ChatPromptTemplate  ──► filled prompt (context + question)     │
│      │                                                          │
│      ▼                                                          │
│  LLM (OpenAI gpt-3.5-turbo OR Ollama llama3)                   │
│      │                                                          │
│      ▼                                                          │
│  StrOutputParser() ──► answer string                            │
│      │                                                          │
│      ▼                                                          │
│  ask_question() returns {answer, sources}                       │
│      │                                                          │
│      ▼                                                          │
│  Streamlit renders answer + source pill badges                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
rag_complex_docs/
├── data/
│   ├── raw_pdfs/          # PDF files (downloaded or user-uploaded)
│   └── vector_db/         # ChromaDB persistent storage (SQLite + parquet)
├── src/
│   ├── document_processor.py   # arxiv download + PDF parsing + chunking
│   ├── embedding_utils.py      # embedding model + vector DB creation/loading
│   └── rag_pipeline.py         # LCEL chain + LLM init + ask_question()
├── app.py                 # Streamlit web interface
├── requirements.txt       # Python dependencies
├── TECHNICAL_DOCS.md      # This file
└── README.md              # User-facing setup guide
```

Each file in `src/` is independently runnable as a script (via `if __name__ == "__main__"`) and also importable as a module by `app.py`.

---

## 4. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `langchain` | ≥0.3.0 | Core framework, orchestration |
| `langchain-community` | ≥0.3.0 | `PyMuPDFLoader`, `Chroma`, `Ollama` integrations |
| `langchain-huggingface` | ≥0.1.0 | `HuggingFaceEmbeddings` adapter |
| `langchain-openai` | ≥0.2.0 | `ChatOpenAI` adapter |
| `langchain-text-splitters` | ≥0.3.0 | `RecursiveCharacterTextSplitter` |
| `chromadb` | ≥0.5.0 | Local vector database |
| `sentence-transformers` | ≥3.0.0 | Runtime backend for the BGE model |
| `pymupdf` | ≥1.24.0 | PDF parsing engine (`fitz`) |
| `arxiv` | ≥2.1.0 | arxiv REST API client |
| `streamlit` | ≥1.38.0 | Web UI framework |
| `python-dotenv` | ≥1.0.0 | Reads `OPENAI_API_KEY` from `.env` |

---

## 5. Module: document_processor.py

**File:** [src/document_processor.py](src/document_processor.py)

### Constants

```python
SEARCH_QUERIES = [
    "superscalar architecture",
    "cache coherence",
    "algebraic topology",
    "differential equations",
]
MAX_RESULTS_PER_QUERY = 2
RAW_PDFS_DIR = BASE_DIR / "data" / "raw_pdfs"
```

### `download_arxiv_papers()`

```python
def download_arxiv_papers(
    queries: list[str] = SEARCH_QUERIES,
    max_per_query: int = MAX_RESULTS_PER_QUERY,
    output_dir: Path = RAW_PDFS_DIR,
) -> list[str]
```

**What it does:**
- Creates an `arxiv.Client()` and runs one `arxiv.Search` per query.
- Sorts results by `SubmittedDate` to get the most recent papers.
- Uses a `seen_ids: set[str]` to deduplicate — if the same paper appears under two queries, it is only downloaded once.
- Sanitizes the paper title to produce a filesystem-safe filename by stripping `/`, `\`, `:`, `?`, `"` and truncating to 80 characters.
- Final filename format: `{arxiv_id}_{sanitized_title}.pdf`
- Skips files that already exist on disk (idempotent — safe to re-run).
- Returns a list of absolute file paths for all downloaded PDFs.

**Error handling:** Each download is wrapped in `try/except`. A failure on one paper does not abort the rest.

---

### `load_pdfs()`

```python
def load_pdfs(pdf_dir: Path = RAW_PDFS_DIR) -> list
```

**What it does:**
- Globs all `.pdf` files from `pdf_dir` in sorted order (deterministic processing).
- Instantiates `PyMuPDFLoader(str(pdf_path))` for each file.
- Calls `.load()` which returns one `Document` object per page.
- Accumulates all pages into a flat list and returns it.

**Why PyMuPDF:**
- Preserves **reading order** in multi-column academic layouts.
- Correctly decodes **unicode math symbols** (∂, ∇, ∑, α, β, etc.).
- Extracts embedded **PDF metadata** (title, author, creation date) into `Document.metadata`.
- More robust than `PyPDF2` or `pdfminer` for complex technical PDFs.

**Output shape:** `list[Document]` where each `Document.page_content` is the extracted text of one page, and `Document.metadata` contains source path, page number, title, author, etc.

---

### `chunk_documents()`

```python
def chunk_documents(
    documents: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list
```

**What it does:**
- Creates a `RecursiveCharacterTextSplitter` with the separator priority chain:
  `["\n\n", "\n", ". ", " ", ""]`
- Calls `.split_documents(documents)` which processes the entire document list and returns a new flat list of `Document` objects (chunks), each retaining the original `metadata` from its parent page.

**Separator priority explained:**
1. `"\n\n"` — split at blank lines (paragraph breaks). Best semantic boundary.
2. `"\n"` — split at line breaks. Used when a paragraph exceeds `chunk_size`.
3. `". "` — split at sentence boundaries. Keeps sentences together.
4. `" "` — split at word boundaries. Last resort before character-level.
5. `""` — raw character split. Absolute fallback (never preferred).

**chunk_overlap=200:** The last 200 characters of chunk N are repeated at the start of chunk N+1. This prevents a concept that spans a boundary (e.g., a theorem statement on one chunk and its proof beginning on the next) from losing context.

---

## 6. Module: embedding_utils.py

**File:** [src/embedding_utils.py](src/embedding_utils.py)

### Constants

```python
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "rag_academic_papers"
VECTOR_DB_DIR = str(BASE_DIR / "data" / "vector_db")
```

### `get_embedding_model()`

```python
def get_embedding_model() -> HuggingFaceEmbeddings
```

Initializes the BGE embedding model with:

```python
HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

- `device="cpu"` — runs entirely on CPU, no GPU required.
- `normalize_embeddings=True` — forces all output vectors to unit length (L2 norm = 1). When vectors are normalized, cosine similarity equals dot product, which is faster to compute and numerically more stable.

The model produces **512-dimensional** dense vectors. On first call it downloads ~130MB of model weights from HuggingFace Hub into `~/.cache/huggingface/`.

---

### `create_vector_db()`

```python
def create_vector_db(
    chunks: list,
    persist_directory: str = VECTOR_DB_DIR
) -> Chroma
```

Calls `Chroma.from_documents()` which:
1. Iterates over every chunk.
2. Calls `get_embedding_model().embed_documents([chunk.page_content])` for each one.
3. Stores the (vector, text, metadata) triple in the ChromaDB collection.
4. Writes the collection to disk at `persist_directory`.

The collection is named `"rag_academic_papers"`. ChromaDB uses this name to separate multiple collections within the same persist directory.

---

### `load_vector_db()`

```python
def load_vector_db(persist_directory: str = VECTOR_DB_DIR) -> Chroma
```

Loads an **existing** Chroma collection from disk without re-embedding anything. Used by `rag_pipeline.py` at startup.

---

## 7. Module: rag_pipeline.py

**File:** [src/rag_pipeline.py](src/rag_pipeline.py)

This is the core of the RAG system. It connects the vector database (retrieval) to the language model (generation) using **LangChain Expression Language (LCEL)**.

### The Prompt Template

```python
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert academic research assistant. Use the following context
from academic papers to answer the question. If you cannot find the answer
in the context, say so clearly.

Context:
{context}

Question: {question}

Provide a detailed, accurate answer based on the context above.
Cite the source papers when possible.
""")
```

**Two variables are injected at runtime:**
- `{context}` — the 4 retrieved chunks joined with `"\n\n---\n\n"` separators.
- `{question}` — the raw user query string.

The instruction *"say so clearly if you cannot find the answer"* is an anti-hallucination directive — it instructs the LLM to admit uncertainty rather than fabricate information from its training data.

---

### `get_llm()`

```python
def get_llm()
```

Priority-based LLM detection:

1. **OpenAI** — checks `os.getenv("OPENAI_API_KEY")`. If set (via `.env` file), returns `ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)`.

2. **Ollama** — makes a real HTTP GET to `http://localhost:11434/api/tags` with a 2-second timeout. Only if `status_code == 200` (Ollama is actually running) does it return `Ollama(model="llama3", temperature=0.2)`.

3. **RuntimeError** — raised if neither backend is available. The app catches this and enters retrieval-only mode rather than crashing.

**Why `temperature=0.2`:** Temperature controls the randomness of token sampling. At 0.0 the model is fully deterministic; at 1.0 it is highly creative. For a factual Q&A system over academic papers, 0.2 provides near-deterministic, accurate answers while allowing slight variation in phrasing.

---

### `build_rag_chain()`

```python
def build_rag_chain(vector_store: Chroma = None) -> tuple[chain, retriever]
```

Constructs the LCEL pipeline:

```python
chain = (
    {
        "context": retriever | _format_docs,
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
```

**How LCEL works here:**

The input to the chain is a plain string (the user query). The `{}` dict at the start is a `RunnableParallel` — it forks execution into two paths simultaneously:

- `"context"`: The query goes through `retriever` (returns `list[Document]`), then through `_format_docs()` (joins them into a single string).
- `"question"`: The query passes through `RunnablePassthrough()` unchanged.

Both outputs are collected into `{"context": "...", "question": "..."}` and passed to `RAG_PROMPT`, which fills the template. The filled prompt goes to `llm`, and `StrOutputParser()` extracts the plain string from the response object.

---

### `_format_docs()`

```python
def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
```

Joins the 4 retrieved chunks with a visible `---` separator. This helps the LLM distinguish where one source ends and another begins within the context window.

---

### `ask_question()`

```python
def ask_question(query: str, chain=None, retriever=None) -> dict
```

The public-facing function called by the Streamlit app. It:

1. Calls `chain.invoke(query)` to get the generated answer string.
2. Calls `retriever.invoke(query)` **a second time** to get the raw `Document` objects (the LCEL chain's internal retrieval doesn't expose them).
3. Deduplicates sources by `"{filename}:p{page}"` key.
4. Returns `{"answer": str, "sources": list[dict]}`.

**Why two retriever calls?** The LCEL chain pipes retrieved docs directly into `_format_docs()`, discarding the `Document` objects. Calling the retriever again outside the chain is the clean way to get the metadata. Both calls hit the same in-memory Chroma index so performance impact is negligible.

---

## 8. Module: app.py

**File:** [app.py](app.py)

### Python Path Injection

```python
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
```

This adds `src/` to the Python module search path before any imports, allowing `from rag_pipeline import ...` to resolve even though `app.py` is in the project root.

---

### `@st.cache_resource` — Startup Caching

```python
@st.cache_resource(show_spinner="Initializing RAG system...")
def init_rag():
    vs = load_vector_store()
    try:
        chain, retriever = build_rag_chain(vs)
        return vs, chain, retriever, True
    except RuntimeError:
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        return vs, None, retriever, False
```

`@st.cache_resource` executes `init_rag()` exactly once per process lifetime and caches the return value in memory. Streamlit reruns the script on every user interaction; without this decorator, the embedding model (~130MB) and vector database would reload on every button click or message.

The function returns the `vector_store` object itself so it can be mutated (new documents added) by the upload feature without requiring a cache invalidation.

---

### `ingest_uploaded_pdf()`

```python
def ingest_uploaded_pdf(pdf_path: Path) -> int:
    loader = PyMuPDFLoader(str(pdf_path))
    pages = loader.load()
    chunks = chunk_documents(pages)
    vector_store.add_documents(chunks)
    return len(chunks)
```

`vector_store.add_documents()` is ChromaDB's incremental ingestion API. It:
1. Embeds each new chunk using the same `BAAI/bge-small-en-v1.5` model.
2. Appends the new vectors to the existing collection.
3. Writes the changes to disk immediately.

Because the `retriever` and `chain` hold a reference to the same `vector_store` object in memory, new documents are immediately available for retrieval without any restart.

---

### Dual Operating Modes

| Mode | Trigger | Behaviour |
|---|---|---|
| **Full RAG** | LLM backend detected | `ask_question()` → retrieval + LLM generation → answer + sources |
| **Retrieval-only** | No LLM available | `retriever.invoke()` → 4 raw chunks shown as styled blockquotes |

The app detects LLM availability once at startup (inside `init_rag`) and stores the boolean `llm_available`. Every render checks this flag to decide which mode to run.

---

### Session State & Chat History

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

Streamlit re-executes the entire script from top to bottom on every user interaction. `st.session_state` is a dictionary that persists across reruns within the same browser session. Without it, the chat history would be wiped after every message.

Each message is stored as `{"role": "user"|"assistant", "content": str}`.

---

## 9. Data Flow: Offline Indexing

This is the setup phase, run once before the app starts.

```
python src/document_processor.py
    │
    ├── arxiv.Client().results(search) × 4 queries
    │       └── result.download_pdf() → data/raw_pdfs/{id}_{title}.pdf
    │
    ├── PyMuPDFLoader(path).load()
    │       └── returns list[Document], one per page
    │           metadata = {source, page, title, author, total_pages, ...}
    │
    └── RecursiveCharacterTextSplitter(1000, 200).split_documents(pages)
            └── returns list[Document] (chunks), inheriting parent metadata

python src/embedding_utils.py
    │
    ├── HuggingFaceEmbeddings("BAAI/bge-small-en-v1.5")
    │       └── downloads model to ~/.cache/huggingface/ (first run only)
    │
    └── Chroma.from_documents(chunks, embeddings, persist_dir)
            ├── embed each chunk.page_content → 512-dim float32 vector
            ├── store (vector, text, metadata) in collection "rag_academic_papers"
            └── persist to data/vector_db/ (SQLite + binary files)
```

---

## 10. Data Flow: Online Querying

This runs on every user message in the Streamlit app.

```
user types: "What is the Poincaré inequality?"
    │
    ▼
chain.invoke("What is the Poincaré inequality?")
    │
    ├── retriever branch:
    │   ├── embed query → 512-dim vector
    │   ├── ChromaDB cosine similarity search (k=4)
    │   │   └── returns 4 Document objects (closest vectors)
    │   └── _format_docs() → joined string of 4 chunk texts
    │
    └── passthrough branch:
        └── "What is the Poincaré inequality?" (unchanged)
    │
    ▼
RAG_PROMPT.format(context="...", question="...")
    │
    ▼
ChatOpenAI("gpt-3.5-turbo") OR Ollama("llama3")
    │
    ▼
StrOutputParser() → plain answer string
    │
    ▼
retriever.invoke(query) [second call, for metadata only]
    └── returns same 4 Documents → extract title + page
    │
    ▼
ask_question() returns:
{
  "answer": "The Poincaré inequality states...",
  "sources": [
    {"title": "Uniformly Bounded Cochain Extensions...", "page": 4, "file": "...pdf"},
    {"title": "Uniformly Bounded Cochain Extensions...", "page": 22, "file": "...pdf"}
  ]
}
    │
    ▼
Streamlit renders answer text + source pill badges
```

---

## 11. Data Flow: User PDF Upload

```
user drops PDF file onto the sidebar uploader
    │
    ▼
uploaded_file.getvalue() → raw bytes
    │
    ▼
dest = RAW_PDFS_DIR / uploaded_file.name
dest.exists()? → "Already indexed" warning (no duplicate processing)
    │
    ▼
dest.write_bytes(raw_bytes)  ← saved to data/raw_pdfs/
    │
    ▼
ingest_uploaded_pdf(dest):
    ├── PyMuPDFLoader(dest).load() → list[Document] (one per page)
    ├── chunk_documents(pages, 1000, 200) → list[Document] (chunks)
    └── vector_store.add_documents(chunks)
        ├── embed each chunk → 512-dim vector
        ├── append to collection "rag_academic_papers"
        └── persist to data/vector_db/
    │
    ▼
st.rerun() → page refreshes
    ├── sidebar paper list re-reads data/raw_pdfs/ → new file appears
    └── live_chunk_count re-queries vector_store._collection.count()
```

The new document is immediately queryable — no restart needed.

---

## 12. Key Technical Decisions

### Why RAG over fine-tuning?

Fine-tuning an LLM on your documents bakes the knowledge into the model weights — expensive ($$$), slow to update, and impossible to audit. RAG keeps the knowledge external and retrievable, making it cheap to add new papers, fully traceable (sources are cited), and accurate on content the base model never saw.

### Why PyMuPDF over PyPDF2 / pdfminer?

| Feature | PyMuPDF | PyPDF2 | pdfminer |
|---|---|---|---|
| Multi-column reading order | Correct | Often scrambled | Sometimes wrong |
| Unicode math symbols | Preserved | Broken | Partial |
| Embedded metadata | Full extraction | Partial | Partial |
| Speed | Fast | Slow | Slow |
| Error tolerance | High | Low | Medium |

Academic papers use multi-column layout, heavy LaTeX math, and custom fonts — exactly where PyMuPDF excels.

### Why RecursiveCharacterTextSplitter over fixed-size splitting?

Fixed-size splitting (e.g., every 1000 characters) blindly cuts text mid-sentence or mid-equation. Recursive splitting tries semantic boundaries (paragraphs → sentences → words) before falling back to characters. This keeps related ideas together within each chunk, which directly improves retrieval quality.

### Why chunk_size=1000, chunk_overlap=200?

- **1000 characters ≈ 150–200 words** — enough to contain a complete mathematical statement with context, but small enough that retrieval precision stays high (a very large chunk would match many queries loosely).
- **200 character overlap** — prevents loss of context at boundaries. If a theorem spans two chunks, both chunks contain the bridge text that links them.

### Why BAAI/bge-small-en-v1.5?

- Ranks in the **top tier** of the MTEB retrieval benchmark for its model size class.
- **CPU-only** — no GPU required. Other high-quality models (e.g., `text-embedding-3-large`) require API calls or GPU.
- **Normalized embeddings** — all vectors have L2 norm = 1, making cosine similarity equivalent to dot product (faster lookup in ChromaDB).
- **512 dimensions** — good balance of expressiveness vs. storage and search speed.

### Why ChromaDB over Pinecone / Weaviate / FAISS?

- **Zero infrastructure** — runs as an embedded library, no server, no Docker, no cloud account.
- **Disk persistence** — survives app restarts, unlike FAISS (in-memory only).
- **Incremental writes** — `add_documents()` appends to an existing collection, enabling the live PDF upload feature.
- **Metadata filtering** — supports filtering by author, page, title, etc. (not yet used, but available).

### Why LCEL over the deprecated RetrievalQA chain?

LangChain v0.3+ deprecated the `RetrievalQA` class in favour of LCEL. LCEL is composable (each `|` is a pipe), readable, type-safe, and supports streaming. The explicit `{context: retriever | _format_docs, question: RunnablePassthrough()}` structure makes it clear exactly what data flows into the prompt, making debugging straightforward.

---

## 13. LangChain Internals

### What is a Document?

```python
class Document:
    page_content: str    # the text of this chunk/page
    metadata: dict       # arbitrary key-value pairs (source, page, title, ...)
```

All LangChain components (loaders, splitters, retrievers, vector stores) exchange `Document` objects. Metadata is preserved through the entire pipeline from loading through to the final source citations.

### What is a Runnable?

LCEL components implement the `Runnable` interface with an `.invoke(input)` method. The `|` operator chains them: `A | B` means "pipe A's output as B's input". This makes the pipeline explicit and debuggable.

### What is a Retriever?

A retriever wraps a vector store and exposes `.invoke(query: str) -> list[Document]`. Internally it embeds the query and runs similarity search. The `search_kwargs={"k": 4}` tells it to return 4 results.

---

## 14. Vector Database Internals

ChromaDB stores data in `data/vector_db/` as:
- A **SQLite** file (`chroma.sqlite3`) — stores metadata, document text, and collection configuration.
- **Binary data files** — store the raw float32 vectors using an HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.

### HNSW Search

HNSW is a graph-based ANN algorithm. Each vector is a node; nodes are connected to their nearest neighbors in a layered graph. Search traverses the graph starting from an entry point, greedily moving to nodes closer to the query vector. This gives **O(log N)** average search time vs. **O(N)** for brute-force cosine scan, with minimal accuracy loss.

### Cosine Similarity

The similarity score between query vector **q** and document vector **d** is:

```
cosine_similarity(q, d) = (q · d) / (||q|| × ||d||)
```

Since both are normalized (||q|| = ||d|| = 1), this simplifies to the dot product `q · d`. Results are ranked by this score; the top 4 are returned.

---

## 15. Embedding Model Internals

`BAAI/bge-small-en-v1.5` is a transformer-based sentence embedding model from the Beijing Academy of Artificial Intelligence. This section explains exactly how it turns a string of text into a 512-dimensional vector, from first principles.

---

### Step 1 — Tokenization

Before any neural processing, raw text is broken into **tokens** using a WordPiece tokenizer (the same vocabulary as BERT, ~30,000 tokens).

```
Input:  "What is the Poincaré inequality?"
Tokens: [CLS] what is the poin ##car ##é inequality ? [SEP]
IDs:    [ 101] 2054 2003 1996 13366 7488 2726 26060 1029  [102]
```

Key details:
- `[CLS]` (ID 101) is a special token prepended to every sequence. Its final hidden state is used by classification tasks, but for sentence embeddings we use **mean pooling** instead (explained below).
- `[SEP]` (ID 102) marks the end of the sequence.
- Rare or unknown words are split into subword pieces: `Poincaré` → `poin`, `##car`, `##é`. The `##` prefix means "this piece continues the previous token without a space."
- Maximum sequence length is **512 tokens**. Longer inputs are truncated.

Each token ID is looked up in an **embedding table** (a 30,000 × 768 matrix of learned floats), yielding one 768-dim vector per token. Positional encodings are added to these so the model knows token order.

---

### Step 2 — Transformer Encoder (Self-Attention)

The BGE-small model has **6 transformer encoder layers**. Each layer applies the same operation to the sequence of token vectors, progressively building richer contextual representations.

#### Self-Attention Mechanism

Each token "attends" to every other token in the sequence. For a sequence of N tokens, each layer computes:

```
For each token i:
    Query vector:  Q_i = W_Q × h_i       (what am I looking for?)
    Key vector:    K_j = W_K × h_j       (what does token j offer?)
    Value vector:  V_j = W_V × h_j       (what information does j carry?)

Attention score: score(i,j) = (Q_i · K_j) / sqrt(d_k)
Attention weight: α(i,j) = softmax(score(i,j))   over all j

New representation: h_i' = Σ_j α(i,j) × V_j
```

Where `W_Q`, `W_K`, `W_V` are learned weight matrices and `d_k` is the key dimension (divided to prevent vanishing gradients).

In plain terms: each token's new representation is a **weighted average of all other tokens' values**, where the weights are learned relevance scores. The word "inequality" will attend strongly to "Poincaré" because that co-occurrence is meaningful.

BGE-small uses **8 attention heads** — 8 parallel attention computations with different learned weight matrices, capturing different types of relationships simultaneously (syntactic, semantic, positional, etc.).

#### Why "Bidirectional"?

Unlike GPT (which only attends to past tokens), BERT-based models attend to **both left and right context simultaneously**. The word "bank" in "river bank" sees both "river" (left) and everything after (right) to resolve its meaning. This bidirectionality is why BERT-style models produce better embeddings than decoder-only models for retrieval.

#### Feed-Forward Network

After attention, each token's representation passes through a position-wise feed-forward network:

```
FFN(h) = ReLU(h × W_1 + b_1) × W_2 + b_2
```

This adds non-linear transformation capacity per token. Layer norm and residual connections wrap both the attention and FFN steps for training stability.

After 6 such layers, each input token has a rich, contextually-aware 768-dim representation that encodes its meaning relative to every other token in the sequence.

---

### Step 3 — Mean Pooling

After the 6 transformer layers, we have one 768-dim vector per token. We need a **single vector** to represent the entire input. BGE uses **mean pooling**:

```
sentence_vector = mean(h_1, h_2, ..., h_N)   [averaged across all N tokens]
```

This averages all token representations into one 768-dim vector. It is slightly better than using just the `[CLS]` token (which was the original BERT approach) because it incorporates information from every token rather than relying on a single position to summarize the whole input.

---

### Step 4 — Linear Projection to 512 Dimensions

BGE-small adds a learned linear projection layer after pooling:

```
embedding = W_proj × sentence_vector   [768-dim → 512-dim]
```

This reduces dimensionality for storage and search efficiency while retaining the most informative dimensions.

---

### Step 5 — L2 Normalization

Finally, the vector is normalized to unit length:

```
embedding_normalized = embedding / ||embedding||_2
```

Where `||v||_2 = sqrt(v_1² + v_2² + ... + v_512²)`.

After normalization, all vectors lie on the surface of a 512-dimensional unit hypersphere. The cosine similarity between any two such vectors equals their dot product — which is the fastest similarity computation available on modern hardware.

---

### Full Inference Pipeline Summary

```
"What is the Poincaré inequality?"
        │
        ▼
WordPiece Tokenizer
[CLS] what is the poin ##car ##é inequality ? [SEP]
        │
        ▼
Token Embedding Lookup (30k vocab × 768)
+ Positional Encodings
        │
        ▼
Transformer Layer 1: Self-Attention (8 heads) + FFN
        │
       ...  (×6 layers total)
        │
        ▼
Transformer Layer 6: contextual token representations
[768-dim vector per token]
        │
        ▼
Mean Pooling across all tokens
[single 768-dim vector]
        │
        ▼
Linear Projection (768 → 512)
        │
        ▼
L2 Normalization
        │
        ▼
[512-dim float32 unit vector]  ← stored in / searched against ChromaDB
```

---

### How the Model Learned: Contrastive Training

The BGE model was not trained on a classification or generation task. It was trained using **contrastive learning** on a massive dataset of (query, positive passage, hard negative passage) triplets.

#### The Training Objective

For each training triplet `(q, p+, p-)`:
- `q` = a natural language question
- `p+` = a passage that correctly answers it (positive)
- `p-` = a passage that looks plausible but does not answer it (hard negative)

The model is trained to minimize the **InfoNCE loss** (also called NT-Xent loss):

```
L = -log( exp(sim(q, p+) / τ) / Σ_j exp(sim(q, p_j) / τ) )
```

Where:
- `sim(a, b) = a · b` (dot product of normalized vectors = cosine similarity)
- `τ` (tau) is a temperature hyperparameter controlling the sharpness of the distribution
- The sum in the denominator runs over the positive and all negatives in the batch

Minimizing this loss **pulls `q` and `p+` together** (increases their cosine similarity) and **pushes `q` and `p-` apart** (decreases their cosine similarity).

#### Why Hard Negatives Matter

"Easy negatives" (completely unrelated passages) are trivially separable even by a random model. Hard negatives — passages that contain related vocabulary but don't actually answer the query — force the model to learn deep semantic understanding rather than surface keyword overlap.

Example of a hard negative for query "What is the Poincaré inequality?":
- ✅ Positive: *"The Poincaré inequality states: for a domain Ω, ∫|∇u|² dx ≥ C∫|u|² dx..."*
- ❌ Hard negative: *"Henri Poincaré was a French mathematician who made contributions to topology..."*

The hard negative mentions Poincaré but answers "who was Poincaré?", not "what is the inequality?". Training on these forces the model to understand the difference.

---

### The Result: Semantic Search

After contrastive training, the embedding space has a remarkable property: **semantic similarity corresponds to geometric proximity**, regardless of vocabulary overlap.

```
sim( embed("What is the Poincaré inequality?"),
     embed("The Poincaré inequality states ∫|∇u|² ≥ C∫|u|²...") )
≈ 0.85   ← high similarity, correct retrieval

sim( embed("What is the Poincaré inequality?"),
     embed("PointTPA uses dynamic parameter adaptation for 3D scenes") )
≈ 0.12   ← low similarity, correctly rejected
```

This is fundamentally different from keyword search (BM25, TF-IDF), which would score "PointTPA" as irrelevant because it shares no words, but would struggle to distinguish "Henri Poincaré the mathematician" from "the Poincaré inequality" since both share the keyword "Poincaré".

---

### Model Size vs. Quality Trade-off

The BGE family spans a range of sizes. This project uses `bge-small`:

| Model | Parameters | Embedding Dim | MTEB Score | RAM | Speed |
|---|---|---|---|---|---|
| `bge-small-en-v1.5` | 33M | 512 | 62.x | ~250 MB | Fast |
| `bge-base-en-v1.5` | 110M | 768 | 64.x | ~440 MB | Medium |
| `bge-large-en-v1.5` | 335M | 1024 | 65.x | ~1.3 GB | Slow |

The quality difference between small and large is modest (~3 MTEB points). For a local CPU-only setup querying a few hundred documents, `bge-small` provides the best trade-off of speed and accuracy.

---

## 16. LLM Backend Configuration

### Option A: OpenAI

Create `rag_complex_docs/.env`:
```
OPENAI_API_KEY=sk-your-key-here
```

The `python-dotenv` library loads this at startup via `load_dotenv()`. `ChatOpenAI` sends the filled prompt to OpenAI's API and returns the response. Costs approximately $0.001–0.002 per query with `gpt-3.5-turbo`.

### Option B: Ollama (local, free)

```bash
ollama pull llama3      # download the model (~4GB)
ollama serve            # start the local API server on port 11434
```

The pipeline checks `http://localhost:11434/api/tags` before attempting to use Ollama. This prevents a `ConnectionRefusedError` if Ollama isn't running.

### Adding a new LLM backend

Edit `get_llm()` in [src/rag_pipeline.py](src/rag_pipeline.py):

```python
# Example: Anthropic Claude
from langchain_anthropic import ChatAnthropic
return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2)
```

Any LangChain-compatible LLM can be dropped in — the rest of the chain is backend-agnostic.

---

## 17. Chunking Strategy Deep Dive

Given an input page of 3000 characters with `chunk_size=1000, chunk_overlap=200`:

```
Original page text (3000 chars):
┌────────────────────────────────────────────────────┐
│  Paragraph A (600 chars)                           │
│                                                    │
│  Paragraph B (800 chars)                           │
│                                                    │
│  Paragraph C (400 chars)                           │
│                                                    │
│  Paragraph D (1200 chars)                          │
└────────────────────────────────────────────────────┘

After splitting:

Chunk 1 (≤1000 chars): Paragraph A + Paragraph B (fits)
         │
         └── overlap: last 200 chars of chunk 1
                           │
Chunk 2 (≤1000 chars): [200 overlap chars] + Paragraph C + start of D
         │
         └── overlap: last 200 chars of chunk 2
                           │
Chunk 3 (≤1000 chars): [200 overlap chars] + rest of Paragraph D
```

The splitter tries `"\n\n"` first. If a paragraph exceeds 1000 chars (Paragraph D), it recurses and tries `"\n"`, then `". "`, etc.

---

## 18. Metadata Schema

Each `Document` object carries a `metadata` dict. The schema as populated by `PyMuPDFLoader`:

| Key | Type | Example | Source |
|---|---|---|---|
| `source` | `str` | `C:\...\2604.04927v1_Uniformly...pdf` | Full path to the PDF file |
| `file_path` | `str` | same as `source` | Duplicate of source |
| `page` | `int` | `3` | 0-indexed page number |
| `total_pages` | `int` | `27` | Total pages in the document |
| `title` | `str` | `"Uniformly Bounded Cochain..."` | From PDF metadata |
| `author` | `str` | `"Jane Doe; John Smith"` | From PDF metadata |
| `subject` | `str` | `""` | From PDF metadata (often empty) |
| `keywords` | `str` | `"topology; Hodge theory"` | From PDF metadata |
| `producer` | `str` | `"pikepdf 8.15.1"` | PDF generation software |
| `creator` | `str` | `"arXiv GenPDF"` | PDF creation tool |
| `creationdate` | `str` | `"2026-04-07T02:01:33+00:00"` | ISO 8601 timestamp |
| `format` | `str` | `"PDF 1.7"` | PDF specification version |

This metadata is preserved through chunking and stored alongside each vector in ChromaDB, enabling the source citation feature.

---

## 19. Failure Modes & Mitigations

| Failure | Cause | Mitigation in code |
|---|---|---|
| arxiv download fails | Network error or rate limit | Per-paper `try/except`; other papers continue |
| PDF parse error | Corrupt PDF or unsupported format | Per-file `try/except` in `load_pdfs()` |
| `cmsOpenProfileFromMem failed` | Embedded ICC color profile in PDF | MuPDF warning only; text extraction still succeeds |
| No LLM backend | OpenAI key missing and Ollama not running | `RuntimeError` caught in `init_rag()`; retrieval-only mode activated |
| Ollama detected but not running | Process cached but crashed | HTTP ping to `:11434/api/tags` with 2s timeout before initializing |
| Duplicate PDF upload | Same filename uploaded twice | `dest.exists()` check before writing; shows "Already indexed" message |
| Upload ingestion fails | Corrupt PDF, disk full, etc. | `try/except` around `ingest_uploaded_pdf()`; uploaded file deleted on failure |
| Windows unicode console error | `cp1252` codec can't encode math chars | `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` |

---

## 20. Performance Characteristics

### Offline Indexing (one-time setup)

| Step | Typical Duration | Bottleneck |
|---|---|---|
| arxiv download (5 papers) | 30–120 seconds | Network bandwidth |
| PyMuPDF parsing (131 pages) | < 2 seconds | Disk I/O |
| Chunking (629 chunks) | < 1 second | CPU, single-threaded |
| BGE embedding (629 chunks) | 60–180 seconds (CPU) | Neural network inference |
| ChromaDB write | < 5 seconds | Disk I/O |

Embedding is the dominant cost. On CPU, `BAAI/bge-small-en-v1.5` processes roughly 5–10 chunks per second. With a GPU (CUDA), this drops to < 1 second total.

---

### Online Querying (per user message)

| Step | Typical Duration | Notes |
|---|---|---|
| Query embedding | 0.05–0.2 seconds (CPU) | Single vector, very fast |
| ChromaDB similarity search | < 0.05 seconds | HNSW index, O(log N) |
| LLM generation (OpenAI) | 1–5 seconds | Network RTT + token generation |
| LLM generation (Ollama/llama3) | 5–30 seconds | Depends on CPU/GPU speed |
| Second retriever call (metadata) | < 0.05 seconds | Same index, cached |

Total perceived latency for a user is dominated entirely by the LLM. The retrieval step (vector search + embedding the query) is under 300ms on CPU.

---

### Streamlit Startup

| Step | Duration |
|---|---|
| Python import + `sys.path` setup | 1–3 seconds |
| TensorFlow (if installed) load warning | 5–20 seconds (first run) |
| BGE model load into memory | 2–5 seconds (first run; cached after) |
| ChromaDB open from disk | < 1 second |
| Ollama health ping | ≤ 2 seconds (timeout) |

`@st.cache_resource` ensures the slow steps (model load, DB open) happen only once. All subsequent rerenders are near-instant.

---

## 21. Memory & Disk Usage

### Disk

| Location | Contents | Typical Size |
|---|---|---|
| `data/raw_pdfs/` | 5 arxiv PDFs | 3–15 MB |
| `data/vector_db/` | ChromaDB SQLite + HNSW binary | ~15–30 MB for 629 vectors |
| `~/.cache/huggingface/` | BGE model weights | ~130 MB (downloaded once) |
| `~/.ollama/models/` | Ollama llama3 weights | ~4 GB |

ChromaDB storage scales linearly: each 512-dim float32 vector takes 2 KB. 629 vectors ≈ 1.3 MB of raw vector data, plus SQLite overhead for text and metadata.

### RAM (at runtime)

| Component | Memory Usage |
|---|---|
| BGE embedding model | ~250 MB |
| ChromaDB collection (629 vectors in memory) | ~10 MB |
| Streamlit + LangChain Python objects | ~100 MB |
| Ollama (llama3, 4-bit quantized) | ~4–6 GB |
| **Total without Ollama** | ~360 MB |
| **Total with Ollama** | ~4.5–6.5 GB |

---

## 22. Configuration Reference

All tuneable parameters and where they live:

### document_processor.py

| Constant / Parameter | Default | Effect |
|---|---|---|
| `SEARCH_QUERIES` | 4 topics | Topics queried on arxiv |
| `MAX_RESULTS_PER_QUERY` | `2` | Papers downloaded per query |
| `chunk_size` | `1000` | Max characters per chunk |
| `chunk_overlap` | `200` | Shared characters between adjacent chunks |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Split priority order |

### embedding_utils.py

| Constant | Default | Effect |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `"BAAI/bge-small-en-v1.5"` | HuggingFace model identifier |
| `COLLECTION_NAME` | `"rag_academic_papers"` | ChromaDB collection name |
| `VECTOR_DB_DIR` | `data/vector_db/` | ChromaDB persist path |
| `device` | `"cpu"` | `"cuda"` for GPU acceleration |
| `normalize_embeddings` | `True` | Enables dot-product cosine similarity |

### rag_pipeline.py

| Parameter | Default | Effect |
|---|---|---|
| `k` in `get_retriever()` | `4` | Number of chunks retrieved per query |
| `search_type` | `"similarity"` | Can also be `"mmr"` (maximal marginal relevance) |
| `temperature` | `0.2` | LLM output randomness (0 = deterministic, 1 = creative) |
| LLM model (OpenAI) | `"gpt-3.5-turbo"` | Swap to `"gpt-4"` for higher quality |
| LLM model (Ollama) | `"llama3"` | Any locally pulled model name |

### .env file (project root)

```env
OPENAI_API_KEY=sk-...          # Enables OpenAI backend
```

---

## 23. Scalability & Limitations

### Current Limits

| Dimension | Current Capacity | Bottleneck |
|---|---|---|
| Documents | ~50–200 PDFs | ChromaDB HNSW stays fast up to ~1M vectors |
| Chunks | Up to ~100,000 comfortably | RAM for the in-memory HNSW index |
| Concurrent users | 1 (Streamlit default) | Single-threaded Python process |
| PDF size | Any (tested up to ~300 pages) | Memory during parsing |
| Query latency | Depends on LLM | Retrieval is always < 300ms |

### ChromaDB Scaling

ChromaDB uses HNSW which scales to millions of vectors on a single machine. The practical limit of this project is RAM: each 512-dim float32 vector takes 2 KB. 100,000 vectors = ~200 MB of RAM for the index, well within typical machine limits.

For **production scale** (millions of documents, many concurrent users), the natural upgrade path is:
- Replace ChromaDB with a dedicated vector database server (Pinecone, Weaviate, Qdrant).
- Run the Streamlit app behind a WSGI server (Gunicorn) or rewrite the frontend in FastAPI.
- Move embedding computation to a GPU worker process.

### Retrieval Quality Limits

The system retrieves `k=4` chunks regardless of whether they are truly relevant. If a query has no related content in the corpus, the LLM receives 4 weakly related chunks and should respond "I cannot find this in the context." In practice, adding a **similarity score threshold** (only return chunks with cosine similarity > 0.7) would improve answer quality for out-of-domain queries.

### LLM Context Window

The 4 retrieved chunks (at most 4 × 1000 = 4000 characters ≈ ~1000 tokens) plus the prompt template fit comfortably within `gpt-3.5-turbo`'s 16,384-token context window and `llama3`'s 8,192-token window. If you increase `k` beyond ~12 or `chunk_size` beyond ~3000, you risk hitting context limits.

---

## 24. How to Extend the System

### Add a new data source

Any document that can be converted to `list[Document]` works. LangChain provides loaders for:

```python
# Web pages
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://arxiv.org/abs/2301.00001")

# Word documents
from langchain_community.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("paper.docx")

# Plain text
from langchain_community.document_loaders import TextLoader
loader = TextLoader("notes.txt")
```

Pass the resulting documents through `chunk_documents()` and then `vector_store.add_documents()`. Everything downstream (retrieval, LLM, UI) works without any other changes.

---

### Add a new LLM

Edit `get_llm()` in [src/rag_pipeline.py](src/rag_pipeline.py). Any class implementing LangChain's `BaseChatModel` or `BaseLLM` interface works:

```python
# Anthropic Claude
from langchain_anthropic import ChatAnthropic
return ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2)

# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

# Azure OpenAI
from langchain_openai import AzureChatOpenAI
return AzureChatOpenAI(azure_deployment="gpt-4", temperature=0.2)

# Groq (fast inference)
from langchain_groq import ChatGroq
return ChatGroq(model="llama3-70b-8192", temperature=0.2)
```

---

### Change the retrieval strategy

In `get_retriever()` in [src/rag_pipeline.py](src/rag_pipeline.py):

```python
# Default: pure similarity search
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# Alternative: Maximal Marginal Relevance (MMR)
# Balances relevance with diversity — avoids returning 4 near-identical chunks
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)

# Alternative: Similarity with score threshold
# Only returns chunks above a minimum relevance score
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 4},
)
```

---

### Change the embedding model

Edit `EMBEDDING_MODEL_NAME` in [src/embedding_utils.py](src/embedding_utils.py):

```python
# Larger, more accurate (requires more RAM & time)
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"   # 1024-dim, ~1.3GB

# Multilingual (for non-English documents)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# OpenAI embeddings (API-based, very high quality)
from langchain_openai import OpenAIEmbeddings
return OpenAIEmbeddings(model="text-embedding-3-small")
```

> **Important:** If you change the embedding model, you must **rebuild the vector database** from scratch. Vectors from different models live in incompatible spaces and cannot be mixed.

---

### Enable streaming responses

Replace `StrOutputParser()` in `build_rag_chain()` with Streamlit's `st.write_stream()`:

```python
# In app.py, replace:
result = ask_question(prompt, chain=chain, retriever=retriever)
st.markdown(result["answer"])

# With streaming:
with st.chat_message("assistant"):
    answer = st.write_stream(chain.stream(prompt))
```

LCEL chains are streaming-compatible by default — `.stream()` yields tokens as they are generated.

---

### Add metadata filtering

ChromaDB supports filtering retrievals by metadata fields. For example, to only search within a specific paper:

```python
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"title": "Uniformly Bounded Cochain Extensions and Uniform Poincaré Inequalities"}
    }
)
```

This enables per-document Q&A without modifying the pipeline.

---

## 25. Security Considerations

### API Key Management

The `OPENAI_API_KEY` is loaded from a `.env` file via `python-dotenv`. This file should:
- **Never** be committed to version control. Add `.env` to `.gitignore`.
- Have restricted file permissions on shared machines (`chmod 600 .env` on Unix).
- Be rotated regularly if the project is shared or deployed.

### PDF Upload Security

The file uploader in `app.py` accepts only `.pdf` files (enforced by `type=["pdf"]` in `st.file_uploader`). However, this is a client-side MIME-type hint, not a server-side validation. The actual parsing is done by PyMuPDF which will fail gracefully on non-PDF content — the `try/except` in `ingest_uploaded_pdf()` catches parse errors and deletes the malformed file.

For a production deployment, add explicit server-side validation:

```python
import magic  # pip install python-magic
mime = magic.from_buffer(uploaded_file.getvalue(256), mime=True)
if mime != "application/pdf":
    raise ValueError(f"Rejected: expected PDF, got {mime}")
```

### Prompt Injection

Academic papers could theoretically contain text designed to manipulate the LLM (e.g., "Ignore all previous instructions and..."). The RAG prompt template wraps retrieved content in a clearly labelled `Context:` block, which reduces but does not eliminate this risk. For high-security deployments, retrieved content should be sanitized before injection into the prompt.

### Local Ollama

Ollama runs as a local HTTP server on `localhost:11434` with no authentication. If the machine is on a shared network, this port should be firewalled to prevent other users from querying the model or reading its logs.

---

## 26. RAG Quality & Evaluation

### What makes RAG answers good?

Answer quality depends on three factors in sequence:

1. **Retrieval precision** — did the right chunks come back? If the retrieved chunks are irrelevant, the LLM has nothing useful to work with.
2. **Context coverage** — do the retrieved chunks contain enough information to answer the question? A question may require information spread across chunks that didn't all rank in the top 4.
3. **LLM faithfulness** — did the LLM answer strictly from the context, or did it mix in training data? Lower temperature and the "say so clearly" instruction in the prompt help.

### How to evaluate retrieval quality

Run known (query, expected source document) pairs and check if the correct document appears in the top-4 results:

```python
# Quick retrieval sanity check
from src.rag_pipeline import load_vector_store

db = load_vector_store()
results = db.similarity_search("Poincaré inequality for convex domains", k=4)
titles = [r.metadata["title"] for r in results]
assert "Uniformly Bounded Cochain Extensions and Uniform Poincaré Inequalities" in titles
```

### How to evaluate answer faithfulness

Check whether the answer text can be traced back to the retrieved chunks:
- Every factual claim in the answer should appear (possibly paraphrased) in at least one retrieved chunk.
- If the LLM cites a paper or author not in the retrieved chunks, it is hallucinating.

### Improving retrieval quality

| Technique | How | Impact |
|---|---|---|
| Increase `k` | `search_kwargs={"k": 8}` | More coverage, slightly more noise |
| Use MMR | `search_type="mmr"` | More diverse chunks, avoids redundancy |
| Add score threshold | `search_type="similarity_score_threshold"` | Rejects low-relevance chunks |
| Larger embedding model | `BAAI/bge-large-en-v1.5` | Better semantic understanding |
| Smaller chunk size | `chunk_size=500` | Higher precision per chunk |
| Query rewriting | Rephrase query before retrieval | Reduces vocabulary mismatch |

---

## 27. Comparison with Alternative Approaches

### RAG vs. Fine-tuning

| Dimension | RAG (this project) | Fine-tuning |
|---|---|---|
| Cost to add new documents | Seconds (re-embed + store) | Hours/days of GPU training |
| Source traceability | Built-in (source citations) | None |
| Knowledge cutoff | None — always up to date | Fixed at training time |
| Compute required | CPU sufficient | GPU required |
| Hallucination risk | Low (grounded in retrieved text) | Higher (baked into weights) |
| Max corpus size | Millions of documents | Limited by training data |

### RAG vs. Long-context LLMs

Modern LLMs like GPT-4o (128k context) or Gemini 1.5 Pro (1M context) can accept entire document sets in a single prompt. Why use RAG?

| Dimension | RAG | Long-context LLM |
|---|---|---|
| Cost | Cheap (small prompt) | Expensive (pay per token) |
| Speed | Fast (only 4 chunks sent) | Slow (process all tokens) |
| Privacy | Documents stay local | Documents sent to API |
| Precision | High (retrieved by relevance) | Degrades with document count ("lost in the middle") |
| Corpus size | Unlimited | Hard limit (~100–1000 pages) |

RAG is better for large, private, or frequently updated corpora. Long-context LLMs are better for small document sets where you want the model to consider everything holistically.

### ChromaDB vs. Other Vector Databases

| Feature | ChromaDB | Pinecone | FAISS | Weaviate |
|---|---|---|---|---|
| Setup | Embedded library | Cloud service | Embedded library | Docker server |
| Persistence | Yes (SQLite) | Yes (cloud) | No (RAM only) | Yes |
| Incremental writes | Yes | Yes | No (rebuild) | Yes |
| Metadata filtering | Yes | Yes | No | Yes |
| Scale | ~1M vectors (local) | Unlimited (cloud) | Billions (RAM) | Unlimited |
| Cost | Free | Paid | Free | Free (self-hosted) |
| Best for | Local / prototype | Production cloud | Research / speed | Production self-hosted |

---

## 28. Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation. An AI pattern that grounds LLM answers in a retrieved document corpus rather than training data alone. |
| **LLM** | Large Language Model. A neural network trained on text (e.g., GPT, llama3) capable of generating human-like language. |
| **Embedding** | A fixed-size vector of floating-point numbers that represents the semantic meaning of a piece of text. Semantically similar texts produce nearby vectors. |
| **Vector Database** | A database optimized for storing and searching embedding vectors by similarity, rather than exact key-value lookup. |
| **Cosine Similarity** | A measure of similarity between two vectors: the cosine of the angle between them. Ranges from -1 (opposite) to 1 (identical). |
| **HNSW** | Hierarchical Navigable Small World. A graph-based algorithm for approximate nearest neighbor search. Used by ChromaDB internally. |
| **Chunk** | A fragment of a document produced by splitting. Typically a few hundred to a few thousand characters. The atomic unit of retrieval. |
| **chunk_overlap** | The number of characters shared between adjacent chunks. Prevents context loss at split boundaries. |
| **Retriever** | A LangChain abstraction that accepts a query string and returns a list of relevant `Document` objects from the vector store. |
| **LCEL** | LangChain Expression Language. A composable, pipe-based syntax for building LangChain pipelines using the `\|` operator. |
| **Temperature** | A parameter controlling LLM output randomness. 0 = deterministic; 1 = highly creative. Set to 0.2 in this project for factual accuracy. |
| **BGE** | BAAI General Embeddings. A family of embedding models from the Beijing Academy of Artificial Intelligence, optimized for retrieval tasks. |
| **MTEB** | Massive Text Embedding Benchmark. The standard leaderboard for evaluating text embedding models across retrieval, classification, and clustering tasks. |
| **Ollama** | An open-source tool for running LLMs locally. Manages model downloads, GPU offloading, and exposes a REST API on `localhost:11434`. |
| **ChromaDB** | An open-source embedded vector database written in Python and C++. Stores vectors on disk using SQLite + HNSW. |
| **PyMuPDF** | A Python binding to the MuPDF rendering library. Used here for PDF text extraction with correct reading order and unicode support. |
| **Prompt Template** | A reusable text template with named placeholders (`{context}`, `{question}`) that is filled at runtime before being sent to the LLM. |
| **Hallucination** | When an LLM generates factually incorrect information confidently. RAG mitigates this by grounding the LLM in retrieved source text. |
| **Contrastive Learning** | A training technique where a model learns to embed similar items close together and dissimilar items far apart in vector space. Used to train BGE. |
| **MMR** | Maximal Marginal Relevance. A retrieval strategy that balances relevance to the query with diversity among results, avoiding redundant chunks. |
| **Session State** | Streamlit's mechanism for persisting Python objects (like chat history) across script reruns within the same browser session. |
| **`@st.cache_resource`** | A Streamlit decorator that executes a function once per process lifetime and caches the result in memory, avoiding expensive reloads. |
| **Normalize** | Scaling a vector so its L2 norm equals 1. Enables cosine similarity to be computed as a simple dot product. |
| **L2 Norm** | The Euclidean length of a vector: `sqrt(x₁² + x₂² + ... + xₙ²)`. Normalized vectors have L2 norm = 1. |
| **Persistent Directory** | The folder where ChromaDB writes its SQLite database and binary vector files to disk, surviving process restarts. |
| **`RunnablePassthrough`** | A LangChain LCEL component that passes its input through unchanged. Used to fork a pipeline so the same input goes to multiple branches. |
