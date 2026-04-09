# RAG for Complex Academic Documents

A Retrieval-Augmented Generation (RAG) system that indexes and queries complex academic papers on topics like superscalar architecture, cache coherence, algebraic topology, and differential equations.

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
    |  top-4 chunks                          |
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
                                      [arxiv PDF Downloads]
```

1. **Data Acquisition**: Papers are downloaded from arxiv using the `arxiv` Python library.
2. **Parsing**: PDFs are loaded with `PyMuPDFLoader`, which extracts text while preserving equations and formatting.
3. **Chunking**: Documents are split into overlapping chunks (1000 chars, 200 overlap) using `RecursiveCharacterTextSplitter`.
4. **Embedding**: Chunks are embedded using the `BAAI/bge-small-en-v1.5` model and stored in a local ChromaDB database.
5. **Retrieval & Generation**: User queries retrieve the top-4 most similar chunks, which are passed as context to an LLM for answer generation.

## Setup & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download papers and build the vector database
python src/document_processor.py
python src/embedding_utils.py

# 3. Configure an LLM backend (choose one):

# Option A: OpenAI — create a .env file in this directory
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Option B: Ollama — install Ollama and pull a model
ollama pull llama3

# 4. Run the Streamlit app
streamlit run app.py
```

The app also works in **retrieval-only mode** (no LLM) — it will display the most relevant document chunks for any query.

## Design Decisions

### Why PyMuPDF?
Academic papers contain complex layouts: multi-column text, mathematical equations, tables, and embedded figures. PyMuPDF (`fitz`) handles these far better than simpler PDF parsers like `pdfminer` or `PyPDF2`. It preserves the reading order and extracts equations as unicode text rather than garbled symbols.

### Why RecursiveCharacterTextSplitter?
Mathematical and hardware architecture texts have deeply nested logical structures (theorem → proof → lemma, pipeline stage → hazard → solution). The recursive splitter respects paragraph and sentence boundaries before falling back to character-level splits, keeping related concepts together within each chunk. The 200-character overlap ensures that context is not lost at chunk boundaries.

### Why BAAI/bge-small-en-v1.5?
This model offers an excellent balance of embedding quality and speed for local processing. It runs on CPU without requiring a GPU, produces normalized embeddings (ideal for cosine similarity), and ranks highly on the MTEB benchmark for retrieval tasks.

## Project Structure

```
rag_complex_docs/
├── data/
│   ├── raw_pdfs/          # Downloaded arxiv papers
│   └── vector_db/         # Persistent ChromaDB storage
├── src/
│   ├── document_processor.py  # Download, parse, and chunk PDFs
│   ├── embedding_utils.py     # Embedding model & vector DB creation
│   └── rag_pipeline.py        # RAG chain (retriever + LLM)
├── app.py                 # Streamlit chat interface
├── requirements.txt
└── README.md
```
