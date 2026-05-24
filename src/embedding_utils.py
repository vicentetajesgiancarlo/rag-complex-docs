"""
Embedding Utilities Module
- Initializes local HuggingFace embedding model (BAAI/bge-small-en-v1.5)
- Ingests chunked documents into a persistent Chroma vector database
"""

import sys
import functools
from pathlib import Path

# Allow running as a standalone script from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"

# Embedding model config
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 512  # output dimensions for BAAI/bge-small-en-v1.5
COLLECTION_NAME = "rag_academic_papers"


@functools.lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Initialize and return the HuggingFace embedding model (cached singleton)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_db(chunks: list, persist_directory: Path = VECTOR_DB_DIR) -> Chroma:
    """Ingest chunked documents into a persistent Chroma vector database.

    Args:
        chunks: List of LangChain Document objects (from document_processor.chunk_documents).
        persist_directory: Path to store the Chroma database on disk.

    Returns:
        The Chroma vector store instance.
    """
    print(f"[INFO] Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = get_embedding_model()

    print(f"[INFO] Creating Chroma vector DB at: {persist_directory}")
    print(f"[INFO] Ingesting {len(chunks)} chunks (this may take a few minutes) ...")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=COLLECTION_NAME,
    )

    count = len(db.get(include=[])["ids"])
    print(f"[DONE] Vector database created with {count} vectors.")
    return db


def load_vector_db(persist_directory: Path = VECTOR_DB_DIR) -> Chroma:
    """Load an existing Chroma database from disk."""
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# ---------------------------------------------------------------------------
# Execution block -- run with: python src/embedding_utils.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from document_processor import load_pdfs, chunk_documents

    print("=" * 60)
    print("PHASE 2 -- Building vector embeddings and Chroma database")
    print("=" * 60)

    print("\n[STEP 1] Loading PDFs ...")
    docs = load_pdfs()

    if not docs:
        print("[ERR] No documents loaded. Run document_processor.py first or add PDFs to data/raw_pdfs/")
        sys.exit(1)

    print("\n[STEP 2] Chunking documents ...")
    chunks = chunk_documents(docs)

    print("\n[STEP 3] Creating vector database ...")
    db = create_vector_db(chunks)

    print("\n[TEST] Running a test query: 'cache coherence protocol'")
    results = db.similarity_search("cache coherence protocol", k=3)
    for i, doc in enumerate(results):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        print(f"  Result {i+1}: {source} (page {page})")
        print(f"    {doc.page_content[:120]}...")
