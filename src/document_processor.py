"""
Document Processor Module
- Parses uploaded PDFs using PyMuPDFLoader
- Chunks documents using RecursiveCharacterTextSplitter
"""

import sys
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PDFS_DIR = BASE_DIR / "data" / "raw_pdfs"


def load_pdfs(pdf_dir: Path = RAW_PDFS_DIR) -> list:
    """Load all PDFs from pdf_dir using PyMuPDFLoader.

    Returns a flat list of LangChain Document objects (one per page).
    """
    all_docs = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("[WARN] No PDF files found in", pdf_dir)
        return all_docs

    for pdf_path in pdf_files:
        print(f"  [LOAD] Loading: {pdf_path.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"  [ERR] Error loading {pdf_path.name}: {e}")

    print(f"[DONE] Total pages loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split documents into overlapping chunks.

    Returns a list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[DONE] Total chunks created: {len(chunks)}")
    return chunks
