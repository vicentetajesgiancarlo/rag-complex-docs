"""
Document Processor Module
- Downloads papers from arxiv using the arxiv Python library
- Parses PDFs using PyMuPDFLoader
- Chunks documents using RecursiveCharacterTextSplitter
"""

import re
import sys
from pathlib import Path

# Allow running as a standalone script from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PDFS_DIR = BASE_DIR / "data" / "raw_pdfs"

SEARCH_QUERIES = [
    "superscalar architecture",
    "cache coherence",
    "algebraic topology",
    "differential equations",
]
MAX_RESULTS_PER_QUERY = 2


def download_arxiv_papers(
    queries: list[str] = SEARCH_QUERIES,
    max_per_query: int = MAX_RESULTS_PER_QUERY,
    output_dir: Path = RAW_PDFS_DIR,
) -> list[str]:
    """Download recent papers from arxiv for the given search queries.

    Returns a list of absolute file paths for all downloaded PDFs.
    Each download is wrapped in try/except — one failure does not abort the rest.
    Files that already exist on disk are skipped (idempotent).
    """
    import arxiv

    output_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client()
    downloaded: list[str] = []
    seen_ids: set[str] = set()

    for query in queries:
        print(f"  [SEARCH] Querying: '{query}'")
        search = arxiv.Search(
            query=query,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        for result in client.results(search):
            paper_id = result.entry_id.split("/")[-1]
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)

            # Sanitize title for filesystem safety
            safe_title = re.sub(r'[\\/:*?"<>|]', "", result.title)[:80]
            filename = f"{paper_id}_{safe_title}.pdf"
            dest = output_dir / filename

            if dest.exists():
                print(f"    [SKIP] Already downloaded: {filename}")
                downloaded.append(str(dest))
                continue

            try:
                result.download_pdf(dirpath=str(output_dir), filename=filename)
                print(f"    [DOWN] Downloaded: {filename}")
                downloaded.append(str(dest))
            except Exception as e:
                print(f"    [ERR] Failed to download '{result.title}': {e}")

    print(f"[DONE] {len(downloaded)} PDFs ready in {output_dir}")
    return downloaded


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


# ---------------------------------------------------------------------------
# Execution block -- run with: python src/document_processor.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1 -- Downloading arxiv papers")
    print("=" * 60)

    paths = download_arxiv_papers()

    if not paths:
        print(
            "[WARN] No papers downloaded. You can still add PDFs manually "
            f"to {RAW_PDFS_DIR}"
        )
    else:
        print(f"\n[INFO] {len(paths)} PDF(s) are ready in {RAW_PDFS_DIR}")

    print("\nNext step: python src/embedding_utils.py")
