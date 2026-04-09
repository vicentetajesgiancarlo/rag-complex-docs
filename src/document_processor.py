"""
Document Processor Module
- Downloads academic papers from arxiv
- Parses PDFs using PyMuPDFLoader
- Chunks documents using RecursiveCharacterTextSplitter
"""

import sys
import arxiv
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Fix Windows console encoding for unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Resolve paths relative to this file's location
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PDFS_DIR = BASE_DIR / "data" / "raw_pdfs"


# ---------------------------------------------------------------------------
# Phase 2: Data Acquisition
# ---------------------------------------------------------------------------

SEARCH_QUERIES = [
    "superscalar architecture",
    "cache coherence",
    "algebraic topology",
    "differential equations",
]

MAX_RESULTS_PER_QUERY = 2  # 2 papers per query -> ~8 total, deduplicated to 5-10


def download_arxiv_papers(
    queries: list[str] = SEARCH_QUERIES,
    max_per_query: int = MAX_RESULTS_PER_QUERY,
    output_dir: Path = RAW_PDFS_DIR,
) -> list[str]:
    """Search arxiv for each query and download PDFs into output_dir.

    Returns a list of downloaded file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    seen_ids: set[str] = set()

    client = arxiv.Client()

    for query in queries:
        print(f"\n[SEARCH] Searching arxiv for: '{query}' ...")
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

            # Build a filesystem-safe filename
            safe_title = (
                result.title.replace("/", "-")
                .replace("\\", "-")
                .replace(":", "-")
                .replace("?", "")
                .replace('"', "")
                [:80]
            )
            filename = f"{paper_id}_{safe_title}.pdf"
            filepath = output_dir / filename

            if filepath.exists():
                print(f"  [OK] Already exists: {filename}")
                downloaded.append(str(filepath))
                continue

            try:
                result.download_pdf(dirpath=str(output_dir), filename=filename)
                print(f"  [DL] Downloaded: {filename}")
                downloaded.append(str(filepath))
            except Exception as e:
                print(f"  [ERR] Failed to download {result.title}: {e}")

    print(f"\n[DONE] Total papers downloaded: {len(downloaded)}")
    return downloaded


# ---------------------------------------------------------------------------
# Phase 3: Parsing & Chunking
# ---------------------------------------------------------------------------

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
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter.

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
    print("PHASE 2 -- Downloading papers from arxiv")
    print("=" * 60)
    download_arxiv_papers()

    print("\n" + "=" * 60)
    print("PHASE 3 -- Parsing & chunking downloaded papers")
    print("=" * 60)
    docs = load_pdfs()
    if docs:
        chunks = chunk_documents(docs)
        print(f"\nSample chunk metadata: {chunks[0].metadata}")
