"""
RAG Pipeline Module
- Loads Chroma vector database
- Builds a retrieval-augmented generation chain using LCEL
- Supports OpenAI API or local Ollama as the LLM backend
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from embedding_utils import get_embedding_model, VECTOR_DB_DIR, COLLECTION_NAME

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


# ---------------------------------------------------------------------------
# Prompt template for RAG
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert academic research assistant. Use the following context \
from academic papers to answer the question. If you cannot find the answer in \
the context, say so clearly.

Context:
{context}

Question: {question}

Provide a detailed, accurate answer based on the context above. \
Cite the source papers when possible."""
)


# ---------------------------------------------------------------------------
# LLM initialization
# ---------------------------------------------------------------------------

def get_llm():
    """Initialize the LLM backend.

    Priority:
      1. If OPENAI_API_KEY is set -> use ChatOpenAI (gpt-3.5-turbo)
      2. Otherwise -> attempt local Ollama (llama3)

    To use OpenAI: create a .env file in the project root with:
        OPENAI_API_KEY=sk-...
    To use Ollama: install Ollama and pull a model (e.g. `ollama pull llama3`)
    """
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        from langchain_openai import ChatOpenAI
        print("[LLM] Using OpenAI (gpt-3.5-turbo)")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            api_key=openai_key,
        )

    # Fallback: try local Ollama (must be running)
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            from langchain_community.llms import Ollama
            print("[LLM] Using local Ollama (llama3)")
            return Ollama(model="llama3", temperature=0.2)
    except Exception:
        pass

    raise RuntimeError(
        "No LLM backend available. Either:\n"
        "  1. Set OPENAI_API_KEY in a .env file, or\n"
        "  2. Install and run Ollama with a model (e.g. `ollama pull llama3`)"
    )


# ---------------------------------------------------------------------------
# Vector store & retriever
# ---------------------------------------------------------------------------

def load_vector_store() -> Chroma:
    """Load the persisted Chroma vector database."""
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def get_retriever(vector_store: Chroma = None, k: int = 4):
    """Return a retriever that fetches the top-k most relevant chunks."""
    if vector_store is None:
        vector_store = load_vector_store()
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ---------------------------------------------------------------------------
# RAG chain (LCEL)
# ---------------------------------------------------------------------------

def _format_docs(docs) -> str:
    """Join retrieved document contents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store: Chroma = None):
    """Build the full RAG chain using LangChain Expression Language."""
    retriever = get_retriever(vector_store)
    llm = get_llm()

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask_question(query: str, chain=None, retriever=None) -> dict:
    """Ask a question against the academic paper corpus.

    Args:
        query: The user's question.
        chain: Optional pre-built LCEL chain.
        retriever: Optional retriever (for fetching source docs).

    Returns:
        dict with keys:
            - "answer": The LLM-generated answer string.
            - "sources": List of dicts with source metadata (title, page, file).
    """
    if chain is None or retriever is None:
        chain, retriever = build_rag_chain()

    # Get answer from the chain
    answer = chain.invoke(query)

    # Retrieve source documents separately for metadata
    source_docs = retriever.invoke(query)

    sources = []
    seen = set()
    for doc in source_docs:
        meta = doc.metadata
        source_file = Path(meta.get("source", "unknown")).name
        page = meta.get("page", "?")
        title = meta.get("title", source_file)
        key = f"{source_file}:p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "title": title,
                "page": page,
                "file": source_file,
            })

    return {
        "answer": answer,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("PHASE 5 -- RAG Pipeline Test")
    print("=" * 60)

    try:
        response = ask_question("What are the main approaches discussed in the papers?")
        print(f"\nAnswer:\n{response['answer']}")
        print(f"\nSources:")
        for s in response["sources"]:
            print(f"  - {s['title']} (page {s['page']})")
    except RuntimeError as e:
        print(f"\n[WARN] {e}")
        print("\nThe pipeline is correctly wired. Provide an LLM backend to test generation.")

        # Still test retrieval independently
        print("\n--- Testing retrieval only ---")
        db = load_vector_store()
        results = db.similarity_search("main approaches discussed", k=4)
        for i, doc in enumerate(results):
            title = doc.metadata.get("title", "?")
            page = doc.metadata.get("page", "?")
            print(f"  [{i+1}] {title} (page {page})")
            print(f"      {doc.page_content[:150]}...\n")
