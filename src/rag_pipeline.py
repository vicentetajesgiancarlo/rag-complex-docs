"""
RAG Pipeline Module
- Loads Chroma vector database
- Builds a retrieval-augmented generation chain using LCEL
- Supports Ollama (local) and OpenAI as LLM backends
"""

import os
import sys
from pathlib import Path

# Allow running as a standalone script from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from embedding_utils import get_embedding_model, VECTOR_DB_DIR, COLLECTION_NAME

# Load .env from project root so OPENAI_API_KEY is available
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Number of chunks to retrieve per query
RETRIEVAL_K = 4


# ---------------------------------------------------------------------------
# Custom exception — raised when no LLM backend is reachable
# ---------------------------------------------------------------------------

class LLMUnavailableError(RuntimeError):
    """Raised when no LLM backend (OpenAI or Ollama) is reachable at startup."""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert research assistant. Use the following context \
from the uploaded documents to answer the question. If you cannot find \
the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Provide a detailed, accurate answer based on the context above. \
Cite the source documents when possible."""
)


# ---------------------------------------------------------------------------
# LLM — OpenAI (Option A) or Ollama (Option B)
# ---------------------------------------------------------------------------

def get_llm():
    """Return an LLM instance, preferring OpenAI then falling back to Ollama.

    Raises LLMUnavailableError if neither backend is reachable.

    OpenAI: set OPENAI_API_KEY in a .env file at the project root.
    Ollama: run `ollama pull llama3` then `ollama serve`.
    """
    # Option A: OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        from langchain_openai import ChatOpenAI
        print("[LLM] Using OpenAI (gpt-3.5-turbo)")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=api_key)

    # Option B: Ollama — verify it's running AND the model is pulled
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            model_name = "llama3"
            available = [m["name"] for m in resp.json().get("models", [])]
            model_pulled = any(
                m == model_name or m.startswith(model_name + ":") for m in available
            )
            if not model_pulled:
                raise LLMUnavailableError(
                    f"Ollama is running but '{model_name}' is not pulled.\n"
                    f"  Run: ollama pull {model_name}\n"
                    f"  Available models: {', '.join(available) or 'none'}"
                )
            from langchain_ollama import OllamaLLM
            print(f"[LLM] Connected to Ollama ({model_name})")
            return OllamaLLM(model=model_name, temperature=0.2)
    except LLMUnavailableError:
        raise
    except Exception:
        pass

    raise LLMUnavailableError(
        "No LLM backend available.\n"
        "  • OpenAI: add OPENAI_API_KEY=sk-... to a .env file in the project root\n"
        "  • Ollama: run 'ollama pull llama3' then 'ollama serve'"
    )


# ---------------------------------------------------------------------------
# Vector store & retriever
# ---------------------------------------------------------------------------

def load_vector_store() -> Chroma:
    """Load the persisted Chroma vector database."""
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def get_retriever(vector_store: Chroma = None, k: int = RETRIEVAL_K):
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
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store: Chroma = None):
    """Build the full RAG chain using LangChain Expression Language.

    Returns (chain, retriever). Raises LLMUnavailableError if no LLM is found.
    """
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
    """Ask a question against the indexed document corpus.

    Uses a single RunnableParallel invocation so the chain and the retriever
    share the same query call — the displayed sources always match the context
    the LLM actually used.

    Returns:
        dict with "answer" (str) and "sources" (list of metadata dicts).
    """
    if chain is None or retriever is None:
        chain, retriever = build_rag_chain()

    runner = RunnableParallel(answer=chain, source_docs=retriever)
    result = runner.invoke(query)

    answer = result["answer"]
    source_docs = result["source_docs"]

    sources = []
    seen: set[str] = set()
    for doc in source_docs:
        meta = doc.metadata
        source_file = Path(meta.get("source", "unknown")).name
        page = meta.get("page", "?")
        title = meta.get("title", source_file)
        key = f"{source_file}:p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({"title": title, "page": page, "file": source_file})

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# CLI test — run with: python src/rag_pipeline.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("RAG Pipeline Test")
    print("=" * 60)

    try:
        response = ask_question("Summarize the main topics in the documents.")
        print(f"\nAnswer:\n{response['answer']}")
        print("\nSources:")
        for s in response["sources"]:
            print(f"  - {s['title']} (page {s['page']})")
    except LLMUnavailableError as e:
        print(f"\n[WARN] {e}")
        print("\nTesting retrieval only ...")
        db = load_vector_store()
        results = db.similarity_search("main topics", k=4)
        for i, doc in enumerate(results):
            print(f"  [{i+1}] {doc.metadata.get('title','?')} (page {doc.metadata.get('page','?')})")
            print(f"      {doc.page_content[:150]}...\n")
