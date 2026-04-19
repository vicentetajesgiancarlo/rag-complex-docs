"""
RAG Pipeline Module
- Loads Chroma vector database
- Builds a retrieval-augmented generation chain using LCEL
- Uses local Ollama as the LLM backend
"""

from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from embedding_utils import get_embedding_model, VECTOR_DB_DIR, COLLECTION_NAME


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
# LLM — Ollama only
# ---------------------------------------------------------------------------

def get_llm():
    """Connect to a locally running Ollama instance.

    Start Ollama with:
        ollama pull llama3
        ollama serve
    """
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            from langchain_community.llms import Ollama
            print("[LLM] Connected to Ollama (llama3)")
            return Ollama(model="llama3", temperature=0.2)
    except Exception:
        pass

    raise RuntimeError(
        "Ollama is not running. Start it with:\n"
        "  ollama pull llama3\n"
        "  ollama serve"
    )


# ---------------------------------------------------------------------------
# Vector store & retriever
# ---------------------------------------------------------------------------

def load_vector_store() -> Chroma:
    """Load (or create) the persisted Chroma vector database."""
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
    """Ask a question against the indexed document corpus.

    Returns:
        dict with "answer" (str) and "sources" (list of metadata dicts).
    """
    if chain is None or retriever is None:
        chain, retriever = build_rag_chain()

    answer = chain.invoke(query)
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
            sources.append({"title": title, "page": page, "file": source_file})

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# CLI test — run with: python src/rag_pipeline.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("RAG Pipeline Test")
    print("=" * 60)

    try:
        response = ask_question("Summarize the main topics in the documents.")
        print(f"\nAnswer:\n{response['answer']}")
        print(f"\nSources:")
        for s in response["sources"]:
            print(f"  - {s['title']} (page {s['page']})")
    except RuntimeError as e:
        print(f"\n[WARN] {e}")
        print("\nTesting retrieval only ...")
        db = load_vector_store()
        results = db.similarity_search("main topics", k=4)
        for i, doc in enumerate(results):
            print(f"  [{i+1}] {doc.metadata.get('title','?')} (page {doc.metadata.get('page','?')})")
            print(f"      {doc.page_content[:150]}...\n")
