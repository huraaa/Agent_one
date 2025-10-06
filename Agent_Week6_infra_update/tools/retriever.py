# tools/retriever.py
# Chroma-based retriever with OpenAI embeddings, input sanitization, and safe batching.

import os
import re
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

from rag.config import CHROMA_DIR, EMBED_MODEL

# --- Sanitize text to avoid tiktoken special-token errors ---
_SPECIAL = re.compile(r"<\|.*?\|>")                  # matches <|...|>
_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

def _clean(s: str) -> str:
    s = s or ""
    s = _SPECIAL.sub(" ", s)                         # drop special tokens
    s = _CTRL.sub(" ", s)                            # drop control chars
    return " ".join(s.split())                       # collapse whitespace


# --- Chroma setup ---
_db = chromadb.PersistentClient(path=CHROMA_DIR)

_embed = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=EMBED_MODEL,
)

COLLECTION_NAME = "docs"
_collection = _db.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embed
)


def add_documents(docs: List[Dict[str, str]]) -> None:
    """
    Add documents in safe batches.
    docs: [{"id": "...", "text": "...", "source": "..."}]
    """
    # Conservative limits to stay well under API per-request token caps.
    BATCH_MAX_CHARS = 180_000    # total chars per add()
    CLIP_CHARS = 4_000           # per-item clip (~1–1.5k tokens typical)

    ids: List[str] = []
    docs_texts: List[str] = []
    metas: List[Dict[str, str]] = []
    budget = 0

    def flush():
        nonlocal ids, docs_texts, metas, budget
        if ids:
            _collection.add(ids=ids, documents=docs_texts, metadatas=metas)
            ids, docs_texts, metas, budget = [], [], [], 0

    for d in docs:
        text = _clean((d.get("text") or "")[:CLIP_CHARS])
        length = len(text)

        # Start a new batch if adding this would exceed the budget.
        if budget and budget + length > BATCH_MAX_CHARS:
            flush()

        ids.append(d["id"])
        docs_texts.append(text)
        metas.append({"source": d.get("source", "")})
        budget += length

    flush()


def query_topk(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Return top-k chunks with their source and distance score.
    """
    res = _collection.query(query_texts=[_clean(query)], n_results=k)
    out: List[Dict[str, Any]] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]  # may be empty if backend doesn't return
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None
        out.append({
            "text": doc,
            "source": meta.get("source", ""),
            "score": dist
        })
    return out


def reset_collection() -> None:
    """Dangerous: drops and recreates the collection."""
    _db.delete_collection(COLLECTION_NAME)
    global _collection
    _collection = _db.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embed
    )

# tools/retriever.py
def confident(results, max_distance=0.25) -> bool:
    if not results: return False
    s = results[0].get("score")
    if s is None: return True  # backend didn’t return distances
    return s <= max_distance

