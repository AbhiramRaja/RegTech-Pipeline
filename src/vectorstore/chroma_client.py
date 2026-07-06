"""
src/vectorstore/chroma_client.py

ChromaDB persistent client with two collections:
  - regulatory_corpus   : regulatory rule chunks
  - internal_policy     : internal policy chunks

Metadata schemas per architecture.md §7:
  regulatory_corpus : doc_id, clause_id, effective_date, superseded_by, source
  internal_policy   : doc_id, clause_id, policy_owner, last_reviewed

Design notes:
  - Embeddings are provided externally (from embedder.py) to keep concerns separated.
  - upsert_chunks() is idempotent — re-running won't duplicate.
  - query_similar() returns dicts that map directly into ComplianceState.retrieved_context.
"""

import logging
from typing import Literal, Optional

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError as e:
    raise ImportError("chromadb is required. Run: pip install chromadb") from e

from config import settings

# Collection names — referenced by graph nodes; don't change without updating nodes.py
REGULATORY_COLLECTION = "regulatory_corpus"
INTERNAL_POLICY_COLLECTION = "internal_policy"

_client: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    """Lazy singleton ChromaDB client."""
    global _client
    if _client is None:
        logger.info("Connecting to ChromaDB at: %s", settings.chroma_persist_dir)
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
        )
    return _client


def get_collection(name: Literal["regulatory_corpus", "internal_policy"]) -> chromadb.Collection:
    """Get (or create) a named collection."""
    client = _get_client()
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity (embeddings are normalized)
    )
    return collection


def upsert_chunks(
    collection_name: Literal["regulatory_corpus", "internal_policy"],
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """
    Upsert chunks into the specified collection.

    Args:
        collection_name: Target collection.
        chunks:          List of clause dicts (must include clause_id, text, and metadata keys).
        embeddings:      Corresponding embedding vectors (same order as chunks).

    Chunk dict keys expected:
        regulatory_corpus: clause_id, text, doc_id, effective_date, superseded_by, source
        internal_policy:   clause_id, text, doc_id, policy_owner, last_reviewed
    """
    if not chunks:
        logger.warning("upsert_chunks called with empty chunk list.")
        return

    assert len(chunks) == len(embeddings), "chunks and embeddings must have the same length"

    collection = get_collection(collection_name)

    ids = [c["clause_id"] for c in chunks]
    documents = [c["text"] for c in chunks]

    if collection_name == REGULATORY_COLLECTION:
        metadatas = [
            {
                "doc_id": c.get("doc_id", ""),
                "clause_id": c["clause_id"],
                "effective_date": c.get("effective_date", ""),
                "superseded_by": c.get("superseded_by", ""),
                "source": c.get("source", ""),
                "page": str(c.get("page", "")),
                "section": c.get("section", ""),
            }
            for c in chunks
        ]
    else:  # internal_policy
        metadatas = [
            {
                "doc_id": c.get("doc_id", ""),
                "clause_id": c["clause_id"],
                "policy_owner": c.get("policy_owner", ""),
                "last_reviewed": c.get("last_reviewed", ""),
                "page": str(c.get("page", "")),
                "section": c.get("section", ""),
            }
            for c in chunks
        ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info("Upserted %d chunks into '%s'.", len(chunks), collection_name)


def query_similar(
    collection_name: Literal["regulatory_corpus", "internal_policy"],
    query_embedding: list[float],
    top_k: int = 5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks.

    Returns:
        List of dicts with keys: chunk_id, text, score, metadata.
        These map directly into ComplianceState.retrieved_context.
    """
    collection = get_collection(collection_name)

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(
            {
                "chunk_id": meta.get("clause_id", ""),
                "text": doc,
                "score": float(1 - dist),  # cosine: distance → similarity
                "metadata": meta,
            }
        )
    return output


def collection_count(collection_name: Literal["regulatory_corpus", "internal_policy"]) -> int:
    """Return number of chunks in a collection."""
    return get_collection(collection_name).count()


def reset_collection(collection_name: Literal["regulatory_corpus", "internal_policy"]) -> None:
    """Delete and recreate a collection. Use only in tests."""
    client = _get_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    logger.info("Collection '%s' reset.", collection_name)
