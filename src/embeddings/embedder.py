"""
src/embeddings/embedder.py

Sentence-transformers wrapper for local embedding generation.
Model: BAAI/bge-small-en-v1.5 (default, free, no API key needed).

Design notes:
  - Singleton pattern: model is loaded once at import time.
  - BGE models require a query prefix for retrieval tasks.
  - Batch processing to keep memory usage reasonable.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("sentence-transformers is required. Run: pip install sentence-transformers") from e

from config import settings

# BGE models: use a query prefix for retrieval, no prefix for passage embedding
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_PASSAGE_PREFIX = ""  # BGE small doesn't need passage prefix

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazy singleton loader — avoids downloading the model at import time."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model_name)
        _model = SentenceTransformer(settings.embedding_model_name)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str], is_query: bool = False, batch_size: int = 32) -> list[list[float]]:
    """
    Embed a list of texts.

    Args:
        texts:      Texts to embed.
        is_query:   If True, prepend the BGE query prefix.
        batch_size: Number of texts to process at once.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    model = _get_model()

    if is_query:
        prefixed = [_QUERY_PREFIX + t for t in texts]
    else:
        prefixed = texts

    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity friendly
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Convenience wrapper for single query embedding."""
    return embed_texts([query], is_query=True)[0]


def embed_passages(passages: list[str], batch_size: int = 32) -> list[list[float]]:
    """Convenience wrapper for passage (document) embedding."""
    return embed_texts(passages, is_query=False, batch_size=batch_size)
