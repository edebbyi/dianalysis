"""Create and cache Qdrant and embedding clients."""

from __future__ import annotations

import os
from functools import lru_cache

try:  # optional dependency
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]

try:  # optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]


DEFAULT_COLLECTION = "dianalysis_products"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def collection_name() -> str:
    """Return the active Qdrant collection name."""
    return os.getenv("DIANALYSIS_QDRANT_COLLECTION", DEFAULT_COLLECTION)


def model_name() -> str:
    """Return the embedding model name."""
    return os.getenv("DIANALYSIS_EMBED_MODEL", DEFAULT_MODEL)


def qdrant_url() -> str:
    """Return Qdrant URL from environment."""
    return os.getenv("QDRANT_URL", "http://localhost:6333")


def retrieval_enabled() -> bool:
    """Return True when semantic retrieval is enabled and deps exist."""
    backend = os.getenv("DIANALYSIS_RETRIEVAL_BACKEND", "heuristic").strip().lower()
    return backend == "qdrant" and QdrantClient is not None and SentenceTransformer is not None


@lru_cache(maxsize=1)
def qdrant_client() -> QdrantClient:
    """Create one cached Qdrant client."""
    if QdrantClient is None:  # pragma: no cover
        raise RuntimeError("qdrant-client is not installed")
    return QdrantClient(url=qdrant_url())


@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    """Create one cached sentence-transformer embedder."""
    if SentenceTransformer is None:  # pragma: no cover
        raise RuntimeError("sentence-transformers is not installed")
    return SentenceTransformer(model_name())


def embedding_dimension() -> int:
    """Return embedding dimension with backward-compatible API calls."""
    emb = embedder()
    if hasattr(emb, "get_embedding_dimension"):
        return int(emb.get_embedding_dimension())
    return int(emb.get_sentence_embedding_dimension())


def ensure_collection_exists(name: str) -> None:
    """Create collection with cosine vectors if it does not exist."""
    client = qdrant_client()
    try:
        client.get_collection(collection_name=name)
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=embedding_dimension(), distance=qmodels.Distance.COSINE),
        )
