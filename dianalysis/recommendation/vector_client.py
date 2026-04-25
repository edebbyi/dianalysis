"""Create and cache Qdrant and embedding clients."""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.util import find_spec
from typing import Any

qmodels: Any
QdrantClientType: Any | None
try:  # optional dependency
    from qdrant_client import QdrantClient as _QdrantClient
    from qdrant_client.http import models as _qmodels

    QdrantClientType = _QdrantClient
    qmodels = _qmodels
except Exception:  # pragma: no cover
    QdrantClientType = None
    qmodels = None

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


def qdrant_api_key() -> str:
    """Return optional Qdrant API key from environment."""
    return str(
        os.getenv("QDRANT_API_KEY", "") or os.getenv("DIANALYSIS_QDRANT_API_KEY", "")
    ).strip()


def retrieval_enabled() -> bool:
    """Return True when semantic retrieval is enabled and deps exist."""
    backend = os.getenv("DIANALYSIS_RETRIEVAL_BACKEND", "qdrant").strip().lower()
    return backend == "qdrant" and QdrantClientType is not None and find_spec("sentence_transformers") is not None


@lru_cache(maxsize=1)
def _sentence_transformer_class() -> Any:
    """Import sentence-transformers lazily so heuristic paths stay lightweight."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


@lru_cache(maxsize=1)
def qdrant_client() -> Any:
    """Create one cached Qdrant client."""
    if QdrantClientType is None:  # pragma: no cover
        raise RuntimeError("qdrant-client is not installed")
    key = qdrant_api_key()
    if key:
        return QdrantClientType(url=qdrant_url(), api_key=key)
    return QdrantClientType(url=qdrant_url())


@lru_cache(maxsize=1)
def embedder() -> Any:
    """Create one cached sentence-transformer embedder."""
    if find_spec("sentence_transformers") is None:  # pragma: no cover
        raise RuntimeError("sentence-transformers is not installed")
    return _sentence_transformer_class()(model_name())


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
