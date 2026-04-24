"""Compatibility layer for vector indexing and semantic search."""

from __future__ import annotations

import pandas as pd

from .recommendation.vector_client import DEFAULT_COLLECTION, DEFAULT_MODEL, retrieval_enabled
from .recommendation.vector_index import (
    embedding_text_for_item,
    index_dataframe as _index_dataframe,
    product_key,
    prune_collection_by_keys,
)
from .recommendation.vector_search import search_similar_candidates


def index_dataframe(
    df: pd.DataFrame,
    *,
    collection_name: str | None = None,
    recreate: bool = False,
    prune_missing: bool = False,
    batch_size: int = 256,
    sync_meta: dict | None = None,
) -> int:
    """Embed and upsert dataframe rows into Qdrant."""
    return _index_dataframe(
        df,
        collection=collection_name,
        recreate=recreate,
        prune_missing=prune_missing,
        batch_size=batch_size,
        sync_meta=sync_meta,
    )


__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_MODEL",
    "embedding_text_for_item",
    "index_dataframe",
    "product_key",
    "prune_collection_by_keys",
    "retrieval_enabled",
    "search_similar_candidates",
]
