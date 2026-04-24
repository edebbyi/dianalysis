"""Build and maintain the Qdrant product index."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .candidate_pool import ensure_group_columns
from .vector_client import (
    collection_name,
    embedder,
    ensure_collection_exists,
    qdrant_client,
    qmodels,
    retrieval_enabled,
)


def embedding_text_for_item(item: dict[str, Any]) -> str:
    """Build text used to embed one product."""
    name = str(item.get("name", "") or "").strip()
    brand = str(item.get("brand", "") or "").strip()
    category = str(item.get("category", "") or "").strip()
    category_main = str(item.get("category_main", "") or category).strip()
    alt_group = str(item.get("alt_group", "") or "").strip()
    alt_group_fine = str(item.get("alt_group_fine", "") or "").strip()
    categories_all = str(item.get("categories_all", "") or "").strip()
    ingredients = str(item.get("ingredients_text", "") or "").strip()
    return (
        f"name: {name} | brand: {brand} | category: {category} | category_main: {category_main} | "
        f"group: {alt_group} | fine_group: {alt_group_fine} | "
        f"categories: {categories_all} | ingredients: {ingredients}"
    )


def product_key(item: dict[str, Any]) -> str:
    """Create a stable product key for joining rows and points."""
    upc = str(item.get("upc", "") or "").strip()
    if upc:
        return f"upc:{upc}"
    name = str(item.get("name", "") or "").strip().lower()
    brand = str(item.get("brand", "") or "").strip().lower()
    return f"namebrand:{name}|{brand}"


def stable_point_id(key: str) -> int:
    """Convert product key to a stable int64 point id."""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) & 0x7FFF_FFFF_FFFF_FFFF


def prune_collection_by_keys(name: str, keep_keys: set[str], batch_size: int = 512) -> int:
    """Delete points missing from the latest dataset."""
    if not retrieval_enabled():
        return 0

    client = qdrant_client()
    offset: Any | None = None
    to_delete: list[int] = []

    while True:
        points, next_offset = client.scroll(
            collection_name=name,
            with_payload=True,
            with_vectors=False,
            limit=batch_size,
            offset=offset,
        )
        for point in points:
            payload = getattr(point, "payload", None) or {}
            key = str(payload.get("product_key", "") or "")
            if key not in keep_keys:
                to_delete.append(int(point.id))
        if next_offset is None:
            break
        offset = next_offset

    if not to_delete:
        return 0

    for start in range(0, len(to_delete), batch_size):
        chunk = to_delete[start : start + batch_size]
        client.delete(
            collection_name=name,
            points_selector=qmodels.PointIdsList(points=chunk),
            wait=True,
        )
    return len(to_delete)


def index_dataframe(
    df: pd.DataFrame,
    *,
    collection: str | None = None,
    recreate: bool = False,
    prune_missing: bool = False,
    batch_size: int = 256,
    sync_meta: dict[str, Any] | None = None,
) -> int:
    """Embed and upsert dataframe rows into Qdrant."""
    if not retrieval_enabled():
        return 0

    target = collection or collection_name()
    client = qdrant_client()

    if recreate:
        try:
            client.delete_collection(collection_name=target)
        except Exception:
            pass

    ensure_collection_exists(target)

    work = df.copy()
    work = ensure_group_columns(work)
    work["_product_key"] = work.apply(lambda r: product_key(r.to_dict()), axis=1)
    work["_point_id"] = work["_product_key"].apply(stable_point_id)
    texts = [embedding_text_for_item(row.to_dict()) for _, row in work.iterrows()]
    vectors = embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)

    total = 0
    for start in range(0, len(work), batch_size):
        end = min(start + batch_size, len(work))
        points = []
        rows = work.iloc[start:end].to_dict(orient="records")
        for local_idx, row in enumerate(rows, start=start):
            payload = {
                "product_key": str(row.get("_product_key", "") or ""),
                "category": str(row.get("category", "") or ""),
                "category_main": str(row.get("category_main", row.get("category", "")) or ""),
                "alt_group": str(row.get("alt_group", "") or ""),
                "alt_group_fine": str(row.get("alt_group_fine", "") or ""),
            }
            if sync_meta:
                if "model_type" in sync_meta:
                    payload["model_type"] = str(sync_meta.get("model_type", "") or "")
                if "model_fingerprint" in sync_meta:
                    payload["model_fingerprint"] = str(sync_meta.get("model_fingerprint", "") or "")
                if "scored_at_utc" in sync_meta:
                    payload["scored_at_utc"] = str(sync_meta.get("scored_at_utc", "") or "")
            points.append(
                qmodels.PointStruct(
                    id=int(row["_point_id"]),
                    vector=vectors[local_idx].tolist(),
                    payload=payload,
                )
            )
        client.upsert(collection_name=target, points=points, wait=False)
        total += len(points)

    if prune_missing and not recreate:
        keep_keys = set(work["_product_key"].astype(str).tolist())
        prune_collection_by_keys(target, keep_keys)

    return total
