"""Search Qdrant for similar products and map results back to rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .vector_client import (
    collection_name,
    embedder,
    ensure_collection_exists,
    qdrant_client,
    qmodels,
    retrieval_enabled,
)
from .vector_index import embedding_text_for_item, product_key


def _search_with_filter(
    query_vector: list[float],
    *,
    query_filter: Any,
    local_limit: int,
    target_collection: str,
) -> list[Any]:
    client = qdrant_client()
    try:
        response = client.query_points(
            collection_name=target_collection,
            query=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=local_limit,
        )
        return list(getattr(response, "points", []))
    except Exception:
        if hasattr(client, "search"):
            return client.search(
                collection_name=target_collection,
                query_vector=query_vector,
                query_filter=query_filter,
                with_payload=True,
                limit=local_limit,
            )
    return []


def query_points(
    query_vector: list[float],
    *,
    target_collection: str,
    limit: int,
    cat: str,
    group_key: str,
) -> list[Any]:
    """Search same group first, then top-up from same category."""
    group_match = qmodels.FieldCondition(key="alt_group", match=qmodels.MatchValue(value=group_key))
    cat_match = qmodels.FieldCondition(key="category", match=qmodels.MatchValue(value=cat))

    strict_filter = qmodels.Filter(must=[group_match])
    strict_hits = _search_with_filter(
        query_vector,
        query_filter=strict_filter,
        local_limit=limit,
        target_collection=target_collection,
    )
    if len(strict_hits) >= limit:
        return strict_hits[:limit]

    category_filter = qmodels.Filter(must=[cat_match])
    category_hits = _search_with_filter(
        query_vector,
        query_filter=category_filter,
        local_limit=max(limit * 3, 30),
        target_collection=target_collection,
    )

    merged = list(strict_hits)
    seen_keys = {
        str((getattr(hit, "payload", None) or {}).get("product_key", "") or "")
        for hit in strict_hits
    }
    for hit in category_hits:
        payload = getattr(hit, "payload", None) or {}
        key = str(payload.get("product_key", "") or "")
        if key in seen_keys:
            continue
        merged.append(hit)
        seen_keys.add(key)
        if len(merged) >= limit:
            break
    return merged


def search_similar_candidates(
    df_all: pd.DataFrame,
    query_item: dict[str, Any],
    *,
    cat: str,
    group_key: str,
    limit: int = 100,
    collection: str | None = None,
) -> pd.DataFrame:
    """Return nearest-neighbor candidates as dataframe rows."""
    if not retrieval_enabled() or df_all.empty:
        return pd.DataFrame()

    target = collection or collection_name()
    try:
        ensure_collection_exists(target)
        query_text = embedding_text_for_item(query_item)
        query_vector = embedder().encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        hits = query_points(
            query_vector,
            target_collection=target,
            limit=limit,
            cat=cat,
            group_key=group_key,
        )
    except Exception:
        return pd.DataFrame()

    keys: list[str] = []
    key_to_score: dict[str, float] = {}
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        key = str(payload.get("product_key", "") or "").strip()
        if key:
            keys.append(key)
            raw_score = getattr(hit, "score", None)
            if raw_score is not None:
                score = float(raw_score)
                prev = key_to_score.get(key)
                if prev is None or score > prev:
                    key_to_score[key] = score
    if not keys:
        return pd.DataFrame()

    work = df_all.copy()
    work["_product_key"] = work.apply(lambda r: product_key(r.to_dict()), axis=1)
    out = work[work["_product_key"].isin(keys)].copy()
    if out.empty:
        return pd.DataFrame()

    rank_map = {key: idx for idx, key in enumerate(keys)}
    out["_key_rank"] = out["_product_key"].map(rank_map).fillna(len(keys)).astype(int)
    out["_retrieval_score"] = out["_product_key"].map(key_to_score).fillna(0.0).astype(float)

    # Drop duplicate identities before returning rows to ranking.
    out = out.sort_values("_key_rank").drop_duplicates(subset=["_product_key"], keep="first")

    # Extra safety: drop duplicates by same name+brand when identity keys collide upstream.
    out["_name_norm"] = out["name"].fillna("").astype(str).str.lower().str.strip()
    out["_brand_norm"] = out["brand"].fillna("").astype(str).str.lower().str.strip()
    out = out.drop_duplicates(subset=["_name_norm", "_brand_norm"], keep="first")

    return out.drop(columns=["_product_key", "_key_rank", "_name_norm", "_brand_norm"])
