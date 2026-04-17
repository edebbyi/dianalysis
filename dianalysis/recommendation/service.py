"""Orchestrate recommendation retrieval, filtering, ranking, and formatting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .candidate_filters import (
    apply_group_priority,
    apply_style_filters,
    drop_duplicate_candidates,
    select_stage_candidates,
)
from .candidate_pool import (
    drop_self_candidates,
    ensure_str,
    normalize_target_group,
    prepare_pool_columns,
    select_pool,
)
from .candidate_ranker import format_alternatives, rank_candidates
from ..model import compute_net_carbs
from .vector_search import search_similar_candidates


def make_alternatives(df_all: pd.DataFrame, this_row: dict[str, Any], score_this: int, k: int = 3) -> list[dict[str, Any]]:
    """Find lower-risk alternatives from the same group, with safe fallbacks."""
    cat, group_key, name_text = normalize_target_group(this_row)

    pool = search_similar_candidates(
        df_all,
        this_row,
        cat=cat,
        group_key=group_key,
        limit=max(60, k * 20),
    )
    if pool.empty:
        pool = select_pool(df_all, cat, group_key)
    if pool.empty:
        return []

    pool = drop_self_candidates(pool, this_row)
    if pool.empty:
        return []

    pool = prepare_pool_columns(pool)

    fiber_this = float(this_row.get("fiber_g", 0) or 0)
    sugar_this = float(this_row.get("sugar_g", 0) or 0)
    this_net = compute_net_carbs(this_row)

    cand = select_stage_candidates(
        pool,
        score_this=score_this,
        fiber_this=fiber_this,
        sugar_this=sugar_this,
        this_net=this_net,
        group_key=group_key,
        k=k,
    )
    if cand.empty:
        return []

    cand = apply_style_filters(cand, group_key=group_key, name_text=name_text)
    cand = drop_duplicate_candidates(cand)
    cand = apply_group_priority(cand, group_key=group_key)

    query_ingredients = ensure_str(this_row.get("ingredients_text", ""))
    cand = rank_candidates(
        cand,
        this_net=this_net,
        sugar_this=sugar_this,
        fiber_this=fiber_this,
        cat=cat,
        group_key=group_key,
        name_text=name_text,
        query_ingredients=query_ingredients,
        k=k,
    )
    return format_alternatives(cand, this_net=this_net, fiber_this=fiber_this)
