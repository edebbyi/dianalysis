"""Filter candidate rows through strict and fallback stages."""

from __future__ import annotations

import re

import pandas as pd

from .candidate_pool import CHIP_KEYWORDS, COOKIE_KEYWORDS, text_contains_any
from .candidate_pool import (
    DAIRY_DRINK_KEYWORDS,
    JUICE_KEYWORDS,
    SOFT_DRINK_KEYWORDS,
    WATER_KEYWORDS,
)


def select_stage_candidates(
    pool: pd.DataFrame,
    *,
    score_this: int,
    fiber_this: float,
    sugar_this: float,
    this_net: float,
    group_key: str,
    k: int,
) -> pd.DataFrame:
    """Pick candidates in strict stage first, then soft and guarded fallbacks."""
    strict = pool[(pool["risk_score"] < score_this) & (pool["fiber_g"].fillna(0) >= fiber_this)].copy()
    strict["_stage"] = "strict"

    cand = strict
    if len(cand) < k:
        soft = pool[
            (pool["risk_score"] < score_this) & ((pool["net_carbs_g"] < this_net) | (pool["sugar_g"] < sugar_this))
        ].copy()
        if not soft.empty:
            soft["_stage"] = "soft"
            cand = pd.concat([cand, soft], ignore_index=False)

    if cand.empty and score_this <= 5:
        lowrisk = pool[
            (pool["risk_score"] <= score_this)
            & ((pool["fiber_g"] > fiber_this) | (pool["net_carbs_g"] < this_net))
        ].copy()
        if not lowrisk.empty:
            lowrisk["_stage"] = "lowrisk"
            cand = lowrisk

    if cand.empty and group_key in {"nuts-seeds", "nut"}:
        nuts = pool[
            (pool["risk_score"] <= score_this)
            & ((pool["fiber_g"] >= fiber_this) | (pool["net_carbs_g"] < this_net))
        ].copy()
        if not nuts.empty:
            nuts["_stage"] = "nuts"
            cand = nuts

    return cand


def apply_style_filters(
    cand: pd.DataFrame, *, group_key: str, name_text: str, fine_group_key: str = ""
) -> pd.DataFrame:
    """Filter snack results by chip-like or cookie-like query style."""
    out = cand
    if out.empty:
        return out

    if group_key == "drink":
        out["_drink_text"] = (
            out["name"].fillna("").astype(str)
            + " "
            + out["brand"].fillna("").astype(str)
            + " "
            + out["categories_all"].fillna("").astype(str)
        ).str.lower()

        normalized_fine = str(fine_group_key or "").strip().lower()
        wants_soft_drink = (normalized_fine == "drink:soft_drink") or text_contains_any(name_text, SOFT_DRINK_KEYWORDS)
        wants_water = (normalized_fine == "drink:water") or text_contains_any(name_text, WATER_KEYWORDS)
        wants_juice = (normalized_fine == "drink:juice") or text_contains_any(name_text, JUICE_KEYWORDS)

        if wants_soft_drink:
            # For soda/cola queries, prefer soda-like swaps and avoid
            # jumping to water or dairy drinks unless no soda-like rows exist.
            out["_is_soft"] = out["_drink_text"].apply(lambda t: text_contains_any(t, SOFT_DRINK_KEYWORDS))
            out["_is_water_like"] = out["_drink_text"].apply(lambda t: text_contains_any(t, WATER_KEYWORDS))
            out["_is_dairy_like"] = out["_drink_text"].apply(lambda t: text_contains_any(t, DAIRY_DRINK_KEYWORDS))
            out = out[out["_is_soft"] & (~out["_is_water_like"]) & (~out["_is_dairy_like"])]
            return out.drop(columns=["_drink_text", "_is_soft", "_is_water_like", "_is_dairy_like"], errors="ignore")

        if wants_water:
            out["_is_water"] = out["_drink_text"].apply(lambda t: text_contains_any(t, WATER_KEYWORDS))
            out = out[out["_is_water"]]
            return out.drop(columns=["_drink_text", "_is_water"], errors="ignore")

        if wants_juice:
            out["_is_juice"] = out["_drink_text"].apply(lambda t: text_contains_any(t, JUICE_KEYWORDS))
            out = out[out["_is_juice"]]
            return out.drop(columns=["_drink_text", "_is_juice"], errors="ignore")

        return out.drop(columns=["_drink_text"], errors="ignore")

    if group_key != "snack":
        return out

    wants_chip = text_contains_any(name_text, CHIP_KEYWORDS)
    wants_cookie = text_contains_any(name_text, COOKIE_KEYWORDS)

    if wants_chip:
        out["_matches_snack"] = out.apply(
            lambda r: text_contains_any(
                f"{r.get('name', '')} {r.get('brand', '')} {r.get('categories_all', '')}",
                CHIP_KEYWORDS,
            ),
            axis=1,
        )
        filtered = out[out["_matches_snack"]]
        out = filtered if not filtered.empty else out
        return out.drop(columns=["_matches_snack"])

    if wants_cookie:
        out["_matches_cookie"] = out.apply(
            lambda r: text_contains_any(
                f"{r.get('name', '')} {r.get('brand', '')} {r.get('categories_all', '')}",
                COOKIE_KEYWORDS,
            ),
            axis=1,
        )
        filtered = out[out["_matches_cookie"]]
        out = filtered if not filtered.empty else out
        return out.drop(columns=["_matches_cookie"])

    return out


def drop_duplicate_candidates(cand: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows by normalized name and brand."""
    out = cand.copy()
    if "upc" in out.columns:
        out["_upc_norm"] = out["upc"].fillna("").astype(str).str.strip()
        out = out[(out["_upc_norm"] == "") | (~out["_upc_norm"].duplicated(keep="first"))]

    def _canon_text(val: object) -> str:
        text = str(val or "").lower().strip()
        return re.sub(r"[^a-z0-9]+", "", text)

    out["_name_norm"] = out["name"].fillna("").apply(_canon_text)
    out["_brand_norm"] = out["brand"].fillna("").apply(_canon_text)
    out = out.drop_duplicates(subset=["_name_norm", "_brand_norm"], keep="first")
    return out.drop(columns=["_name_norm", "_brand_norm", "_upc_norm"], errors="ignore")


def apply_group_priority(cand: pd.DataFrame, *, group_key: str) -> pd.DataFrame:
    """Prefer rows from the same semantic group when available."""
    out = cand
    if group_key in {"nuts-seeds", "nut"}:
        nut_mask = (out["alt_group"].fillna("").str.lower() == "nuts-seeds") | (
            out["category"].fillna("").str.lower() == "nut"
        )
        nut_cand = out[nut_mask]
        if not nut_cand.empty:
            out = nut_cand

    if group_key == "snack":
        snack_mask = (out["alt_group"].fillna("").str.lower() == "snack") | (
            out["category"].fillna("").str.lower() == "snack"
        )
        snack_only = out[snack_mask]
        if not snack_only.empty:
            out = snack_only

    return out
