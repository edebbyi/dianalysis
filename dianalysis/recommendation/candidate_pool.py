"""Prepare candidate pools before filtering and ranking."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from ..model import compute_net_carbs


CHIP_KEYWORDS = ["chip", "chips", "crisps", "nacho", "tortilla", "pretzel"]
COOKIE_KEYWORDS = ["cookie", "cookies", "biscuit", "sandwich", "oreo", "wafer"]
NUT_KEYWORDS = [
    "nut",
    "nuts",
    "almond",
    "cashew",
    "peanut",
    "pistachio",
    "walnut",
    "hazelnut",
    "seed",
    "trail mix",
]


def text_contains_any(text: str, keywords: Sequence[str]) -> bool:
    """Return True when any keyword appears in text."""
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def ensure_str(val: Any) -> str:
    """Convert any value to string, keeping None as empty string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return str(val)


def normalize_target_group(this_row: dict[str, Any]) -> tuple[str, str, str]:
    """Return normalized category, group key, and lookup text."""
    cat = ensure_str(this_row.get("category"))
    group_key = ensure_str(this_row.get("alt_group")) or cat
    name_text = " ".join(
        [
            ensure_str(this_row.get("name", "")),
            ensure_str(this_row.get("brand", "")),
            ensure_str(this_row.get("categories_all", "")),
        ]
    )

    if group_key == "snack" and text_contains_any(name_text, NUT_KEYWORDS):
        group_key = "nuts-seeds"
        this_row["alt_group"] = group_key
        this_row["category"] = "nut"
        cat = ensure_str(this_row["category"])

    return cat, group_key, name_text


def select_pool(df_all: pd.DataFrame, cat: str, group_key: str) -> pd.DataFrame:
    """Return same-group rows, then same-category rows when group is empty."""
    if "alt_group" in df_all.columns:
        pool = df_all[df_all["alt_group"].fillna(df_all["category"]) == group_key].copy()
        if pool.empty:
            pool = df_all[df_all["category"] == cat].copy()
        return pool
    return df_all[df_all["category"] == cat].copy()


def drop_self_candidates(pool: pd.DataFrame, this_row: dict[str, Any]) -> pd.DataFrame:
    """Remove rows that are the same product as the query row."""
    upc = ensure_str(this_row.get("upc")).strip()
    name = ensure_str(this_row.get("name")).strip().lower()
    brand = ensure_str(this_row.get("brand")).strip().lower()

    out = pool
    if "upc" in out.columns and upc:
        out = out[out["upc"].astype(str).fillna("") != upc]

    if {"name", "brand"}.issubset(out.columns) and name and brand:
        out = out[
            ~((out["name"].str.lower().fillna("") == name) & (out["brand"].str.lower().fillna("") == brand))
        ]
    return out


def prepare_pool_columns(pool: pd.DataFrame) -> pd.DataFrame:
    """Create numeric columns needed by filtering and ranking."""
    out = pool.copy()
    if "net_carbs_g" not in out.columns:
        out["net_carbs_g"] = out.apply(compute_net_carbs, axis=1)
    out["fiber_g"] = pd.to_numeric(out["fiber_g"], errors="coerce").fillna(0.0)
    out["sugar_g"] = pd.to_numeric(out["sugar_g"], errors="coerce").fillna(float("inf"))
    return out
