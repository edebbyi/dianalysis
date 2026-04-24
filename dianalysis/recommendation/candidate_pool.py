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
SOFT_DRINK_KEYWORDS = ["soda", "cola", "coke", "soft drink", "diet", "schweppes"]
WATER_KEYWORDS = ["water", "sparkling", "seltzer", "club soda"]
JUICE_KEYWORDS = ["juice", "lemonade"]
DAIRY_DRINK_KEYWORDS = [
    "milk",
    "shake",
    "protein",
    "soy",
    "almond",
    "oat",
    "fairlife",
    "core power",
]

CANON_CATEGORY_BY_ALT_GROUP = {
    "oats": "grain",
    "rice": "grain",
    "quinoa": "grain",
    "pasta-noodles": "grain",
    "cereal": "cereal",
    "granola": "cereal",
    "bread": "bread",
    "drink": "drink",
    "ice-cream": "dessert",
    "dairy": "dairy",
    "nuts-seeds": "nut",
    "snack": "snack",
}

BAGEL_KEYWORDS = ["bagel"]
WRAP_KEYWORDS = ["wrap", "flatbread", "tortilla", "pita", "naan"]
CRISPBREAD_KEYWORDS = ["crispbread", "wasa", "crackerbread"]
ROLL_BUN_KEYWORDS = ["roll", "bun"]
MUFFIN_KEYWORDS = ["english muffin", "muffin"]
SOURDOUGH_KEYWORDS = ["sourdough"]
RYE_KEYWORDS = ["rye"]
WHOLE_WHEAT_KEYWORDS = ["whole wheat", "wholemeal", "whole grain"]
WHITE_BREAD_KEYWORDS = ["white bread"]
POPCORN_KEYWORDS = ["popcorn"]


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


def _contains_any_phrase(text: str, phrases: Sequence[str]) -> bool:
    """Return True when any phrase is found in text (case-insensitive)."""
    if not text:
        return False
    lower = text.lower()
    return any(p in lower for p in phrases)


def _normalize_category_main(this_row: dict[str, Any]) -> str:
    """Resolve the broad category used for fallback retrieval."""
    category_main = ensure_str(this_row.get("category_main")).strip().lower()
    if category_main:
        return category_main
    alt_group = ensure_str(this_row.get("alt_group")).strip().lower()
    if alt_group in CANON_CATEGORY_BY_ALT_GROUP:
        return CANON_CATEGORY_BY_ALT_GROUP[alt_group]
    category = ensure_str(this_row.get("category")).strip().lower()
    return category or "snack"


def infer_alt_group_fine(
    *,
    category_main: str,
    alt_group: str,
    name_text: str,
    categories_all: str = "",
    ingredients_text: str = "",
) -> str:
    """
    Build a fine-grained retrieval group for semantic matching.

    Examples:
    - bread:bagel
    - drink:soft_drink
    - drink:water
    """
    cat_main = ensure_str(category_main).strip().lower() or "snack"
    group_key = ensure_str(alt_group).strip().lower() or cat_main
    lookup = " ".join([ensure_str(name_text), ensure_str(categories_all), ensure_str(ingredients_text)]).lower()

    if group_key == "bread":
        if _contains_any_phrase(lookup, BAGEL_KEYWORDS):
            return "bread:bagel"
        if _contains_any_phrase(lookup, WRAP_KEYWORDS):
            return "bread:wrap"
        if _contains_any_phrase(lookup, CRISPBREAD_KEYWORDS):
            return "bread:crispbread"
        if _contains_any_phrase(lookup, ROLL_BUN_KEYWORDS):
            return "bread:roll_bun"
        if _contains_any_phrase(lookup, MUFFIN_KEYWORDS):
            return "bread:muffin_english"
        if _contains_any_phrase(lookup, SOURDOUGH_KEYWORDS):
            return "bread:sourdough"
        if _contains_any_phrase(lookup, RYE_KEYWORDS):
            return "bread:rye"
        if _contains_any_phrase(lookup, WHOLE_WHEAT_KEYWORDS):
            return "bread:whole_wheat"
        if _contains_any_phrase(lookup, WHITE_BREAD_KEYWORDS):
            return "bread:white"
        return "bread:other"

    if group_key == "drink":
        # Use title/category context first so "carbonated water" in ingredients
        # does not force sodas into the water bucket.
        drink_surface = " ".join([ensure_str(name_text), ensure_str(categories_all)]).lower()
        if text_contains_any(drink_surface, SOFT_DRINK_KEYWORDS):
            return "drink:soft_drink"
        if text_contains_any(drink_surface, JUICE_KEYWORDS):
            return "drink:juice"
        if text_contains_any(drink_surface, DAIRY_DRINK_KEYWORDS):
            return "drink:dairy_drink"
        if text_contains_any(drink_surface, WATER_KEYWORDS):
            return "drink:water"
        # Last chance: use wider text (including ingredients).
        if text_contains_any(lookup, SOFT_DRINK_KEYWORDS):
            return "drink:soft_drink"
        if text_contains_any(lookup, JUICE_KEYWORDS):
            return "drink:juice"
        if text_contains_any(lookup, DAIRY_DRINK_KEYWORDS):
            return "drink:dairy_drink"
        if text_contains_any(lookup, WATER_KEYWORDS):
            return "drink:water"
        return "drink:other"

    if group_key in {"oats", "rice", "quinoa", "pasta-noodles"}:
        return f"grain:{group_key.replace('-', '_')}"

    if group_key in {"cereal", "granola"}:
        return f"cereal:{group_key}"

    if group_key == "snack":
        if _contains_any_phrase(lookup, POPCORN_KEYWORDS):
            return "snack:popcorn"
        if text_contains_any(lookup, CHIP_KEYWORDS):
            return "snack:chips"
        if text_contains_any(lookup, COOKIE_KEYWORDS):
            return "snack:cookies"
        if text_contains_any(lookup, NUT_KEYWORDS):
            return "snack:nut_seed"
        return "snack:other"

    if group_key == "nuts-seeds":
        return "nut:nut_seed"
    if group_key == "ice-cream":
        return "dessert:ice_cream"
    if group_key == "dairy":
        return "dairy:other"
    return f"{cat_main}:{group_key.replace('-', '_')}"


def ensure_row_group_fields(this_row: dict[str, Any]) -> dict[str, Any]:
    """Populate `category_main` and `alt_group_fine` on one row."""
    out = dict(this_row)
    category_main = _normalize_category_main(out)
    group_key = ensure_str(out.get("alt_group")).strip().lower() or category_main
    name_text = " ".join(
        [
            ensure_str(out.get("name", "")),
            ensure_str(out.get("brand", "")),
            ensure_str(out.get("categories_all", "")),
        ]
    )
    fine = infer_alt_group_fine(
        category_main=category_main,
        alt_group=group_key,
        name_text=name_text,
        categories_all=ensure_str(out.get("categories_all", "")),
        ingredients_text=ensure_str(out.get("ingredients_text", "")),
    )
    out["category_main"] = category_main
    out["alt_group"] = group_key
    out["alt_group_fine"] = fine
    if not ensure_str(out.get("category")).strip():
        out["category"] = category_main
    return out


def ensure_group_columns(df_all: pd.DataFrame) -> pd.DataFrame:
    """Ensure candidate dataframe has normalized group columns for retrieval + ranking."""
    out = df_all.copy()
    if out.empty:
        if "category_main" not in out.columns:
            out["category_main"] = ""
        if "alt_group" not in out.columns:
            out["alt_group"] = out["category"] if "category" in out.columns else ""
        if "alt_group_fine" not in out.columns:
            out["alt_group_fine"] = ""
        return out

    if "category_main" not in out.columns:
        out["category_main"] = ""
    if "alt_group" not in out.columns and "category" in out.columns:
        out["alt_group"] = out["category"]
    elif "alt_group" not in out.columns:
        out["alt_group"] = ""
    if "alt_group_fine" not in out.columns:
        out["alt_group_fine"] = ""

    needs_fill = (
        out["category_main"].fillna("").astype(str).str.strip().eq("")
        | out["alt_group_fine"].fillna("").astype(str).str.strip().eq("")
    )
    if not needs_fill.any():
        return out

    refreshed_rows = [ensure_row_group_fields(r) for r in out.to_dict(orient="records")]
    extra_cols = [c for c in refreshed_rows[0].keys() if c not in out.columns]
    ordered_cols = out.columns.tolist() + extra_cols
    return pd.DataFrame(refreshed_rows, columns=ordered_cols)


def normalize_target_group(this_row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return normalized category_main, alt_group, alt_group_fine, and lookup text."""
    out = ensure_row_group_fields(this_row)
    cat_main = ensure_str(out.get("category_main")).strip().lower() or "snack"
    group_key = ensure_str(out.get("alt_group")).strip().lower() or cat_main
    name_text = " ".join(
        [
            ensure_str(out.get("name", "")),
            ensure_str(out.get("brand", "")),
            ensure_str(out.get("categories_all", "")),
        ]
    )

    if group_key == "snack" and text_contains_any(name_text, NUT_KEYWORDS):
        out["alt_group"] = "nuts-seeds"
        out["category_main"] = "nut"
        out["category"] = "nut"
        out["alt_group_fine"] = "nut:nut_seed"
        cat_main = "nut"
        group_key = "nuts-seeds"

    this_row.clear()
    this_row.update(out)
    fine_group = ensure_str(out.get("alt_group_fine")).strip().lower()
    if not fine_group:
        fine_group = infer_alt_group_fine(
            category_main=cat_main,
            alt_group=group_key,
            name_text=name_text,
            categories_all=ensure_str(out.get("categories_all", "")),
            ingredients_text=ensure_str(out.get("ingredients_text", "")),
        )
        this_row["alt_group_fine"] = fine_group

    return cat_main, group_key, fine_group, name_text


def select_pool(df_all: pd.DataFrame, category_main: str, group_key: str, fine_group: str) -> pd.DataFrame:
    """Return same fine group first, then same broad group, then same category."""
    work = ensure_group_columns(df_all)
    fine_pool = work[work["alt_group_fine"].fillna("").astype(str).str.lower() == ensure_str(fine_group).lower()].copy()
    if not fine_pool.empty:
        return fine_pool

    group_pool = work[
        work["alt_group"].fillna(work["category_main"]).astype(str).str.lower() == ensure_str(group_key).lower()
    ].copy()
    if not group_pool.empty:
        return group_pool
    return work[work["category_main"].fillna("").astype(str).str.lower() == ensure_str(category_main).lower()].copy()


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
    out = ensure_group_columns(pool)
    if "net_carbs_g" not in out.columns:
        out["net_carbs_g"] = out.apply(compute_net_carbs, axis=1)
    out["fiber_g"] = pd.to_numeric(out["fiber_g"], errors="coerce").fillna(0.0)
    out["sugar_g"] = pd.to_numeric(out["sugar_g"], errors="coerce").fillna(float("inf"))
    return out
