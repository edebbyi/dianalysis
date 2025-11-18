"""
Scoring logic for food items with human-readable explanations and alternatives.
"""

import pandas as pd
import numpy as np
from .model import NUM_COLS, CAT_COLS, compute_net_carbs, rule_points_and_reasons


# Display rules for formatting nutrition values
DISPLAY_RULES = {
    "carbs_g": {"label": "Carbs", "unit": "g"},
    "fiber_g": {"label": "Fiber", "unit": "g"},
    "sugar_g": {"label": "Total sugar", "unit": "g"},
    "added_sugar_g": {"label": "Added sugar", "unit": "g"},
    "sugar_alcohols_g": {"label": "Sugar alcohols", "unit": "g"},
    "protein_g": {"label": "Protein", "unit": "g"},
    "fat_g": {"label": "Fat", "unit": "g"},
    "sodium_mg": {"label": "Sodium", "unit": "mg"},
    "calories": {"label": "Calories", "unit": "kcal"},
}


def format_risk_display(prob: float) -> str:
    """Format probability as user-friendly risk display string."""
    if prob is None:
        return "—"
    p = float(prob)
    if p < 0.005:  # <0.5%
        return "Very low (<1)"
    if p > 0.995:  # >99.5%
        return "Very high (>99)"
    return str(int(round(100 * p)))


def _format_display_value(v, unit: str) -> str:
    """Format a nutrition value with appropriate unit."""
    if v is None:
        return "not listed"
    try:
        v = float(v)
    except Exception:
        return "not listed"
    if unit == "mg":
        return f"{int(round(v))}mg"
    if unit == "kcal":
        return f"{int(round(v))}kcal"
    return f"{v:.1f}g"


def ensure_display(row: dict) -> dict:
    """Ensure item has __display dict for label-faithful formatting."""
    disp = row.get("__display") or {}
    if disp:  # already present
        return row

    new_disp = {}
    for field, rule in DISPLAY_RULES.items():
        new_disp[field] = _format_display_value(row.get(field), rule["unit"])
    row["__display"] = new_disp
    return row


def ensure_alt_group(row: dict) -> dict:
    """Default alt_group to category if missing."""
    if not row.get("alt_group"):
        row["alt_group"] = row.get("category")
    return row


def human_reasons(row):
    """
    Generate human-readable reasons for the risk score.
    Uses numeric values for thresholds, but shows label-faithful strings from __display.
    """
    reasons = []
    disp = row.get("__display", {}) or {}

    # Net carbs (always numeric display)
    net = row.get("net_carbs_g", compute_net_carbs(row))
    reasons.append(
        f"High net carbs ({net:.1f}g > 20g)" if net > 20
        else f"Net carbs within target ({net:.1f}g ≤ 20g)"
    )

    # Added sugar
    added_v = row.get("added_sugar_g")
    added_txt = disp.get("added_sugar_g") or (
        "not listed" if added_v is None else f"{float(added_v):.1f}g"
    )
    if added_v is None:
        reasons.append("Added sugar not listed")
    elif float(added_v) >= 8:
        reasons.append(f"High added sugar ({added_txt} ≥ 8g)")
    else:
        reasons.append(f"Low added sugar ({added_txt})")

    # Sodium (mg)
    sodium_v = row.get("sodium_mg")
    sodium_txt = disp.get("sodium_mg") or (
        "not listed" if sodium_v is None else f"{int(round(float(sodium_v)))}mg"
    )
    if sodium_v is None:
        reasons.append("Sodium not listed")
    else:
        s = float(sodium_v)
        if s >= 500:
            reasons.append(f"High sodium ({sodium_txt} ≥ 500mg)")
        elif s <= 140:
            reasons.append(f"Low sodium ({sodium_txt} ≤ 140mg)")
        else:
            reasons.append(f"Moderate sodium ({sodium_txt} < 500mg)")

    # Fiber
    fiber_v = row.get("fiber_g")
    fiber_txt = disp.get("fiber_g") or (
        "not listed" if fiber_v is None else f"{float(fiber_v):.1f}g"
    )
    if fiber_v is None:
        reasons.append("Fiber not listed (some labels show '<1g' for trace amounts)")
    elif float(fiber_v) >= 5:
        reasons.append(f"Good fiber ({fiber_txt} ≥ 5g)")
    else:
        reasons.append(f"Low fiber ({fiber_txt} < 5g)")

    # Protein
    protein_v = row.get("protein_g")
    protein_txt = disp.get("protein_g") or (
        "not listed" if protein_v is None else f"{float(protein_v):.1f}g"
    )
    if protein_v is None:
        reasons.append("Protein not listed")
    elif float(protein_v) >= 12:
        reasons.append(f"Higher protein ({protein_txt} ≥ 12g)")
    else:
        reasons.append(f"Moderate protein ({protein_txt} < 12g)")

    return reasons


def data_notes(row: dict) -> list:
    """Generate notes explaining 'not listed' or inferred values."""
    disp = row.get("__display", {}) or {}
    notes = []
    for field, rule in DISPLAY_RULES.items():
        shown = disp.get(field)
        label = rule.get("label", field)

        if shown == "not listed":
            notes.append(
                f"{label}: source didn't include a per-serving value. "
                f"Some packages print trace amounts; we show that when per-100g data supports it."
            )
        elif isinstance(shown, str) and shown.startswith("<"):
            notes.append(f"{label}: '{shown}' inferred from per-100g × serving size.")
    return notes


CHIP_KEYWORDS = ["chip", "chips", "crisps", "nacho", "tortilla", "pretzel"]
COOKIE_KEYWORDS = ["cookie", "cookies", "biscuit", "sandwich", "oreo", "wafer"]
NUT_KEYWORDS = ["nut", "nuts", "almond", "cashew", "peanut", "pistachio", "walnut", "hazelnut", "seed", "trail mix"]


def _text_contains_any(text: str, keywords) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _ensure_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return str(val)


def make_alternatives(df_all: pd.DataFrame, this_row: dict, score_this: int, k: int = 3):
    """
    Find better alternatives in the same category/alt_group.
    
    FIXED: Now properly uses alt_group to find alternatives within the same food group
    (e.g., oats alternatives for oats, not mixing with rice or pasta).
    
    Logic:
    - Prefer same alt_group; fall back to same category
    - Candidates must have lower risk AND >= fiber (strict rule)
    - Fallback for very low risk items (<=5): allow same risk if fiber/net_carbs improve
    - Ranking: lowest risk, then highest fiber, then biggest net-carb reduction
    """
    cat = this_row.get("category")
    group_key = this_row.get("alt_group") or cat
    name_text = " ".join(
        [
            _ensure_str(this_row.get("name", "")),
            _ensure_str(this_row.get("brand", "")),
            _ensure_str(this_row.get("categories_all", "")),
        ]
    )
    if group_key == "snack" and _text_contains_any(name_text, NUT_KEYWORDS):
        group_key = "nuts-seeds"
        this_row["alt_group"] = group_key
        this_row["category"] = "nut"
        cat = this_row["category"]

    # FIXED: prefer same alt_group; fall back to same category
    if "alt_group" in df_all.columns:
        pool = df_all[df_all["alt_group"].fillna(df_all["category"]) == group_key].copy()
        if pool.empty:
            pool = df_all[df_all["category"] == cat].copy()
    else:
        pool = df_all[df_all["category"] == cat].copy()

    if pool.empty:
        return []

    # Avoid self-suggesting by UPC or exact name+brand
    upc = (this_row.get("upc") or "").strip()
    name = (this_row.get("name") or "").strip().lower()
    brand = (this_row.get("brand") or "").strip().lower()

    if "upc" in pool.columns and upc:
        pool = pool[pool["upc"].astype(str).fillna("") != upc]

    if {"name", "brand"}.issubset(pool.columns) and name and brand:
        pool = pool[
            ~(
                (pool["name"].str.lower().fillna("") == name) &
                (pool["brand"].str.lower().fillna("") == brand)
            )
        ]

    if pool.empty:
        return []

    # Ensure net_carbs_g exists
    if "net_carbs_g" not in pool.columns:
        pool["net_carbs_g"] = pool.apply(compute_net_carbs, axis=1)

    fiber_this = float(this_row.get("fiber_g", 0) or 0)
    sugar_this = float(this_row.get("sugar_g", 0) or 0)
    this_net = compute_net_carbs(this_row)

    # Strict rule: lower risk AND >= fiber (or allow same risk for nuts later)
    cand = pool[
        (pool["risk_score"] < score_this) &
        (pool["fiber_g"].fillna(0) >= fiber_this)
    ].copy()

    # Fallback only when item already very low risk (<=5)
    if cand.empty and score_this <= 5:
        cand = pool[
            (pool["risk_score"] <= score_this) &
            (
                (pool["fiber_g"].fillna(0) > fiber_this) |
                (pool["net_carbs_g"] < this_net)
            )
        ].copy()

    # Additional fallback for nuts: allow same risk if fiber >= or if net carbs matching but fiber better
    if cand.empty and group_key in {"nuts-seeds", "nut"}:
        cand = pool[
            (pool["risk_score"] <= score_this) &
            (
                (pool["fiber_g"].fillna(0) >= fiber_this) |
                (pool["net_carbs_g"] < this_net)
            )
        ].copy()

    if cand.empty:
        return []

    # When the target group is snack, prefer other chip/crisp-style foods
    if group_key == "snack":
        wants_chip_style = _text_contains_any(name_text, CHIP_KEYWORDS)
        wants_cookie_style = _text_contains_any(name_text, COOKIE_KEYWORDS)
        if wants_chip_style and not cand.empty:
            cand["_matches_snack"] = cand.apply(
                lambda r: _text_contains_any(
                    f"{r.get('name', '')} {r.get('brand', '')} {r.get('categories_all', '')}",
                    CHIP_KEYWORDS,
                ),
                axis=1,
            )
            filtered = cand[cand["_matches_snack"]]
            if not filtered.empty:
                cand = filtered
            cand = cand.drop(columns=["_matches_snack"])
        elif wants_cookie_style and not cand.empty:
            cand["_matches_cookie"] = cand.apply(
                lambda r: _text_contains_any(
                    f"{r.get('name', '')} {r.get('brand', '')} {r.get('categories_all', '')}",
                    COOKIE_KEYWORDS,
                ),
                axis=1,
            )
            filtered = cand[cand["_matches_cookie"]]
            if not filtered.empty:
                cand = filtered
            cand = cand.drop(columns=["_matches_cookie"])

    # Remove duplicate candidates (same name + brand)
    cand["_name_norm"] = cand["name"].fillna("").str.lower().str.strip()
    cand["_brand_norm"] = cand["brand"].fillna("").str.lower().str.strip()
    cand = cand.drop_duplicates(subset=["_name_norm", "_brand_norm"])
    cand = cand.drop(columns=["_name_norm", "_brand_norm"])

    # When searching for nuts, prioritize candidates with alt_group/category nuts-seeds.
    if group_key in {"nuts-seeds", "nut"}:
        nut_mask = (
            cand["alt_group"].fillna("").str.lower() == "nuts-seeds"
        ) | (
            cand["category"].fillna("").str.lower() == "nut"
        )
        nut_cand = cand[nut_mask]
        if not nut_cand.empty:
            cand = nut_cand

    if group_key in {"nuts-seeds", "nut"}:
        nut_mask = (
            cand["alt_group"].fillna("").str.lower() == "nuts-seeds"
        ) | (
            cand["category"].fillna("").str.lower() == "nut"
        )
        nut_only = cand[nut_mask]
        if not nut_only.empty:
            cand = nut_only
    elif group_key == "snack":
        snack_mask = (
            cand["alt_group"].fillna("").str.lower() == "snack"
        ) | (
            cand["category"].fillna("").str.lower() == "snack"
        )
        snack_only = cand[snack_mask]
        if not snack_only.empty:
            cand = snack_only

    # Rank: lowest risk, then highest fiber, then biggest net-carb reduction
    cand["delta_net"] = this_net - cand["net_carbs_g"].astype(float)  # positive = improvement
    cand["sugar_g"] = pd.to_numeric(cand["sugar_g"], errors="coerce").fillna(0.0)
    cand["fiber_g"] = pd.to_numeric(cand["fiber_g"], errors="coerce").fillna(0.0)
    cand["sugar_diff"] = (cand["sugar_g"] - sugar_this).abs()
    cand["context_score"] = cand["risk_score"] * 5 + cand["sugar_diff"]

    # Rank by blended risk + macro similarity, then fiber and net-carb gain
    cand = cand.sort_values(
        by=["context_score", "fiber_g", "delta_net"],
        ascending=[True, False, False]
    ).head(k)

    tiers = ["Good", "Better", "Best"]
    out = []
    for i, (_, r) in enumerate(cand.iterrows()):
        why_bits = []
        if r["net_carbs_g"] < this_net:
            why_bits.append(f"-{(this_net - r['net_carbs_g']):.0f}g net carbs")
        if r["fiber_g"] > fiber_this:
            why_bits.append(f"+{(r['fiber_g'] - fiber_this):.0f}g fiber")

        out.append({
            "tier": tiers[min(i, 2)],
            "name": r.get("name", f"Alt {i+1}"),
            "brand": r.get("brand"),
            "category": r["category"],
            "risk_score": int(r["risk_score"]),
            "risk_display": r["risk_display"],
            "why": ", ".join(why_bits) if why_bits else "Lower risk in same category"
        })

    return out


def score_item(item: dict, model, df_candidates: pd.DataFrame = None) -> dict:
    """
    Core scoring pipeline for one food item.
    
    Args:
        item: Dictionary with food nutrition data
        model: Trained calibrated classifier
        df_candidates: DataFrame of alternative products (optional)
        
    Returns:
        Dictionary with risk score, reasons, alternatives, and display data
    """
    row = item.copy()
    row["net_carbs_g"] = compute_net_carbs(row)

    # Ensure display formatting
    row = ensure_display(row)
    row = ensure_alt_group(row)

    # Get model probability → 0–100 risk score
    x = pd.DataFrame([{k: row.get(k, np.nan) for k in (NUM_COLS + CAT_COLS)}])
    prob = float(model.predict_proba(x)[0, 1])
    risk = int(round(100 * prob))

    # Generate human reasons
    pts, _ = rule_points_and_reasons(row)
    reasons = human_reasons(row)
    notes = data_notes(row)

    # Find alternatives
    alts = []
    if df_candidates is not None and not df_candidates.empty:
        # Add alt_group column if missing
        candidates = df_candidates.copy()
        if "alt_group" not in candidates.columns:
            candidates["alt_group"] = candidates["category"]

        # Score candidates
        X_cand = candidates[NUM_COLS + CAT_COLS]
        candidates["risk_prob"] = model.predict_proba(X_cand)[:, 1]
        candidates["risk_score"] = (candidates["risk_prob"] * 100).round().astype(int)
        candidates["risk_display"] = candidates["risk_prob"].apply(format_risk_display)

        alts = make_alternatives(candidates, row, risk, k=3)

    return {
        "item_name": row.get("name"),
        "item_brand": row.get("brand"),
        "item_category": row.get("category"),
        "item_alt_group": row.get("alt_group"),
        "risk_score": risk,
        "risk_display": format_risk_display(prob),
        "prob_risky": prob,
        "rule_points": pts,
        "reasons": reasons,
        "alternatives": alts,
        "display": row.get("__display", {}),
        "notes": notes,
    }


def score_by_barcode(barcode: str, model, df_candidates: pd.DataFrame = None):
    """
    Score a food item by barcode lookup (requires off_pipeline module).
    
    Args:
        barcode: UPC/EAN barcode
        model: Trained calibrated classifier
        df_candidates: DataFrame of alternative products (optional)
        
    Returns:
        Scoring result dictionary or error
    """
    try:
        from .off_pipeline import fetch_and_normalize_off, infer_alt_group_for_item, fetch_category_products, OFF_TAGS_MULTI
        
        # Fetch from Open Food Facts
        item = fetch_and_normalize_off(barcode)
        item = infer_alt_group_for_item(item)
        
        # Score with smart candidate pool selection
        pool = None
        ag = item.get("alt_group", "").lower()
        alternatives_source = "database"
        alternatives_count = 0
        
        if df_candidates is not None and not df_candidates.empty:
            if ag and "alt_group" in df_candidates.columns:
                pool = df_candidates[df_candidates["alt_group"].str.lower() == ag]
                if not pool.empty:
                    alternatives_count = len(pool)
        
        # If no candidates found in database, fetch real alternatives from OpenFoodFacts
        if pool is None or pool.empty:
            print(f"No alternatives in database for {ag}, fetching from OpenFoodFacts...")
            if ag and ag in OFF_TAGS_MULTI:
                # Fetch real alternatives for this food group
                search_tags = OFF_TAGS_MULTI.get(ag, [ag])
                real_alts = []
                for tag in search_tags[:2]:  # Try first 2 tags to get diverse results
                    try:
                        print(f"Fetching products for tag: {tag}")
                        products = fetch_category_products(tag, limit=30)
                        print(f"Fetched {len(products)} products for tag {tag}")
                        real_alts.extend(products)
                        if len(real_alts) >= 50:  # Got enough
                            break
                    except Exception as e:
                        print(f"Error fetching {tag}: {e}")
                        continue
                
                if real_alts:
                    # Filter products to ensure they match the target alt_group
                    filtered_alts = []
                    for product in real_alts:
                        product_ag = (product.get("alt_group") or "").lower()
                        product_cat = (product.get("category") or "").lower()
                        # Only include if alt_group matches exactly
                        if product_ag == ag or (not product_ag and product_cat == item.get("category", "").lower()):
                            filtered_alts.append(product)
                    
                    print(f"Filtered to {len(filtered_alts)} products matching alt_group '{ag}'")
                    
                    if filtered_alts:
                        pool = pd.DataFrame(filtered_alts)
                        # Ensure alt_group is set correctly
                        if "alt_group" not in pool.columns:
                            pool["alt_group"] = pool["category"]
                        # Force alt_group to match target
                        pool["alt_group"] = ag
                        alternatives_source = "dynamic"
                        alternatives_count = len(pool)
                        print(f"Created pool with {alternatives_count} alternatives from OpenFoodFacts")
        
        result = score_item(item, model, pool if pool is not None and not pool.empty else df_candidates)
        result["alternatives_source"] = alternatives_source
        result["alternatives_count"] = alternatives_count
        return result
        
    except ImportError:
        return {"barcode": barcode, "error": "off_pipeline module not available"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"barcode": barcode, "error": f"lookup failed: {e}"}
