"""Score one item and build recommendation output."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .explanations import data_notes, ensure_alt_group, ensure_display, format_risk_display, human_reasons
from ..model import CAT_COLS, NUM_COLS, compute_net_carbs, rule_points_and_reasons
from ..recommender import make_alternatives
from ..type_defs import ModelLike


def score_item(item: dict, model: ModelLike, df_candidates: pd.DataFrame | None = None) -> dict:
    """Score one food item and return risk, reasons, and alternatives."""
    row = item.copy()
    row["net_carbs_g"] = compute_net_carbs(row)

    row = ensure_display(row)
    row = ensure_alt_group(row)

    x = pd.DataFrame([{k: row.get(k, np.nan) for k in (NUM_COLS + CAT_COLS)}])
    prob = float(model.predict_proba(x)[0, 1])
    risk = int(round(100 * prob))

    pts, _ = rule_points_and_reasons(row)
    reasons = human_reasons(row)
    notes = data_notes(row)

    alts: list[dict] = []
    if df_candidates is not None and not df_candidates.empty:
        candidates = df_candidates.copy()
        if "alt_group" not in candidates.columns:
            candidates["alt_group"] = candidates["category"]

        has_prescore = {"risk_prob", "risk_score", "risk_display"}.issubset(candidates.columns)
        if not has_prescore:
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
