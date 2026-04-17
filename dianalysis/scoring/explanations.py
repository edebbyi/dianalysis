"""
Human-readable explanation and display formatting helpers.

Why:
- Keep user-facing wording and nutrition display rules in one module.
- Avoid mixing presentation text logic with recommendation/model orchestration.
"""

from __future__ import annotations

from typing import Any

from ..model import compute_net_carbs


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


def _format_display_value(v: Any, unit: str) -> str:
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
    """Ensure item has a `__display` dict for label-faithful formatting."""
    disp = row.get("__display")
    if isinstance(disp, dict) and disp:
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


def human_reasons(row: dict) -> list[str]:
    """
    Generate human-readable reasons for the risk score.
    Uses numeric thresholds, but shows label-faithful strings from __display.
    """
    reasons: list[str] = []
    disp = row.get("__display", {}) or {}

    net = row.get("net_carbs_g", compute_net_carbs(row))
    reasons.append(
        f"High net carbs ({net:.1f}g > 20g)" if net > 20 else f"Net carbs within target ({net:.1f}g ≤ 20g)"
    )

    added_v = row.get("added_sugar_g")
    added_txt = disp.get("added_sugar_g") or ("not listed" if added_v is None else f"{float(added_v):.1f}g")
    if added_v is None:
        reasons.append("Added sugar not listed")
    elif float(added_v) >= 8:
        reasons.append(f"High added sugar ({added_txt} ≥ 8g)")
    else:
        reasons.append(f"Low added sugar ({added_txt})")

    sodium_v = row.get("sodium_mg")
    sodium_txt = disp.get("sodium_mg") or ("not listed" if sodium_v is None else f"{int(round(float(sodium_v)))}mg")
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

    fiber_v = row.get("fiber_g")
    fiber_txt = disp.get("fiber_g") or ("not listed" if fiber_v is None else f"{float(fiber_v):.1f}g")
    if fiber_v is None:
        reasons.append("Fiber not listed (some labels show '<1g' for trace amounts)")
    elif float(fiber_v) >= 5:
        reasons.append(f"Good fiber ({fiber_txt} ≥ 5g)")
    else:
        reasons.append(f"Low fiber ({fiber_txt} < 5g)")

    protein_v = row.get("protein_g")
    protein_txt = disp.get("protein_g") or ("not listed" if protein_v is None else f"{float(protein_v):.1f}g")
    if protein_v is None:
        reasons.append("Protein not listed")
    elif float(protein_v) >= 12:
        reasons.append(f"Higher protein ({protein_txt} ≥ 12g)")
    else:
        reasons.append(f"Moderate protein ({protein_txt} < 12g)")

    return reasons


def data_notes(row: dict) -> list[str]:
    """Generate notes explaining 'not listed' or inferred values."""
    disp = row.get("__display", {}) or {}
    notes: list[str] = []
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
