"""
Human-readable explanation and display formatting helpers.

Why:
- Keep user-facing wording and nutrition display rules in one module.
- Avoid mixing presentation text logic with recommendation/model orchestration.
"""

from __future__ import annotations

from typing import Any

from ..model import compute_net_carbs
from ..model_components import (
    ADDED_SUGAR_RISK_G,
    BEVERAGE_CARBS_RISK_G,
    EMPTY_CALORIE_SUGAR_G,
    FIBER_PROTECTIVE_G,
    PROTEIN_PROTECTIVE_G,
    SODIUM_RISK_MG,
    TOTAL_SUGAR_INFERRED_RISK_G,
    TOTAL_CARBS_RISK_G,
    _is_beverage_row,
    _supports_inferred_sugar_rule,
)
from ..recommendation.candidate_pool import ensure_row_group_fields


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
    """Ensure item has a `__display` dict for package-label style formatting."""
    disp = row.get("__display")
    if isinstance(disp, dict) and disp:
        return row

    new_disp = {}
    for field, rule in DISPLAY_RULES.items():
        new_disp[field] = _format_display_value(row.get(field), rule["unit"])
    row["__display"] = new_disp
    return row


def ensure_alt_group(row: dict) -> dict:
    """Ensure group fields exist for retrieval and ranking."""
    return ensure_row_group_fields(row)


def human_reasons(row: dict) -> list[str]:
    """
    Generate human-readable reasons for the risk score.
    Uses numeric thresholds, but shows package-label style formatting from `__display`.
    """
    reasons: list[str] = []
    disp = row.get("__display", {}) or {}

    carbs_v = row.get("carbs_g")
    carbs_txt = disp.get("carbs_g") or ("not listed" if carbs_v is None else f"{float(carbs_v):.1f}g")
    carbs_threshold = BEVERAGE_CARBS_RISK_G if _is_beverage_row(row) else TOTAL_CARBS_RISK_G
    if carbs_v is None:
        reasons.append("Total carbs not listed")
    elif float(carbs_v) >= carbs_threshold:
        reasons.append(f"High total carbs ({carbs_txt} ≥ {carbs_threshold:.0f}g)")
    else:
        reasons.append(f"Total carbs below high threshold ({carbs_txt} < {carbs_threshold:.0f}g)")

    sugar_v = row.get("sugar_g")
    sugar_txt = disp.get("sugar_g") or ("not listed" if sugar_v is None else f"{float(sugar_v):.1f}g")

    added_v = row.get("added_sugar_g")
    added_txt = disp.get("added_sugar_g") or ("not listed" if added_v is None else f"{float(added_v):.1f}g")
    if added_v is None:
        if _supports_inferred_sugar_rule(row) and sugar_v is not None and float(sugar_v) >= TOTAL_SUGAR_INFERRED_RISK_G:
            reasons.append(
                f"Added sugar not listed; inferred risk from total sugar ({sugar_txt} ≥ {TOTAL_SUGAR_INFERRED_RISK_G:.0f}g)"
            )
        else:
            reasons.append("Added sugar not listed")
    elif float(added_v) >= ADDED_SUGAR_RISK_G:
        reasons.append(f"High added sugar ({added_txt} ≥ {ADDED_SUGAR_RISK_G:.0f}g)")
    else:
        # When added sugar is low but total sugar is high, show total sugar as the context signal.
        if sugar_v is not None and float(sugar_v) >= 20.0:
            reasons.append(f"Total sugar ({sugar_txt})")

    sodium_v = row.get("sodium_mg")
    sodium_txt = disp.get("sodium_mg") or ("not listed" if sodium_v is None else f"{int(round(float(sodium_v)))}mg")
    if sodium_v is None:
        reasons.append("Sodium not listed")
    else:
        s = float(sodium_v)
        if s >= SODIUM_RISK_MG:
            reasons.append(f"High sodium ({sodium_txt} ≥ {SODIUM_RISK_MG:.0f}mg)")
        elif s <= 140:
            reasons.append(f"Low sodium ({sodium_txt} ≤ 140mg)")
        else:
            reasons.append(f"Moderate sodium ({sodium_txt} < {SODIUM_RISK_MG:.0f}mg)")

    fiber_v = row.get("fiber_g")
    fiber_txt = disp.get("fiber_g") or ("not listed" if fiber_v is None else f"{float(fiber_v):.1f}g")
    if fiber_v is None:
        reasons.append("Fiber not listed (some labels show '<1g' for trace amounts)")
    elif float(fiber_v) >= FIBER_PROTECTIVE_G:
        reasons.append(f"Good fiber ({fiber_txt} ≥ {FIBER_PROTECTIVE_G:.1f}g)")
    else:
        reasons.append(f"Low fiber ({fiber_txt} < {FIBER_PROTECTIVE_G:.1f}g)")

    protein_v = row.get("protein_g")
    protein_txt = disp.get("protein_g") or ("not listed" if protein_v is None else f"{float(protein_v):.1f}g")
    if protein_v is None:
        reasons.append("Protein not listed")
    elif float(protein_v) >= PROTEIN_PROTECTIVE_G:
        reasons.append(f"Higher protein ({protein_txt} ≥ {PROTEIN_PROTECTIVE_G:.0f}g)")
    else:
        reasons.append(f"Moderate protein ({protein_txt} < {PROTEIN_PROTECTIVE_G:.0f}g)")

    # Explain empty-calorie pattern when sugar is high and buffers are low.
    fat_v = row.get("fat_g")
    sugar_num = None if sugar_v is None else float(sugar_v)
    fiber_num = 0.0 if fiber_v is None else float(fiber_v)
    protein_num = 0.0 if protein_v is None else float(protein_v)
    fat_num = 0.0 if fat_v is None else float(fat_v)
    if (
        sugar_num is not None
        and sugar_num > EMPTY_CALORIE_SUGAR_G
        and fiber_num <= 0.1
        and protein_num <= 0.1
        and fat_num <= 1.0
    ):
        reasons.append("High sugar with little or no fiber/protein/fat buffering")

    # Net carbs are a model feature and context signal, not a primary label trigger.
    net = row.get("net_carbs_g", compute_net_carbs(row))
    reasons.append(f"Net carbs = {float(net):.1f}g (estimated)")

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

    # Clarify a common source-data mismatch for barcode lookups.
    try:
        total_sugar = row.get("sugar_g")
        added_sugar = row.get("added_sugar_g")
        if total_sugar is not None and added_sugar is not None:
            if float(total_sugar) >= 20.0 and float(added_sugar) == 0.0:
                notes.append(
                    "Added sugar is reported as 0g in the source, while total sugar is high. "
                    "This can happen when source data is incomplete for added sugar."
                )
    except Exception:
        pass

    notes.append(
        "Net carbs estimate uses: carbs - fiber - sugar alcohols. "
        "If a value is missing, it is treated as 0 for this estimate."
    )
    return notes
