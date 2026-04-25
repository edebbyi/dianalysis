"""Score one item and build recommendation output."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .explanations import data_notes, ensure_alt_group, ensure_display, format_risk_display, human_reasons
from ..model import CAT_COLS, NUM_COLS, compute_net_carbs
from ..model_components import (
    ADDED_SUGAR_RISK_G,
    BEVERAGE_CARBS_RISK_G,
    SODIUM_RISK_MG,
    TOTAL_CARBS_RISK_G,
    TOTAL_SUGAR_INFERRED_RISK_G,
    _is_beverage_row,
    _supports_inferred_sugar_rule,
    rule_points_reasons_meta,
)
from ..recommender import make_alternatives
from ..type_defs import ModelLike


CARB_ONLY_DISPLAY_CAP = 85


def _enforce_lower_risk_alternatives(alts: list[dict], current_risk: int, k: int = 3) -> list[dict]:
    """
    Keep only alternatives with strictly lower risk than the current item.

    This guard prevents UI cases like "risk: +2" where an alternative is
    accidentally worse than the item being scored.
    """
    valid = []
    for alt in alts:
        try:
            alt_risk = float(alt.get("risk_score", 100))
        except Exception:
            continue
        if alt_risk < float(current_risk):
            valid.append(alt)

    # Always show best-first by risk, then by net carbs, then by higher fiber.
    def _safe_float(val: object, default: float) -> float:
        if val is None:
            return default
        try:
            if isinstance(val, (int, float, str)):
                return float(val)
        except (TypeError, ValueError):
            return default
        return default

    valid = sorted(
        valid,
        key=lambda a: (
            _safe_float(a.get("risk_score"), 100.0),
            _safe_float(a.get("net_carbs_g"), 1e9),
            -_safe_float(a.get("fiber_g"), 0.0),
        ),
    )
    return valid[:k]


def _should_recompute_candidate_scores(candidates: pd.DataFrame) -> bool:
    """
    Decide whether candidate rows should be rescored at runtime.

    Runtime rescoring is used as a safety fallback only when:
    - score columns are missing, or
    - a promoted model fingerprint is set and candidate fingerprint metadata
      is missing/mismatched.
    """
    has_prescore = {"risk_prob", "risk_score", "risk_display"}.issubset(candidates.columns)
    if not has_prescore:
        return True

    expected_fingerprint = str(os.getenv("DIANALYSIS_MODEL_FINGERPRINT", "") or "").strip()
    if not expected_fingerprint:
        return False

    if "model_fingerprint" not in candidates.columns:
        return True

    seen = {
        str(v).strip()
        for v in candidates["model_fingerprint"].fillna("").astype(str).tolist()
        if str(v).strip()
    }
    return seen != {expected_fingerprint}


def _to_float_or_none(val: object) -> float | None:
    """Parse a numeric value or return None when missing/unusable."""
    if val is None:
        return None
    try:
        if isinstance(val, (int, float, str)):
            out = float(val)
        else:
            return None
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return out


def _display_score_for_item(row: dict, *, rule_points: int, rule_meta: dict[str, object], risk_score_raw: int) -> tuple[int, bool]:
    """
    Return display score and whether a display cap was applied.

    Why:
    - Keep ranking/filtering logic based on the raw model score.
    - Keep user-facing score less extreme for carb-only positives
      (no high added sugar signal and no high sodium signal).
    """
    if risk_score_raw <= CARB_ONLY_DISPLAY_CAP or rule_points < 2:
        return risk_score_raw, False

    carbs = _to_float_or_none(row.get("carbs_g"))
    sugar = _to_float_or_none(row.get("sugar_g"))
    added = _to_float_or_none(row.get("added_sugar_g"))
    sodium = _to_float_or_none(row.get("sodium_mg"))

    carbs_threshold = BEVERAGE_CARBS_RISK_G if _is_beverage_row(row) else TOTAL_CARBS_RISK_G
    carb_triggered = carbs is not None and carbs >= carbs_threshold

    sugar_triggered = (added is not None and added >= ADDED_SUGAR_RISK_G) or (
        added is None and _supports_inferred_sugar_rule(row) and sugar is not None and sugar >= TOTAL_SUGAR_INFERRED_RISK_G
    )
    # If total sugar is high in processed-food categories, do not treat the item
    # as "carb-only" for display capping, even when added sugar is reported as 0.
    if _supports_inferred_sugar_rule(row) and sugar is not None and sugar >= TOTAL_SUGAR_INFERRED_RISK_G:
        sugar_triggered = True
    if bool(rule_meta.get("inferred_added_sugar")):
        sugar_triggered = True

    sodium_triggered = sodium is not None and sodium >= SODIUM_RISK_MG

    if carb_triggered and not sugar_triggered and not sodium_triggered:
        return CARB_ONLY_DISPLAY_CAP, True
    return risk_score_raw, False


def score_item(item: dict, model: ModelLike, df_candidates: pd.DataFrame | None = None) -> dict:
    """Score one food item and return risk, reasons, and alternatives."""
    row = item.copy()
    row["net_carbs_g"] = compute_net_carbs(row)

    row = ensure_display(row)
    row = ensure_alt_group(row)

    x = pd.DataFrame([{k: row.get(k, np.nan) for k in (NUM_COLS + CAT_COLS)}])
    prob = float(model.predict_proba(x)[0, 1])
    risk = int(round(100 * prob))

    pts, _rule_reasons, rule_meta = rule_points_reasons_meta(row)
    data_confidence = str(rule_meta.get("data_confidence", "high") or "high")
    confidence_notes = list(rule_meta.get("confidence_notes", []) or [])

    # Guardrail: do not let inferred high-sugar/empty-calorie cases appear as low-risk.
    if bool(rule_meta.get("inferred_added_sugar")) or bool(rule_meta.get("empty_calorie_penalty")):
        if pts >= 4:
            floor = 85
        elif pts >= 3:
            floor = 70
        elif pts >= 2:
            floor = 55
        elif pts >= 1:
            floor = 25
        else:
            floor = 0
        if floor > risk:
            risk = floor
            prob = risk / 100.0

    risk_score_display, display_cap_applied = _display_score_for_item(
        row,
        rule_points=pts,
        rule_meta=rule_meta,
        risk_score_raw=risk,
    )

    reasons = human_reasons(row)
    notes = data_notes(row)
    if data_confidence == "low":
        notes.append("Data confidence: low. One or more critical nutrition fields were missing in source data.")
        for c_note in confidence_notes:
            notes.append(f"Confidence note: {c_note}.")
    if display_cap_applied:
        notes.append(
            "Displayed risk score is capped for carb-heavy items without high added sugar or high sodium."
        )

    alts: list[dict] = []
    if df_candidates is not None and not df_candidates.empty:
        candidates = df_candidates.copy()
        if "alt_group" not in candidates.columns:
            candidates["alt_group"] = candidates["category"]

        if _should_recompute_candidate_scores(candidates):
            for col in (NUM_COLS + CAT_COLS):
                if col not in candidates.columns:
                    candidates[col] = np.nan
            X_cand = candidates[NUM_COLS + CAT_COLS]
            candidates["risk_prob"] = model.predict_proba(X_cand)[:, 1]
            candidates["risk_score"] = (candidates["risk_prob"] * 100).round().astype(int)
            candidates["risk_display"] = candidates["risk_prob"].apply(format_risk_display)

        alts = make_alternatives(candidates, row, risk, k=3)
        alts = _enforce_lower_risk_alternatives(alts, current_risk=risk, k=3)

    return {
        "item_name": row.get("name"),
        "item_brand": row.get("brand"),
        "item_category": row.get("category"),
        "item_category_main": row.get("category_main", row.get("category")),
        "item_alt_group": row.get("alt_group"),
        "item_alt_group_fine": row.get("alt_group_fine"),
        "item_net_carbs_g": float(row.get("net_carbs_g", 0.0) or 0.0),
        "item_fiber_g": float(row.get("fiber_g", 0.0) or 0.0),
        "risk_score": risk,
        "risk_score_display": risk_score_display,
        "risk_display": format_risk_display(risk_score_display / 100.0),
        "display_cap_applied": display_cap_applied,
        "prob_risky": prob,
        "data_confidence": data_confidence,
        "data_confidence_notes": confidence_notes,
        "rule_points": pts,
        "reasons": reasons,
        "alternatives": alts,
        "display": row.get("__display", {}),
        "notes": notes,
    }
