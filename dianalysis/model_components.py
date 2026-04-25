"""
Shared model components: feature definitions, labeling rules, and pipeline builders.

Why:
- Keep feature/labeling and model-family construction logic reusable.
- Keep `model.py` focused on training orchestration and artifact persistence.
"""

from __future__ import annotations

from typing import Any, Literal

import math

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = [
    "serving_g",
    "calories",
    "carbs_g",
    "fiber_g",
    "sugar_g",
    "added_sugar_g",
    "sugar_alcohols_g",
    "protein_g",
    "fat_g",
    "sodium_mg",
    "net_carbs_g",
]
CAT_COLS = ["category"]
ModelType = Literal["logreg", "xgboost"]

# Balanced rule-set thresholds (per serving)
TOTAL_CARBS_RISK_G = 30.0
BEVERAGE_CARBS_RISK_G = 20.0
ADDED_SUGAR_RISK_G = 10.0
EXTREME_ADDED_SUGAR_G = 25.0
TOTAL_SUGAR_INFERRED_RISK_G = 10.0
SODIUM_RISK_MG = 460.0
FIBER_PROTECTIVE_G = 5.6
PROTEIN_PROTECTIVE_G = 10.0
EMPTY_CALORIE_SUGAR_G = 10.0
LABEL_POSITIVE_THRESHOLD = 2
NUTRIENT_VALUE_MARGIN_G = 1.0

INFERRED_SUGAR_CATEGORIES = {"drink", "snack", "dessert"}


def _to_float_or_none(val: Any) -> float | None:
    """Return float value or None when value is missing/unusable."""
    if val is None:
        return None
    try:
        out = float(val)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def _is_beverage_row(row: dict[str, Any]) -> bool:
    """Return True when row should use beverage-specific carb threshold."""
    category_main = str(row.get("category_main", "") or "").strip().lower()
    category = str(row.get("category", "") or "").strip().lower()
    alt_group = str(row.get("alt_group", "") or "").strip().lower()
    fine_group = str(row.get("alt_group_fine", "") or "").strip().lower()
    return (
        category_main == "drink"
        or category == "drink"
        or alt_group == "drink"
        or fine_group.startswith("drink:")
    )


def _supports_inferred_sugar_rule(row: dict[str, Any]) -> bool:
    """Return True when missing added sugar can be inferred from total sugar."""
    category_main = str(row.get("category_main", "") or "").strip().lower()
    category = str(row.get("category", "") or "").strip().lower()
    alt_group = str(row.get("alt_group", "") or "").strip().lower()
    return (
        category_main in INFERRED_SUGAR_CATEGORIES
        or category in INFERRED_SUGAR_CATEGORIES
        or alt_group in {"drink", "snack", "ice-cream"}
    )


def rule_data_confidence(row: dict[str, Any]) -> tuple[str, list[str]]:
    """
    Return data confidence level for risk interpretation.

    `low` means one or more critical risk fields are missing.
    """
    notes: list[str] = []
    carbs = _to_float_or_none(row.get("carbs_g"))
    sugar = _to_float_or_none(row.get("sugar_g"))
    added = _to_float_or_none(row.get("added_sugar_g"))
    sodium = _to_float_or_none(row.get("sodium_mg"))
    fiber = _to_float_or_none(row.get("fiber_g"))
    protein = _to_float_or_none(row.get("protein_g"))

    if carbs is None:
        notes.append("Total carbs missing")
    if sugar is None:
        notes.append("Total sugar missing")
    if sodium is None:
        notes.append("Sodium missing")
    if carbs is not None and sugar is not None and sugar > (carbs + NUTRIENT_VALUE_MARGIN_G):
        notes.append("Total sugar exceeds total carbs (possible source-data error)")
    if carbs is not None and added is not None and added > (carbs + NUTRIENT_VALUE_MARGIN_G):
        notes.append("Added sugar exceeds total carbs (possible source-data error)")
    if sugar is not None and added is not None and added > (sugar + NUTRIENT_VALUE_MARGIN_G):
        notes.append("Added sugar exceeds total sugar (possible source-data error)")
    if _supports_inferred_sugar_rule(row) and added is None:
        notes.append("Added sugar missing in processed-food category")
    if sugar is not None and sugar >= TOTAL_SUGAR_INFERRED_RISK_G:
        if fiber is None:
            notes.append("Fiber missing with high sugar")
        if protein is None:
            notes.append("Protein missing with high sugar")
    return ("low", notes) if notes else ("high", [])


def rule_points_reasons_meta(row: dict[str, Any]) -> tuple[int, list[str], dict[str, Any]]:
    """Score a row using nutrition rules and return points, reasons, and metadata."""
    pts = 0
    reasons: list[str] = []
    meta: dict[str, Any] = {
        "beverage_threshold_used": False,
        "inferred_added_sugar": False,
        "empty_calorie_penalty": False,
        "extreme_added_sugar": False,
        "implausible_sugar_values": False,
    }

    confidence, confidence_notes = rule_data_confidence(row)
    meta["data_confidence"] = confidence
    meta["confidence_notes"] = confidence_notes

    carbs = _to_float_or_none(row.get("carbs_g"))
    carbs_threshold = BEVERAGE_CARBS_RISK_G if _is_beverage_row(row) else TOTAL_CARBS_RISK_G
    if carbs_threshold == BEVERAGE_CARBS_RISK_G:
        meta["beverage_threshold_used"] = True
    if carbs is not None and carbs >= carbs_threshold:
        pts += 2
        reasons.append(f"High total carbs ({carbs:.1f}g ≥ {carbs_threshold:.0f}g)")

    sugar = _to_float_or_none(row.get("sugar_g"))
    added = _to_float_or_none(row.get("added_sugar_g"))
    high_added_sugar = False
    if added is not None:
        if added >= ADDED_SUGAR_RISK_G:
            pts += 2
            reasons.append(f"High added sugar ({added:.1f}g ≥ {ADDED_SUGAR_RISK_G:.0f}g)")
            high_added_sugar = True
        if added >= EXTREME_ADDED_SUGAR_G:
            pts += 2
            meta["extreme_added_sugar"] = True
            reasons.append(f"Very high added sugar ({added:.1f}g ≥ {EXTREME_ADDED_SUGAR_G:.0f}g)")
    elif _supports_inferred_sugar_rule(row) and sugar is not None and sugar >= TOTAL_SUGAR_INFERRED_RISK_G:
        pts += 2
        meta["inferred_added_sugar"] = True
        reasons.append(
            f"Added sugar not listed; inferred risk from total sugar ({sugar:.1f}g ≥ {TOTAL_SUGAR_INFERRED_RISK_G:.0f}g)"
        )

    implausible_sugar_values = (
        (carbs is not None and sugar is not None and sugar > (carbs + NUTRIENT_VALUE_MARGIN_G))
        or (carbs is not None and added is not None and added > (carbs + NUTRIENT_VALUE_MARGIN_G))
        or (sugar is not None and added is not None and added > (sugar + NUTRIENT_VALUE_MARGIN_G))
    )
    if implausible_sugar_values:
        pts += 1
        meta["implausible_sugar_values"] = True
        reasons.append("Sugar values are internally inconsistent; risk raised pending data verification")

    sodium = _to_float_or_none(row.get("sodium_mg"))
    if sodium is not None and sodium >= SODIUM_RISK_MG:
        pts += 1
        reasons.append(f"High sodium ({sodium:.0f}mg ≥ {SODIUM_RISK_MG:.0f}mg)")

    fiber = _to_float_or_none(row.get("fiber_g"))
    if fiber is not None and fiber >= FIBER_PROTECTIVE_G:
        if high_added_sugar:
            pts -= 1
            reasons.append(
                f"Fiber benefit is limited when added sugar is high ({fiber:.1f}g ≥ {FIBER_PROTECTIVE_G:.1f}g)"
            )
        else:
            pts -= 2
            reasons.append(f"Protective fiber ({fiber:.1f}g ≥ {FIBER_PROTECTIVE_G:.1f}g)")

    protein = _to_float_or_none(row.get("protein_g"))
    if protein is not None and protein >= PROTEIN_PROTECTIVE_G:
        if high_added_sugar:
            reasons.append(
                f"Protein present ({protein:.1f}g), but high added sugar keeps risk elevated"
            )
        else:
            pts -= 1
            reasons.append(f"Protein helps satiety ({protein:.1f}g ≥ {PROTEIN_PROTECTIVE_G:.0f}g)")

    fat = _to_float_or_none(row.get("fat_g"))
    fiber_for_penalty = 0.0 if fiber is None else max(fiber, 0.0)
    protein_for_penalty = 0.0 if protein is None else max(protein, 0.0)
    fat_for_penalty = 0.0 if fat is None else max(fat, 0.0)
    if (
        sugar is not None
        and sugar > EMPTY_CALORIE_SUGAR_G
        and fiber_for_penalty <= 0.1
        and protein_for_penalty <= 0.1
        and fat_for_penalty <= 1.0
    ):
        pts += 1
        meta["empty_calorie_penalty"] = True
        reasons.append("High sugar with little or no fiber/protein/fat buffering")

    return pts, reasons, meta


def compute_net_carbs(row: dict) -> float:
    """Net carbs approximate carbohydrate impact on glucose."""
    carbs = max(float(row.get("carbs_g", 0) or 0), 0.0)
    fiber = max(float(row.get("fiber_g", 0) or 0), 0.0)
    sugar_alc = max(float(row.get("sugar_alcohols_g", 0) or 0), 0.0)
    return max(carbs - fiber - sugar_alc, 0.0)


def rule_points_and_reasons(row: dict) -> tuple[int, list[str]]:
    """Score a row using simple nutrition-based rules."""
    pts, reasons, _ = rule_points_reasons_meta(row)
    return pts, reasons


def weak_label(row: dict) -> int:
    """Convert rule-based score points into a binary training label."""
    pts, _ = rule_points_and_reasons(row)
    return 1 if pts >= LABEL_POSITIVE_THRESHOLD else 0


def make_preprocessor(add_indicator: bool = True) -> ColumnTransformer:
    """ColumnTransformer for numeric and categorical features."""
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
            ("scale", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipeline, NUM_COLS),
            ("cat", cat_pipeline, CAT_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(
    *,
    model_type: ModelType = "logreg",
    class_weight: str | dict | None = None,
    C: float = 1.0,
    max_iter: int = 1000,
    add_indicator: bool = True,
    random_state: int = 42,
    xgb_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build model pipeline with configurable preprocessing and classifier family."""
    if model_type == "logreg":
        base_clf = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            C=C,
            random_state=random_state,
        )
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:  #Keep training from failing if xgboost is not installed
            raise ImportError(
                "model_type='xgboost' requires the 'xgboost' package. "
                "Install it with `pip install xgboost` (or via requirements)."
            ) from e

        xgb_cfg = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": random_state,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        if xgb_params:
            xgb_cfg.update(xgb_params)
        if class_weight == "balanced" and "scale_pos_weight" not in xgb_cfg:
            xgb_cfg["scale_pos_weight"] = 2.0
        base_clf = XGBClassifier(**xgb_cfg)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(
        [
            ("pre", make_preprocessor(add_indicator=add_indicator)),
            ("clf", base_clf),
        ]
    )
