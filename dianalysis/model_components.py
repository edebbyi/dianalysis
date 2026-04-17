"""
Shared model components: feature definitions, labeling rules, and pipeline builders.

Why:
- Keep feature/labeling and model-family construction logic reusable.
- Keep `model.py` focused on training orchestration and artifact persistence.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
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


def compute_net_carbs(row: dict) -> float:
    """Net carbs approximate carbohydrate impact on glucose."""
    carbs = max(float(row.get("carbs_g", 0) or 0), 0.0)
    fiber = max(float(row.get("fiber_g", 0) or 0), 0.0)
    sugar_alc = max(float(row.get("sugar_alcohols_g", 0) or 0), 0.0)
    return max(carbs - fiber - sugar_alc, 0.0)


def rule_points_and_reasons(row: dict) -> tuple[int, list[str]]:
    """Score a row using simple nutrition-based rules."""
    pts = 0
    reasons: list[str] = []

    net = row.get("net_carbs_g", compute_net_carbs(row))
    if net > 20:
        pts += 2
        reasons.append(f"High net carbs ({net:.1f}g > 20g)")

    added = float(row.get("added_sugar_g", 0) or 0)
    if added >= 8:
        pts += 2
        reasons.append(f"High added sugar ({added:.1f}g ≥ 8g)")

    sodium = float(row.get("sodium_mg", 0) or 0)
    if sodium >= 500:
        pts += 1
        reasons.append(f"High sodium ({sodium:.0f}mg ≥ 500mg)")

    fiber = float(row.get("fiber_g", 0) or 0)
    if fiber >= 5:
        pts -= 2
        reasons.append(f"Protective fiber ({fiber:.1f}g ≥ 5g)")

    protein = float(row.get("protein_g", 0) or 0)
    if protein >= 12:
        pts -= 1
        reasons.append(f"Protein helps satiety ({protein:.1f}g ≥ 12g)")

    return pts, reasons


def weak_label(row: dict) -> int:
    """Convert rule-based score points into a binary training label."""
    pts, _ = rule_points_and_reasons(row)
    return 1 if pts >= 2 else 0


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
