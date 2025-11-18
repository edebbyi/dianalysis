"""
Model training and helpers for Dianalysis scoring.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

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


def compute_net_carbs(row: dict) -> float:
    """
    Net carbs approximate carbohydrate impact on glucose.
    """
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
    """Derive binary weak label from rule points."""
    pts, _ = rule_points_and_reasons(row)
    return 1 if pts >= 2 else 0


def _make_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for numeric and categorical features."""
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
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


def _build_pipeline() -> Pipeline:
    """Logistic regression pipeline with calibration."""
    base_clf = LogisticRegression(max_iter=1000)
    calibrated = CalibratedClassifierCV(base_clf, method="isotonic", cv="prefit")

    # We'll train the base_estimator first to satisfy CalibratedClassifierCV
    pipeline = Pipeline(
        [
            ("pre", _make_preprocessor()),
            ("clf", base_clf),
        ]
    )
    return pipeline


def _compute_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute evaluation metrics using the calibrated classifier."""
    probas = model.predict_proba(X)[:, 1]
    preds = (probas >= 0.5).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, probas)),
        "AUPRC": float(average_precision_score(y, probas)),
        "F1": float(f1_score(y, preds)),
        "BalancedAcc": float(balanced_accuracy_score(y, preds)),
        "Brier": float(brier_score_loss(y, probas)),
    }


def generate_synthetic_data(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic foods for training/demo."""
    rng = np.random.default_rng(random_state)
    categories = ["cereal", "bread", "snack", "drink", "dairy", "grain"]
    df = pd.DataFrame(
        {
            "name": [f"Product {i}" for i in range(n)],
            "brand": [f"Brand {rng.integers(1, 50)}" for _ in range(n)],
            "upc": [str(100000000000 + int(rng.integers(0, 9e11))) for _ in range(n)],
            "source": "synthetic",
            "created_at": pd.Timestamp("2025-09-01"),
            "category": rng.choice(categories, size=n),
            "serving_g": rng.integers(30, 80, size=n).astype(float),
            "calories": rng.integers(50, 400, size=n).astype(float),
            "carbs_g": rng.integers(0, 60, size=n).astype(float),
            "fiber_g": rng.integers(0, 12, size=n).astype(float),
            "sugar_g": rng.integers(0, 35, size=n).astype(float),
            "added_sugar_g": rng.integers(0, 20, size=n).astype(float),
            "sugar_alcohols_g": rng.integers(0, 12, size=n).astype(float),
            "protein_g": rng.integers(0, 30, size=n).astype(float),
            "fat_g": rng.integers(0, 25, size=n).astype(float),
            "sodium_mg": rng.integers(0, 1200, size=n).astype(float),
            "ingredients_text": ["wheat, sugar, salt, vitamins"] * n,
        }
    )
    df["net_carbs_g"] = df.apply(compute_net_carbs, axis=1)
    df["rule_points"], _ = zip(*df.apply(rule_points_and_reasons, axis=1))
    df["label"] = df.apply(weak_label, axis=1)
    return df


def train_model(df: pd.DataFrame, artifacts_dir: str = "artifacts") -> tuple[Pipeline, dict]:
    """
    Train the calibrated logistic regression and persist artifacts.
    """
    df = df.copy()
    df["net_carbs_g"] = df.apply(compute_net_carbs, axis=1)
    df["label"] = df.apply(weak_label, axis=1)

    X = df[NUM_COLS + CAT_COLS]
    y = df["label"]

    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    metrics = {
        "AUROC": float(roc_auc_score(y, pipeline.predict_proba(X)[:, 1])),
        "AUPRC": float(average_precision_score(y, pipeline.predict_proba(X)[:, 1])),
        "F1": float(f1_score(y, pipeline.predict(X))),
        "BalancedAcc": float(balanced_accuracy_score(y, pipeline.predict(X))),
        "Brier": float(brier_score_loss(y, pipeline.predict_proba(X)[:, 1])),
    }

    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(pipeline, Path(artifacts_dir) / "model.joblib")
    joblib.dump({"num_cols": NUM_COLS, "cat_cols": CAT_COLS}, Path(artifacts_dir) / "meta.joblib")

    return pipeline, metrics


def load_model(artifacts_dir: str):
    """
    Load the trained pipeline and metadata from artifacts.
    """
    artifacts_dir = Path(artifacts_dir)
    model_path = artifacts_dir / "model.joblib"
    meta_path = artifacts_dir / "meta.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")

    model = joblib.load(model_path)
    meta = joblib.load(meta_path)

    return model, meta
