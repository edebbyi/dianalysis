"""
Model training, preprocessing, and artifact I/O.

Why:
- Centralize how the classifier is built, trained, evaluated, and persisted.
- Provide stable model interfaces used by app, experiments, and notebook code.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from .model_components import (
    CAT_COLS,
    NUM_COLS,
    ModelType,
    build_pipeline,
    compute_net_carbs,
    rule_points_and_reasons,
    weak_label,
)

class XGBServingModel:
    """Serve predictions by applying a fitted preprocessor then a native XGBoost booster."""

    def __init__(self, preprocessor: Any, booster: Any) -> None:
        self.preprocessor = preprocessor
        self.booster = booster

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities in sklearn-style `[p0, p1]` format."""
        try:
            import xgboost as xgb
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "XGBoost serving artifact requires the 'xgboost' package at inference time."
            ) from e

        X_t = self.preprocessor.transform(X)
        dmat = xgb.DMatrix(X_t)
        p1 = np.asarray(self.booster.predict(dmat), dtype=float).reshape(-1)
        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])


def _compute_metrics_from_probas(y_true: pd.Series, probas: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute core classification metrics from probabilities."""
    preds = (probas >= threshold).astype(int)
    return {
        "F1": float(f1_score(y_true, preds, zero_division=0)),
        "AUPRC": float(average_precision_score(y_true, probas)),
        "Brier": float(brier_score_loss(y_true, probas)),
    }


def _compute_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute core classification metrics from a fitted model."""
    probas = model.predict_proba(X)[:, 1]
    return _compute_metrics_from_probas(y, probas)


def generate_synthetic_data(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic foods for training/demo."""
    rng = np.random.default_rng(random_state)
    categories = ["cereal", "bread", "snack", "drink", "dairy", "grain"]
    df = pd.DataFrame(
        {
            "name": [f"Product {i}" for i in range(n)],
            "brand": [f"Brand {rng.integers(1, 50)}" for _ in range(n)],
            "upc": [str(100000000000 + int(rng.integers(0, 900_000_000_000))) for _ in range(n)],
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


def train_model(
    df: pd.DataFrame,
    artifacts_dir: str = "artifacts",
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
    cv_folds: int = 5,
    model_type: ModelType = "logreg",
    class_weight: str | dict | None = None,
    C: float = 1.0,
    add_indicator: bool = True,
    xgb_params: dict[str, Any] | None = None,
) -> tuple[Pipeline, dict]:
    """
    Train model with CV and holdout diagnostics.
    """
    df = df.copy()
    df["net_carbs_g"] = df.apply(compute_net_carbs, axis=1)
    df["label"] = df.apply(weak_label, axis=1)

    X = df[NUM_COLS + CAT_COLS].copy()
    y = df["label"].astype(int).copy()

    # Split data once to keep a fully untouched holdout test set.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Split train+val again to create a validation set for model-selection diagnostics.
    val_ratio_in_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio_in_trainval,
        stratify=y_trainval,
        random_state=random_state,
    )

    # Run CV on the training split to estimate metric stability.
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_metrics: list[dict] = []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        fold_model = build_pipeline(
            model_type=model_type,
            class_weight=class_weight,
            C=C,
            add_indicator=add_indicator,
            random_state=random_state,
            xgb_params=xgb_params,
        )
        fold_model.fit(X_tr, y_tr)
        fold_probas = fold_model.predict_proba(X_va)[:, 1]
        fold_metrics.append(_compute_metrics_from_probas(y_va, fold_probas))

    cv_mean = {
        metric: float(np.mean([m[metric] for m in fold_metrics]))
        for metric in ("F1", "AUPRC", "Brier")
    }
    cv_std = {
        metric: float(np.std([m[metric] for m in fold_metrics], ddof=1))
        for metric in ("F1", "AUPRC", "Brier")
    }

    # Fit on training data and evaluate on validation data.
    validation_model = build_pipeline(
        model_type=model_type,
        class_weight=class_weight,
        C=C,
        add_indicator=add_indicator,
        random_state=random_state,
        xgb_params=xgb_params,
    )
    validation_model.fit(X_train, y_train)
    validation_metrics = _compute_metrics(validation_model, X_val, y_val)

    # Fit final model on train+val and evaluate once on untouched test data.
    pipeline = build_pipeline(
        model_type=model_type,
        class_weight=class_weight,
        C=C,
        add_indicator=add_indicator,
        random_state=random_state,
        xgb_params=xgb_params,
    )
    pipeline.fit(X_trainval, y_trainval)
    test_metrics = _compute_metrics(pipeline, X_test, y_test)

    metrics = {
        "cv": {"mean": cv_mean, "std": cv_std, "folds": cv_folds},
        "validation": validation_metrics,
        "test": test_metrics,
    }

    try:
        # Import this here (not at file top) to prevent a circular import:
        # recommendation_eval -> scoring -> model.
        from .recommendation_eval import compute_recommendation_eval

        recommendation_eval = compute_recommendation_eval(
            pipeline,
            df,
            sample_size=120,
            k=3,
            random_state=random_state,
        )
    except Exception as e:  # pragma: no cover - Keep training running if recommendation eval fails.
        recommendation_eval = {
            "error": f"recommendation eval failed: {e}",
            "evaluated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }

    os.makedirs(artifacts_dir, exist_ok=True)
    artifacts_path = Path(artifacts_dir)
    if model_type == "xgboost":
        clf = pipeline.named_steps["clf"]
        pre = pipeline.named_steps["pre"]
        try:
            booster = clf.get_booster()
        except Exception as e:  # pragma: no cover - defensive guard
            raise RuntimeError("Expected fitted XGBoost classifier with accessible booster.") from e
        joblib.dump(pre, artifacts_path / "preprocessor.joblib")
        booster.save_model(str(artifacts_path / "xgb_model.json"))
    else:
        joblib.dump(pipeline, artifacts_path / "model.joblib")

    joblib.dump(
        {
            "num_cols": NUM_COLS,
            "cat_cols": CAT_COLS,
            "metrics": metrics,
            "model_type": model_type,
            "class_weight": class_weight,
            "C": C,
            "add_indicator": add_indicator,
            "xgb_params": xgb_params or {},
            "recommendation_eval": recommendation_eval,
        },
        artifacts_path / "meta.joblib",
    )

    return pipeline, metrics


def load_model(artifacts_dir: str) -> tuple[Any, dict[str, Any]]:
    """
    Load the trained pipeline and metadata from artifacts.
    """
    artifacts_path = Path(artifacts_dir)
    meta_path = artifacts_path / "meta.joblib"

    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")

    meta = joblib.load(meta_path)
    model_type = str(meta.get("model_type", "logreg"))

    if model_type == "xgboost":
        pre_path = artifacts_path / "preprocessor.joblib"
        booster_path = artifacts_path / "xgb_model.json"
        if not pre_path.exists():
            raise FileNotFoundError(f"{pre_path} not found")
        if not booster_path.exists():
            raise FileNotFoundError(f"{booster_path} not found")
        pre = joblib.load(pre_path)
        try:
            import xgboost as xgb
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "Loading XGBoost artifacts requires the 'xgboost' package."
            ) from e
        booster = xgb.Booster()
        booster.load_model(str(booster_path))
        model = XGBServingModel(preprocessor=pre, booster=booster)
        return model, meta

    model_path = artifacts_path / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")
    model = joblib.load(model_path)

    return model, meta


def _artifact_files_for_meta(artifacts_dir: str, meta: dict[str, Any]) -> list[Path]:
    """Return the exact artifact files that define the active model binary."""
    artifacts_path = Path(artifacts_dir)
    model_type = str(meta.get("model_type", "logreg")).strip().lower()
    files = [artifacts_path / "meta.joblib"]
    if model_type == "xgboost":
        files.extend([artifacts_path / "preprocessor.joblib", artifacts_path / "xgb_model.json"])
    else:
        files.append(artifacts_path / "model.joblib")
    return files


def compute_model_fingerprint(artifacts_dir: str, *, meta: dict[str, Any] | None = None) -> str:
    """
    Compute a stable SHA-256 fingerprint for active model artifacts.

    The hash includes both filenames and file bytes so it changes whenever
    model binaries or metadata change.
    """
    artifacts_path = Path(artifacts_dir)
    if meta is None:
        meta_path = artifacts_path / "meta.joblib"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found")
        meta = joblib.load(meta_path)

    files = _artifact_files_for_meta(artifacts_dir, meta)
    hasher = hashlib.sha256()
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    return hasher.hexdigest()


def model_identity(artifacts_dir: str) -> dict[str, str]:
    """Return model identity fields used for retrieval-sync verification."""
    artifacts_path = Path(artifacts_dir)
    meta_path = artifacts_path / "meta.joblib"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")
    meta = joblib.load(meta_path)
    model_type = str(meta.get("model_type", "logreg")).strip().lower()
    return {
        "model_type": model_type,
        "model_fingerprint": compute_model_fingerprint(artifacts_dir, meta=meta),
    }
