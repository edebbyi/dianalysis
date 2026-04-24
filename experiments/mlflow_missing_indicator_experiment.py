"""
MLflow experiment runner for missing-indicator and config sensitivity.

Why:
- Compare targeted experimental knobs in a reproducible tracked workflow.
- Persist comparable run metrics/artifacts outside the notebook.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dianalysis.model import CAT_COLS, NUM_COLS, ModelType, build_pipeline, weak_label


def load_dataset() -> pd.DataFrame:
    csv_path = Path("data/products_off_clean.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype={"upc": str})
        source = "openfoodfacts_clean"
    else:
        raise FileNotFoundError("Expected data/products_off_clean.csv for this experiment")

    if "__display" in df.columns:
        df = df.drop(columns=["__display"])
    if "label" not in df.columns:
        df["label"] = df.apply(weak_label, axis=1)
    return df.assign(_source=source)


def evaluate_config(
    df: pd.DataFrame,
    *,
    model_type: str,
    class_weight: str | dict | None,
    C: float,
    add_indicator: bool,
    xgb_params: dict[str, Any] | None = None,
    random_state: int = 42,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> dict:
    X = df[NUM_COLS + CAT_COLS].copy()
    y = df["label"].astype(int).copy()
    rkf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    f1_vals: list[float] = []
    auprc_vals: list[float] = []
    brier_vals: list[float] = []

    for tr_idx, va_idx in rkf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = build_pipeline(
            model_type=cast(ModelType, model_type),
            class_weight=class_weight,
            C=C,
            add_indicator=add_indicator,
            random_state=random_state,
            xgb_params=xgb_params if model_type == "xgboost" else None,
        )
        model.fit(X_tr, y_tr)
        p_va = model.predict_proba(X_va)[:, 1]
        y_hat = (p_va >= 0.5).astype(int)

        f1_vals.append(f1_score(y_va, y_hat, zero_division=0))
        auprc_vals.append(average_precision_score(y_va, p_va))
        brier_vals.append(brier_score_loss(y_va, p_va))

    return {
        "F1_mean": float(np.mean(f1_vals)),
        "F1_std": float(np.std(f1_vals, ddof=1)),
        "AUPRC_mean": float(np.mean(auprc_vals)),
        "AUPRC_std": float(np.std(auprc_vals, ddof=1)),
        "Brier_mean": float(np.mean(brier_vals)),
        "Brier_std": float(np.std(brier_vals, ddof=1)),
    }


def snapshot_notebook_state(nb_path: Path) -> dict:
    """Extract current summary-table outputs from notebook for pre-MLflow snapshot."""
    if not nb_path.exists():
        return {"snapshot_available": False}
    payload = json.loads(nb_path.read_text())
    summary: dict = {"snapshot_available": True, "cells": {}}
    for idx, cell in enumerate(payload.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "results_df" in src and "cv_mean" in src:
            summary["cells"]["split_metrics_cell"] = idx
        if "variance_tuning_df" in src and "RepeatedStratifiedKFold" in src:
            summary["cells"]["variance_cell"] = idx
        if "threshold_results_df" in src and "best_f1_threshold" in src:
            summary["cells"]["threshold_cell"] = idx
    return summary


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("dianalysis_missing_indicator")

    df = load_dataset()
    snapshot = snapshot_notebook_state(Path("dianalysis/dianalysis.ipynb"))

    configs: list[dict[str, Any]] = [
        {"name": "logreg_baseline_C1.0", "model_type": "logreg", "class_weight": None, "C": 1.0},
        {"name": "logreg_balanced_C0.3", "model_type": "logreg", "class_weight": "balanced", "C": 0.3},
        {"name": "xgboost_balanced", "model_type": "xgboost", "class_weight": "balanced", "C": 1.0},
    ]

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        for add_indicator in [False, True]:
            run_name = f"{cfg['name']}_add_indicator_{add_indicator}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("config", cfg["name"])
                mlflow.log_param("model_type", cfg["model_type"])
                mlflow.log_param("class_weight", str(cfg["class_weight"]))
                mlflow.log_param("C", cfg["C"])
                mlflow.log_param("add_indicator", add_indicator)
                mlflow.log_param("dataset_source", df["_source"].iloc[0])
                mlflow.log_param("dataset_rows", int(len(df)))
                mlflow.log_param("cv_scheme", "RepeatedStratifiedKFold")
                mlflow.log_param("cv_splits", 5)
                mlflow.log_param("cv_repeats", 5)

                metrics = evaluate_config(
                    df,
                    model_type=cfg["model_type"],
                    class_weight=cfg["class_weight"],
                    C=cfg["C"],
                    add_indicator=add_indicator,
                    xgb_params={
                        "n_estimators": 300,
                        "max_depth": 4,
                        "learning_rate": 0.05,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                    },
                )
                mlflow.log_metrics(metrics)

                row = {
                    "config": cfg["name"],
                    "model_type": cfg["model_type"],
                    "class_weight": str(cfg["class_weight"]),
                    "C": cfg["C"],
                    "add_indicator": add_indicator,
                    **metrics,
                }
                rows.append(row)

                summary_path = Path("reports") / f"{run_name}_snapshot.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(snapshot, indent=2))
                mlflow.log_artifact(str(summary_path))

    out = pd.DataFrame(rows).sort_values(["config", "add_indicator"]).reset_index(drop=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "mlflow_missing_indicator_summary.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved summary -> {out_path}")
    print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
