"""
CLI entrypoint to train model artifacts.

Why:
- Provide one reproducible command (`python train.py`) to rebuild artifacts.
- Keep training invocation separate from reusable training logic in `dianalysis.model`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dianalysis.model import generate_synthetic_data, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dianalysis model artifacts.")
    parser.add_argument("--data", type=Path, default=Path("data/products_off_clean.csv"))
    parser.add_argument("--use-synthetic", action="store_true", help="Train on synthetic demo data.")
    parser.add_argument("--synthetic-n", type=int, default=1000)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--model-type", choices=["logreg", "xgboost"], default="logreg")
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--C", type=float, default=0.3)
    parser.add_argument("--with-missing-indicator", dest="with_missing_indicator", action="store_true", default=True)
    parser.add_argument("--no-missing-indicator", dest="with_missing_indicator", action="store_false")
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    return parser.parse_args()


def dedupe_products(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate products by UPC, with name+brand fallback when UPC is missing."""
    out = df.copy()
    upc = out["upc"].fillna("").astype(str).str.strip() if "upc" in out.columns else pd.Series("", index=out.index)
    name = out["name"].fillna("").astype(str).str.lower().str.strip() if "name" in out.columns else pd.Series("", index=out.index)
    brand = out["brand"].fillna("").astype(str).str.lower().str.strip() if "brand" in out.columns else pd.Series("", index=out.index)
    has_upc = upc != ""
    key = pd.Series(index=out.index, dtype="string")
    key.loc[has_upc] = "upc:" + upc.loc[has_upc]
    key.loc[~has_upc] = "namebrand:" + name.loc[~has_upc] + "|" + brand.loc[~has_upc]
    out["_dedupe_key"] = key
    out = out.drop_duplicates(subset=["_dedupe_key"], keep="first").drop(columns=["_dedupe_key"])
    return out


def main() -> None:
    args = parse_args()
    if not args.use_synthetic and args.data.exists():
        df = pd.read_csv(args.data, dtype={"upc": str})
        rows_before = len(df)
        df = dedupe_products(df)
        print({"train_rows_before_dedupe": rows_before, "train_rows_after_dedupe": len(df)})
    else:
        df = generate_synthetic_data(n=args.synthetic_n, random_state=args.random_state)

    class_weight: str | dict | None
    class_weight = None if args.class_weight.lower() == "none" else args.class_weight
    xgb_params = {
        "n_estimators": args.xgb_n_estimators,
        "max_depth": args.xgb_max_depth,
        "learning_rate": args.xgb_learning_rate,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample_bytree,
    }
    _, metrics = train_model(
        df,
        artifacts_dir=args.artifacts_dir,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        model_type=args.model_type,
        class_weight=class_weight,
        C=args.C,
        add_indicator=args.with_missing_indicator,
        xgb_params=xgb_params if args.model_type == "xgboost" else None,
    )
    print("Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
