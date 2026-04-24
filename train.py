"""
CLI entrypoint to train model artifacts.

Why:
- Provide one reproducible command (`python train.py`) to rebuild artifacts.
- Keep training invocation separate from reusable training logic in `dianalysis.model`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dianalysis.model import generate_synthetic_data, train_model
from dianalysis.run_config import cfg_get, load_runtime_config, write_json


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    bootstrap.add_argument("--profile", type=Path, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    cfg = load_runtime_config(bootstrap_args.config, bootstrap_args.profile)

    default_data = Path(str(cfg_get(cfg, "paths", "input_csv", default="data/products_off_clean.csv")))
    default_use_synth = bool(cfg_get(cfg, "training", "use_synthetic", default=False))
    default_synth_n = int(cfg_get(cfg, "training", "synthetic_n", default=1000))
    default_artifacts_dir = str(cfg_get(cfg, "paths", "artifacts_dir", default="artifacts"))
    default_random_state = int(cfg_get(cfg, "project", "random_state", default=42))
    default_cv_folds = int(cfg_get(cfg, "training", "cv_folds", default=5))
    default_model_type = str(cfg_get(cfg, "model", "model_type", default="logreg"))
    default_class_weight = str(cfg_get(cfg, "model", "class_weight", default="balanced"))
    default_c = float(cfg_get(cfg, "model", "C", default=0.3))
    default_missing_indicator = bool(cfg_get(cfg, "model", "with_missing_indicator", default=True))
    default_xgb_n_estimators = int(cfg_get(cfg, "model", "xgb", "n_estimators", default=300))
    default_xgb_max_depth = int(cfg_get(cfg, "model", "xgb", "max_depth", default=4))
    default_xgb_learning_rate = float(cfg_get(cfg, "model", "xgb", "learning_rate", default=0.05))
    default_xgb_subsample = float(cfg_get(cfg, "model", "xgb", "subsample", default=0.9))
    default_xgb_colsample = float(cfg_get(cfg, "model", "xgb", "colsample_bytree", default=0.9))

    parser = argparse.ArgumentParser(description="Train Dianalysis model artifacts.")
    parser.add_argument("--config", type=Path, default=bootstrap_args.config)
    parser.add_argument("--profile", type=Path, default=bootstrap_args.profile)
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument(
        "--use-synthetic",
        action=argparse.BooleanOptionalAction,
        default=default_use_synth,
        help="Train on synthetic demo data.",
    )
    parser.add_argument("--synthetic-n", type=int, default=default_synth_n)
    parser.add_argument("--artifacts-dir", type=str, default=default_artifacts_dir)
    parser.add_argument("--random-state", type=int, default=default_random_state)
    parser.add_argument("--cv-folds", type=int, default=default_cv_folds)
    parser.add_argument("--model-type", choices=["logreg", "xgboost"], default=default_model_type)
    parser.add_argument("--class-weight", type=str, default=default_class_weight)
    parser.add_argument("--C", type=float, default=default_c)
    parser.add_argument(
        "--with-missing-indicator",
        dest="with_missing_indicator",
        action="store_true",
        default=default_missing_indicator,
    )
    parser.add_argument("--no-missing-indicator", dest="with_missing_indicator", action="store_false")
    parser.add_argument("--xgb-n-estimators", type=int, default=default_xgb_n_estimators)
    parser.add_argument("--xgb-max-depth", type=int, default=default_xgb_max_depth)
    parser.add_argument("--xgb-learning-rate", type=float, default=default_xgb_learning_rate)
    parser.add_argument("--xgb-subsample", type=float, default=default_xgb_subsample)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=default_xgb_colsample)
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
    cfg = load_runtime_config(args.config, args.profile)
    os.environ["DIANALYSIS_CONFIG"] = str(args.config)
    if args.profile:
        os.environ["DIANALYSIS_PROFILE"] = str(args.profile)
    else:
        os.environ.pop("DIANALYSIS_PROFILE", None)
    print(
        {
            "config_path": str(args.config),
            "profile_path": (str(args.profile) if args.profile else None),
            "config_loaded": bool(cfg),
        }
    )
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

    snapshot_path = Path(
        str(
            cfg_get(
                cfg,
                "paths",
                "train_resolved_config",
                default=f"{args.artifacts_dir}/run_configs/train_resolved_config.json",
            )
        )
    )
    write_json(
        snapshot_path,
        {
            "sources": {
                "config": str(args.config),
                "profile": (str(args.profile) if args.profile else None),
            },
            "cli_args": vars(args),
            "resolved_config": cfg,
            "train_invocation": {
                "data": str(args.data),
                "use_synthetic": bool(args.use_synthetic),
                "synthetic_n": int(args.synthetic_n),
                "artifacts_dir": str(args.artifacts_dir),
                "random_state": int(args.random_state),
                "cv_folds": int(args.cv_folds),
                "model_type": str(args.model_type),
                "class_weight": str(args.class_weight),
                "C": float(args.C),
                "with_missing_indicator": bool(args.with_missing_indicator),
                "xgb_params": xgb_params,
            },
        },
    )
    print(f"Wrote resolved config snapshot -> {snapshot_path}")


if __name__ == "__main__":
    main()
