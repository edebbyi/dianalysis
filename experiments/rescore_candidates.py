"""
Batch-score the candidate CSV with a trained model.

Why:
- Avoid re-scoring the full candidate pool on every app request.
- Keep a reproducible post-training step that refreshes recommendation inputs.
- Optionally run train + rescore + reindex from one command.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from dianalysis.model import (
    CAT_COLS,
    NUM_COLS,
    compute_model_fingerprint,
    generate_synthetic_data,
    load_model,
    train_model,
)
from dianalysis.run_config import cfg_get, load_runtime_config, write_json
from dianalysis.recommendation.candidate_pool import ensure_group_columns
from dianalysis.scoring import format_risk_display
from dianalysis.vector_retrieval import index_dataframe, retrieval_enabled


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    bootstrap.add_argument("--profile", type=Path, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    cfg = load_runtime_config(bootstrap_args.config, bootstrap_args.profile)

    default_input = Path(str(cfg_get(cfg, "paths", "input_csv", default="data/products_off_clean.csv")))
    default_output = Path(str(cfg_get(cfg, "paths", "scored_csv", default="data/products_off_clean_scored.csv")))
    default_artifacts = str(cfg_get(cfg, "paths", "artifacts_dir", default="artifacts"))
    default_train = bool(cfg_get(cfg, "training", "train_before_rescore", default=False))
    default_auto_train = bool(cfg_get(cfg, "training", "auto_train_if_missing", default=False))
    default_use_synth = bool(cfg_get(cfg, "training", "use_synthetic", default=False))
    default_synth_n = int(cfg_get(cfg, "training", "synthetic_n", default=1000))
    default_model_type = str(cfg_get(cfg, "model", "model_type", default="logreg"))
    default_class_weight = str(cfg_get(cfg, "model", "class_weight", default="balanced"))
    default_c = float(cfg_get(cfg, "model", "C", default=0.3))
    default_missing_indicator = bool(cfg_get(cfg, "model", "with_missing_indicator", default=True))
    default_cv_folds = int(cfg_get(cfg, "training", "cv_folds", default=5))
    default_random_state = int(cfg_get(cfg, "project", "random_state", default=42))
    default_xgb_n_estimators = int(cfg_get(cfg, "model", "xgb", "n_estimators", default=300))
    default_xgb_max_depth = int(cfg_get(cfg, "model", "xgb", "max_depth", default=4))
    default_xgb_learning_rate = float(cfg_get(cfg, "model", "xgb", "learning_rate", default=0.05))
    default_xgb_subsample = float(cfg_get(cfg, "model", "xgb", "subsample", default=0.9))
    default_xgb_colsample = float(cfg_get(cfg, "model", "xgb", "colsample_bytree", default=0.9))
    default_qdrant_mode = str(cfg_get(cfg, "retrieval", "qdrant_mode", default="none"))

    parser = argparse.ArgumentParser(description="Refresh recommendation assets (optional train + rescore + index).")
    parser.add_argument("--config", type=Path, default=bootstrap_args.config)
    parser.add_argument("--profile", type=Path, default=bootstrap_args.profile)
    parser.add_argument("--input-csv", type=Path, default=default_input)
    parser.add_argument("--output-csv", type=Path, default=default_output)
    parser.add_argument("--artifacts-dir", type=str, default=default_artifacts)
    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=default_train,
        help="Train model before rescoring candidates.",
    )
    parser.add_argument(
        "--auto-train-if-missing",
        action=argparse.BooleanOptionalAction,
        default=default_auto_train,
        help="Train automatically only if model artifacts are missing.",
    )
    parser.add_argument(
        "--use-synthetic",
        action=argparse.BooleanOptionalAction,
        default=default_use_synth,
        help="Train using synthetic data.",
    )
    parser.add_argument("--synthetic-n", type=int, default=default_synth_n)
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
    parser.add_argument("--cv-folds", type=int, default=default_cv_folds)
    parser.add_argument("--random-state", type=int, default=default_random_state)
    parser.add_argument("--xgb-n-estimators", type=int, default=default_xgb_n_estimators)
    parser.add_argument("--xgb-max-depth", type=int, default=default_xgb_max_depth)
    parser.add_argument("--xgb-learning-rate", type=float, default=default_xgb_learning_rate)
    parser.add_argument("--xgb-subsample", type=float, default=default_xgb_subsample)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=default_xgb_colsample)
    parser.add_argument(
        "--qdrant-mode",
        choices=["none", "upsert", "recreate", "prune"],
        default=default_qdrant_mode,
        help="Qdrant index action after rescoring.",
    )
    return parser.parse_args()


def artifacts_exist(artifacts_dir: str) -> bool:
    """Return True when either logreg or xgboost artifact layout is present."""
    base = Path(artifacts_dir)
    has_meta = (base / "meta.joblib").exists()
    has_logreg = (base / "model.joblib").exists()
    has_xgb = (base / "preprocessor.joblib").exists() and (base / "xgb_model.json").exists()
    return has_meta and (has_logreg or has_xgb)


def score_candidates(df: pd.DataFrame, artifacts_dir: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Score all candidate rows and append reusable risk + sync identity columns."""
    model, meta = load_model(artifacts_dir)
    model_type = str(meta.get("model_type", "unknown")).strip().lower()
    model_fingerprint = compute_model_fingerprint(artifacts_dir, meta=meta)
    scored_at_utc = pd.Timestamp.now(tz="UTC").isoformat()
    print(
        f"Loaded model from '{artifacts_dir}' (model_type={model_type}, fingerprint={model_fingerprint[:12]}...)"
    )

    out = ensure_group_columns(df)
    if "alt_group" not in out.columns and "category" in out.columns:
        out["alt_group"] = out["category"]

    X = out.reindex(columns=NUM_COLS + CAT_COLS)
    probs = model.predict_proba(X)[:, 1]
    out["risk_prob"] = probs
    out["risk_score"] = (out["risk_prob"] * 100).round().astype(int)
    out["risk_display"] = out["risk_prob"].apply(format_risk_display)
    out["model_type"] = model_type
    out["model_fingerprint"] = model_fingerprint
    out["scored_at_utc"] = scored_at_utc
    return out, {
        "model_type": model_type,
        "model_fingerprint": model_fingerprint,
        "scored_at_utc": scored_at_utc,
    }


def maybe_train(args: argparse.Namespace) -> None:
    """Optionally train model artifacts before rescoring."""
    should_train = args.train or (args.auto_train_if_missing and not artifacts_exist(args.artifacts_dir))
    if not should_train:
        return

    if args.use_synthetic:
        train_df = generate_synthetic_data(n=args.synthetic_n, random_state=args.random_state)
        print(f"Training on synthetic data (n={len(train_df)})")
    else:
        if not args.input_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {args.input_csv}")
        train_df = pd.read_csv(args.input_csv, dtype={"upc": str})
        print(f"Training on csv data (n={len(train_df)}) from {args.input_csv}")

    class_weight: str | dict[str, float] | None
    class_weight = None if args.class_weight.lower() == "none" else args.class_weight
    xgb_params: dict[str, Any] = {
        "n_estimators": args.xgb_n_estimators,
        "max_depth": args.xgb_max_depth,
        "learning_rate": args.xgb_learning_rate,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample_bytree,
    }
    _, metrics = train_model(
        train_df,
        artifacts_dir=args.artifacts_dir,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        model_type=args.model_type,
        class_weight=class_weight,
        C=args.C,
        add_indicator=args.with_missing_indicator,
        xgb_params=xgb_params if args.model_type == "xgboost" else None,
    )
    print(
        "Training complete:",
        f"test_F1={metrics['test']['F1']:.4f},",
        f"test_AUPRC={metrics['test']['AUPRC']:.4f},",
        f"test_Brier={metrics['test']['Brier']:.4f}",
    )


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
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    maybe_train(args)

    df = pd.read_csv(args.input_csv, dtype={"upc": str})
    scored, sync_meta = score_candidates(df, artifacts_dir=args.artifacts_dir)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(scored)} scored candidates to: {args.output_csv}")

    if args.qdrant_mode != "none":
        if retrieval_enabled():
            recreate = args.qdrant_mode == "recreate"
            prune_missing = args.qdrant_mode == "prune"
            n = index_dataframe(scored, recreate=recreate, prune_missing=prune_missing, sync_meta=sync_meta)
            action = "recreated" if recreate else ("upserted+pruned" if prune_missing else "upserted")
            print(f"Qdrant index {action} with {n} rows.")
        else:
            print("Skipped Qdrant indexing (retrieval backend is not set to qdrant).")

    snapshot_path = Path(
        str(cfg_get(cfg, "paths", "rescore_resolved_config", default="reports/rescore_resolved_config.json"))
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
            "rescore_invocation": {
                "input_csv": str(args.input_csv),
                "output_csv": str(args.output_csv),
                "artifacts_dir": str(args.artifacts_dir),
                "train": bool(args.train),
                "auto_train_if_missing": bool(args.auto_train_if_missing),
                "use_synthetic": bool(args.use_synthetic),
                "synthetic_n": int(args.synthetic_n),
                "model_type": str(args.model_type),
                "class_weight": str(args.class_weight),
                "C": float(args.C),
                "with_missing_indicator": bool(args.with_missing_indicator),
                "cv_folds": int(args.cv_folds),
                "random_state": int(args.random_state),
                "qdrant_mode": str(args.qdrant_mode),
            },
        },
    )
    print(f"Wrote resolved config snapshot -> {snapshot_path}")


if __name__ == "__main__":
    main()
