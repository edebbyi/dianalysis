"""
Batch-score the candidate CSV with a trained model.

Why:
- Avoid re-scoring the full candidate pool on every app request.
- Keep a reproducible post-training step that refreshes recommendation inputs.
- Optionally run train + rescore + reindex from one command.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from dianalysis.model import CAT_COLS, NUM_COLS, generate_synthetic_data, load_model, train_model
from dianalysis.scoring import format_risk_display
from dianalysis.vector_retrieval import index_dataframe, retrieval_enabled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh recommendation assets (optional train + rescore + index).")
    parser.add_argument("--input-csv", type=Path, default=Path("data/products_off_clean.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/products_off_clean_scored.csv"))
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--train", action="store_true", help="Train model before rescoring candidates.")
    parser.add_argument(
        "--auto-train-if-missing",
        action="store_true",
        help="Train automatically only if model artifacts are missing.",
    )
    parser.add_argument("--use-synthetic", action="store_true", help="Train using synthetic data.")
    parser.add_argument("--synthetic-n", type=int, default=1000)
    parser.add_argument("--model-type", choices=["logreg", "xgboost"], default="logreg")
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--C", type=float, default=0.3)
    parser.add_argument("--with-missing-indicator", dest="with_missing_indicator", action="store_true", default=True)
    parser.add_argument("--no-missing-indicator", dest="with_missing_indicator", action="store_false")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument(
        "--qdrant-mode",
        choices=["none", "upsert", "recreate", "prune"],
        default="none",
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


def score_candidates(df: pd.DataFrame, artifacts_dir: str) -> pd.DataFrame:
    """Score all candidate rows and append reusable risk columns."""
    model, meta = load_model(artifacts_dir)
    model_type = str(meta.get("model_type", "unknown"))
    print(f"Loaded model from '{artifacts_dir}' (model_type={model_type})")

    out = df.copy()
    if "alt_group" not in out.columns and "category" in out.columns:
        out["alt_group"] = out["category"]

    X = out.reindex(columns=NUM_COLS + CAT_COLS)
    probs = model.predict_proba(X)[:, 1]
    out["risk_prob"] = probs
    out["risk_score"] = (out["risk_prob"] * 100).round().astype(int)
    out["risk_display"] = out["risk_prob"].apply(format_risk_display)
    return out


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
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    maybe_train(args)

    df = pd.read_csv(args.input_csv, dtype={"upc": str})
    scored = score_candidates(df, artifacts_dir=args.artifacts_dir)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(scored)} scored candidates to: {args.output_csv}")

    if args.qdrant_mode != "none":
        if retrieval_enabled():
            recreate = args.qdrant_mode == "recreate"
            prune_missing = args.qdrant_mode == "prune"
            n = index_dataframe(scored, recreate=recreate, prune_missing=prune_missing)
            action = "recreated" if recreate else ("upserted+pruned" if prune_missing else "upserted")
            print(f"Qdrant index {action} with {n} rows.")
        else:
            print("Skipped Qdrant indexing (retrieval backend is not set to qdrant).")


if __name__ == "__main__":
    main()
