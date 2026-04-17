"""
Model quality gate script.

Why:
- Provide one automated pass/fail checkpoint for core classification and ranking metrics.
- Output a reproducible JSON report for CI and review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dianalysis.model import train_model, weak_label
from dianalysis.scoring import ndcg_at_k_for_alternatives, score_item
from dianalysis.type_defs import ModelLike


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, dtype={"upc": str})
    if "__display" in df.columns:
        # Stored serialized display blobs are not needed for model training/eval.
        df = df.drop(columns=["__display"])
    if "label" not in df.columns:
        df["label"] = df.apply(weak_label, axis=1)
    return df


def evaluate_ranking_ndcg(
    model: ModelLike,
    df: pd.DataFrame,
    *,
    k: int = 3,
    sample_size: int = 120,
    random_state: int = 42,
) -> dict:
    candidates = df.copy()
    sample_n = min(sample_size, len(candidates))
    queries = candidates.sample(n=sample_n, random_state=random_state)

    ndcg_vals: list[float] = []
    has_any_alt = 0
    for _, row in queries.iterrows():
        result = score_item(row.to_dict(), model, candidates)
        alts = result.get("alternatives", [])
        if alts:
            has_any_alt += 1
        ndcg_vals.append(ndcg_at_k_for_alternatives(row.to_dict(), alts, k=k))

    return {
        "queries_evaluated": sample_n,
        "coverage_with_alternatives": float(has_any_alt / sample_n) if sample_n else 0.0,
        "ndcg_at_k_mean": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        "ndcg_at_k_std": float(np.std(ndcg_vals)) if ndcg_vals else 0.0,
        "k": k,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the model quality gate checks.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/products_off_clean.csv"),
        help="CSV path for evaluation dataset.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--model-type", choices=["logreg", "xgboost"], default="logreg")
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--C", type=float, default=0.3)
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument(
        "--with-missing-indicator",
        dest="with_missing_indicator",
        action="store_true",
        default=True,
        help="Use missingness indicators in numeric preprocessing.",
    )
    parser.add_argument(
        "--no-missing-indicator",
        dest="with_missing_indicator",
        action="store_false",
        help="Disable missingness indicators in numeric preprocessing.",
    )
    parser.add_argument("--min-test-f1", type=float, default=0.60)
    parser.add_argument("--min-test-auprc", type=float, default=0.75)
    parser.add_argument("--max-test-brier", type=float, default=0.03)
    parser.add_argument("--min-ndcg", type=float, default=0.80)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/model_quality_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_dataset(args.data)

    class_weight = None if args.class_weight.lower() == "none" else args.class_weight
    xgb_params = {
        "n_estimators": args.xgb_n_estimators,
        "max_depth": args.xgb_max_depth,
        "learning_rate": args.xgb_learning_rate,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample_bytree,
    }

    model, metrics = train_model(
        df,
        artifacts_dir="artifacts",
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        model_type=args.model_type,
        class_weight=class_weight,
        C=args.C,
        add_indicator=args.with_missing_indicator,
        xgb_params=xgb_params if args.model_type == "xgboost" else None,
    )

    ranking = evaluate_ranking_ndcg(
        model,
        df,
        k=3,
        sample_size=120,
        random_state=args.random_state,
    )

    test_metrics = metrics["test"]
    checks = {
        "test_f1": test_metrics["F1"] >= args.min_test_f1,
        "test_auprc": test_metrics["AUPRC"] >= args.min_test_auprc,
        "test_brier": test_metrics["Brier"] <= args.max_test_brier,
        "ranking_ndcg": ranking["ndcg_at_k_mean"] >= args.min_ndcg,
        "ranking_coverage": ranking["coverage_with_alternatives"] >= args.min_coverage,
    }
    passed = all(checks.values())

    report = {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "thresholds": {
            "min_test_f1": args.min_test_f1,
            "min_test_auprc": args.min_test_auprc,
            "max_test_brier": args.max_test_brier,
            "min_ndcg": args.min_ndcg,
            "min_coverage": args.min_coverage,
        },
        "config": {
            "model_type": args.model_type,
            "class_weight": class_weight,
            "C": args.C,
            "with_missing_indicator": args.with_missing_indicator,
            "cv_folds": args.cv_folds,
            "random_state": args.random_state,
            "xgb_params": xgb_params if args.model_type == "xgboost" else {},
        },
        "metrics": metrics,
        "ranking": ranking,
        "dataset": {
            "path": str(args.data),
            "rows": int(len(df)),
        },
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved report -> {args.report_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
