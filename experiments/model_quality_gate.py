"""
Model quality gate script.

Why:
- Provide one automated pass/fail checkpoint for core classification and ranking metrics.
- Output a reproducible JSON report for CI and review.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dianalysis.model import train_model, weak_label
from dianalysis.model_components import CAT_COLS, NUM_COLS
from dianalysis.recommendation.candidate_pool import (
    drop_self_candidates,
    ensure_group_columns,
    normalize_target_group,
    select_pool,
)
from dianalysis.run_config import cfg_get, load_runtime_config, write_json
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
    candidates = ensure_group_columns(df.copy())
    candidate_inputs = candidates[NUM_COLS + CAT_COLS].copy()
    candidate_probas = model.predict_proba(candidate_inputs)[:, 1]
    candidates["risk_score"] = np.rint(np.asarray(candidate_probas, dtype=float) * 100.0).astype(int)

    sample_n = min(sample_size, len(candidates))
    queries = candidates.sample(n=sample_n, random_state=random_state)

    ndcg_vals: list[float] = []
    ndcg_non_empty: list[float] = []
    has_any_alt = 0
    eligible_queries = 0
    covered_eligible_queries = 0
    for _, row in queries.iterrows():
        query = row.to_dict()
        result = score_item(dict(query), model, candidates)
        alts = result.get("alternatives", [])
        query_risk = float(result.get("risk_score", 0.0) or 0.0)

        # Eligibility = at least one lower-risk item exists in retrieval scope.
        query_scope = dict(query)
        cat_main, group_key, fine_group_key, _ = normalize_target_group(query_scope)
        pool = select_pool(candidates, cat_main, group_key, fine_group_key)
        pool = drop_self_candidates(pool, query_scope)
        is_eligible = bool(
            (pd.to_numeric(pool.get("risk_score", pd.Series([], dtype=float)), errors="coerce").fillna(1000.0) < query_risk).any()
        )
        if is_eligible:
            eligible_queries += 1

        if alts:
            has_any_alt += 1
            ndcg = ndcg_at_k_for_alternatives(query, alts, k=k)
            ndcg_non_empty.append(ndcg)
            if is_eligible:
                covered_eligible_queries += 1
        else:
            ndcg = 0.0
        ndcg_vals.append(ndcg)

    coverage_all = float(has_any_alt / sample_n) if sample_n else 0.0
    eligibility_rate = float(eligible_queries / sample_n) if sample_n else 0.0
    coverage_given_eligible = float(covered_eligible_queries / eligible_queries) if eligible_queries else 1.0

    return {
        "queries_evaluated": sample_n,
        "eligible_queries": int(eligible_queries),
        "eligibility_rate": eligibility_rate,
        "covered_eligible_queries": int(covered_eligible_queries),
        "coverage_with_alternatives": coverage_all,
        "coverage_given_eligible": coverage_given_eligible,
        "ndcg_at_k_mean": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        "ndcg_at_k_std": float(np.std(ndcg_vals)) if ndcg_vals else 0.0,
        "ndcg_given_non_empty": float(np.mean(ndcg_non_empty)) if ndcg_non_empty else 0.0,
        "k": k,
    }


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    bootstrap.add_argument("--profile", type=Path, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    cfg = load_runtime_config(bootstrap_args.config, bootstrap_args.profile)

    default_data = Path(str(cfg_get(cfg, "paths", "input_csv", default="data/products_off_clean.csv")))
    default_random_state = int(cfg_get(cfg, "project", "random_state", default=42))
    default_cv_folds = int(cfg_get(cfg, "training", "cv_folds", default=5))
    default_model_type = str(cfg_get(cfg, "model", "model_type", default="logreg"))
    default_class_weight = str(cfg_get(cfg, "model", "class_weight", default="balanced"))
    default_c = float(cfg_get(cfg, "model", "C", default=0.3))
    default_xgb_n_estimators = int(cfg_get(cfg, "model", "xgb", "n_estimators", default=300))
    default_xgb_max_depth = int(cfg_get(cfg, "model", "xgb", "max_depth", default=4))
    default_xgb_learning_rate = float(cfg_get(cfg, "model", "xgb", "learning_rate", default=0.05))
    default_xgb_subsample = float(cfg_get(cfg, "model", "xgb", "subsample", default=0.9))
    default_xgb_colsample = float(cfg_get(cfg, "model", "xgb", "colsample_bytree", default=0.9))
    default_missing_indicator = bool(cfg_get(cfg, "model", "with_missing_indicator", default=True))
    default_min_test_f1 = float(cfg_get(cfg, "quality_gate", "min_test_f1", default=0.60))
    default_min_test_auprc = float(cfg_get(cfg, "quality_gate", "min_test_auprc", default=0.75))
    default_max_test_brier = float(cfg_get(cfg, "quality_gate", "max_test_brier", default=0.03))
    default_min_ndcg = float(cfg_get(cfg, "quality_gate", "min_ndcg", default=0.80))
    default_min_coverage = float(cfg_get(cfg, "quality_gate", "min_coverage", default=0.80))
    default_ndcg_k = int(cfg_get(cfg, "quality_gate", "ndcg_k", default=3))
    default_sample_size = int(cfg_get(cfg, "quality_gate", "sample_size", default=120))
    default_artifacts_dir = str(cfg_get(cfg, "paths", "artifacts_dir", default="artifacts"))
    default_report = Path(str(cfg_get(cfg, "paths", "quality_report", default="reports/model_quality_report.json")))

    parser = argparse.ArgumentParser(description="Run the model quality gate checks.")
    parser.add_argument("--config", type=Path, default=bootstrap_args.config)
    parser.add_argument("--profile", type=Path, default=bootstrap_args.profile)
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data,
        help="CSV path for evaluation dataset.",
    )
    parser.add_argument("--random-state", type=int, default=default_random_state)
    parser.add_argument("--cv-folds", type=int, default=default_cv_folds)
    parser.add_argument("--model-type", choices=["logreg", "xgboost"], default=default_model_type)
    parser.add_argument("--class-weight", type=str, default=default_class_weight)
    parser.add_argument("--C", type=float, default=default_c)
    parser.add_argument("--xgb-n-estimators", type=int, default=default_xgb_n_estimators)
    parser.add_argument("--xgb-max-depth", type=int, default=default_xgb_max_depth)
    parser.add_argument("--xgb-learning-rate", type=float, default=default_xgb_learning_rate)
    parser.add_argument("--xgb-subsample", type=float, default=default_xgb_subsample)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=default_xgb_colsample)
    parser.add_argument("--artifacts-dir", type=str, default=default_artifacts_dir)
    parser.add_argument(
        "--with-missing-indicator",
        dest="with_missing_indicator",
        action="store_true",
        default=default_missing_indicator,
        help="Use missingness indicators in numeric preprocessing.",
    )
    parser.add_argument(
        "--no-missing-indicator",
        dest="with_missing_indicator",
        action="store_false",
        help="Disable missingness indicators in numeric preprocessing.",
    )
    parser.add_argument("--min-test-f1", type=float, default=default_min_test_f1)
    parser.add_argument("--min-test-auprc", type=float, default=default_min_test_auprc)
    parser.add_argument("--max-test-brier", type=float, default=default_max_test_brier)
    parser.add_argument("--min-ndcg", type=float, default=default_min_ndcg)
    parser.add_argument("--min-coverage", type=float, default=default_min_coverage)
    parser.add_argument("--ndcg-k", type=int, default=default_ndcg_k)
    parser.add_argument("--sample-size", type=int, default=default_sample_size)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=default_report,
    )
    return parser.parse_args()


def main() -> int:
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
        artifacts_dir=args.artifacts_dir,
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
        k=args.ndcg_k,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    test_metrics = metrics["test"]
    checks = {
        "test_f1": test_metrics["F1"] >= args.min_test_f1,
        "test_auprc": test_metrics["AUPRC"] >= args.min_test_auprc,
        "test_brier": test_metrics["Brier"] <= args.max_test_brier,
        "ranking_ndcg": ranking["ndcg_given_non_empty"] >= args.min_ndcg,
        "ranking_coverage": ranking["coverage_given_eligible"] >= args.min_coverage,
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
            "config_path": str(args.config),
            "profile_path": (str(args.profile) if args.profile else None),
            "model_type": args.model_type,
            "class_weight": class_weight,
            "C": args.C,
            "with_missing_indicator": args.with_missing_indicator,
            "cv_folds": args.cv_folds,
            "random_state": args.random_state,
            "xgb_params": xgb_params if args.model_type == "xgboost" else {},
            "artifacts_dir": args.artifacts_dir,
            "ndcg_k": args.ndcg_k,
            "sample_size": args.sample_size,
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
    snapshot_path = Path(
        str(cfg_get(cfg, "paths", "quality_resolved_config", default="reports/model_quality_resolved_config.json"))
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
        },
    )
    print(json.dumps(report, indent=2))
    print(f"\nSaved report -> {args.report_path}")
    print(f"Saved resolved config snapshot -> {snapshot_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
