"""
A/B comparison for heuristic vs Qdrant semantic retrieval backends.

Outputs:
- reports/retrieval_ab_test.csv
- reports/retrieval_ab_test.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from dianalysis.model import load_model
from dianalysis.recommendation_eval import compute_recommendation_eval


def run_eval(
    *,
    backend: str,
    model: Any,
    df: pd.DataFrame,
    sample_size: int,
    k: int,
    random_state: int,
    qdrant_url: str,
) -> dict[str, Any]:
    os.environ["DIANALYSIS_RETRIEVAL_BACKEND"] = backend
    if backend == "qdrant":
        os.environ["QDRANT_URL"] = qdrant_url
    metrics = compute_recommendation_eval(
        model,
        df,
        sample_size=sample_size,
        k=k,
        random_state=random_state,
    )
    return {"backend": backend, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval backend A/B test.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--input-csv", default="data/products_off_clean_scored.csv")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--qdrant-url", default="http://localhost:6335")
    parser.add_argument("--out-csv", default="reports/retrieval_ab_test.csv")
    parser.add_argument("--out-json", default="reports/retrieval_ab_test.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, meta = load_model(args.artifacts_dir)
    print(f"Loaded model_type={meta.get('model_type', 'unknown')} from {args.artifacts_dir}")

    df = pd.read_csv(args.input_csv, dtype={"upc": str})
    rows = []
    for backend in ("heuristic", "qdrant"):
        rows.append(
            run_eval(
                backend=backend,
                model=model,
                df=df,
                sample_size=args.sample_size,
                k=args.k,
                random_state=args.random_state,
                qdrant_url=args.qdrant_url,
            )
        )

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

