"""Export worst recommendation examples for manual label fixes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dianalysis.model import load_model
from dianalysis.scoring import score_item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a retrieval failure audit table.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--input-csv", default="data/products_off_clean_scored_labeled_v2.csv")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-bad", type=int, default=50)
    parser.add_argument("--out-csv", default="reports/retrieval_failure_audit.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _ = load_model(args.artifacts_dir)
    df = pd.read_csv(args.input_csv, dtype={"upc": str})

    n = min(args.sample_size, len(df))
    sample = df.sample(n=n, random_state=args.random_state)

    rows: list[dict] = []
    for _, row in sample.iterrows():
        query = row.to_dict()
        result = score_item(query, model, df)
        query_cat = str(result.get("item_category", "") or "").lower()
        query_group = str(result.get("item_alt_group", "") or query_cat).lower()
        alts = result.get("alternatives", [])

        mismatch_count = 0
        for alt in alts:
            alt_cat = str(alt.get("category", "") or "").lower()
            alt_group = str(alt.get("alt_group", "") or alt_cat).lower()
            same_scope = (alt_group == query_group) or (alt_cat == query_cat)
            if not same_scope:
                mismatch_count += 1

        rows.append(
            {
                "query_name": result.get("item_name"),
                "query_brand": result.get("item_brand"),
                "query_category": query_cat,
                "query_alt_group": query_group,
                "alternatives_count": len(alts),
                "mismatch_count": mismatch_count,
                "coverage_fail": int(len(alts) == 0),
                "alt_1": alts[0].get("name") if len(alts) > 0 else "",
                "alt_2": alts[1].get("name") if len(alts) > 1 else "",
                "alt_3": alts[2].get("name") if len(alts) > 2 else "",
            }
        )

    out = pd.DataFrame(rows)
    out["badness_score"] = out["coverage_fail"] * 3 + out["mismatch_count"]
    out = out.sort_values(["badness_score", "mismatch_count"], ascending=[False, False]).head(args.top_bad)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote failure audit: {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
