"""
Deduplicate source product CSV by stable product identity keys.

Why:
- Prevent duplicate recommendations caused by repeated product rows.
- Keep the best row when duplicates exist by preferring fuller nutrition metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


QUALITY_COLS = [
    "ingredients_text",
    "categories_all",
    "calories",
    "carbs_g",
    "fiber_g",
    "sugar_g",
    "protein_g",
    "fat_g",
    "sodium_mg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate source CSV rows.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/products_off_clean.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/products_off_clean.csv"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/dedupe_report.json"))
    return parser.parse_args()


def _series_str(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[col].fillna("").astype(str)


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv, dtype={"upc": str})
    rows_before = int(len(df))

    upc = _series_str(df, "upc").str.strip()
    name = _series_str(df, "name").str.lower().str.strip()
    brand = _series_str(df, "brand").str.lower().str.strip()

    has_upc = upc != ""
    identity_key = pd.Series(index=df.index, dtype="string")
    identity_key.loc[has_upc] = "upc:" + upc.loc[has_upc]
    identity_key.loc[~has_upc] = "namebrand:" + name.loc[~has_upc] + "|" + brand.loc[~has_upc]

    work = df.copy()
    work["_identity_key"] = identity_key
    quality_cols = [c for c in QUALITY_COLS if c in work.columns]
    if quality_cols:
        work["_quality_score"] = work[quality_cols].notna().sum(axis=1)
    else:
        work["_quality_score"] = 0

    # Keep richer rows when identity duplicates exist.
    deduped = (
        work.sort_values(["_identity_key", "_quality_score"], ascending=[True, False])
        .drop_duplicates(subset=["_identity_key"], keep="first")
        .drop(columns=["_identity_key", "_quality_score"])
    )

    rows_after = int(len(deduped))
    removed = rows_before - rows_after

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(args.output_csv, index=False)

    report = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": removed,
    }
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
