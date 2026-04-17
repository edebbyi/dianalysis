"""
Safely normalize category/alt_group labels into a new CSV file.

Safety guarantees:
- Never overwrites input by default.
- Validates row count and UPC set consistency.
- Writes audit reports for changed rows and before/after distribution.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


CEREAL_STRONG_PATTERNS = [
    r"\bcereal(s)?\b",
    r"\bbreakfast[- ]?cereal(s)?\b",
    r"\bmuesli\b",
    r"\bgranola\b",
    r"\bweetabix\b",
    r"\bcheerios\b",
    r"\bcorn[- ]?flakes\b",
    r"\bbran[- ]?flakes\b",
]

PLAIN_OATS_PATTERNS = [
    r"\boatmeal\b",
    r"\brolled oats?\b",
    r"\bquick oats?\b",
    r"\bsteel[- ]?cut\b",
    r"\bold[- ]?fashioned oats?\b",
    r"\bporridge oats?\b",
]

BREAD_STRONG_PATTERNS = [
    r"\bbread(s)?\b",
    r"\bbagel(s)?\b",
    r"\bbun(s)?\b",
    r"\broll(s)?\b",
    r"\bciabatta\b",
    r"\bnaan\b",
    r"\bpita\b",
    r"\bflatbread(s)?\b",
    r"\bwrap(s)?\b",
    r"\benglish muffin(s)?\b",
]


def _safe_text(*values: Any) -> str:
    return " ".join(str(v or "") for v in values).lower()


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def normalize_row(row: dict[str, Any]) -> tuple[str, str]:
    """Return normalized (category, alt_group) for one row."""
    category = str(row.get("category") or "").strip().lower()
    alt_group = str(row.get("alt_group") or "").strip().lower()
    text = _safe_text(row.get("name"), row.get("categories_all"), row.get("ingredients_text"))

    has_cereal_signal = _matches_any(text, CEREAL_STRONG_PATTERNS)
    has_plain_oats_signal = _matches_any(text, PLAIN_OATS_PATTERNS)
    has_granola_signal = bool(re.search(r"\b(granola|muesli)\b", text))
    has_bread_signal = _matches_any(text, BREAD_STRONG_PATTERNS)

    # Map bread-like products (bagels, buns, wraps, etc.) to bread before cereal lock-in.
    if has_bread_signal and not has_granola_signal:
        return ("bread", "bread")

    # Keep explicit cereal/granola labels internally consistent.
    if alt_group in {"cereal", "granola"}:
        return ("cereal", alt_group)

    # Promote cereal-like items that were mis-labeled by broad nut/oat signals.
    if has_cereal_signal:
        if alt_group == "oats" and has_plain_oats_signal and not has_granola_signal and "cereal" not in text:
            return ("grain", "oats")
        return ("cereal", "granola" if has_granola_signal else "cereal")

    # Fallback consistency rule.
    if category == "cereal" and alt_group == "oats":
        return ("cereal", "cereal")

    return (category, alt_group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely normalize labels into a new CSV.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/products_off_clean.csv"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/products_off_clean_labeled_v2.csv"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/label_normalization_report.json"),
    )
    parser.add_argument(
        "--changes-csv",
        type=Path,
        default=Path("reports/label_changes_sample.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv, dtype={"upc": str})
    out = df.copy()
    before = out[["category", "alt_group"]].copy()

    normalized = out.apply(lambda r: normalize_row(r.to_dict()), axis=1)
    out["category"] = normalized.apply(lambda t: t[0])
    out["alt_group"] = normalized.apply(lambda t: t[1])

    # Safety checks.
    if len(out) != len(df):
        raise RuntimeError("Row count changed unexpectedly.")
    if "upc" in df.columns:
        in_upc = set(df["upc"].fillna("").astype(str))
        out_upc = set(out["upc"].fillna("").astype(str))
        if in_upc != out_upc:
            raise RuntimeError("UPC set changed unexpectedly.")

    changed_mask = (before["category"] != out["category"]) | (before["alt_group"] != out["alt_group"])
    changed = out[changed_mask].copy()
    changed["old_category"] = before.loc[changed_mask, "category"].values
    changed["old_alt_group"] = before.loc[changed_mask, "alt_group"].values

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.changes_csv.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(args.output_csv, index=False)
    changed_cols = [
        "name",
        "brand",
        "upc",
        "old_category",
        "old_alt_group",
        "category",
        "alt_group",
    ]
    changed[changed_cols].head(200).to_csv(args.changes_csv, index=False)

    before_counts = (
        before.value_counts()
        .rename_axis(["category", "alt_group"])
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    after_counts = (
        out[["category", "alt_group"]]
        .value_counts()
        .rename_axis(["category", "alt_group"])
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    report = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "rows_input": int(len(df)),
        "rows_output": int(len(out)),
        "rows_changed": int(changed_mask.sum()),
        "changes_csv": str(args.changes_csv),
        "top_before": before_counts.head(12).to_dict(orient="records"),
        "top_after": after_counts.head(12).to_dict(orient="records"),
    }
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote normalized CSV: {args.output_csv}")
    print(f"Changed rows: {int(changed_mask.sum())}")
    print(f"Wrote changes sample: {args.changes_csv}")
    print(f"Wrote report: {args.report_json}")


if __name__ == "__main__":
    main()
