#!/usr/bin/env python3
"""
Reproduce weak-label threshold comparison rates on a source CSV.

This script computes the exact comparison rows documented in
`docs/labeling_logic/rule_grounding.md` and writes machine-readable + human-readable
outputs with dataset metadata for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "carbs_g",
    "fiber_g",
    "added_sugar_g",
    "sugar_alcohols_g",
    "protein_g",
    "sodium_mg",
]

RULESET_NOTES = {
    "legacy": "`net>20,+2`; `add>=8,+2`; `sod>=500,+1`; `fiber>=5,-2`; `protein>=12,-1`",
    "balanced_candidate": "`net>=25,+1`; `add>=10,+2`; `sod>=460,+1`; `fiber>=5.6,-2`; `protein>=10,-1`",
    "current_adopted": "`carbs>=30,+2`; `add>=10,+2`; `sod>=460,+1`; `fiber>=5.6,-2`; `protein>=10,-1`",
}

COMPARISON_ROWS = [
    {
        "rule_id": "legacy",
        "label": f"Legacy rules ({RULESET_NOTES['legacy']})",
        "trigger": 2,
    },
    {
        "rule_id": "balanced_candidate",
        "label": f"Balanced rules ({RULESET_NOTES['balanced_candidate']})",
        "trigger": 2,
    },
    {
        "rule_id": "balanced_candidate",
        "label": "Balanced rules (same thresholds, stricter trigger)",
        "trigger": 3,
    },
    {
        "rule_id": "current_adopted",
        "label": f"Current adopted rules ({RULESET_NOTES['current_adopted']})",
        "trigger": 2,
    },
    {
        "rule_id": "current_adopted",
        "label": "Current adopted rules (same thresholds, stricter trigger)",
        "trigger": 3,
    },
]


def _sha256(path: Path) -> str:
    """Return SHA256 hex digest for a file.

    This is used to prove which exact dataset file was evaluated.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare numeric nutrient fields needed by every ruleset.

    Behavior:
    - Validates all required columns exist.
    - Coerces columns to numeric.
    - Treats missing/non-numeric values as `0.0` for deterministic comparisons.
    - Recomputes `net_carbs_g = max(carbs - fiber - sugar_alcohols, 0)`.

    Returns a copy so the caller's DataFrame is unchanged.
    """
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["net_carbs_g"] = (
        out["carbs_g"] - out["fiber_g"] - out["sugar_alcohols_g"]
    ).clip(lower=0.0)
    return out


def _legacy_points(df: pd.DataFrame) -> pd.Series:
    """Compute points per row using the legacy rule definition."""
    return (
        (df["net_carbs_g"] > 20.0).astype(int) * 2
        + (df["added_sugar_g"] >= 8.0).astype(int) * 2
        + (df["sodium_mg"] >= 500.0).astype(int) * 1
        - (df["fiber_g"] >= 5.0).astype(int) * 2
        - (df["protein_g"] >= 12.0).astype(int) * 1
    )


def _balanced_candidate_points(df: pd.DataFrame) -> pd.Series:
    """Compute points per row using the balanced candidate definition."""
    return (
        (df["net_carbs_g"] >= 25.0).astype(int) * 1
        + (df["added_sugar_g"] >= 10.0).astype(int) * 2
        + (df["sodium_mg"] >= 460.0).astype(int) * 1
        - (df["fiber_g"] >= 5.6).astype(int) * 2
        - (df["protein_g"] >= 10.0).astype(int) * 1
    )


def _current_adopted_points(df: pd.DataFrame) -> pd.Series:
    """Compute points per row using the currently adopted definition."""
    return (
        (df["carbs_g"] >= 30.0).astype(int) * 2
        + (df["added_sugar_g"] >= 10.0).astype(int) * 2
        + (df["sodium_mg"] >= 460.0).astype(int) * 1
        - (df["fiber_g"] >= 5.6).astype(int) * 2
        - (df["protein_g"] >= 10.0).astype(int) * 1
    )


def _points_for_rule(df: pd.DataFrame, rule_id: str) -> pd.Series:
    """Dispatch to the points function for a rule id.

    Args:
        df: Prepared input DataFrame from `_to_numeric`.
        rule_id: One of `legacy`, `balanced_candidate`, `current_adopted`.
    """
    if rule_id == "legacy":
        return _legacy_points(df)
    if rule_id == "balanced_candidate":
        return _balanced_candidate_points(df)
    if rule_id == "current_adopted":
        return _current_adopted_points(df)
    raise ValueError(f"Unknown rule_id: {rule_id}")


def _format_pct(value: float, decimals: int) -> str:
    """Format a float percentage with fixed decimal places."""
    return f"{value:.{decimals}f}%"


def _build_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render comparison rows as a Markdown table."""
    header = "| Rule set | Positive trigger | Positive label rate | Positive count | Total |\n"
    divider = "|---|---:|---:|---:|---:|\n"
    body = "".join(
        f"| {row['label']} | `>= {row['trigger']}` | {row['positive_rate_pct']} | {row['positive_count']} | {row['total_count']} |\n"
        for row in rows
    )
    return header + divider + body


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison run."""
    parser = argparse.ArgumentParser(description="Run threshold comparison for label rules.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/products_off_clean.csv"),
        help="Input dataset CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/labeling_logic/results/threshold_comparison_products_off_clean.json"),
        help="Machine-readable output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/labeling_logic/results/threshold_comparison_products_off_clean.md"),
        help="Human-readable markdown output path.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for percentages.",
    )
    return parser.parse_args()


def main() -> None:
    """Run threshold comparison and write reproducible JSON + Markdown outputs.

    Output files include:
    - Dataset metadata (`path`, `sha256`, row/column counts).
    - Per-ruleset positive rates for each trigger threshold tested.
    - A command hint that can be re-run to regenerate the same report.
    """
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()

    df = pd.read_csv(input_csv)
    prepared = _to_numeric(df)

    comparison_rows: list[dict[str, Any]] = []
    for spec in COMPARISON_ROWS:
        rule_id = str(spec["rule_id"])
        raw_trigger = spec["trigger"]
        if not isinstance(raw_trigger, int):
            raise TypeError(f"Trigger must be int, got: {type(raw_trigger)!r}")
        trigger = raw_trigger
        points = _points_for_rule(prepared, rule_id)
        positive = int((points >= trigger).sum())
        total = int(len(points))
        rate = (positive / total) * 100.0 if total else 0.0
        comparison_rows.append(
            {
                "rule_id": rule_id,
                "label": str(spec["label"]),
                "trigger": trigger,
                "positive_count": positive,
                "total_count": total,
                "positive_rate": rate,
                "positive_rate_pct": _format_pct(rate, args.decimals),
            }
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "dataset": {
            "path": str(input_csv),
            "sha256": _sha256(input_csv),
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
        },
        "rulesets": RULESET_NOTES,
        "comparison_rows": comparison_rows,
        "command_hint": (
            "python experiments/threshold_comparison.py "
            f"--input-csv {args.input_csv} --output-json {args.output_json} --output-md {args.output_md}"
        ),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    table = _build_markdown_table(comparison_rows)
    md = (
        "# Threshold Comparison Results (Reproducible Run)\n\n"
        f"Generated (UTC): `{generated_at}`\n\n"
        f"Dataset: `{args.input_csv}` (`n={len(df)}`)\n\n"
        f"Dataset SHA256: `{payload['dataset']['sha256']}`\n\n"
        f"Re-run command: `{payload['command_hint']}`\n\n"
        "Note: rows labeled as \"same thresholds, stricter trigger\" use identical nutrient rules,\n"
        "but raise the positive cutoff from `>= 2` to `>= 3`.\n\n"
        f"{table}"
    )
    output_md.write_text(md, encoding="utf-8")

    print(f"Wrote JSON: {output_json}")
    print(f"Wrote Markdown: {output_md}")
    for row in comparison_rows:
        print(
            f"- {row['label']} | >= {row['trigger']} | "
            f"{row['positive_rate_pct']} ({row['positive_count']}/{row['total_count']})"
        )


if __name__ == "__main__":
    main()
