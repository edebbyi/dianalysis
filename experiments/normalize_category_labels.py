"""Clean up mislabeled category columns in the source CSV.

This script is intentionally conservative:
- It only applies high-confidence relabels for known noisy cases.
- It aligns `category` with known `alt_group` values when those are present.

How to use it:
- Make targets:
  make normalize-labels
  make normalize-labels-write
  make normalize-labels-inplace
- Preview only (no file is written):
  python experiments/normalize_category_labels.py
- Write to a new file (safe default):
  python experiments/normalize_category_labels.py --write
- Overwrite the input file:
  python experiments/normalize_category_labels.py --write --in-place
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


CANON_BY_GROUP: dict[str, str] = {
    "oats": "grain",
    "rice": "grain",
    "quinoa": "grain",
    "pasta-noodles": "grain",
    "cereal": "cereal",
    "granola": "cereal",
    "bread": "bread",
    "drink": "drink",
    "ice-cream": "dessert",
    "dairy": "dairy",
    "nuts-seeds": "nut",
    "snack": "snack",
}

BREAD_SIGNALS = (
    "bread",
    "bagel",
    "bagels",
    "bun",
    "buns",
    "roll",
    "rolls",
    "muffin",
    "muffins",
    "tortilla",
    "tortillas",
    "wrap",
    "wraps",
    "pita",
    "naan",
    "ciabatta",
    "sourdough",
    "flatbread",
    "flatbreads",
    "loaf",
    "crispbread",
    "breadstick",
    "breadsticks",
    "crostini",
    "crackers",
)

SNACK_SIGNALS = ("popcorn", "chip", "chips", "crisps", "nacho", "pretzel")


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Return `True` when at least one keyword appears in `text`."""
    return any(k in text for k in keywords)


def normalize_row(row: dict[str, str]) -> tuple[dict[str, str], str | None]:
    """Apply label cleanup rules to one CSV row.

    The function may update `category` and/or `alt_group`.
    It returns the updated row plus a short reason code when a change is made.
    If no change is needed, reason is `None`.
    """
    out = dict(row)
    category = (out.get("category") or "").strip().lower()
    alt_group = (out.get("alt_group") or "").strip().lower()
    name = (out.get("name") or "").strip().lower()
    cats = (out.get("categories_all") or "").strip().lower()
    text = f"{name} {cats}"

    # Rule 1: Bread-like products mislabeled as cereal.
    if (
        category == "cereal"
        and alt_group == "cereal"
        and _has_any(text, BREAD_SIGNALS)
        and "cereal" not in text
    ):
        out["category"] = "bread"
        out["alt_group"] = "bread"
        return out, "cereal_to_bread_by_text"

    # Rule 2: Snack-like products mislabeled as bread.
    if (
        category == "bread"
        and alt_group == "bread"
        and _has_any(text, SNACK_SIGNALS)
        and not _has_any(text, BREAD_SIGNALS)
    ):
        out["category"] = "snack"
        out["alt_group"] = "snack"
        return out, "bread_to_snack_by_text"

    # Rule 3: For known groups, keep category aligned to the expected value.
    expected_category = CANON_BY_GROUP.get(alt_group)
    if expected_category and category and category != expected_category:
        out["category"] = expected_category
        return out, "category_canonicalized_from_alt_group"

    return out, None


def main() -> int:
    """Run the command-line flow for label cleanup.

    Steps:
    - read input CSV
    - apply `normalize_row` to every row
    - print a short change summary
    - optionally write a cleaned CSV file
    """
    parser = argparse.ArgumentParser(
        description=(
            "Clean category/alt_group labels with safe rules "
            "(bread/snack fixes plus expected category alignment for known groups)."
        )
    )
    parser.add_argument("--input", default="data/products_off_clean.csv", help="Input CSV path")
    parser.add_argument("--write", action="store_true", help="Write cleaned CSV output")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <input_stem>_normalized.csv)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSV (use only when you intend to replace source data)",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"missing input file: {path}")
    if args.in_place and args.output:
        parser.error("use either --in-place or --output, not both")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("input CSV has no header")
        rows = list(reader)

    changed_rows: list[tuple[int, str, str, str, str, str, str]] = []
    reason_counts: Counter[str] = Counter()
    out_rows: list[dict[str, str]] = []

    for line_no, row in enumerate(rows, start=2):
        before_c = (row.get("category") or "").strip()
        before_g = (row.get("alt_group") or "").strip()
        updated, reason = normalize_row(row)
        after_c = (updated.get("category") or "").strip()
        after_g = (updated.get("alt_group") or "").strip()

        if reason is not None and (before_c != after_c or before_g != after_g):
            reason_counts[reason] += 1
            changed_rows.append(
                (
                    line_no,
                    row.get("name", ""),
                    row.get("brand", ""),
                    before_c,
                    before_g,
                    after_c,
                    after_g,
                )
            )
        out_rows.append(updated)

    print(f"rows: {len(rows)}")
    print(f"changed: {len(changed_rows)}")
    for reason, count in reason_counts.items():
        print(f"{reason}: {count}")

    for rec in changed_rows[:80]:
        print(
            f"line {rec[0]} | {rec[1]} | {rec[2]} | "
            f"{rec[3]}/{rec[4]} -> {rec[5]}/{rec[6]}"
        )

    if args.write:
        if args.in_place:
            out_path = path
        elif args.output:
            out_path = Path(args.output)
        else:
            out_path = path.with_name(f"{path.stem}_normalized{path.suffix or '.csv'}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"wrote cleaned CSV to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
