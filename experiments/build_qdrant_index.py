"""
Build a Qdrant semantic index from the product dataset.

Usage:
    DIANALYSIS_RETRIEVAL_BACKEND=qdrant python experiments/build_qdrant_index.py
"""

from __future__ import annotations

import argparse

import pandas as pd

from dianalysis.vector_retrieval import index_dataframe, retrieval_enabled


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant index for product retrieval.")
    parser.add_argument("--data-path", default="data/products_off_clean.csv")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection.")
    args = parser.parse_args()

    if not retrieval_enabled():
        print(
            "Qdrant retrieval is not enabled. Set DIANALYSIS_RETRIEVAL_BACKEND=qdrant "
            "and install qdrant-client + sentence-transformers."
        )
        return

    df = pd.read_csv(args.data_path, dtype={"upc": str})
    n = index_dataframe(df, recreate=args.recreate)
    print(f"Indexed {n} products into Qdrant.")


if __name__ == "__main__":
    main()

