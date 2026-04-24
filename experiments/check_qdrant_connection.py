"""
Check connectivity and authentication to Qdrant.

Usage:
    PYTHONPATH=. python experiments/check_qdrant_connection.py
    PYTHONPATH=. python experiments/check_qdrant_connection.py --require-collection
"""

from __future__ import annotations

import argparse
import sys

from dianalysis.recommendation.vector_client import (
    collection_name,
    qdrant_api_key,
    qdrant_client,
    qdrant_url,
)


def _mask_api_key(value: str) -> str:
    if not value:
        return "not set"
    return f"set (length {len(value)})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Qdrant connectivity and auth.")
    parser.add_argument(
        "--collection",
        default=collection_name(),
        help="Collection name to verify (default: DIANALYSIS_QDRANT_COLLECTION or dianalysis_products).",
    )
    parser.add_argument(
        "--require-collection",
        action="store_true",
        help="Fail if the target collection is missing.",
    )
    args = parser.parse_args()

    print(f"QDRANT_URL: {qdrant_url()}")
    print(f"QDRANT_API_KEY: {_mask_api_key(qdrant_api_key())}")
    print(f"Target collection: {args.collection}")

    try:
        client = qdrant_client()
        collections_response = client.get_collections()
    except Exception as exc:
        print(f"Connection failed: {exc}")
        return 1

    names: list[str] = []
    try:
        names = [str(c.name) for c in getattr(collections_response, "collections", [])]
    except Exception:
        names = []

    print(f"Connection OK. Found {len(names)} collection(s).")
    if names:
        preview = ", ".join(sorted(names)[:10])
        print(f"Collections: {preview}")

    exists = args.collection in set(names)
    if exists:
        print(f"Collection '{args.collection}' exists.")
        return 0

    msg = f"Collection '{args.collection}' not found."
    if args.require_collection:
        print(msg)
        return 2

    print(msg + " Connection/auth are still valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

