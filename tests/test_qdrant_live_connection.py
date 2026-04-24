from __future__ import annotations

"""Optional live integration test for Qdrant Cloud connectivity and barcode retrieval."""

import os
from pathlib import Path
import unittest

import pandas as pd

from dianalysis.model import load_model
from dianalysis.recommendation.vector_client import collection_name, qdrant_client
from dianalysis.scoring.barcode import score_by_barcode


def _live_enabled() -> bool:
    """Return True when the caller explicitly enables live cloud tests."""
    return str(os.getenv("DIANALYSIS_LIVE_QDRANT_TEST", "")).strip().lower() in {"1", "true", "yes"}


def _live_barcodes() -> list[str]:
    """Return comma-separated test barcodes from env or defaults."""
    raw = str(os.getenv("DIANALYSIS_LIVE_BARCODES", "049000028904,5000436049135"))
    return [x.strip() for x in raw.split(",") if x.strip()]


@unittest.skipUnless(_live_enabled(), "Set DIANALYSIS_LIVE_QDRANT_TEST=1 to run live Qdrant integration tests.")
class QdrantLiveConnectionTests(unittest.TestCase):
    """Tests for cloud connection plus retrieval over real barcode flows."""

    @classmethod
    def setUpClass(cls) -> None:
        required = ("QDRANT_URL", "QDRANT_API_KEY")
        missing = [name for name in required if not str(os.getenv(name, "")).strip()]
        if missing:
            raise unittest.SkipTest(f"Missing required env var(s) for live test: {', '.join(missing)}")

        # Force semantic retrieval path for this integration test.
        os.environ["DIANALYSIS_RETRIEVAL_BACKEND"] = "qdrant"

        scored_path = Path("data/products_off_clean_scored.csv")
        if not scored_path.exists():
            raise unittest.SkipTest(
                "Missing data/products_off_clean_scored.csv. Run rescore/sync before live integration test."
            )

        cls.model, _ = load_model("artifacts")
        cls.df_candidates = pd.read_csv(scored_path, dtype={"upc": str})

    def test_connection_and_collection_available(self) -> None:
        """Qdrant client should connect and expose the target collection."""
        client = qdrant_client()
        response = client.get_collections()
        names = [str(c.name) for c in getattr(response, "collections", [])]
        self.assertIn(collection_name(), names)

    def test_barcode_smoke_returns_ranked_alternatives(self) -> None:
        """Known barcodes should score without error and return alternatives from indexed data."""
        for barcode in _live_barcodes():
            with self.subTest(barcode=barcode):
                result = score_by_barcode(barcode, self.model, self.df_candidates)
                self.assertIsNone(result.get("error"))
                self.assertFalse(bool(result.get("insufficient_data")))
                self.assertEqual(str(result.get("alternatives_source", "")), "database")
                self.assertGreaterEqual(len(result.get("alternatives", []) or []), 1)


if __name__ == "__main__":
    unittest.main()

