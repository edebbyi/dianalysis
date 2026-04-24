from __future__ import annotations

"""Tests for vector retrieval query flow and de-duplication behavior."""

from dataclasses import dataclass
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dianalysis.recommendation.vector_search import query_points, search_similar_candidates
from dianalysis.recommendation.vector_index import product_key


@dataclass
class FakeHit:
    """Simple stand-in for a vector-search hit object."""

    payload: dict
    score: float


class VectorSearchTests(unittest.TestCase):
    """Covers fallback query filters and result dedupe by product key."""

    def test_query_points_uses_fine_group_then_category_fallback(self) -> None:
        """Search should try fine-group filter first, then broaden to category-level."""
        calls: list = []

        def fake_search(query_vector, *, query_filter, local_limit, target_collection):
            calls.append(query_filter)
            if len(calls) == 1:
                return []  # no fine-group hits
            return []

        with patch("dianalysis.recommendation.vector_search._search_with_filter", side_effect=fake_search):
            _ = query_points(
                [0.1, 0.2, 0.3],
                target_collection="demo",
                limit=5,
                cat="bread",
                group_key="bread",
                fine_group_key="bread:bagel",
            )

        self.assertGreaterEqual(len(calls), 2)
        self.assertIn("alt_group_fine", str(calls[0]))
        self.assertTrue(any("category_main" in str(c) or "category" in str(c) for c in calls[1:]))

    def test_search_similar_candidates_dedupes_product_keys(self) -> None:
        """Duplicate rows with the same product key should collapse to one candidate."""
        df = pd.DataFrame(
            [
                {
                    "name": "Brown Rice & Quinoa",
                    "brand": "Minute",
                    "upc": "17400140380",
                    "category": "grain",
                    "alt_group": "rice",
                },
                {
                    "name": "Brown Rice & Quinoa",
                    "brand": "Minute",
                    "upc": "17400140380",
                    "category": "grain",
                    "alt_group": "rice",
                },
                {"name": "Quinoa Blend", "brand": "BrandX", "upc": "111", "category": "grain", "alt_group": "rice"},
            ]
        )
        query_item = df.iloc[0].to_dict()

        k1 = product_key(df.iloc[0].to_dict())
        k2 = product_key(df.iloc[2].to_dict())
        fake_hits = [
            FakeHit(payload={"product_key": k1}, score=0.99),
            FakeHit(payload={"product_key": k2}, score=0.80),
        ]

        class FakeEmbed:
            def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
                return np.array([[0.1, 0.2, 0.3]])

        with (
            patch("dianalysis.recommendation.vector_search.retrieval_enabled", return_value=True),
            patch("dianalysis.recommendation.vector_search.ensure_collection_exists"),
            patch("dianalysis.recommendation.vector_search.embedder", return_value=FakeEmbed()),
            patch("dianalysis.recommendation.vector_search.query_points", return_value=fake_hits),
        ):
            out = search_similar_candidates(
                df,
                query_item,
                cat="grain",
                group_key="rice",
                fine_group_key="grain:rice",
                limit=5,
            )

        names = out["name"].tolist()
        self.assertEqual(names.count("Brown Rice & Quinoa"), 1)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
