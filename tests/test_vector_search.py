from __future__ import annotations

from dataclasses import dataclass
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dianalysis.recommendation.vector_search import search_similar_candidates
from dianalysis.recommendation.vector_index import product_key


@dataclass
class FakeHit:
    payload: dict
    score: float


class VectorSearchTests(unittest.TestCase):
    def test_search_similar_candidates_dedupes_product_keys(self) -> None:
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
            out = search_similar_candidates(df, query_item, cat="grain", group_key="rice", limit=5)

        names = out["name"].tolist()
        self.assertEqual(names.count("Brown Rice & Quinoa"), 1)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
