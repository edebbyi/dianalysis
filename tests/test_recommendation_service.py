from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from dianalysis.recommendation.service import make_alternatives


class RecommendationServiceTests(unittest.TestCase):
    def test_make_alternatives_drops_duplicate_name_brand(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "name": "Query Oats",
                    "brand": "Q",
                    "upc": "1",
                    "category": "grain",
                    "alt_group": "oats",
                    "risk_score": 8,
                    "risk_display": "high",
                    "fiber_g": 2.0,
                    "net_carbs_g": 20.0,
                    "sugar_g": 8.0,
                    "ingredients_text": "oats, sugar",
                },
                {
                    "name": "Alt Oats A",
                    "brand": "A",
                    "upc": "2",
                    "category": "grain",
                    "alt_group": "oats",
                    "risk_score": 4,
                    "risk_display": "low",
                    "fiber_g": 5.0,
                    "net_carbs_g": 10.0,
                    "sugar_g": 2.0,
                    "ingredients_text": "oats",
                },
                {
                    "name": "Alt Oats A",
                    "brand": "A",
                    "upc": "3",
                    "category": "grain",
                    "alt_group": "oats",
                    "risk_score": 4,
                    "risk_display": "low",
                    "fiber_g": 5.0,
                    "net_carbs_g": 10.0,
                    "sugar_g": 2.0,
                    "ingredients_text": "oats",
                },
                {
                    "name": "Alt Oats B",
                    "brand": "B",
                    "upc": "4",
                    "category": "grain",
                    "alt_group": "oats",
                    "risk_score": 3,
                    "risk_display": "low",
                    "fiber_g": 4.0,
                    "net_carbs_g": 11.0,
                    "sugar_g": 1.0,
                    "ingredients_text": "oats",
                },
            ]
        )

        query = df.iloc[0].to_dict()
        with patch("dianalysis.recommendation.service.search_similar_candidates", return_value=pd.DataFrame()):
            out = make_alternatives(df, query, score_this=8, k=3)

        names = [x["name"] for x in out]
        self.assertEqual(names.count("Alt Oats A"), 1)


if __name__ == "__main__":
    unittest.main()
