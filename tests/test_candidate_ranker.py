from __future__ import annotations

import unittest

import pandas as pd

from dianalysis.recommendation.candidate_ranker import ingredient_overlap_score, rank_candidates


class CandidateRankerTests(unittest.TestCase):
    def test_ingredient_overlap_score_basic(self) -> None:
        s = ingredient_overlap_score("oats water salt", "oats salt")
        self.assertGreater(s, 0.3)

    def test_rank_candidates_penalizes_cross_category(self) -> None:
        cand = pd.DataFrame(
            [
                {
                    "name": "Same Group",
                    "brand": "A",
                    "category": "grain",
                    "alt_group": "oats",
                    "risk_score": 4,
                    "risk_display": "low",
                    "fiber_g": 5,
                    "net_carbs_g": 9,
                    "sugar_g": 2,
                    "categories_all": "oats",
                    "ingredients_text": "oats water",
                    "_stage": "strict",
                    "_retrieval_score": 0.20,
                },
                {
                    "name": "Cross Category",
                    "brand": "B",
                    "category": "dessert",
                    "alt_group": "ice-cream",
                    "risk_score": 3,
                    "risk_display": "low",
                    "fiber_g": 6,
                    "net_carbs_g": 8,
                    "sugar_g": 1,
                    "categories_all": "dessert",
                    "ingredients_text": "milk sugar",
                    "_stage": "strict",
                    "_retrieval_score": 0.95,
                },
            ]
        )

        ranked = rank_candidates(
            cand,
            this_net=20.0,
            sugar_this=8.0,
            fiber_this=2.0,
            cat="grain",
            group_key="oats",
            name_text="query oats",
            query_ingredients="oats sugar",
            k=2,
        )
        self.assertEqual(ranked.iloc[0]["name"], "Same Group")


if __name__ == "__main__":
    unittest.main()
