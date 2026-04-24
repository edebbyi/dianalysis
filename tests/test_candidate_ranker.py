from __future__ import annotations

"""Tests for candidate ranking and scoring robustness."""

import unittest

import pandas as pd

from dianalysis.recommendation.candidate_ranker import ingredient_overlap_score, rank_candidates


class CandidateRankerTests(unittest.TestCase):
    """Validates ranking order and type-safety in ranking features."""

    def test_ingredient_overlap_score_basic(self) -> None:
        """Simple overlap should produce a non-trivial similarity score."""
        s = ingredient_overlap_score("oats water salt", "oats salt")
        self.assertGreater(s, 0.3)

    def test_rank_candidates_penalizes_cross_category(self) -> None:
        """Cross-category candidates should not outrank same-group matches."""
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
            fine_group_key="grain:oats",
            name_text="query oats",
            query_ingredients="oats sugar",
            k=2,
        )
        self.assertEqual(ranked.iloc[0]["name"], "Same Group")

    def test_rank_candidates_handles_string_typed_retrieval_scores(self) -> None:
        """Ranking should coerce string-typed numeric fields without crashing."""
        cand = pd.DataFrame(
            [
                {
                    "name": "Candidate A",
                    "brand": "A",
                    "category": "drink",
                    "alt_group": "drink",
                    "risk_score": "1",
                    "risk_display": "1",
                    "fiber_g": "0",
                    "net_carbs_g": "0",
                    "sugar_g": "0",
                    "categories_all": "soft drink cola",
                    "ingredients_text": "carbonated water",
                    "_stage": "strict",
                    "_retrieval_score": "0.7",
                }
            ]
        )

        ranked = rank_candidates(
            cand,
            this_net=10.0,
            sugar_this=8.0,
            fiber_this=0.0,
            cat="drink",
            group_key="drink",
            fine_group_key="drink:soft_drink",
            name_text="Schweppes",
            query_ingredients="carbonated water sugar",
            k=1,
        )
        self.assertEqual(len(ranked), 1)
        self.assertTrue(pd.notna(ranked.iloc[0]["final_score"]))


if __name__ == "__main__":
    unittest.main()
