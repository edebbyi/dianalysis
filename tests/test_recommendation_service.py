from __future__ import annotations

"""Behavior tests for end-to-end alternative selection service."""

import os
import unittest
from unittest.mock import patch

import pandas as pd

from dianalysis.recommendation.service import make_alternatives


class RecommendationServiceTests(unittest.TestCase):
    """Verifies dedupe, subtype preference, and strict soft-drink outputs."""

    def test_make_alternatives_drops_duplicate_name_brand(self) -> None:
        """Duplicate name+brand rows should not appear twice in returned alternatives."""
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

    def test_make_alternatives_prefers_same_bread_subtype_when_available(self) -> None:
        """Bagel queries should prioritize bagel candidates before generic bread matches."""
        query = {
            "name": "Tesco Original Bagels 5 Pack",
            "brand": "Tesco",
            "upc": "5000436049135",
            "category": "bread",
            "alt_group": "bread",
            "carbs_g": 44.4,
            "fiber_g": 2.3,
            "sugar_g": 4.5,
            "added_sugar_g": 2.2,
            "protein_g": 7.9,
            "fat_g": 1.3,
            "sodium_mg": 201.0,
            "ingredients_text": "wheat flour water yeast",
            "categories_all": "bread|bagels",
        }
        df = pd.DataFrame(
            [
                {
                    "name": "Bagels Plain",
                    "brand": "THOMAS'",
                    "upc": "a1",
                    "category": "bread",
                    "alt_group": "bread",
                    "risk_score": 11,
                    "risk_display": "11",
                    "fiber_g": 3.0,
                    "net_carbs_g": 39.0,
                    "sugar_g": 4.0,
                    "ingredients_text": "wheat flour water yeast",
                    "categories_all": "bread|bagels",
                },
                {
                    "name": "Organic Bagels EPIC EVERYTHING",
                    "brand": "Dave's Killer Bread",
                    "upc": "a2",
                    "category": "bread",
                    "alt_group": "bread",
                    "risk_score": 2,
                    "risk_display": "2",
                    "fiber_g": 5.0,
                    "net_carbs_g": 38.0,
                    "sugar_g": 5.0,
                    "ingredients_text": "wheat flour water seeds",
                    "categories_all": "bread|bagels",
                },
                {
                    "name": "Thin-Sliced Organic Bread Good Seed",
                    "brand": "Dave's Killer Bread",
                    "upc": "b1",
                    "category": "bread",
                    "alt_group": "bread",
                    "risk_score": 1,
                    "risk_display": "1",
                    "fiber_g": 3.0,
                    "net_carbs_g": 10.0,
                    "sugar_g": 2.0,
                    "ingredients_text": "wheat flour seeds",
                    "categories_all": "bread",
                },
                {
                    "name": "LIGHT RYE CRISPBREAD SWEDISH STYLE",
                    "brand": "wasa",
                    "upc": "b2",
                    "category": "bread",
                    "alt_group": "bread",
                    "risk_score": 0,
                    "risk_display": "Very low (<1)",
                    "fiber_g": 4.0,
                    "net_carbs_g": 9.0,
                    "sugar_g": 1.0,
                    "ingredients_text": "rye flour",
                    "categories_all": "bread|crispbread",
                },
            ]
        )

        with patch("dianalysis.recommendation.service.search_similar_candidates", return_value=df):
            out = make_alternatives(df, query, score_this=99, k=3)

        self.assertTrue(out)
        names = [x["name"] for x in out]
        self.assertIn("Bagels Plain", names[:2])
        self.assertIn("Organic Bagels EPIC EVERYTHING", names[:2])

    def test_make_alternatives_returns_empty_for_soft_drink_when_only_dairy_or_water(self) -> None:
        """Strict soft-drink lookup should return empty when only water/dairy candidates exist."""
        query = {
            "name": "Schweppes",
            "brand": "Schweppes",
            "upc": "5449000046413",
            "category": "drink",
            "alt_group": "drink",
            "category_main": "drink",
            "alt_group_fine": "drink:soft_drink",
            "carbs_g": 1.2,
            "fiber_g": 0.0,
            "sugar_g": 1.2,
            "added_sugar_g": 1.2,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "sodium_mg": 10.0,
            "ingredients_text": "carbonated water, sugar",
            "categories_all": "",
        }
        df = pd.DataFrame(
            [
                {
                    "name": "Organic Soy",
                    "brand": "Silk",
                    "upc": "d1",
                    "category": "drink",
                    "alt_group": "drink",
                    "category_main": "drink",
                    "alt_group_fine": "drink:dairy_drink",
                    "risk_score": 0,
                    "risk_display": "Very low (<1)",
                    "fiber_g": 2.0,
                    "net_carbs_g": 1.0,
                    "sugar_g": 0.0,
                    "ingredients_text": "soy milk",
                    "categories_all": "soy milk beverage",
                },
                {
                    "name": "Sparkling Water",
                    "brand": "Brand B",
                    "upc": "d2",
                    "category": "drink",
                    "alt_group": "drink",
                    "category_main": "drink",
                    "alt_group_fine": "drink:water",
                    "risk_score": 0,
                    "risk_display": "Very low (<1)",
                    "fiber_g": 0.0,
                    "net_carbs_g": 0.0,
                    "sugar_g": 0.0,
                    "ingredients_text": "carbonated water",
                    "categories_all": "water beverage",
                },
            ]
        )

        with patch("dianalysis.recommendation.service.search_similar_candidates", return_value=df):
            out = make_alternatives(df, query, score_this=1, k=3)

        self.assertEqual(out, [])

    def test_make_alternatives_uses_configurable_retrieval_limit(self) -> None:
        """Retrieval candidate limit should be configurable via environment variable."""
        query = {
            "name": "Query",
            "brand": "Q",
            "upc": "999",
            "category": "bread",
            "alt_group": "bread",
            "carbs_g": 30.0,
            "fiber_g": 1.0,
            "sugar_g": 2.0,
            "added_sugar_g": 0.0,
            "protein_g": 4.0,
            "fat_g": 1.0,
            "sodium_mg": 100.0,
            "ingredients_text": "flour water yeast",
            "categories_all": "bread",
        }
        df = pd.DataFrame(
            [
                {
                    "name": "Alt Bread",
                    "brand": "A",
                    "upc": "1000",
                    "category": "bread",
                    "alt_group": "bread",
                    "risk_score": 0,
                    "risk_display": "Very low (<1)",
                    "fiber_g": 2.0,
                    "net_carbs_g": 10.0,
                    "sugar_g": 1.0,
                    "ingredients_text": "flour water",
                    "categories_all": "bread",
                }
            ]
        )

        with (
            patch.dict(os.environ, {"DIANALYSIS_RETRIEVAL_CANDIDATE_LIMIT": "80"}, clear=False),
            patch("dianalysis.recommendation.service.search_similar_candidates", return_value=pd.DataFrame()) as mocked,
        ):
            _ = make_alternatives(df, query, score_this=99, k=3)

        self.assertTrue(mocked.called)
        self.assertEqual(int(mocked.call_args.kwargs.get("limit", 0)), 80)


if __name__ == "__main__":
    unittest.main()
