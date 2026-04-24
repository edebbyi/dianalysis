from __future__ import annotations

"""Edge-case tests for candidate filtering and fine-group inference."""

import unittest

import pandas as pd

from dianalysis.recommendation.candidate_pool import infer_alt_group_fine
from dianalysis.recommendation.candidate_filters import apply_style_filters, drop_duplicate_candidates


class CandidateFilterEdgeTests(unittest.TestCase):
    """Covers duplicate cleanup and soft-drink style filtering behavior."""

    def test_drop_duplicate_candidates_normalizes_brand_punctuation(self) -> None:
        """Brand punctuation variants should collapse into one logical product."""
        cand = pd.DataFrame(
            [
                {"name": "Coca-Cola Zero", "brand": "Coca-Cola", "upc": "111"},
                {"name": "Coca-Cola Zero", "brand": "Coca Cola", "upc": "222"},
                {"name": "Coca-Cola Zero Sugar", "brand": "Coca-Cola", "upc": "333"},
            ]
        )
        out = drop_duplicate_candidates(cand)
        names = out["name"].tolist()
        self.assertEqual(len(out), 2)
        self.assertIn("Coca-Cola Zero", names)
        self.assertIn("Coca-Cola Zero Sugar", names)

    def test_soft_drink_filter_keeps_soda_like_and_excludes_water_and_dairy(self) -> None:
        """Soft-drink queries should keep soda-like items, not water or dairy drinks."""
        cand = pd.DataFrame(
            [
                {
                    "name": "Organic Soy",
                    "brand": "Silk",
                    "categories_all": "soy milk beverage",
                    "category": "drink",
                    "alt_group": "drink",
                },
                {
                    "name": "Diet Cola",
                    "brand": "Brand A",
                    "categories_all": "soft drink cola",
                    "category": "drink",
                    "alt_group": "drink",
                },
                {
                    "name": "Sparkling Water",
                    "brand": "Brand B",
                    "categories_all": "water beverage",
                    "category": "drink",
                    "alt_group": "drink",
                },
                {
                    "name": "Cola Zero Sugar",
                    "brand": "Brand C",
                    "categories_all": "soft drink cola",
                    "category": "drink",
                    "alt_group": "drink",
                },
            ]
        )
        out = apply_style_filters(cand, group_key="drink", name_text="Coca-Cola Original Taste Soda")
        names = out["name"].tolist()
        self.assertIn("Diet Cola", names)
        self.assertIn("Cola Zero Sugar", names)
        self.assertNotIn("Sparkling Water", names)
        self.assertNotIn("Organic Soy", names)

    def test_infer_alt_group_fine_labels_schweppes_as_soft_drink(self) -> None:
        """Schweppes-style ingredient text should map to soft-drink fine group."""
        fine = infer_alt_group_fine(
            category_main="drink",
            alt_group="drink",
            name_text="Schweppes",
            categories_all="",
            ingredients_text="EAU GAZEIFIEE, SUCRE, JUS DE CITRON",
        )
        self.assertEqual(fine, "drink:soft_drink")

    def test_soft_drink_filter_returns_empty_when_only_water_or_dairy_exist(self) -> None:
        """If no soda-like items exist, strict soft-drink filtering returns no candidates."""
        cand = pd.DataFrame(
            [
                {
                    "name": "Organic Soy",
                    "brand": "Silk",
                    "categories_all": "soy milk beverage",
                    "category": "drink",
                    "alt_group": "drink",
                },
                {
                    "name": "Sparkling Water",
                    "brand": "Brand B",
                    "categories_all": "water beverage",
                    "category": "drink",
                    "alt_group": "drink",
                },
            ]
        )
        out = apply_style_filters(
            cand,
            group_key="drink",
            name_text="Schweppes",
            fine_group_key="drink:soft_drink",
        )
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
