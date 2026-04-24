from __future__ import annotations

"""Edge-case tests for barcode scoring behavior with sparse nutrition fields."""

import os
import unittest
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("DIANALYSIS_RETRIEVAL_BACKEND", "heuristic")

from dianalysis.scoring.barcode import score_by_barcode


class BarcodeEdgeTests(unittest.TestCase):
    """Checks score visibility rules when barcode nutrition data is incomplete."""

    def test_sparse_barcode_data_hides_score(self) -> None:
        """Hide score when critical nutrition values are missing, but keep alternatives."""
        item = {
            "name": "Diet Cola",
            "brand": "Brand A",
            "category": "drink",
            "alt_group": "drink",
            "carbs_g": 0.0,
            "sugar_g": None,
            "added_sugar_g": None,
            "fiber_g": None,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "sodium_mg": 40.0,
            "__display": {
                "carbs_g": "0g",
                "sugar_g": "not listed",
                "added_sugar_g": "not listed",
                "fiber_g": "not listed",
                "protein_g": "0g",
                "fat_g": "0g",
                "sodium_mg": "40mg",
            },
        }
        fake_scored = {
            "risk_score": 1,
            "risk_display": "Very low (<1)",
            "reasons": ["Low sodium"],
            "alternatives": [{"name": "Alt Drink", "risk_score": 0}],
            "notes": [],
        }

        with (
            patch("dianalysis.off_pipeline.fetch_and_normalize_off", return_value=item),
            patch("dianalysis.off_pipeline.infer_alt_group_for_item", side_effect=lambda x: x),
            patch("dianalysis.off_pipeline.fetch_category_products", return_value=[]),
            patch("dianalysis.scoring.barcode.score_item", return_value=fake_scored),
        ):
            result = score_by_barcode("049000028904", model=object(), df_candidates=pd.DataFrame())

        self.assertTrue(result.get("insufficient_data"))
        self.assertEqual(result.get("risk_display"), "—")
        self.assertEqual(result.get("reasons"), [])
        self.assertTrue(any("Score hidden" in str(n) for n in result.get("notes", [])))
        self.assertTrue(result.get("alternatives"))

    def test_sufficient_barcode_data_keeps_score(self) -> None:
        """Keep score visible when enough nutrition data is present."""
        item = {
            "name": "Bread",
            "brand": "Brand B",
            "category": "bread",
            "alt_group": "bread",
            "carbs_g": 30.0,
            "sugar_g": 4.0,
            "added_sugar_g": 1.0,
            "fiber_g": 3.0,
            "protein_g": 5.0,
            "fat_g": 2.0,
            "sodium_mg": 210.0,
            "__display": {
                "carbs_g": "30g",
                "sugar_g": "4g",
                "added_sugar_g": "1g",
                "fiber_g": "3g",
                "protein_g": "5g",
                "fat_g": "2g",
                "sodium_mg": "210mg",
            },
        }
        fake_scored = {
            "risk_score": 55,
            "risk_display": "55",
            "reasons": ["High carbs"],
            "alternatives": [],
            "notes": [],
        }

        with (
            patch("dianalysis.off_pipeline.fetch_and_normalize_off", return_value=item),
            patch("dianalysis.off_pipeline.infer_alt_group_for_item", side_effect=lambda x: x),
            patch("dianalysis.off_pipeline.fetch_category_products", return_value=[]),
            patch("dianalysis.scoring.barcode.score_item", return_value=fake_scored),
        ):
            result = score_by_barcode("014100074670", model=object(), df_candidates=pd.DataFrame())

        self.assertFalse(bool(result.get("insufficient_data")))
        self.assertEqual(result.get("risk_display"), "55")


if __name__ == "__main__":
    unittest.main()
