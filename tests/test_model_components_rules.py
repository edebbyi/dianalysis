from __future__ import annotations

"""Rule-level tests for labeling points and confidence metadata."""

import unittest

from dianalysis.model_components import rule_data_confidence, rule_points_reasons_meta


class ModelComponentRuleTests(unittest.TestCase):
    """Validates rule behavior for missing data and beverage-specific logic."""

    def test_inferred_added_sugar_for_processed_category(self) -> None:
        """Processed categories can infer added-sugar risk from total sugar when missing."""
        row = {
            "category": "drink",
            "alt_group": "drink",
            "category_main": "drink",
            "carbs_g": 11.7,
            "sugar_g": 11.7,
            "added_sugar_g": None,
            "fiber_g": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "sodium_mg": None,
        }
        pts, reasons, meta = rule_points_reasons_meta(row)
        self.assertGreaterEqual(pts, 3)
        self.assertTrue(meta.get("inferred_added_sugar"))
        self.assertTrue(any("inferred risk from total sugar" in r for r in reasons))

    def test_beverage_uses_lower_carb_threshold(self) -> None:
        """Beverage rows should use the lower carb threshold branch."""
        row = {
            "category": "drink",
            "alt_group": "drink",
            "category_main": "drink",
            "carbs_g": 22.0,
            "sugar_g": 0.0,
            "added_sugar_g": 0.0,
            "fiber_g": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "sodium_mg": 10.0,
        }
        pts, _reasons, meta = rule_points_reasons_meta(row)
        self.assertGreaterEqual(pts, 2)
        self.assertTrue(meta.get("beverage_threshold_used"))

    def test_non_processed_category_does_not_infer_added_sugar(self) -> None:
        """Non-processed categories should not infer added sugar from total sugar."""
        row = {
            "category": "dairy",
            "alt_group": "dairy",
            "category_main": "dairy",
            "carbs_g": 12.0,
            "sugar_g": 12.0,
            "added_sugar_g": None,
            "fiber_g": 0.0,
            "protein_g": 8.0,
            "fat_g": 8.0,
            "sodium_mg": 100.0,
        }
        pts, _reasons, meta = rule_points_reasons_meta(row)
        self.assertLess(pts, 2)
        self.assertFalse(meta.get("inferred_added_sugar"))

    def test_data_confidence_low_when_critical_fields_missing(self) -> None:
        """Missing key nutrition fields should lower data confidence."""
        row = {
            "category": "drink",
            "alt_group": "drink",
            "category_main": "drink",
            "carbs_g": 10.0,
            "sugar_g": 12.0,
            "added_sugar_g": None,
            "fiber_g": None,
            "protein_g": None,
            "fat_g": 0.0,
            "sodium_mg": None,
        }
        confidence, notes = rule_data_confidence(row)
        self.assertEqual(confidence, "low")
        self.assertTrue(notes)


if __name__ == "__main__":
    unittest.main()
