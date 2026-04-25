from __future__ import annotations

"""Edge tests for scoring pipeline outputs and alternative guardrails."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

os.environ.setdefault("DIANALYSIS_RETRIEVAL_BACKEND", "heuristic")

from dianalysis.scoring.pipeline import _enforce_lower_risk_alternatives, score_item


class FakeModel:
    """Tiny deterministic model used to keep tests fast and predictable."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        carbs = pd.to_numeric(X.get("carbs_g"), errors="coerce").fillna(0.0).clip(lower=0.0)
        p1 = (carbs / 100.0).clip(0.0, 1.0).to_numpy(dtype=float)
        return np.column_stack([1.0 - p1, p1])


def _row(
    *,
    name: str,
    brand: str,
    upc: str,
    category: str,
    alt_group: str,
    carbs_g: float,
    fiber_g: float = 0.0,
    sugar_g: float = 0.0,
    added_sugar_g: float = 0.0,
    sodium_mg: float = 0.0,
) -> dict:
    """Build a minimal product row with defaults for scoring tests."""
    return {
        "name": name,
        "brand": brand,
        "upc": upc,
        "source": "test",
        "created_at": "2026-01-01T00:00:00Z",
        "category": category,
        "alt_group": alt_group,
        "serving_g": 100.0,
        "calories": 100.0,
        "carbs_g": float(carbs_g),
        "fiber_g": float(fiber_g),
        "sugar_g": float(sugar_g),
        "added_sugar_g": float(added_sugar_g),
        "sugar_alcohols_g": 0.0,
        "protein_g": 0.0,
        "fat_g": 0.0,
        "sodium_mg": float(sodium_mg),
        "ingredients_text": "water",
        "categories_all": category,
    }


class ScoringPipelineEdgeTests(unittest.TestCase):
    """Tests strict lower-risk filtering, sorting, and missing-data protections."""

    def test_lower_risk_enforcer_keeps_zero_risk_and_sorts_first(self) -> None:
        """Alternatives should be strictly lower risk and sorted ascending by risk."""
        alts = [
            {"name": "Alt Mid", "risk_score": 20, "net_carbs_g": 10, "fiber_g": 0},
            {"name": "Alt Zero", "risk_score": 0, "net_carbs_g": 0, "fiber_g": 1},
            {"name": "Alt High", "risk_score": 100, "net_carbs_g": 5, "fiber_g": 0},
        ]
        out = _enforce_lower_risk_alternatives(alts, current_risk=99, k=3)
        self.assertEqual([a["name"] for a in out], ["Alt Zero", "Alt Mid"])

    def test_alternatives_are_strictly_lower_risk_and_sorted(self) -> None:
        """End-to-end scoring should return only lower-risk alternatives in sorted order."""
        model = FakeModel()
        item = _row(
            name="Query Cola",
            brand="Q",
            upc="1",
            category="drink",
            alt_group="drink",
            carbs_g=80.0,
            sugar_g=39.0,
            sodium_mg=45.0,
        )
        df = pd.DataFrame(
            [
                {
                    **_row(name="Alt Cola 30", brand="A", upc="2", category="drink", alt_group="drink", carbs_g=30.0),
                    "categories_all": "soft drink cola",
                },
                {
                    **_row(name="Alt Cola 10", brand="B", upc="3", category="drink", alt_group="drink", carbs_g=10.0),
                    "categories_all": "soft drink cola",
                },
                {
                    **_row(name="Alt Cola 20", brand="C", upc="4", category="drink", alt_group="drink", carbs_g=20.0),
                    "categories_all": "soft drink cola",
                },
                {
                    **_row(name="Alt Cola 90", brand="D", upc="5", category="drink", alt_group="drink", carbs_g=90.0),
                    "categories_all": "soft drink cola",
                },
            ]
        )

        result = score_item(item, model, df)
        risk = float(result["risk_score"])
        alts = result["alternatives"]
        self.assertTrue(alts)
        alt_risks = [float(a["risk_score"]) for a in alts]
        self.assertTrue(all(r < risk for r in alt_risks))
        self.assertEqual(alt_risks, sorted(alt_risks))

    def test_fingerprint_mismatch_triggers_candidate_rescore(self) -> None:
        """Stale pre-scored candidates must be recomputed when model fingerprint changes."""
        model = FakeModel()
        item = _row(
            name="Query Snack",
            brand="Q",
            upc="11",
            category="snack",
            alt_group="snack",
            carbs_g=60.0,
            sugar_g=20.0,
        )
        df = pd.DataFrame(
            [
                {
                    **_row(name="Alt A", brand="A", upc="12", category="snack", alt_group="snack", carbs_g=30.0),
                    "risk_prob": 1.0,
                    "risk_score": 100,
                    "risk_display": "Very high (>99)",
                    "model_fingerprint": "stale-fingerprint",
                },
                {
                    **_row(name="Alt B", brand="B", upc="13", category="snack", alt_group="snack", carbs_g=20.0),
                    "risk_prob": 1.0,
                    "risk_score": 100,
                    "risk_display": "Very high (>99)",
                    "model_fingerprint": "stale-fingerprint",
                },
            ]
        )

        with patch.dict(os.environ, {"DIANALYSIS_MODEL_FINGERPRINT": "fresh-fingerprint"}, clear=False):
            result = score_item(item, model, df)

        # If stale prescores were trusted, alternatives would be filtered out.
        self.assertTrue(result["alternatives"])
        self.assertTrue(all(float(a["risk_score"]) < float(result["risk_score"]) for a in result["alternatives"]))

    def test_inferred_sugar_guardrail_prevents_false_low_risk(self) -> None:
        """Missing added sugar should not allow sugary drinks to appear very low risk."""
        model = FakeModel()
        item = _row(
            name="Pepsi mini",
            brand="Pepsi",
            upc="99",
            category="drink",
            alt_group="drink",
            carbs_g=11.7,
            sugar_g=11.7,
            added_sugar_g=0.0,
            sodium_mg=0.0,
        )
        # Simulate source missing critical fields; avoid coercing to zero risk.
        item["added_sugar_g"] = None
        item["sodium_mg"] = None
        item["fiber_g"] = None
        item["fat_g"] = None
        item["categories_all"] = "soft drink cola"
        item["ingredients_text"] = "carbonated water sugar phosphoric acid"

        result = score_item(item, model, pd.DataFrame())

        self.assertEqual(result.get("data_confidence"), "low")
        self.assertGreaterEqual(float(result["risk_score"]), 55.0)
        self.assertTrue(any("Data confidence" in str(n) for n in result.get("notes", [])))

    def test_high_carb_high_added_sugar_floor_prevents_very_low_score(self) -> None:
        """High-carb + high-added-sugar items should not score in an ultra-low range."""

        class LowProbModel:
            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N802
                p1 = np.full(len(X), 0.02, dtype=float)
                return np.column_stack([1.0 - p1, p1])

        model = LowProbModel()
        item = _row(
            name="Frosted Cereal Example",
            brand="Test",
            upc="203",
            category="cereal",
            alt_group="cereal",
            carbs_g=37.0,
            fiber_g=3.0,
            sugar_g=14.0,
            added_sugar_g=10.0,
            sodium_mg=240.0,
        )
        result = score_item(item, model, pd.DataFrame())
        self.assertGreaterEqual(float(result["risk_score"]), 45.0)
        self.assertTrue(any("high carbs + high added sugar" in str(n).lower() for n in result.get("notes", [])))

    def test_display_score_is_capped_for_carb_only_positive(self) -> None:
        """Display score is capped for carb-only positives while raw score stays intact."""

        class HighProbModel:
            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N802
                p1 = np.full(len(X), 0.99, dtype=float)
                return np.column_stack([1.0 - p1, p1])

        model = HighProbModel()
        item = _row(
            name="Bagel Example",
            brand="Test",
            upc="200",
            category="bread",
            alt_group="bread",
            carbs_g=44.4,
            fiber_g=2.3,
            sugar_g=4.5,
            added_sugar_g=2.2,
            sodium_mg=201.0,
        )
        result = score_item(item, model, pd.DataFrame())
        self.assertEqual(int(result["risk_score"]), 99)
        self.assertEqual(int(result["risk_score_display"]), 85)
        self.assertTrue(bool(result.get("display_cap_applied")))

    def test_display_score_is_not_capped_when_added_sugar_is_high(self) -> None:
        """Cap should not apply when a strong sugar risk signal is present."""

        class HighProbModel:
            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N802
                p1 = np.full(len(X), 0.99, dtype=float)
                return np.column_stack([1.0 - p1, p1])

        model = HighProbModel()
        item = _row(
            name="Sugary Drink Example",
            brand="Test",
            upc="201",
            category="drink",
            alt_group="drink",
            carbs_g=39.0,
            fiber_g=0.0,
            sugar_g=39.0,
            added_sugar_g=39.0,
            sodium_mg=45.0,
        )
        result = score_item(item, model, pd.DataFrame())
        self.assertEqual(int(result["risk_score"]), 99)
        self.assertEqual(int(result["risk_score_display"]), 99)
        self.assertFalse(bool(result.get("display_cap_applied")))

    def test_display_score_is_not_capped_when_total_sugar_is_high_for_processed_food(self) -> None:
        """Cap should not apply when total sugar is high in processed categories."""

        class HighProbModel:
            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N802
                p1 = np.full(len(X), 0.99, dtype=float)
                return np.column_stack([1.0 - p1, p1])

        model = HighProbModel()
        item = _row(
            name="Cola Example",
            brand="Test",
            upc="202",
            category="drink",
            alt_group="drink",
            carbs_g=39.0,
            fiber_g=0.0,
            sugar_g=39.0,
            added_sugar_g=0.0,
            sodium_mg=45.0,
        )
        result = score_item(item, model, pd.DataFrame())
        self.assertEqual(int(result["risk_score"]), 99)
        self.assertEqual(int(result["risk_score_display"]), 99)
        self.assertFalse(bool(result.get("display_cap_applied")))


if __name__ == "__main__":
    unittest.main()
