from __future__ import annotations

"""Tests for merging base config, profile overrides, and CLI precedence."""

import argparse
import tempfile
from pathlib import Path
import unittest

from dianalysis.run_config import cfg_get, load_runtime_config


class RunConfigTests(unittest.TestCase):
    """Ensures runtime config layering behaves as expected."""

    def test_profile_overrides_base_and_preserves_other_keys(self) -> None:
        """Profile values should override matching keys while keeping non-overridden ones."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs_dir = root / "configs"
            profiles_dir = configs_dir / "profiles"
            profiles_dir.mkdir(parents=True)

            base = configs_dir / "base.toml"
            base.write_text(
                """
[model]
model_type = "logreg"
C = 0.3

[recommendation.weights]
similarity_alpha = 0.35
health_beta = 0.65
""".strip()
            )

            profile = profiles_dir / "xgboost.toml"
            profile.write_text(
                """
[model]
model_type = "xgboost"

[recommendation.weights]
similarity_alpha = 0.55
""".strip()
            )

            merged = load_runtime_config(base, Path("profiles/xgboost.toml"))
            self.assertEqual(cfg_get(merged, "model", "model_type"), "xgboost")
            self.assertEqual(cfg_get(merged, "model", "C"), 0.3)
            self.assertEqual(cfg_get(merged, "recommendation", "weights", "similarity_alpha"), 0.55)
            self.assertEqual(cfg_get(merged, "recommendation", "weights", "health_beta"), 0.65)

    def test_cli_override_beats_profile_default(self) -> None:
        """CLI flags should take priority over config-derived parser defaults."""
        merged = {"model": {"model_type": "xgboost"}}

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--model-type", default=cfg_get(merged, "model", "model_type", default="logreg"))

        args_default = parser.parse_args([])
        self.assertEqual(args_default.model_type, "xgboost")

        args_cli = parser.parse_args(["--model-type", "logreg"])
        self.assertEqual(args_cli.model_type, "logreg")


if __name__ == "__main__":
    unittest.main()
