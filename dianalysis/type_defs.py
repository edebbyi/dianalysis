"""
Shared typing aliases and protocols.

Why:
- Define common type contracts once and reuse across modules.
- Keep annotations consistent without repeating ad-hoc local aliases.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

import numpy as np
import pandas as pd

FoodRow: TypeAlias = dict[str, Any]
AltRow: TypeAlias = dict[str, Any]
AltList: TypeAlias = list[AltRow]


class ModelLike(Protocol):
    """Minimal protocol for model objects used by scoring/evaluation paths."""

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...
