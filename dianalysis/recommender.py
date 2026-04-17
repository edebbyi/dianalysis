"""Compatibility entrypoint for recommendation generation."""

from __future__ import annotations

from .recommendation.service import make_alternatives

__all__ = ["make_alternatives"]
