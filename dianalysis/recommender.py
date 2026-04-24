"""Compatibility entrypoint for recommendation generation."""

from __future__ import annotations

from typing import Any


def make_alternatives(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Lazy import wrapper to avoid eager loading of retrieval dependencies."""
    from .recommendation.service import make_alternatives as _make_alternatives

    return _make_alternatives(*args, **kwargs)

__all__ = ["make_alternatives"]
