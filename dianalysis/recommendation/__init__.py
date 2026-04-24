"""Recommendation package: retrieval, filtering, ranking, and service orchestration."""

from __future__ import annotations

from typing import Any


def make_alternatives(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Lazy import wrapper to avoid loading vector stack at package import time."""
    from .service import make_alternatives as _make_alternatives

    return _make_alternatives(*args, **kwargs)


__all__ = ["make_alternatives"]
