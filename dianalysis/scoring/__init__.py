"""Public scoring API used by app, notebook, and experiments."""

from __future__ import annotations

from .explanations import format_risk_display
from .barcode import score_by_barcode
from .pipeline import score_item
from ..recommendation.ranking_metrics import alternative_proxy_relevance, ndcg_at_k_for_alternatives
from ..recommendation.service import make_alternatives

__all__ = [
    "alternative_proxy_relevance",
    "format_risk_display",
    "make_alternatives",
    "ndcg_at_k_for_alternatives",
    "score_by_barcode",
    "score_item",
]
