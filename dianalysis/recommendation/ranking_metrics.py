"""
Ranking-evaluation metrics for alternative recommendations.

Why:
- Evaluate recommendation order quality (for example NDCG) separately from ranking logic.
- Keep offline metric definitions reusable across notebook/experiments/quality gates.
"""

from __future__ import annotations

from sklearn.metrics import ndcg_score

from ..model import compute_net_carbs


def alternative_proxy_relevance(this_row: dict, alt_row: dict) -> float:
    """
    Graded relevance proxy for ranking metrics like NDCG.

    Higher is better:
    - Same category/alt_group
    - Lower risk than query item
    - Higher/equal fiber
    - Lower net carbs
    """
    this_risk = float(this_row.get("risk_score", 0.0) or 0.0)
    this_fiber = float(this_row.get("fiber_g", 0.0) or 0.0)
    this_net = float(this_row.get("net_carbs_g", compute_net_carbs(this_row)) or 0.0)
    this_cat = str(this_row.get("category") or "").lower()
    this_group = str(this_row.get("alt_group") or this_cat).lower()

    alt_risk = float(alt_row.get("risk_score", this_risk) or this_risk)
    alt_fiber = float(alt_row.get("fiber_g", 0.0) or 0.0)
    alt_net = float(alt_row.get("net_carbs_g", 0.0) or 0.0)
    alt_cat = str(alt_row.get("category") or "").lower()
    alt_group = str(alt_row.get("alt_group") or alt_cat).lower()

    relevance = 0.0
    if alt_group == this_group or alt_cat == this_cat:
        relevance += 1.0
    if alt_risk < this_risk:
        relevance += 2.0
    if alt_fiber >= this_fiber:
        relevance += 1.0
    if alt_net < this_net:
        relevance += 1.0
    return relevance


def ndcg_at_k_for_alternatives(this_row: dict, alternatives: list[dict], k: int = 3) -> float:
    """Compute NDCG@k for an already-ranked alternatives list using proxy relevance."""
    if not alternatives:
        return 0.0

    relevances = [alternative_proxy_relevance(this_row, alt) for alt in alternatives]
    if not any(relevances):
        return 0.0
    if len(relevances) == 1:
        return 1.0 if relevances[0] > 0 else 0.0

    # Reflect existing rank order with descending rank scores.
    rank_scores = list(range(len(relevances), 0, -1))
    return float(ndcg_score([relevances], [rank_scores], k=min(k, len(relevances))))
