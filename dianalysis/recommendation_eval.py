"""
Recommendation-eval summary computation used by training/reporting flows.

Why:
- Keep ranking diagnostic aggregation out of model-training core code.
- Produce traceable metadata (coverage + NDCG summary) for artifacts/reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import ndcg_at_k_for_alternatives, score_item
from .type_defs import ModelLike


def compute_recommendation_eval(
    model: ModelLike,
    df: pd.DataFrame,
    *,
    sample_size: int = 120,
    k: int = 3,
    random_state: int = 42,
) -> dict[str, float | int | str]:
    """
    Compute lightweight ranking diagnostics for recommendation traceability.

    Notes:
    - Uses proxy relevance (NDCG) based on current recommendation heuristics.
    - Returns coverage and NDCG summary; does not mutate model or data.
    """
    candidates = df.copy()
    if "__display" in candidates.columns:
        candidates = candidates.drop(columns=["__display"])

    sample_n = min(sample_size, len(candidates))
    if sample_n == 0:
        return {
            "queries_evaluated": 0,
            "coverage_with_alternatives": 0.0,
            "ndcg_at_3_mean": 0.0,
            "ndcg_at_3_std": 0.0,
            "k": k,
            "sample_size_requested": sample_size,
            "evaluated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }

    queries = candidates.sample(n=sample_n, random_state=random_state)
    ndcg_vals: list[float] = []
    has_any_alt = 0

    for _, row in queries.iterrows():
        item = row.to_dict()
        result = score_item(item, model, candidates)
        alts = result.get("alternatives", [])
        if alts:
            has_any_alt += 1
        ndcg_vals.append(ndcg_at_k_for_alternatives(item, alts, k=k))

    return {
        "queries_evaluated": int(sample_n),
        "coverage_with_alternatives": float(has_any_alt / sample_n),
        "ndcg_at_3_mean": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        "ndcg_at_3_std": float(np.std(ndcg_vals)) if ndcg_vals else 0.0,
        "k": k,
        "sample_size_requested": int(sample_size),
        "evaluated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
