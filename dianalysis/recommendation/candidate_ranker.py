"""Rank candidates and format recommendation payloads."""

from __future__ import annotations

import re

import pandas as pd

from .config import DEFAULT_RANKING_WEIGHTS, TEXT_STOPWORDS


def token_set(text: str) -> set[str]:
    """Tokenize text into normalized words for overlap checks."""
    toks = {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}
    return {t for t in toks if t not in TEXT_STOPWORDS}


def ingredient_overlap_score(query_ingredients: str, candidate_ingredients: str) -> float:
    """Compute ingredient overlap score in the range [0, 1]."""
    q = token_set(query_ingredients)
    c = token_set(candidate_ingredients)
    if not q or not c:
        return 0.0
    inter = len(q.intersection(c))
    union = len(q.union(c))
    return 0.0 if union == 0 else inter / union


def rank_candidates(
    cand: pd.DataFrame,
    *,
    this_net: float,
    sugar_this: float,
    fiber_this: float,
    cat: str,
    group_key: str,
    name_text: str,
    query_ingredients: str,
    k: int,
) -> pd.DataFrame:
    """Rank candidates by blended similarity, health, and alignment signals."""
    cfg = DEFAULT_RANKING_WEIGHTS
    out = cand.copy()

    out["delta_net"] = this_net - out["net_carbs_g"].astype(float)
    out["sugar_g"] = pd.to_numeric(out["sugar_g"], errors="coerce").fillna(float("inf"))
    out["fiber_g"] = pd.to_numeric(out["fiber_g"], errors="coerce").fillna(0.0)

    out["improves_net"] = out["net_carbs_g"] < this_net
    out["improves_sugar"] = out["sugar_g"] < sugar_this
    out["improves_both"] = (out["improves_net"] & out["improves_sugar"]).astype(int)
    out["sugar_diff"] = (out["sugar_g"] - sugar_this).abs()
    out["context_score"] = out["risk_score"] * 5 + out["sugar_diff"]

    stage_rank = {"strict": 0, "soft": 1, "lowrisk": 2, "nuts": 3}
    out["stage_rank"] = out["_stage"].map(stage_rank).fillna(9)

    retrieval_col = out["_retrieval_score"] if "_retrieval_score" in out.columns else pd.Series(0.0, index=out.index)
    out["similarity_score"] = pd.to_numeric(retrieval_col, errors="coerce").fillna(0.0)
    out["similarity_score"] = ((out["similarity_score"] + 1.0) / 2.0).clip(0.0, 1.0)

    risk_norm = (1.0 - (pd.to_numeric(out["risk_score"], errors="coerce").fillna(10.0) / 10.0)).clip(0.0, 1.0)
    net_improve = (out["delta_net"].clip(lower=0.0) / (abs(float(this_net)) + 1.0)).clip(0.0, 1.0)
    sugar_improve = ((float(sugar_this) - out["sugar_g"]).clip(lower=0.0) / (abs(float(sugar_this)) + 1.0)).clip(0.0, 1.0)
    fiber_improve = ((out["fiber_g"] - float(fiber_this)).clip(lower=0.0) / (abs(float(fiber_this)) + 1.0)).clip(0.0, 1.0)
    out["health_score"] = ((0.50 * risk_norm) + (0.20 * net_improve) + (0.20 * sugar_improve) + (0.10 * fiber_improve)).clip(
        0.0, 1.0
    )

    cand_group = out["alt_group"].fillna(out["category"]).astype(str).str.lower()
    cand_cat = out["category"].fillna("").astype(str).str.lower()
    query_group = str(group_key).lower()
    query_cat = str(cat).lower()
    out["category_penalty"] = cfg.cross_category_penalty
    out.loc[cand_cat == query_cat, "category_penalty"] = cfg.same_category_penalty
    out.loc[cand_group == query_group, "category_penalty"] = 0.00

    out["stage_penalty"] = out["stage_rank"] * cfg.stage_penalty_step

    query_tokens = token_set(name_text)
    categories_all = out["categories_all"] if "categories_all" in out.columns else pd.Series("", index=out.index)
    ingredients_text = out["ingredients_text"] if "ingredients_text" in out.columns else pd.Series("", index=out.index)
    cand_text = out["name"].fillna("").astype(str) + " " + out["brand"].fillna("").astype(str) + " " + categories_all.fillna("").astype(str)
    out["text_align_score"] = 0.0
    if query_tokens:
        out["text_align_score"] = cand_text.apply(
            lambda s: len(query_tokens.intersection(token_set(s))) / max(len(query_tokens), 1)
        ).astype(float)

    out["ingredient_score"] = ingredients_text.fillna("").astype(str).apply(
        lambda s: ingredient_overlap_score(query_ingredients, s)
    )

    out["final_score"] = (
        (cfg.similarity_alpha * out["similarity_score"])
        + (cfg.health_beta * out["health_score"])
        + (cfg.text_align_gamma * out["text_align_score"])
        + (cfg.ingredient_gamma * out["ingredient_score"])
        - out["category_penalty"]
        - out["stage_penalty"]
    )

    return out.sort_values(
        by=[
            "final_score",
            "stage_rank",
            "improves_both",
            "context_score",
            "fiber_g",
            "delta_net",
            "improves_net",
            "improves_sugar",
        ],
        ascending=[False, True, False, True, False, False, False, False],
    ).head(k)


def format_alternatives(cand: pd.DataFrame, *, this_net: float, fiber_this: float) -> list[dict]:
    """Convert ranked rows into the API response format."""
    # Ranked rows are returned best-to-worst; tier labels must follow that order.
    tiers = ["Best", "Better", "Good"]
    out: list[dict] = []
    for i, (_, r) in enumerate(cand.iterrows()):
        why_bits = []
        if r["net_carbs_g"] < this_net:
            why_bits.append(f"-{(this_net - r['net_carbs_g']):.0f}g net carbs")
        if r["fiber_g"] > fiber_this:
            why_bits.append(f"+{(r['fiber_g'] - fiber_this):.0f}g fiber")

        out.append(
            {
                "tier": tiers[min(i, 2)],
                "name": r.get("name", f"Alt {i + 1}"),
                "brand": r.get("brand"),
                "category": r["category"],
                "alt_group": r.get("alt_group"),
                "risk_score": int(r["risk_score"]),
                "risk_display": r["risk_display"],
                "fiber_g": float(r.get("fiber_g", 0.0) or 0.0),
                "net_carbs_g": float(r.get("net_carbs_g", 0.0) or 0.0),
                "why": ", ".join(why_bits) if why_bits else "Lower risk in same category",
            }
        )
    return out
