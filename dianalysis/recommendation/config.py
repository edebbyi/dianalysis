"""Store recommendation ranking weights in one place."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankingWeights:
    """Hold weights and penalties used by candidate ranking."""

    similarity_alpha: float = 0.35
    health_beta: float = 0.65
    text_align_gamma: float = 0.20
    ingredient_gamma: float = 0.20
    same_category_penalty: float = 0.20
    cross_category_penalty: float = 0.85
    stage_penalty_step: float = 0.05


DEFAULT_RANKING_WEIGHTS = RankingWeights()

TEXT_STOPWORDS = {
    "the",
    "and",
    "with",
    "for",
    "of",
    "in",
    "a",
    "an",
    "to",
    "style",
    "original",
    "organic",
    "pack",
    "food",
    "foods",
    "co",
    "company",
}
