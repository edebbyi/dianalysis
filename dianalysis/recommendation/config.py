"""Store recommendation ranking weights in one place."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from ..run_config import cfg_get, load_runtime_config


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


@lru_cache(maxsize=8)
def _runtime_cfg_for_sources(config_src: str, profile_src: str) -> dict:
    profile_path = Path(profile_src) if profile_src else None
    return load_runtime_config(Path(config_src), profile_path)


def _runtime_cfg() -> dict:
    config_src = os.getenv("DIANALYSIS_CONFIG", "configs/base.toml")
    profile_src = os.getenv("DIANALYSIS_PROFILE", "")
    return _runtime_cfg_for_sources(config_src, profile_src)


def _load_weights_from_cfg(cfg: dict) -> RankingWeights:
    return RankingWeights(
        similarity_alpha=float(
            cfg_get(cfg, "recommendation", "weights", "similarity_alpha", default=RankingWeights.similarity_alpha)
        ),
        health_beta=float(cfg_get(cfg, "recommendation", "weights", "health_beta", default=RankingWeights.health_beta)),
        text_align_gamma=float(
            cfg_get(cfg, "recommendation", "weights", "text_align_gamma", default=RankingWeights.text_align_gamma)
        ),
        ingredient_gamma=float(
            cfg_get(cfg, "recommendation", "weights", "ingredient_gamma", default=RankingWeights.ingredient_gamma)
        ),
        same_category_penalty=float(
            cfg_get(
                cfg,
                "recommendation",
                "weights",
                "same_category_penalty",
                default=RankingWeights.same_category_penalty,
            )
        ),
        cross_category_penalty=float(
            cfg_get(
                cfg,
                "recommendation",
                "weights",
                "cross_category_penalty",
                default=RankingWeights.cross_category_penalty,
            )
        ),
        stage_penalty_step=float(
            cfg_get(cfg, "recommendation", "weights", "stage_penalty_step", default=RankingWeights.stage_penalty_step)
        ),
    )


DEFAULT_RANKING_WEIGHTS = _load_weights_from_cfg(_runtime_cfg())

_DEFAULT_STOPWORDS = {
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


def _load_stopwords_from_cfg(cfg: dict) -> set[str]:
    cfg_stopwords = cfg_get(cfg, "recommendation", "text_stopwords", default=None)
    if isinstance(cfg_stopwords, list) and cfg_stopwords:
        return {str(w).strip().lower() for w in cfg_stopwords if str(w).strip()}
    return _DEFAULT_STOPWORDS


TEXT_STOPWORDS = _load_stopwords_from_cfg(_runtime_cfg())


def get_ranking_weights() -> RankingWeights:
    """Return active ranking weights from runtime config with safe defaults."""
    return _load_weights_from_cfg(_runtime_cfg())


def get_text_stopwords() -> set[str]:
    """Return active token stopwords from runtime config with safe defaults."""
    return _load_stopwords_from_cfg(_runtime_cfg())
