"""
Config for region_forecast_ctx - a contextualization-focused extension of
the region_forecast baseline (see region_forecast/config.py, unmodified).

The baseline's P1 (contextualize) step asks an LLM to freehand a narrative
report from the raw lookback series. This project keeps that same P1 -> P3
("TimeCP") skeleton but enriches P1 with a feature calculator whose output
is serialized into the prompt, plus optional spatio-temporal and
experience-retrieval context. Every added component is independently
toggleable so its contribution can be ablated:

  context_level 1  -> target-series statistical context only (data
                       overview, momentum, trend/seasonality, volatility/
                       persistence, correlation - each independently
                       switchable via the ctx_* flags below).
  context_level 2  -> level 1 + spatio-temporal (geographic neighbor)
                       context, using haversine distance over
                       latitude/longitude to find the top-k nearest regions.
  context_level 3  -> level 2 + experience retrieval from the training pool
                       (placeholder - see features/experience.py).

A component only takes effect once *both* `context_level` has reached its
tier *and* its own ctx_* flag is True, so you can e.g. run at
context_level=3 with ctx_correlation=False to ablate just correlation
within the full stack.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from region_forecast.config import Config as BaseConfig


@dataclass
class Config(BaseConfig):
    mode: str = 'timecp_ctx'  # 'timeseries' (baseline passthrough) | 'timecp_ctx' (enriched P1 -> P3)

    # --- ablation switch: how far up the context stack to go -------------
    context_level: int = 1  # 1, 2, or 3 - see module docstring

    # --- level-1 component toggles (target-series statistical context) ---
    ctx_data_overview: bool = True
    ctx_momentum: bool = True
    ctx_trend_seasonal: bool = True
    ctx_volatility_persistence: bool = True
    ctx_correlation: bool = True

    # level-1 feature-calculator parameters
    momentum_short_months: int = 3
    momentum_compare_months: int = 3
    stl_period: int = 12
    acf_lags: Tuple[int, ...] = (1, 3, 6, 12)
    volatility_recent_months: int = 6
    correlation_top_n: int = 3
    correlation_min_abs: float = 0.3

    # --- level-2 component toggle + params (spatio-temporal context) -----
    ctx_spatial: bool = True
    spatial_k: int = 5
    spatial_max_km: Optional[float] = None
    latitude_column: str = 'latitude'
    longitude_column: str = 'longitude'

    # --- level-3 component toggle + params (experience retrieval) --------
    ctx_experience: bool = True
    experience_k: int = 3

    def validate(self):
        if self.mode not in ('timeseries', 'timecp_ctx'):
            raise ValueError(f'unknown mode: {self.mode}')
        frac_sum = self.train_frac + self.val_frac + self.test_frac
        if abs(frac_sum - 1.0) > 1e-6:
            raise ValueError(f'train/val/test fractions must sum to 1.0, got {frac_sum}')
        if self.lookback <= 0 or self.horizon <= 0:
            raise ValueError('lookback and horizon must be positive')
        if self.context_level not in (1, 2, 3):
            raise ValueError('context_level must be 1, 2, or 3')

    @property
    def spatial_enabled(self):
        return self.context_level >= 2 and self.ctx_spatial

    @property
    def experience_enabled(self):
        return self.context_level >= 3 and self.ctx_experience

    @property
    def ablation_tag(self):
        """
        Compact, deterministic encoding of exactly which context components
        are active. Used as a results subdirectory so that toggling a
        component and re-running against the same --results-dir can never
        silently resume from checkpoints computed under a *different*
        ablation configuration - each distinct on/off combination gets its
        own checkpoint tree.
        """
        parts = [f'L{self.context_level}']
        comp = []
        if self.ctx_data_overview:
            comp.append('overview')
        if self.ctx_momentum:
            comp.append('momentum')
        if self.ctx_trend_seasonal:
            comp.append('trend')
        if self.ctx_volatility_persistence:
            comp.append('vol')
        if self.ctx_correlation:
            comp.append('corr')
        parts.append('+'.join(comp) if comp else 'none')
        if self.context_level >= 2:
            parts.append('spatial' if self.ctx_spatial else 'nospatial')
        if self.context_level >= 3:
            parts.append('exp' if self.ctx_experience else 'noexp')
        return '_'.join(parts)
