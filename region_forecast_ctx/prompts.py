"""
Prompt templates for region_forecast_ctx.

Only P1 (contextualize) changes relative to the region_forecast baseline:
instead of asking the LLM to freehand a narrative purely from the raw
lookback series, the prompt is assembled from a fixed list of sections -
the raw series plus whichever feature-calculator blocks are enabled by the
ablation config (target-series stats always available at context_level>=1,
spatial context at >=2, experience precedents at >=3). Each section is
independently toggleable so its marginal contribution can be measured.

P2/P3 (predict from series-only / series+text) are unchanged from the
baseline and imported directly, so the predict stage's prompt contract -
and therefore the model's forecasting task - stays identical across
ablation levels; only what P1 hands to P3 as `text` differs.
"""

from region_forecast.prompts import (  # noqa: F401 - re-exported for pipeline.py
    display_name, target_label, format_series, indicator_block, region_label,
    forecast_instruction, predict_time_prompt, predict_text_prompt,
)
from region_forecast_ctx.features import serialize as S


def _target_context_sections(cfg, feats):
    sections = []
    if cfg.ctx_data_overview:
        sections.append(('Data overview', S.data_overview_block(feats['overview'])))
    if cfg.ctx_momentum:
        sections.append(('Short-term momentum', S.momentum_block(
            feats['momentum'], cfg.momentum_short_months, cfg.momentum_compare_months)))
    if cfg.ctx_trend_seasonal:
        sections.append(('Long-term trend & seasonal structure', S.trend_seasonal_block(feats['trend_seasonal'])))
    if cfg.ctx_volatility_persistence:
        sections.append(('Volatility & persistence', S.volatility_persistence_block(
            feats['volatility_persistence'], cfg.volatility_recent_months)))
    if cfg.ctx_correlation:
        sections.append(('Correlated indicators', S.correlation_block(feats['correlation'])))
    return sections


def contextualize_prompt(cfg, region_id, window, feats, neighbor_ctx=None, experience_cases=None):
    label = region_label(region_id)

    system_prompt = (
        "Your job is to act as a professional regional real-estate market analyst. You will write a "
        "high-quality report that is informative and helps in understanding the current regional "
        "housing market situation, grounded in the quantitative statistics provided rather than the "
        "raw series alone."
    )

    sections = [(f'Raw monthly indicator series (last {cfg.lookback} months)', indicator_block(cfg, window))]
    sections += _target_context_sections(cfg, feats)

    if cfg.spatial_enabled:
        sections.append(('Spatio-temporal context (geographic neighbors)', S.spatial_block(neighbor_ctx)))

    if cfg.experience_enabled:
        sections.append(('Similar historical precedents', S.experience_block(experience_cases)))

    body = '\n\n'.join(f'{title}:\n{text}' for title, text in sections)

    closing_hint = 'pricing momentum, supply and demand balance, market tightness'
    if cfg.spatial_enabled:
        closing_hint += ", and how this region compares to its geographic neighbors"

    user_prompt = (
        f"Your task is to analyze key housing market indicators for {label} over the last {cfg.lookback} "
        f"months, using both the raw time-series data and the pre-computed statistics below.\n\n"
        f"{body}\n\n"
        f"Based on all of the information above, write a concise report that provides insights crucial "
        f"for understanding the current regional housing market situation. Your report should be limited "
        f"to six sentences, yet comprehensive, highlighting key trends (e.g. {closing_hint}) and "
        f"considering their potential impact on home prices in this region over the coming months."
    )
    return system_prompt, user_prompt
