"""
Text templates for region_forecast_ctx.

P1 (contextualization) is *not* an LLM call in this project. The feature
calculator's output is serialized directly into a compact, structured text
block (build_context_text) that is used as-is as the `text` argument to
P3's predict_text_prompt (imported unchanged from the baseline) - i.e. a
straight data -> prompt -> prediction pipeline, with nothing in between
that could distort, drop, or hallucinate on top of the computed statistics.
This also makes contextualization free (no API cost) and fully
deterministic (same input always yields the same context).

De-identification: everything that reaches the forecasting LLM - this text
plus the unchanged P2/P3 templates - is built without region identifiers
(zipcode) or exact calendar dates, only relative statistics (medians,
slopes, correlations, neighbor rank/distance/direction). This prevents the
model from substituting memorized real-world knowledge about a specific
ZIP/period for genuine reasoning over the provided data. Geographic
neighbors are anonymized to "Neighbor 1", "Neighbor 2", ... in rank order
(see features/serialize.py) rather than named by zipcode. Internal
checkpoint files on disk (features/*.json) still carry region_id/dates for
our own bookkeeping and evaluation - de-identification only applies to text
that is actually sent to the LLM.

P2/P3 (predict from series-only / series+text) are unchanged from the
baseline and imported directly for the pipeline to use; their prompt text
already never surfaces zipcode or exact dates (region_forecast/prompts.py
computes a `label` but never interpolates it into the prompt strings for
these two templates), so no changes were needed there.
"""

from region_forecast.prompts import predict_time_prompt, predict_text_prompt  # noqa: F401 - re-exported for pipeline.py
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


def build_context_text(cfg, feats, neighbor_ctx=None, experience_cases=None):
    """
    Deterministic replacement for the P1 LLM narrative: a compact,
    de-identified statistics report assembled from whichever components the
    ablation config enables. The returned string is written to
    `summaries/<start>.txt` and consumed verbatim as the `text` argument to
    predict_text_prompt - no LLM is involved in producing it.
    """
    sections = _target_context_sections(cfg, feats)

    if cfg.spatial_enabled:
        sections.append(('Spatio-temporal context (geographic neighbors)', S.spatial_block(neighbor_ctx)))

    if cfg.experience_enabled:
        sections.append(('Similar historical precedents', S.experience_block(experience_cases)))

    if not sections:
        return 'No contextual statistics available (all context components disabled).'

    return '\n\n'.join(f'{title}:\n{text}' for title, text in sections)
