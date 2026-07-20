"""
Text templates for region_forecast_ctx.

P1 (contextualization) is *not* an LLM call in this project. The feature
calculator's output is serialized directly into a compact, structured text
block (build_context_text) that is used as-is as the `text` argument to
P3's predict_text_prompt_structured (below) - i.e. a straight data -> prompt
-> prediction pipeline, with nothing in between
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

P2/P3 (predict from series-only / series+text) reuse the baseline's series
formatting helpers (region_forecast.prompts.display_name/format_series/
indicator_block - unmodified) but are otherwise rebuilt here as
predict_time_prompt_structured / predict_text_prompt_structured, with two
region_forecast_ctx-only enrichments the baseline templates don't have:

  - "thinking mode" (cfg.predict_thinking, off by default): when False, the
    output format asked of the LLM is the same bare '|'-separated number
    string the baseline uses. When True, the LLM is instead asked for a
    small JSON object - {"prediction": [...], "rationale": "...",
    "information_source": [...]} - so a forecast's reasoning and which
    categories of information it says it drew on (target series / geographic
    neighbors / historical precedent) are inspectable after the fact, not
    just its numbers. Either way, region_forecast_ctx/parsing.py always
    recovers `prediction` even if the model didn't comply with whichever
    format was asked for.
  - if ctx_correlation (+ ctx_show_correlated_series) found indicator(s)
    strongly correlated with the target for this window, their own lookback
    series is shown next to the target's, not just a correlation-coefficient
    sentence in the P1 text - see indicator_block_with_correlated.

Neither template ever surfaces zipcode or exact calendar dates - same
de-identification rule as build_context_text below.
"""

from region_forecast import prompts as BP
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
    predict_text_prompt_structured - no LLM is involved in producing it.
    """
    sections = _target_context_sections(cfg, feats)

    if cfg.spatial_enabled:
        sections.append(('Spatio-temporal context (geographic neighbors)', S.spatial_block(neighbor_ctx)))

    if cfg.experience_enabled:
        sections.append(('Similar historical precedents', S.experience_block(experience_cases)))

    if not sections:
        return 'No contextual statistics available (all context components disabled).'

    return '\n\n'.join(f'{title}:\n{text}' for title, text in sections)


# ---------------------------------------------------------------------------
# P2/P3 predict prompts - structured JSON output (prediction + rationale +
# information_source), region_forecast_ctx-only rebuild of the baseline
# templates. See module docstring for why.
# ---------------------------------------------------------------------------

def _available_information_sources(cfg):
    """
    Which information_source categories are actually possible to have used,
    given the current mode/ablation - offering a category the run has no way
    of having supplied would just invite the model to claim it anyway.
    """
    sources = ['target_series']
    if cfg.mode == 'timecp_ctx':
        if cfg.spatial_enabled:
            sources.append('neighbor_information')
        if cfg.experience_enabled:
            sources.append('historical_precedent')
    return sources


def _structured_output_instruction(cfg):
    sources = _available_information_sources(cfg)
    sources_str = ', '.join(f'"{s}"' for s in sources)
    return (
        f"respond with a single JSON object and nothing else (no markdown code fences, no text before or "
        f"after it), with exactly these keys:\n"
        f'  "prediction": an array of exactly {cfg.horizon} numeric values (no units or currency symbols), '
        f"forecasting the {BP.target_label(cfg)} for each of the next {cfg.horizon} months in chronological order,\n"
        f'  "rationale": a concise (2-3 sentence) explanation of the reasoning behind your forecast,\n'
        f'  "information_source": an array naming which of the following you actually relied on when '
        f"forecasting (any subset, or an empty array if none applied): {sources_str}."
    )


def _output_instruction(cfg):
    """
    cfg.predict_thinking gates whether the model is asked for anything
    beyond the prediction itself - see module docstring.
    """
    return _structured_output_instruction(cfg) if cfg.predict_thinking else BP.forecast_instruction(cfg)


def indicator_block_with_correlated(cfg, window, correlated_names):
    """
    Target series plus the actual lookback series of whichever indicator(s)
    the feature calculator found strongly correlated with the target for
    this window (target_features.compute_correlation), so the forecasting
    LLM sees the numbers behind the "Correlated indicators:" narrative
    sentence in the P1 text, not just the correlation coefficients.
    """
    lines = [f"- {BP.display_name(cfg.target_column)}: {BP.format_series(window['target'])}"]
    for name in correlated_names or []:
        vals = window['indicators'].get(name)
        if vals is None:
            continue
        lines.append(f"- {BP.display_name(name)}: {BP.format_series(vals)}")
    return '\n'.join(lines)


def predict_time_prompt_structured(cfg, region_id, window, forecast_dates):
    system_prompt = (
        f"Your job is to act as a professional regional real-estate forecaster. You will be given "
        f"time-series data of housing market indicators from the past {cfg.lookback} months for a specific "
        f"region. Based on this information, your task is to forecast the {BP.target_label(cfg)} for "
        f"each of the next {cfg.horizon} months."
    )
    user_prompt = (
        f"Your task is to forecast the {BP.target_label(cfg)} in one region for each of the next "
        f"{cfg.horizon} months. Review the {BP.target_label(cfg)} data provided for the last {cfg.lookback} "
        f"months. The historical time series consists of monthly values separated by a '|' token: \n\n"
        f"{BP.indicator_block(cfg, window, target_only=True)}\n\n"
        f"Based on this information, {_output_instruction(cfg)}"
    )
    return system_prompt, user_prompt


def predict_text_prompt_structured(cfg, region_id, window, text, forecast_dates, correlated_names=None):
    system_prompt = (
        f"Your job is to act as a professional regional real-estate forecaster. You will be given "
        f"time-series data and a written summary of the housing market over the past {cfg.lookback} "
        f"months for a specific region. Based on this information, your task is to forecast the "
        f"{BP.target_label(cfg)} for each of the next {cfg.horizon} months."
    )
    series_block = (indicator_block_with_correlated(cfg, window, correlated_names) if correlated_names
                     else BP.indicator_block(cfg, window, target_only=True))
    user_prompt = (
        f"Your task is to forecast the {BP.target_label(cfg)} in one region for each of the next "
        f"{cfg.horizon} months. Review the data provided for the last {cfg.lookback} months below - the "
        f"historical time series consists of monthly values separated by a '|' token. The target series is "
        f"always first"
        + (", followed by the series of indicator(s) found strongly correlated with it:\n\n"
           if correlated_names else ":\n\n")
        + f"{series_block}\n\n"
        f"In addition, the housing market situation of the last {cfg.lookback} months is summarized "
        f"as follows:\n\n{text}\n\n"
        f"Based on both the time-series data and the summary above, {_output_instruction(cfg)}"
    )
    return system_prompt, user_prompt
