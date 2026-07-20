"""
Feature calculator for region_forecast_ctx's enriched contextualization
step. Each submodule computes plain-dict statistics from numeric arrays;
`region_forecast_ctx.prompts` turns the dicts this package returns into
compact text blocks (see `serialize.py`) and assembles them into the P1
prompt, gated by the ablation toggles on Config.

- target_features: level-1 (target-series) statistics.
- spatial_features: level-2 (geographic neighbor) statistics.
- experience: level-3 (training-pool retrieval) placeholder.
- serialize: dict -> compact prompt-text rendering for all of the above.
"""

from region_forecast_ctx.features import target_features as TF


def compute_target_features(cfg, window, history):
    """
    window: this window's lookback-only slice (dict with 'target'/'indicators'), from D.extract_window.
    history: the full causal history up to target_start (same shape), from D.extract_history.
    Only computes the components whose ctx_* toggle is on, so disabled
    components incur no cost and are simply absent from the result dict.
    """
    out = {}
    if cfg.ctx_data_overview:
        out['overview'] = TF.compute_data_overview(history['target'], history['indicators'], window['target'])
    if cfg.ctx_momentum:
        out['momentum'] = TF.compute_momentum(
            history['target'], cfg.momentum_short_months, cfg.momentum_compare_months)
    if cfg.ctx_trend_seasonal:
        out['trend_seasonal'] = TF.compute_trend_seasonal(history['target'], cfg.stl_period)
    if cfg.ctx_volatility_persistence:
        out['volatility_persistence'] = TF.compute_volatility_persistence(
            history['target'], cfg.volatility_recent_months, cfg.acf_lags)
    if cfg.ctx_correlation:
        out['correlation'] = TF.compute_correlation(
            history['target'], history['indicators'], cfg.correlation_top_n, cfg.correlation_min_abs)
    return out
