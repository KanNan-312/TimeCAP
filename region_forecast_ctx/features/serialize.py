"""
Renders the dicts produced by target_features / spatial_features /
experience into compact bullet-point text blocks for the P1 prompt. Kept
separate from the calculators themselves so the numeric computation and its
textual presentation can evolve independently (e.g. swapping units or
verbosity without touching the stats).
"""

from region_forecast.prompts import display_name


def _fmt(v, nd=2):
    if v is None:
        return 'n/a'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


def data_overview_block(d):
    return (
        f"- Observations: {d['n_observations_valid']}/{d['n_observations_total']} valid "
        f"({_fmt((d['missing_ratio_total'] or 0) * 100, 1)}% missing), {d['n_variables']} variable(s) tracked\n"
        f"- Level: lookback-window median={_fmt(d['lookback_median'])}, std={_fmt(d['lookback_std'])}; "
        f"full-history median={_fmt(d['history_median'])}, std={_fmt(d['history_std'])}"
    )


def momentum_block(m, short_m, compare_m):
    if m.get('status') != 'ok':
        return '- Momentum: insufficient history to compute'
    return (
        f"- Short-term momentum: last {short_m}mo median={_fmt(m['recent_median'])} vs prior "
        f"{compare_m}mo median={_fmt(m['prior_median'])} "
        f"({_fmt(m['recent_vs_prior_pct'])}% change, direction={m['direction']}); "
        f"vs full-history median={_fmt(m['overall_median'])} ({_fmt(m['recent_vs_overall_pct'])}% change)"
    )


def trend_seasonal_block(t):
    lines = [
        f"- Long-term trend: slope={_fmt(t.get('trend_slope_per_month'), 4)}/month "
        f"({_fmt(t.get('trend_slope_pct_per_month'), 3)}%/month), linear-fit R^2={_fmt(t.get('trend_r2'), 3)}"
    ]
    if t.get('seasonal_strength') is not None:
        lines.append(
            f"- Seasonal structure (STL): seasonality strength={_fmt(t['seasonal_strength'], 3)} (0-1), "
            f"trend strength={_fmt(t.get('trend_strength_stl'), 3)} (0-1), "
            f"seasonal amplitude={_fmt(t.get('seasonal_amplitude'))}"
        )
    else:
        lines.append(f"- Seasonal structure (STL): {t.get('stl_note', 'not available')}")
    return '\n'.join(lines)


def volatility_persistence_block(v, recent_m):
    acf_items = v.get('acf', {})
    acf_str = ', '.join(f"lag{k.split('_')[-1]}={_fmt(val, 3)}" for k, val in acf_items.items()) or 'n/a'
    return (
        f"- Volatility: full-history std of monthly % change={_fmt(v.get('volatility_full_pct'), 3)}%, "
        f"last {recent_m}mo std={_fmt(v.get('volatility_recent_pct'), 3)}%\n"
        f"- Persistence (autocorrelation): {acf_str}\n"
        f"- Mean-reversion: {_fmt(v.get('median_crossings'), 0)} median-crossings over history "
        f"(rate={_fmt(v.get('median_crossing_rate'), 3)} per month)"
    )


def correlation_block(c):
    if not c.get('correlations'):
        return '- Correlated indicators: none of the tracked indicators are strongly correlated with the target'
    parts = [f"{display_name(name)} (r={_fmt(r, 3)})" for name, r in c['correlations']]
    return '- Strongly correlated indicators: ' + ', '.join(parts)


def spatial_block(s):
    if not s or s.get('status') != 'ok':
        reason = (s or {}).get('status', 'unavailable')
        return f'- Spatial context: {reason}'
    dc = s['direction_counts']
    lines = [
        f"- Nearest {s['k']} geographic neighbor region(s): " + ', '.join(
            f"{n['region_id']} ({n['distance_km']}km, {n['direction']})" for n in s['neighbors']
        ),
        f"- Neighbor trend agreement: {dc['up']} up / {dc['down']} down / {dc['flat']} flat (of {s['k']})",
    ]
    if s.get('neighbor_median_momentum_pct') is not None:
        qm = (s.get('query_momentum') or {}).get('recent_vs_prior_pct')
        lines.append(
            f"- This region's short-term momentum vs neighbor median momentum: "
            f"{_fmt(qm)}% vs {_fmt(s['neighbor_median_momentum_pct'])}%"
        )
    if s.get('region_price_percentile_among_neighbors') is not None:
        lines.append(
            f"- This region's current price percentile among its neighbors: "
            f"{s['region_price_percentile_among_neighbors']}th percentile"
        )
    return '\n'.join(lines)


def experience_block(cases):
    if not cases:
        return ('- Similar historical cases: none retrieved (experience retrieval is a placeholder at this '
                'ablation level and is not yet implemented)')
    lines = ['- Similar historical cases:']
    for i, c in enumerate(cases, 1):
        lines.append(f"  {i}. {c.get('summary', c)}")
    return '\n'.join(lines)
