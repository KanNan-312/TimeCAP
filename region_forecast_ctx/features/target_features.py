"""
Level-1 ablation: statistics describing the target series itself. Every
function here takes plain numeric arrays (not dataframes) so the same code
can be reused for the query region and, in spatial_features.py, for
neighboring regions.

Inputs named `history` are the causal history-to-date (series start ->
target_start, see data.extract_history) - deliberately longer than the
`cfg.lookback` window so trend/seasonality/volatility/correlation estimates
aren't starved of data by a short lookback.
"""

import numpy as np
import pandas as pd


def _arr(values):
    return np.asarray(values, dtype=float)


def _round(v, nd=4):
    if v is None:
        return None
    v = float(v)
    return None if np.isnan(v) else round(v, nd)


def _direction(a, b, eps=1e-9):
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return 'unknown'
    if a > b * (1 + eps):
        return 'up'
    if a < b * (1 - eps):
        return 'down'
    return 'flat'


# ---------------------------------------------------------------------------
# Data overview
# ---------------------------------------------------------------------------

def compute_data_overview(history_target, history_indicators):
    """
    Summarizes the *entire* causal history to date (see data.extract_history)
    - deliberately not restricted to the lookback window, so this reflects
    everything known about the series, not just its most recent slice.
    """
    h = _arr(history_target)
    n_total = int(len(h))
    n_valid = int(np.sum(~np.isnan(h)))
    missing_ratio = float((n_total - n_valid) / n_total) if n_total else None
    return {
        'n_observations_total': n_total,
        'n_observations_valid': n_valid,
        'missing_ratio_total': _round(missing_ratio),
        'n_variables': 1 + len(history_indicators),
        'history_median': _round(np.nanmedian(h)) if h.size and not np.all(np.isnan(h)) else None,
        'history_std': _round(np.nanstd(h)) if h.size and not np.all(np.isnan(h)) else None,
    }


# ---------------------------------------------------------------------------
# Short-term momentum
# ---------------------------------------------------------------------------

def compute_momentum(history_target, short_months=3, compare_months=3):
    h = _arr(history_target)
    if len(h) < short_months + compare_months or np.all(np.isnan(h)):
        return {'status': 'insufficient_history'}

    recent = h[-short_months:]
    prior = h[-(short_months + compare_months):-short_months]
    recent_median = np.nanmedian(recent) if not np.all(np.isnan(recent)) else np.nan
    prior_median = np.nanmedian(prior) if not np.all(np.isnan(prior)) else np.nan
    overall_median = np.nanmedian(h)

    def pct(a, b):
        if a is None or b is None or np.isnan(a) or np.isnan(b) or b == 0:
            return None
        return round(float((a - b) / abs(b) * 100.0), 2)

    return {
        'status': 'ok',
        'recent_median': _round(recent_median),
        'prior_median': _round(prior_median),
        'overall_median': _round(overall_median),
        'recent_vs_prior_pct': pct(recent_median, prior_median),
        'recent_vs_overall_pct': pct(recent_median, overall_median),
        'direction': _direction(recent_median, prior_median),
    }


# ---------------------------------------------------------------------------
# Long-term trend and seasonal structure
# ---------------------------------------------------------------------------

def compute_trend_seasonal(history_target, period=12):
    h = _arr(history_target)
    mask = ~np.isnan(h)
    valid = h[mask]
    out = {}

    if len(valid) >= 3:
        x = np.arange(len(h))
        slope, intercept = np.polyfit(x[mask], h[mask], 1)
        pred = slope * x[mask] + intercept
        ss_res = float(np.sum((h[mask] - pred) ** 2))
        ss_tot = float(np.sum((h[mask] - np.mean(h[mask])) ** 2))
        r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else None
        level = float(np.mean(h[mask]))
        slope_pct = (float(slope) / level * 100.0) if level else None
        out['trend_slope_per_month'] = _round(slope)
        out['trend_slope_pct_per_month'] = _round(slope_pct, 3)
        out['trend_r2'] = _round(r2, 3)
    else:
        out['trend_slope_per_month'] = None
        out['trend_slope_pct_per_month'] = None
        out['trend_r2'] = None

    if len(valid) >= max(2 * period, 8):
        try:
            from statsmodels.tsa.seasonal import STL
            s = pd.Series(h).interpolate(limit_direction='both')
            res = STL(s, period=period, robust=True).fit()
            resid_var = float(np.var(res.resid))
            trend_var = float(np.var(res.trend + res.resid))
            seas_var = float(np.var(res.seasonal + res.resid))
            f_trend = max(0.0, 1 - resid_var / trend_var) if trend_var > 0 else 0.0
            f_seasonal = max(0.0, 1 - resid_var / seas_var) if seas_var > 0 else 0.0
            out['seasonal_strength'] = _round(f_seasonal, 3)
            out['trend_strength_stl'] = _round(f_trend, 3)
            out['seasonal_amplitude'] = _round(float(np.max(res.seasonal) - np.min(res.seasonal)))
        except Exception:
            out['seasonal_strength'] = None
            out['trend_strength_stl'] = None
            out['seasonal_amplitude'] = None
            out['stl_note'] = 'STL decomposition failed'
    else:
        out['seasonal_strength'] = None
        out['trend_strength_stl'] = None
        out['seasonal_amplitude'] = None
        out['stl_note'] = f'insufficient history for STL (need >= {max(2 * period, 8)} valid points)'

    return out


# ---------------------------------------------------------------------------
# Volatility and persistence
# ---------------------------------------------------------------------------

def compute_volatility_persistence(history_target, recent_months=6, acf_lags=(1, 3, 6, 12)):
    h = _arr(history_target)
    mask = ~np.isnan(h)
    valid = h[mask]
    out = {}

    if len(valid) >= 2:
        rets = np.diff(valid) / valid[:-1]
        out['volatility_full_pct'] = _round(np.nanstd(rets) * 100, 3)
    else:
        out['volatility_full_pct'] = None

    if len(valid) >= recent_months + 1:
        recent = valid[-(recent_months + 1):]
        rets_recent = np.diff(recent) / recent[:-1]
        out['volatility_recent_pct'] = _round(np.nanstd(rets_recent) * 100, 3)
    else:
        out['volatility_recent_pct'] = None

    acf_out = {}
    max_lag = max(acf_lags) if acf_lags else 0
    if len(valid) >= max_lag + 2:
        s = pd.Series(h).interpolate(limit_direction='both').to_numpy()
        s = s - np.mean(s)
        denom = float(np.sum(s ** 2))
        for lag in acf_lags:
            if denom > 0 and len(s) > lag:
                num = float(np.sum(s[:-lag] * s[lag:]))
                acf_out[f'acf_lag_{lag}'] = _round(num / denom, 3)
            else:
                acf_out[f'acf_lag_{lag}'] = None
    else:
        for lag in acf_lags:
            acf_out[f'acf_lag_{lag}'] = None
    out['acf'] = acf_out

    if len(valid) >= 4:
        med = np.median(valid)
        signs = np.sign(valid - med)
        signs = signs[signs != 0]
        crossings = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
        out['median_crossings'] = crossings
        out['median_crossing_rate'] = _round(crossings / max(1, len(valid) - 1), 3)
    else:
        out['median_crossings'] = None
        out['median_crossing_rate'] = None

    return out


# ---------------------------------------------------------------------------
# Correlation with other indicators
# ---------------------------------------------------------------------------

def compute_correlation(history_target, history_indicators, top_n=3, min_abs=0.3):
    t = _arr(history_target)
    results = []
    for name, vals in history_indicators.items():
        v = _arr(vals)
        n = min(len(t), len(v))
        if n < 4:
            continue
        a, b = t[:n], v[:n]
        pair_mask = ~(np.isnan(a) | np.isnan(b))
        if pair_mask.sum() < 4:
            continue
        a2, b2 = a[pair_mask], b[pair_mask]
        if np.std(a2) == 0 or np.std(b2) == 0:
            continue
        r = float(np.corrcoef(a2, b2)[0, 1])
        if np.isnan(r):
            continue
        results.append((name, round(r, 3)))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    strong = [r for r in results if abs(r[1]) >= min_abs][:top_n]
    return {'correlations': strong, 'all_correlations': results}
