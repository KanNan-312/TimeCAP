"""
Data helpers for region_forecast_ctx. Splitting/window logic is identical to
the baseline (region_forecast/data.py, unmodified) and is reused directly;
this module only adds what the enriched contextualization step needs:

  - extract_history: the *entire* causal history available at prediction
    time (series start -> target_start), not just the `cfg.lookback`-month
    window. Used by trend/seasonality/volatility/correlation features that
    benefit from more context than a 12-month window - still strictly
    causal since it never reads past target_start.
  - region_locations: per-region (latitude, longitude), used to build the
    haversine distance index for spatial context (level >= 2).
"""

from region_forecast import data as base

load_dataset = base.load_dataset
list_region_ids = base.list_region_ids
get_region_frame = base.get_region_frame
compute_boundaries = base.compute_boundaries
compute_target_starts = base.compute_target_starts
extract_window = base.extract_window


def resolve_indicator_columns(df, cfg):
    """
    Same auto-detection as the baseline, plus excluding the *configured*
    latitude/longitude columns (not just the literal names 'latitude'/
    'longitude' the baseline hardcodes) - those are consumed separately by
    the spatial index, never as an auxiliary indicator series.
    """
    cols = base.resolve_indicator_columns(df, cfg)
    coord_cols = {cfg.latitude_column, cfg.longitude_column}
    return [c for c in cols if c not in coord_cols]


def extract_history(frame, cfg, indicator_cols, end):
    """
    Everything known up to (but not including) row `end`. Pass
    end=target_start to get the full causal history for a test/train
    window - this deliberately reaches further back than the lookback
    window alone, spanning train/val/test split boundaries, since all of
    it is legitimately "the past" relative to the forecast being made.
    """
    sub = frame.iloc[:end]
    return {
        'dates': sub[cfg.date_column].dt.strftime('%Y-%m-%d').tolist(),
        'target': sub[cfg.target_column].astype(float).tolist(),
        'indicators': {c: sub[c].astype(float).tolist() for c in indicator_cols},
    }


def region_static_location(df, cfg, region_id):
    sub = df[df[cfg.region_column] == str(region_id)]
    if cfg.latitude_column not in sub.columns or cfg.longitude_column not in sub.columns:
        return None
    lat = sub[cfg.latitude_column].dropna()
    lon = sub[cfg.longitude_column].dropna()
    if lat.empty or lon.empty:
        return None
    return float(lat.iloc[0]), float(lon.iloc[0])


def region_locations(df, cfg, region_ids):
    """{region_id: (lat, lon)} for every region that has valid coordinates."""
    out = {}
    for rid in region_ids:
        loc = region_static_location(df, cfg, rid)
        if loc is not None:
            out[rid] = loc
    return out
