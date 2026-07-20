"""
Level-3 ablation: accumulate "experience" from the training pool and query
similar historical cases at test time, to condition the forecast on
precedent - analogous in spirit to TimeCAP's in-context retrieval, but
retrieved cases feed the *contextualization* text as an aggregated,
de-identified outcome summary (see serialize.experience_block), not spliced
into the predict prompt as raw series+outcome pairs the way TimeCAP's P4
does.

Two retrieval methods, selected via cfg.experience_method:

  'features' (default) - k-NN in a small, scale-invariant feature space
      (momentum %, trend slope %, seasonal/trend strength, volatility %,
      ACF, mean-reversion rate) built from the same statistics as level 1,
      but computed independently of the ctx_* *display* toggles and of
      cfg.acf_lags, so retrieval quality never silently shifts when those
      are ablated for what's *shown* in the prompt. Cheap, and works across
      regions at very different price levels since every dimension is
      already relative.

  'shape' - distance between z-normalized lookback series (Euclidean, or
      DTW via cfg.experience_shape_metric). Complements 'features': two
      windows can have near-identical summary statistics (same trend slope,
      same volatility) but a different *shape* (front-loaded vs
      back-loaded change, one bump vs two), which only a direct series
      comparison catches. DTW is O(lookback^2) per comparison - fine at
      typical lookback sizes (~12) but noticeably slower than Euclidean
      over a large pool, hence Euclidean is the default.

Both methods:
  - draw candidates from every region's training pool, not just the query
    region's own history - cross-region analogs, since this is a
    spatio-temporal panel and the pool is far richer than any one region's
    own history. (Caveat: this also means matches can cluster around
    shared macro cycles - e.g. many regions moving together during a broad
    market shift - rather than reflecting idiosyncratic similarity. Not
    corrected for here.)
  - de-duplicate near-identical overlapping windows from the same region by
    enforcing a minimum start-index gap
    (cfg.effective_experience_min_gap_months, default = cfg.lookback, i.e.
    no two selected analogs from the same region may overlap) between any
    two selected analogs drawn from the same source region.
  - report outcomes as the *relative* % change from the last known value to
    each retrieved case's own horizon, not raw price levels, and never
    surface which region/period a match came from - only its rank,
    distance, and outcome (see serialize.py). Internal-only bookkeeping
    keys (region_id/start, prefixed with '_') are kept on each result for
    our own checkpoint files but are never read by the text serializer.

The fitted store is persisted to disk (<prefix>.json + <prefix>.npz) by
pipeline.run_experience_stage, mirroring the baseline's TimeCAP
embeddings_path convention, so a standalone `--stage contextualize` run can
load a previously-fitted store without needing to refit it.
"""

import os

import numpy as np
import pandas as pd

from region_forecast.utils import ensure_dir, read_json, write_json
from region_forecast_ctx.features import target_features as TF

# Fixed schema for 'features' mode - deliberately independent of cfg.acf_lags
# / cfg.ctx_* so retrieval quality never silently shifts when those are
# ablated for display purposes.
_ACF_LAGS = (1, 3, 6, 12)
_VECTOR_DIMS = (
    'momentum_recent_vs_prior_pct', 'momentum_recent_vs_overall_pct',
    'trend_slope_pct_per_month', 'trend_r2', 'seasonal_strength', 'trend_strength_stl',
    'volatility_full_pct', 'volatility_recent_pct',
    'acf_lag_1', 'acf_lag_3', 'acf_lag_6', 'acf_lag_12',
    'median_crossing_rate',
)


def _feature_vector(history, cfg):
    mom = TF.compute_momentum(history['target'], cfg.momentum_short_months, cfg.momentum_compare_months)
    trend = TF.compute_trend_seasonal(history['target'], cfg.stl_period)
    vol = TF.compute_volatility_persistence(history['target'], cfg.volatility_recent_months, _ACF_LAGS)
    acf = vol.get('acf', {})
    raw = {
        'momentum_recent_vs_prior_pct': mom.get('recent_vs_prior_pct'),
        'momentum_recent_vs_overall_pct': mom.get('recent_vs_overall_pct'),
        'trend_slope_pct_per_month': trend.get('trend_slope_pct_per_month'),
        'trend_r2': trend.get('trend_r2'),
        'seasonal_strength': trend.get('seasonal_strength'),
        'trend_strength_stl': trend.get('trend_strength_stl'),
        'volatility_full_pct': vol.get('volatility_full_pct'),
        'volatility_recent_pct': vol.get('volatility_recent_pct'),
        'acf_lag_1': acf.get('acf_lag_1'), 'acf_lag_3': acf.get('acf_lag_3'),
        'acf_lag_6': acf.get('acf_lag_6'), 'acf_lag_12': acf.get('acf_lag_12'),
        'median_crossing_rate': vol.get('median_crossing_rate'),
    }
    return np.array([np.nan if raw[k] is None else raw[k] for k in _VECTOR_DIMS], dtype=float)


def _shape_vector(history, lookback):
    """z-normalized last `lookback` values of history['target'], or None if unavailable."""
    tail = history['target'][-lookback:]
    if len(tail) < lookback:
        return None
    s = pd.Series(tail, dtype=float).interpolate(limit_direction='both').to_numpy()
    if np.any(np.isnan(s)):
        return None
    mu, sigma = float(np.mean(s)), float(np.std(s))
    return (s - mu) / (sigma if sigma > 1e-9 else 1.0)


def _dtw_distance(a, b):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def _outcome_stats(last_known, outcome):
    if last_known is None or np.isnan(last_known) or last_known == 0:
        return None
    pct = [(v - last_known) / abs(last_known) * 100.0 for v in outcome if v is not None and not np.isnan(v)]
    if not pct:
        return None
    cum = pct[-1]
    direction = 'up' if cum > 1e-6 else ('down' if cum < -1e-6 else 'flat')
    return {
        'cum_pct_change': round(float(cum), 2),
        'median_step_pct_change': round(float(np.median(pct)), 2),
        'direction': direction,
    }


class ExperienceStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.method = cfg.experience_method
        self._fitted = False
        self._cases = []       # per-case metadata: region_id, start, outcome stats
        self._matrix = None    # (n, d) representation, method-dependent
        self._feat_mean = None
        self._feat_std = None

    def __len__(self):
        return len(self._cases)

    def fit(self, train_cases):
        """
        train_cases: list of dicts, one per training-pool (region, window),
        each {'region_id', 'start', 'window', 'history', 'outcome'} (see
        pipeline.run_experience_stage). Cases whose outcome can't be
        expressed as a relative % change (e.g. a zero/missing last-known
        value) are dropped.
        """
        self._cases, vecs = [], []
        for case in train_cases:
            last_known = case['history']['target'][-1] if case['history']['target'] else None
            outcome_stats = _outcome_stats(last_known, case['outcome'])
            if outcome_stats is None:
                continue

            vec = (_shape_vector(case['history'], self.cfg.lookback) if self.method == 'shape'
                   else _feature_vector(case['history'], self.cfg))
            if vec is None or np.all(np.isnan(vec)):
                continue

            vecs.append(vec)
            self._cases.append({'region_id': case['region_id'], 'start': case['start'], **outcome_stats})

        self._fitted = True
        if not vecs:
            return self

        self._matrix = np.vstack(vecs)
        if self.method == 'features':
            self._feat_mean = np.nanmean(self._matrix, axis=0)
            self._feat_std = np.nanstd(self._matrix, axis=0)
            self._feat_std = np.where(self._feat_std < 1e-9, 1.0, self._feat_std)
            filled = np.where(np.isnan(self._matrix), self._feat_mean, self._matrix)
            self._matrix = (filled - self._feat_mean) / self._feat_std
        return self

    def _query_vector(self, history):
        if self.method == 'shape':
            return _shape_vector(history, self.cfg.lookback)
        qvec = _feature_vector(history, self.cfg)
        if qvec is None or self._feat_mean is None:
            return qvec
        qvec = np.where(np.isnan(qvec), self._feat_mean, qvec)
        return (qvec - self._feat_mean) / self._feat_std

    def query(self, region_id, start, history, k):
        """
        Returns up to `k` precedent cases most similar to the query
        region's current situation, nearest first, as de-identified dicts:
        {'rank', 'distance', 'horizon', 'cum_pct_change',
        'median_step_pct_change', 'direction', 'method'} plus internal
        '_region_id'/'_start' bookkeeping (never rendered into prompt text -
        see serialize.experience_block). Returns [] if the store hasn't
        been fitted, has no usable cases, or the query window itself lacks
        enough valid history to build a representation.
        """
        if not self._fitted or self._matrix is None or not self._cases:
            return []

        qvec = self._query_vector(history)
        if qvec is None or np.all(np.isnan(qvec)):
            return []

        if self.method == 'shape' and self.cfg.experience_shape_metric == 'dtw':
            dists = np.array([_dtw_distance(qvec, row) for row in self._matrix])
        else:
            dists = np.linalg.norm(self._matrix - qvec, axis=1)

        order = np.argsort(dists)
        min_gap = self.cfg.effective_experience_min_gap_months

        selected_starts_by_region = {}
        results = []
        for idx in order:
            if len(results) >= k:
                break
            case = self._cases[idx]
            cid, cstart = case['region_id'], case['start']
            prior = selected_starts_by_region.get(cid, [])
            if any(abs(cstart - s) < min_gap for s in prior):
                continue
            selected_starts_by_region.setdefault(cid, []).append(cstart)
            results.append({
                'rank': len(results) + 1,
                'distance': round(float(dists[idx]), 4),
                'horizon': self.cfg.horizon,
                'cum_pct_change': case['cum_pct_change'],
                'median_step_pct_change': case['median_step_pct_change'],
                'direction': case['direction'],
                'method': self.method,
                '_region_id': cid, '_start': cstart,
            })
        return results

    def save(self, path_prefix):
        ensure_dir(os.path.dirname(path_prefix))
        write_json(path_prefix + '.json', {'method': self.method, 'cases': self._cases})
        if self._matrix is not None:
            kwargs = {'matrix': self._matrix}
            if self._feat_mean is not None:
                kwargs['feat_mean'] = self._feat_mean
                kwargs['feat_std'] = self._feat_std
            np.savez(path_prefix + '.npz', **kwargs)

    def load(self, path_prefix):
        meta = read_json(path_prefix + '.json')
        self.method = meta['method']
        self._cases = meta['cases']
        if self._cases and os.path.exists(path_prefix + '.npz'):
            npz = np.load(path_prefix + '.npz')
            self._matrix = npz['matrix']
            if 'feat_mean' in npz:
                self._feat_mean = npz['feat_mean']
                self._feat_std = npz['feat_std']
        self._fitted = True
        return self
