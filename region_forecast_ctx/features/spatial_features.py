"""
Level-2 ablation: spatio-temporal (geographic neighbor) context.

Regions are located with the dataset's `latitude`/`longitude` columns and
compared pairwise with the haversine great-circle distance. For a query
region + target_start, we pull its top-k geographic neighbors, compute each
neighbor's own short-term momentum (reusing target_features.compute_momentum
on the neighbor's own causal history, aligned by *date* rather than row
index so it's still correct if regions have gappy/misaligned panels), and
aggregate: how many neighbors are trending the same direction, how the
query region's momentum compares to the neighbor median, and where the
query region's current price level sits relative to its neighbors.
"""

import numpy as np

from region_forecast_ctx.features import target_features as TF

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class SpatialIndex:
    """Precomputed pairwise haversine distance matrix over region_locations."""

    def __init__(self, locations):
        self.ids = list(locations.keys())
        self._pos = {rid: i for i, rid in enumerate(self.ids)}
        lat = np.array([locations[i][0] for i in self.ids])
        lon = np.array([locations[i][1] for i in self.ids])
        n = len(self.ids)
        self.dist = np.zeros((n, n), dtype=float)
        for i in range(n):
            self.dist[i] = haversine_km(lat[i], lon[i], lat, lon)

    def neighbors(self, region_id, k, max_km=None):
        if region_id not in self._pos:
            return []
        i = self._pos[region_id]
        order = np.argsort(self.dist[i])
        out = []
        for j in order:
            rid = self.ids[j]
            if rid == region_id:
                continue
            d = float(self.dist[i, j])
            if max_km is not None and d > max_km:
                continue
            out.append((rid, d))
            if len(out) >= k:
                break
        return out


def compute_neighbor_context(cfg, region_id, target_start, frames, date_idx, spatial_index):
    """
    frames: {region_id: region_frame} for every planned region (date-sorted).
    date_idx: {region_id: {date_str: row_idx}} precomputed once per region.
    """
    neighbors = spatial_index.neighbors(region_id, cfg.spatial_k, cfg.spatial_max_km)
    if not neighbors:
        return {'status': 'no_neighbors_found'}

    if target_start <= 0:
        return {'status': 'no_history_before_window'}

    query_frame = frames[region_id]
    anchor_date = query_frame[cfg.date_column].dt.strftime('%Y-%m-%d').iloc[target_start - 1]
    query_momentum = TF.compute_momentum(
        query_frame[cfg.target_column].astype(float).iloc[:target_start].tolist(),
        cfg.momentum_short_months, cfg.momentum_compare_months)
    region_price = float(query_frame[cfg.target_column].astype(float).iloc[target_start - 1])

    rows, directions, momenta_pct, price_levels = [], [], [], []
    for nid, dist_km in neighbors:
        nframe = frames.get(nid)
        if nframe is None:
            continue
        n_idx = date_idx.get(nid, {}).get(anchor_date)
        if n_idx is None:
            continue
        n_hist = nframe[cfg.target_column].astype(float).iloc[:n_idx + 1].tolist()
        m = TF.compute_momentum(n_hist, cfg.momentum_short_months, cfg.momentum_compare_months)
        if m.get('status') != 'ok':
            continue
        directions.append(m['direction'])
        if m['recent_vs_prior_pct'] is not None:
            momenta_pct.append(m['recent_vs_prior_pct'])
        if n_hist:
            price_levels.append(n_hist[-1])
        rows.append({
            'region_id': nid, 'distance_km': round(dist_km, 2),
            'direction': m['direction'], 'recent_vs_prior_pct': m['recent_vs_prior_pct'],
        })

    if not rows:
        return {'status': 'no_alignable_neighbors', 'neighbors_considered': [n for n, _ in neighbors]}

    price_percentile = None
    if price_levels:
        price_percentile = round(100.0 * float(np.mean([p <= region_price for p in price_levels])), 1)

    return {
        'status': 'ok',
        'k': len(rows),
        'neighbors': rows,
        'direction_counts': {
            'up': directions.count('up'), 'down': directions.count('down'), 'flat': directions.count('flat'),
        },
        'query_momentum': query_momentum,
        'neighbor_median_momentum_pct': round(float(np.median(momenta_pct)), 2) if momenta_pct else None,
        'region_price_percentile_among_neighbors': price_percentile,
    }
