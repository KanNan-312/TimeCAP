import pandas as pd

# Columns that are numeric but should never be auto-selected as auxiliary
# indicators (identifiers / geo-coordinates / already-modeled elsewhere).
DEFAULT_EXCLUDE = {'state', 'longitude', 'latitude'}


def load_dataset(cfg):
    df = pd.read_csv(cfg.csv_path)
    df[cfg.date_column] = pd.to_datetime(df[cfg.date_column])
    df[cfg.region_column] = df[cfg.region_column].astype(str)
    df = df.sort_values([cfg.region_column, cfg.date_column]).reset_index(drop=True)
    return df


def resolve_indicator_columns(df, cfg):
    if cfg.indicator_columns:
        return list(cfg.indicator_columns)
    exclude = {
        cfg.region_column, cfg.date_column, cfg.city_column,
        cfg.metro_column, cfg.target_column,
    } | DEFAULT_EXCLUDE
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def list_region_ids(df, cfg):
    ids = sorted(df[cfg.region_column].unique().tolist())
    if cfg.zipcodes:
        wanted = {str(z) for z in cfg.zipcodes}
        ids = [i for i in ids if i in wanted]
    if cfg.max_regions:
        ids = ids[:cfg.max_regions]
    return ids


def get_region_frame(df, cfg, region_id):
    sub = df[df[cfg.region_column] == str(region_id)]
    sub = sub.sort_values(cfg.date_column).drop_duplicates(subset=[cfg.date_column], keep='last')
    return sub.reset_index(drop=True)


def compute_boundaries(T, cfg):
    num_train = int(T * cfg.train_frac)
    num_test = int(T * cfg.test_frac)
    num_vali = T - num_train - num_test
    return num_train, num_vali, num_test


def compute_target_starts(T, cfg, need_train_pool):
    """
    A window is identified by `target_start`, the position of the first
    forecasted month. Its lookback range is [target_start - lookback,
    target_start) and its forecast range is [target_start, target_start +
    horizon).

    - Train-pool windows (used only as TimeCAP in-context examples) are kept
      fully inside the train range on both ends, so no val/test information
      ever leaks into a retrieved example.
    - Test windows start only once the val range has ended, but their
      lookback range is allowed to dip back into the train/val history -
      that's the "first test samples can look back into train/val" behavior.
    """
    num_train, num_vali, num_test = compute_boundaries(T, cfg)
    L, H = cfg.lookback, cfg.horizon

    train_starts = []
    if need_train_pool:
        train_hi = num_train - H  # last valid start s.t. start + H <= num_train
        if train_hi >= L:
            train_starts = list(range(L, train_hi + 1, max(1, cfg.train_stride)))

    test_lo = num_train + num_vali
    test_hi = T - H  # last valid start s.t. start + H <= T
    test_starts = list(range(test_lo, test_hi + 1, max(1, cfg.test_stride))) if test_hi >= test_lo else []

    return {
        'num_train': num_train, 'num_vali': num_vali, 'num_test': num_test,
        'train_starts': train_starts, 'test_starts': test_starts,
    }


def extract_window(frame, cfg, indicator_cols, start, length):
    sub = frame.iloc[start:start + length]
    return {
        'dates': sub[cfg.date_column].dt.strftime('%Y-%m-%d').tolist(),
        'target': sub[cfg.target_column].astype(float).tolist(),
        'indicators': {c: sub[c].astype(float).tolist() for c in indicator_cols},
    }
