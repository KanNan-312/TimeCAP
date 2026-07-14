import numpy as np

_EPS = 1e-6


def compute_metrics(preds, trues):
    """
    preds/trues: flat sequences of forecasted vs actual price values
    (already flattened across all horizon steps / windows / regions the
    caller wants aggregated together).
    """
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    if preds.shape != trues.shape or len(preds) == 0:
        raise ValueError('preds and trues must be non-empty and equal length')

    err = preds - trues
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(trues), _EPS, None)) * 100.0)

    n_nonpositive = int(np.sum(preds <= 0))
    log_preds = np.log(np.clip(preds, _EPS, None))
    log_trues = np.log(np.clip(trues, _EPS, None))
    log_err = log_preds - log_trues
    log_mae = float(np.mean(np.abs(log_err)))
    log_rmse = float(np.sqrt(np.mean(log_err ** 2)))

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'log_mae': log_mae,
        'log_rmse': log_rmse,
        'n_points': int(len(preds)),
        'n_nonpositive_preds': n_nonpositive,
    }
