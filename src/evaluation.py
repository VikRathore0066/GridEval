import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE": round(float(mape), 2),
    }


def compute_crps_empirical(samples: np.ndarray, y_true: np.ndarray, seed: int = 42) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS) from forecast samples.

    Uses the energy score identity:
    CRPS = E|X - y| - 0.5 * E|X - X'|
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=float)  # shape: [num_samples, num_timestamps]
    y_true = np.asarray(y_true, dtype=float)    # shape: [num_timestamps]

    n_samples, _ = samples.shape

    # Mean absolute error across samples
    e1 = np.mean(np.abs(samples - y_true[None, :]), axis=0)

    # Sample pairwise differences
    pair_size = min(200, n_samples * (n_samples - 1) // 2) if n_samples > 1 else 1
    idx1 = rng.choice(n_samples, size=pair_size, replace=True)
    idx2 = rng.choice(n_samples, size=pair_size, replace=True)
    e2 = np.mean(np.abs(samples[idx1] - samples[idx2]), axis=0)

    crps_per_step = e1 - 0.5 * e2
    return round(float(np.mean(crps_per_step)), 2)


def compute_picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Compute Prediction Interval Coverage Probability (percentage inside bounds)."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return round(float(np.mean((y_true >= lower) & (y_true <= upper)) * 100.0), 2)


def compute_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Compute mean interval width across forecast steps."""
    return round(float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float))), 2)


def compute_ece_and_reliability(samples: np.ndarray, y_true: np.ndarray, num_bins: int = 19):
    """
    Compute Expected Calibration Error (ECE) and empirical calibration curve.
    """
    nominal_levels = np.linspace(0.05, 0.95, num_bins)
    empirical_frequencies = []

    y_true = np.asarray(y_true, dtype=float)
    for q in nominal_levels:
        q_vals = np.percentile(samples, q * 100.0, axis=0)
        empirical_coverage = np.mean(y_true < q_vals)
        empirical_frequencies.append(empirical_coverage)

    empirical_frequencies = np.array(empirical_frequencies)
    ece = float(np.mean(np.abs(empirical_frequencies - nominal_levels)))

    return round(ece, 4), nominal_levels, empirical_frequencies
