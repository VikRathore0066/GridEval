import pandas as pd
import numpy as np


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
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


def fix_quantile_crossing(q_values: np.ndarray) -> np.ndarray:
    """Ensure quantile forecasts are monotonically sorted along the quantile axis to correct quantile crossing."""
    q = np.asarray(q_values, dtype=float)
    if q.ndim == 2:
        return np.sort(q, axis=0)
    elif q.ndim == 1:
        return np.sort(q)
    return q

def compute_crps_quantile(q_values: np.ndarray, alphas: np.ndarray, y_true: np.ndarray, correct_crossing: bool = True) -> float:
    """
    Compute CRPS from quantile forecasts using the pinball loss integral / weighted quantile formulation,
    or proper probability spacing (alpha differences).
    CRPS = 2 * mean_over_alphas ( (y - q_alpha) * (alpha - I(y < q_alpha)) * w_alpha )
    or weighted pairwise absolute differences with probability weights.
    """
    q = np.asarray(q_values, dtype=float)
    if correct_crossing:
        q = fix_quantile_crossing(q)
    alphas = np.asarray(alphas, dtype=float)
    y = np.asarray(y_true, dtype=float)    # shape: [num_timestamps]

    if q.ndim == 1:
        q = q[:, None]

    order = np.argsort(alphas)
    alphas = alphas[order]
    q = q[order]

    # Compute spacing weights delta_alpha
    # For a set of quantiles, weight between alpha_{k-1} and alpha_k is alpha_k - alpha_{k-1}
    n_alphas = len(alphas)
    if n_alphas == 1:
        # Fallback to pinball loss for single quantile
        alpha = alphas[0]
        pinball = np.maximum(alpha * (y - q[0]), (1 - alpha) * (q[0] - y))
        return round(float(2.0 * np.mean(pinball)), 2)

    # Interval weights
    weights = np.zeros_like(alphas)
    weights[0] = alphas[0] + 0.5 * (alphas[1] - alphas[0])
    for i in range(1, n_alphas - 1):
        weights[i] = 0.5 * (alphas[i+1] - alphas[i-1])
    weights[-1] = (1.0 - alphas[-1]) + 0.5 * (alphas[-1] - alphas[-2])

    crps_steps = np.zeros(q.shape[1])
    for i, alpha in enumerate(alphas):
        qi = q[i]
        pinball = np.maximum(alpha * (y - qi), (1 - alpha) * (qi - y))
        crps_steps += 2.0 * weights[i] * pinball

    return round(float(np.mean(crps_steps)), 2)


def compute_picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Compute Prediction Interval Coverage Probability (percentage inside bounds)."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return round(float(np.mean((y_true >= lower) & (y_true <= upper)) * 100.0), 2)


def check_and_correct_quantile_crossing(q_values: np.ndarray) -> np.ndarray:
    """
    Detect and correct quantile crossing by sorting quantiles monotonically along the quantile axis.
    q_values shape: [num_alphas, num_timestamps] or [num_timestamps, num_alphas] depending on convention.
    Here we ensure monotonic non-decreasing order across alphas for each timestamp.
    """
    q = np.asarray(q_values, dtype=float)
    if q.ndim == 2:
        # Sort along axis 0 (quantiles)
        return np.sort(q, axis=0)
    return q



def compute_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Compute mean interval width across forecast steps."""
    return round(float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float))), 2)


def compute_ece_and_reliability(quantile_grid: np.ndarray, alphas: np.ndarray, y_true: np.ndarray, num_bins: int = 10):
    """
    Compute Expected Calibration Error (ECE) and empirical calibration curve
    checking each predicted quantile against its nominal level alpha.
    quantile_grid shape: [num_alphas, num_timestamps]
    """
    alphas = np.asarray(alphas, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    quantile_grid = np.asarray(quantile_grid, dtype=float)

    empirical_coverages = []
    for i, alpha in enumerate(alphas):
        q_vals = quantile_grid[i]
        # empirical frequency where y_true <= q_alpha should approximate alpha
        coverage = np.mean(y_true <= q_vals)
        empirical_coverages.append(coverage)

    empirical_coverages = np.array(empirical_coverages)
    ece = float(np.mean(np.abs(empirical_coverages - alphas)))

    return round(ece, 4), alphas, empirical_coverages

def get_rolling_schedule(test_index, horizon: int = 24, step: int = 24) -> list:
    """
    Generate ordered rolling forecast block schedules with strict validation.
    Returns a list of dicts with:
    - origin_idx: integer position in test_index
    - origin_identity: identity key at origin
    - target_indices: list of integer positions in test_index for the block
    - target_identities: list of identity keys for the block
    """
    if horizon <= 0 or step <= 0:
        raise ValueError("Horizon and step must be positive integers.")
    if test_index is None or len(test_index) == 0:
        return []

    if test_index.duplicated().any():
        raise ValueError("test_index contains duplicate entries.")

    if isinstance(test_index, pd.MultiIndex):
        flat_test = list(test_index)
        if flat_test != sorted(flat_test):
            raise ValueError("test_index must be strictly sorted in chronological order.")
    else:
        if not test_index.is_monotonic_increasing:
            raise ValueError("test_index must be strictly sorted in chronological order.")

    test_len = len(test_index)
    schedule = []

    curr_i = 0
    while curr_i < test_len:
        h = min(horizon, test_len - curr_i)
        target_idxs = list(range(curr_i, curr_i + h))
        target_idents = [test_index[idx] for idx in target_idxs]

        schedule.append({
            "origin_idx": curr_i,
            "origin_identity": test_index[curr_i],
            "target_indices": target_idxs,
            "target_identities": target_idents
        })
        curr_i += step

    return schedule


def validate_calibration_alignment(test_f_index, test_raw_index, y_test_arr, test_raw_arr, res_ch, horizon=24, step=24):
    """Strict pure dependency-free helper to validate calibration identity and value alignment."""
    if list(test_f_index) != list(test_raw_index):
        raise ValueError("test_f.index and test_raw.index must match exactly in order.")
    if y_test_arr.shape != test_raw_arr.shape or not np.array_equal(y_test_arr, test_raw_arr):
        raise ValueError("y_test_arr and test_raw_arr values and shapes must match exactly.")
    if not isinstance(res_ch, dict):
        raise TypeError("res_ch must be a dictionary.")
    if "test_identities" not in res_ch or "schedule" not in res_ch:
        raise ValueError("Chronos calibration evaluation result missing required test_identities or schedule.")
    if list(res_ch["test_identities"]) != list(test_raw_index):
        raise ValueError("Calibration Chronos test_identities do not match test_raw.index.")

    expected_schedule = get_rolling_schedule(test_raw_index, horizon=horizon, step=step)
    if res_ch["schedule"] != expected_schedule:
        raise ValueError("Chronos evaluation schedule does not match expected rolling schedule.")
    if not expected_schedule:
        raise ValueError("Rolling schedule cannot be empty.")
    return True
