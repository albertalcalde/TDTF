import numpy as np

def normalize(data):
    """
    Normalize the data into the range [-1, 1].

    Args:
        data (ndarray): The data to be normalized.

    Returns:
        tuple: A tuple containing the normalized data, minimum values, and maximum values.
    """
    vmax = np.max(data, axis=(0, 2), keepdims=True)
    vmin = np.min(data, axis=(0, 2), keepdims=True)
    target = 2 * (data - 0.5 * (vmax + vmin)) / (vmax - vmin)
    target = np.clip(target, -1, 1)
    return target, vmin, vmax

def standarize(data):
    """
    Standardize the data to zero mean and unit variance.

    Args:
        data (ndarray): The data to be standardized.

    Returns:
        tuple: A tuple containing the standardized data, mean values, and standard deviation values.
    """
    mean = np.mean(data, axis=(0, 2), keepdims=True)
    std = np.std(data, axis=(0, 2), keepdims=True) + 1e-12
    target = (data - mean) / std
    return target, mean, std

def reconstruct_TDDMD(A_TD, x0_td, vmin=None, vmax=None, mean=None, std=None, steps=1, normalization="minmax"):
    """
    Predict with time-delayed DMD using true data for the first n steps,
    then recursively predict up to `steps` steps.

    Args:
        A_TD: (nd, nd) time-delay DMD matrix
        x0_td: (d, n) initial time-delay states (columns are x_{0}^j,...,x_{n-1}^j)
        vmin: (1, d, 1) minimum values for min-max normalization
        vmax: (1, d, 1) maximum values for min-max normalization
        mean: (1, d, 1) mean values for standardization
        std: (1, d, 1) std values for standardization
        steps: total number of steps to predict
        normalization: "minmax" or "standard"
    Returns:
        (d, steps) predicted states
    """
    d, n = x0_td.shape
    X_pred = np.zeros((d, steps))

    if normalization == "minmax":
        # Normalize
        x0_td = 2 * (x0_td - 0.5 * (vmax + vmin)) / (vmax - vmin)
    elif normalization == "standard":
        x0_td = (x0_td - mean) / std
    else:
        raise ValueError("normalization must be 'minmax' or 'standard'")

    # Fill in the first n columns with the initial data
    X_pred[:, :n] = x0_td

    # Flatten initial delay vector for recursion
    x0_td = x0_td.flatten(order='F')  # shape (nd,)

    for k in range(n, steps):
        x0_td = A_TD @ x0_td
        X_pred[:, k] = x0_td[-d:]

    # Unnormalize the predictions
    if normalization == "minmax":
        X_pred = X_pred * 0.5 * (vmax - vmin) + 0.5 * (vmax + vmin)
    elif normalization == "standard":
        X_pred = X_pred * std + mean

    return X_pred


def compute_lobe_switches(x, T, threshold=2.0):
    """
    Count lobe switches for Lorenz-like x(t) using sign(x) once |x| > threshold.

    Args:
        x: 1D array, shape (K,)
        T: 1D array of times, shape (K,)
        threshold: float, defines "left" vs "right" lobes via sign(x) when |x| > threshold

    Returns:
        n_switches: int
        freq: float, switches per unit time
    """
    x = np.asarray(x)
    T = np.asarray(T)
    assert x.shape == T.shape

    state = None   # +1 for right lobe, -1 for left lobe
    last_switch_time = T[0]
    switches = 0

    for xi, ti in zip(x, T):
        if abs(xi) < threshold:
            # in the "middle region", don't count anything
            continue

        s = 1 if xi > 0 else -1

        if state is None:
            state = s
            last_switch_time = ti
        elif s != state:
            switches += 1
            state = s
            last_switch_time = ti

    total_time = T[-1] - T[0]
    freq = switches / total_time if total_time > 0 else np.nan
    return switches, freq


def find_peaks_simple(x):
    """
    Very simple 1D peak detector: local maxima x[i-1] < x[i] > x[i+1].

    Returns:
        indices of peaks (np.ndarray)
    """
    x = np.asarray(x)
    if x.size < 3:
        return np.array([], dtype=int)
    mask = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    return np.where(mask)[0] + 1


def peak_stats(x, T):
    """
    Compute basic peak statistics:
        - number of peaks
        - mean inter-peak time (if any)

    Args:
        x: 1D array, shape (K,)
        T: 1D array, shape (K,)

    Returns:
        n_peaks: int
        mean_inter_peak_time: float or np.nan
    """
    x = np.asarray(x)
    T = np.asarray(T)

    idx = find_peaks_simple(x)
    n_peaks = len(idx)

    if n_peaks < 2:
        mean_inter_peak_time = np.nan
    else:
        peak_times = T[idx]
        inter_peak_times = np.diff(peak_times)
        mean_inter_peak_time = np.mean(inter_peak_times)

    return n_peaks, mean_inter_peak_time


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))