"""
Regenerate Lorenz63 switches/peaks statistics from saved prediction files.

This script does NOT train models. It loads saved predictions from prior Lorenz63
runs for TD-DMD, TD-TF, TD-MLP, recomputes the relevant statistics, and
writes a compact CSV for the table in the paper.

Expected prediction shape after loading:
    [N_test, d, K]  or  [N_test, K]  or  [K]

The script is deliberately permissive about filenames and file formats.
It supports .npy, .npz, .pt, and .pth files and infers method / n / h / seed
from filenames when possible.

Examples of accepted filenames:
    lorenz63_tddmd_n4.npy
    TDDMD_n4_predictions.npz
    TDTF_n4_h100_seed0_predictions.npy
    TDMLP_n4_h100_seed3.pt

If your filenames differ, edit METHOD_PATTERNS and/or parse_metadata_from_path().
"""

import os
import re
import glob
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.evaluate import peak_stats, compute_lobe_switches


# ----------------------------
# Configuration
# ----------------------------

FILENAME = "lorenz63/data/lorenz63X.npy"

t_span = (0, 100)
dt = 0.01
tau = 16
train_frac = 0.9

lobe_threshold = 0.1

RESULTS_DIR = "lorenz63/results"
PREDICTIONS_DIRS = [
    "lorenz63/results",
    "lorenz63/predictions",
    "lorenz63/saved",
    "results/lorenz63",
]

CSV_PATH = os.path.join(RESULTS_DIR, "lorenz_switches_peaks_stats.csv")

METHOD_ORDER = ["True", "TD-DMD", "TD-TF", "TD-MLP"]

METHOD_PATTERNS = {
    "TD-DMD": ["tddmd", "td-dmd", "td_dmd"],
    "TD-TF": ["tdtf", "td-tf", "td_tf"],
    "TD-MLP": ["delay_mlp"]
    }

SUPPORTED_EXTENSIONS = (".npy", ".npz", ".pt", ".pth")

TABLE_CONFIGS = None


# ----------------------------
# Data loading
# ----------------------------

def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_prepare_test_data(
    filename: str,
    dt: float,
    t_span: Tuple[float, float],
    tau: int,
    train_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Lorenz63 data exactly as in the original statistics script:
    discard transients, subsample by tau, keep only x-coordinate, then use
    the held-out test trajectories.
    """
    T_full = np.arange(t_span[0], t_span[1], dt)
    T_full = T_full[len(T_full) // 2 :]
    T = T_full[::tau]

    data = np.load(filename)
    data = data.transpose(0, 2, 1)      # [N_all, d, K]
    data = data[:, 0:1, :]              # keep x only: [N_all, 1, K]

    train_split = int(train_frac * data.shape[0])
    test_trajectories = data[train_split:]
    test_tau = test_trajectories[:, :, ::tau]

    K = test_tau.shape[-1]
    return test_tau, T[:K]


# ----------------------------
# Prediction-file discovery
# ----------------------------

def infer_method(path: str) -> Optional[str]:
    name = os.path.basename(path).lower()
    for method, patterns in METHOD_PATTERNS.items():
        if any(pattern in name for pattern in patterns):
            return method
    return None


def parse_int_token(name: str, token: str) -> Optional[int]:
    """
    Extract integer metadata from filename tokens such as:
        n4, n=4, n_4, n-4
        h100, h=100, seed3, seed_3
    """
    pattern = rf"(?:^|[_\-.]){token}[=_\-]?(\d+)(?:[_\-.]|$)"
    match = re.search(pattern, name.lower())
    if match:
        return int(match.group(1))
    return None


def parse_metadata_from_path(path: str) -> Dict[str, Any]:
    name = os.path.basename(path)
    method = infer_method(path)

    return {
        "path": path,
        "method": method,
        "n": parse_int_token(name, "n"),
        "h": parse_int_token(name, "h"),
        "seed": parse_int_token(name, "seed"),
    }


def find_prediction_files(prediction_dirs: Iterable[str]) -> List[Dict[str, Any]]:
    files: List[str] = []

    for directory in prediction_dirs:
        if not os.path.isdir(directory):
            continue

        for ext in SUPPORTED_EXTENSIONS:
            files.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))

    records = []
    for path in sorted(set(files)):
        meta = parse_metadata_from_path(path)
        if meta["method"] is None:
            continue

        # Avoid accidentally loading aggregate CSVs or unrelated checkpoints.
        # Keep .pt/.pth because some workflows save prediction tensors that way.
        lower = os.path.basename(path).lower()
        if any(bad in lower for bad in ["optimizer", "state_dict", "checkpoint"]):
            continue

        records.append(meta)

    return records


def config_is_allowed(method: str, n: Optional[int], h: Optional[int]) -> bool:
    if TABLE_CONFIGS is None:
        return True

    for cfg in TABLE_CONFIGS:
        if cfg.get("method") != method:
            continue
        if cfg.get("n") is not None and cfg.get("n") != n:
            continue
        if cfg.get("h") is not None and cfg.get("h") != h:
            continue
        return True

    return False


# ----------------------------
# Prediction loading
# ----------------------------

def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def first_array_from_mapping(obj: Dict[str, Any]) -> np.ndarray:
    """
    Try common prediction keys first. Fall back to the first array-like item.
    """
    preferred_keys = [
        "X_pred",
        "x_pred",
        "pred",
        "preds",
        "prediction",
        "predictions",
        "X_pred_TF",
        "X_pred_TD",
        "y_pred",
        "Y_pred",
    ]

    for key in preferred_keys:
        if key in obj:
            return to_numpy(obj[key])

    for key, value in obj.items():
        try:
            arr = to_numpy(value)
        except Exception:
            continue

        if arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number):
            warnings.warn(
                f"No standard prediction key found. Using array under key '{key}'."
            )
            return arr

    raise ValueError("Could not find an array-like prediction in the saved file.")


def load_prediction_file(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        obj = np.load(path, allow_pickle=True)

        # Handles np.save(dict).
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
            obj = obj.item()
            if isinstance(obj, dict):
                return first_array_from_mapping(obj)

        return to_numpy(obj)

    if ext == ".npz":
        obj = np.load(path, allow_pickle=True)
        return first_array_from_mapping({key: obj[key] for key in obj.files})

    if ext in [".pt", ".pth"]:
        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict):
            return first_array_from_mapping(obj)

        return to_numpy(obj)

    raise ValueError(f"Unsupported file extension: {ext}")


def standardize_prediction_shape(
    pred: np.ndarray,
    n_test: int,
    K_true: int,
) -> np.ndarray:
    """
    Convert prediction array to [N_test, K].

    Accepted common shapes:
        [N_test, d, K]
        [N_test, K, d]
        [N_test, K]
        [d, K]
        [K]
    """
    pred = np.asarray(pred, dtype=float)
    pred = np.squeeze(pred)

    if pred.ndim == 1:
        pred = pred[None, :]

    elif pred.ndim == 2:
        # Already likely [N_test, K].
        # If shape is [K, N_test], transpose it.
        if pred.shape[0] == K_true and pred.shape[1] == n_test:
            pred = pred.T

    elif pred.ndim == 3:
        # [N_test, d, K]
        if pred.shape[0] == n_test and pred.shape[-1] == K_true:
            pred = pred[:, 0, :]

        # [N_test, K, d]
        elif pred.shape[0] == n_test and pred.shape[1] == K_true:
            pred = pred[:, :, 0]

        # [d, N_test, K]
        elif pred.shape[1] == n_test and pred.shape[-1] == K_true:
            pred = pred[0, :, :]

        else:
            raise ValueError(f"Unsupported 3D prediction shape: {pred.shape}")

    else:
        raise ValueError(f"Unsupported prediction shape: {pred.shape}")

    if pred.shape[0] == 1 and n_test > 1:
        warnings.warn(
            "Prediction file contains one trajectory only. "
            "Statistics will be computed on the first test trajectory."
        )

    if pred.shape[0] not in [1, n_test]:
        raise ValueError(
            f"Prediction trajectory count does not match test set. "
            f"Got {pred.shape[0]}, expected {n_test}."
        )

    return pred


# ----------------------------
# Statistics
# ----------------------------

def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def evaluate_series(x: np.ndarray, T: np.ndarray) -> Dict[str, float]:
    K_min = min(len(x), len(T))
    x_i = np.asarray(x[:K_min], dtype=float)
    T_i = np.asarray(T[:K_min], dtype=float)

    switches, switch_freq = compute_lobe_switches(
        x_i,
        T_i,
        threshold=lobe_threshold,
    )
    n_peaks, mean_dt_peak = peak_stats(x_i, T_i)

    return {
        "lobe_switches": float(switches),
        "switch_freq": float(switch_freq),
        "n_peaks": float(n_peaks),
        "mean_dt_peak": float(mean_dt_peak),
    }


def evaluate_predictions(
    pred: np.ndarray,
    test_tau: np.ndarray,
    T: np.ndarray,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean ± std over available test trajectories for one saved run.
    """
    n_available = pred.shape[0]
    n_eval = min(n_available, test_tau.shape[0])

    per_traj = []
    for i in range(n_eval):
        per_traj.append(evaluate_series(pred[i], T))

    output = {}
    for metric in ["lobe_switches", "switch_freq", "n_peaks", "mean_dt_peak"]:
        output[metric] = mean_std(row[metric] for row in per_traj)

    return output


def evaluate_true(test_tau: np.ndarray, T: np.ndarray) -> Dict[str, Tuple[float, float]]:
    per_traj = []
    for i in range(test_tau.shape[0]):
        x_true = test_tau[i, 0]
        per_traj.append(evaluate_series(x_true, T))

    output = {}
    for metric in ["lobe_switches", "switch_freq", "n_peaks", "mean_dt_peak"]:
        output[metric] = mean_std(row[metric] for row in per_traj)

    return output


def row_from_stats(
    method: str,
    n: Optional[int],
    h: Optional[int],
    seed_count: int,
    stats: Dict[str, Tuple[float, float]],
) -> Dict[str, Any]:
    """
    Compact row for the paper table.
    """
    return {
        "method": method,
        "n": n,
        "h": h,
        "R": seed_count,

        "lobe_switches_mean": stats["lobe_switches"][0],
        "lobe_switches_std": stats["lobe_switches"][1],

        "switch_freq_mean": stats["switch_freq"][0],
        "switch_freq_std": stats["switch_freq"][1],

        "n_peaks_mean": stats["n_peaks"][0],
        "n_peaks_std": stats["n_peaks"][1],

        "mean_dt_peak_mean": stats["mean_dt_peak"][0],
        "mean_dt_peak_std": stats["mean_dt_peak"][1],
    }


def aggregate_seed_runs(
    run_stats: List[Dict[str, Tuple[float, float]]],
) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate several seeds for the same method/n/h.

    The table reports mean ± std across seeds, where each seed value is already
    the mean over test trajectories. This matches the R=5 revision setup.
    """
    output = {}

    for metric in ["lobe_switches", "switch_freq", "n_peaks", "mean_dt_peak"]:
        seed_means = [stats[metric][0] for stats in run_stats]
        output[metric] = mean_std(seed_means)

    return output


def method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ensure_dirs()

    test_tau, T = load_and_prepare_test_data(
        FILENAME,
        dt=dt,
        t_span=t_span,
        tau=tau,
        train_frac=train_frac,
    )

    n_test = test_tau.shape[0]
    K_true = test_tau.shape[-1]

    rows: List[Dict[str, Any]] = []

    true_stats = evaluate_true(test_tau, T)
    rows.append(row_from_stats("True", n=None, h=None, seed_count=1, stats=true_stats))

    prediction_files = find_prediction_files(PREDICTIONS_DIRS)

    if not prediction_files:
        raise FileNotFoundError(
            "No saved prediction files found. Check PREDICTIONS_DIRS and filename patterns."
        )

    grouped: Dict[Tuple[str, Optional[int], Optional[int]], List[Dict[str, Any]]] = {}

    for meta in prediction_files:
        method = meta["method"]
        n = meta["n"]
        h = meta["h"]

        if method is None:
            continue

        if not config_is_allowed(method, n, h):
            continue

        key = (method, n, h)
        grouped.setdefault(key, []).append(meta)

    if not grouped:
        raise RuntimeError(
            "Prediction files were found, but none matched TABLE_CONFIGS. "
            "Set TABLE_CONFIGS=None or update the filter."
        )

    for (method, n, h), metas in grouped.items():
        print(f"\n[{method}] n={n}, h={h}, files={len(metas)}")

        run_stats = []

        for meta in sorted(metas, key=lambda x: (-1 if x["seed"] is None else x["seed"], x["path"])):
            path = meta["path"]
            seed = meta["seed"]

            print(f"  loading seed={seed}: {path}")

            pred_raw = load_prediction_file(path)
            pred = standardize_prediction_shape(pred_raw, n_test=n_test, K_true=K_true)

            stats = evaluate_predictions(pred, test_tau, T)
            run_stats.append(stats)

            print(
                "    "
                f"switches={stats['lobe_switches'][0]:.3f}±{stats['lobe_switches'][1]:.3f}, "
                f"freq={stats['switch_freq'][0]:.5f}±{stats['switch_freq'][1]:.5f}, "
                f"peaks={stats['n_peaks'][0]:.3f}±{stats['n_peaks'][1]:.3f}, "
                f"dt_peak={stats['mean_dt_peak'][0]:.5f}±{stats['mean_dt_peak'][1]:.5f}"
            )

        # Deterministic methods such as TD-DMD usually have one file.
        # If multiple files are present, they are treated as repeated runs.
        agg_stats = aggregate_seed_runs(run_stats)
        rows.append(
            row_from_stats(
                method=method,
                n=n,
                h=h,
                seed_count=len(run_stats),
                stats=agg_stats,
            )
        )

    df = pd.DataFrame(rows)

    df["_method_order"] = df["method"].map(method_sort_key)
    df = df.sort_values(
        by=["_method_order", "n", "h"],
        na_position="first",
    ).drop(columns=["_method_order"])

    df.to_csv(CSV_PATH, index=False)

    print(f"\n[Saved] {CSV_PATH}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()