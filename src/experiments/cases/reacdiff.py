from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiments.arrays import check_case_data
from src.experiments.engine import CaseData


def load_case(
    *,
    data_dir: str | Path = "reacDiff/data",
    filename: str | Path | None = None,
    D: float = 0.0322307,
    pod_modes: int = 3,
    tau: int = 5,
) -> CaseData:
    if tau < 1:
        raise ValueError(f"tau must be >= 1, got tau={tau}")

    if filename is None:
        filename = Path(data_dir) / f"PODcoeffs_d={pod_modes}_D={D:.8f}.npz"
    else:
        filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Reaction-diffusion POD file not found: {filename}")

    with np.load(filename) as z:
        eta = np.asarray(z["pod_coeffs"])
        T = np.asarray(z["t"]).reshape(-1)

    # Current format is [d, K]. Canonicalize to [1, d, K].
    if eta.ndim == 2:
        if eta.shape[0] == pod_modes:
            data = eta[None, :, :]
        elif eta.shape[1] == pod_modes:
            data = eta.T[None, :, :]
        else:
            raise ValueError(
                f"Could not interpret POD coefficients with shape {eta.shape}."
            )
    elif eta.ndim == 3:
        if eta.shape[1] == pod_modes:
            data = eta
        elif eta.shape[2] == pod_modes:
            data = np.transpose(eta, (0, 2, 1))
        else:
            raise ValueError(
                f"Could not interpret POD coefficients with shape {eta.shape}."
            )
    else:
        raise ValueError(
            f"Expected POD coefficients with 2 or 3 dimensions, got {eta.shape}."
        )

    data_tau = data[:, :, ::tau]
    T_tau = T[::tau]

    K = min(data_tau.shape[-1], T_tau.shape[0])
    data_tau = data_tau[:, :, :K]
    T_tau = T_tau[:K]

    train_data, eval_data, T_tau = check_case_data(data_tau, data_tau.copy(), T_tau)

    pod_ylim = [(-5, 2.5), (-0.2, 0.8), (-0.3, 0.3)]
    coord_labels = [r"$w_1$", r"$w_2$", r"$w_3$"]

    if pod_modes != 3:
        pod_ylim = None
        coord_labels = [rf"$w_{{{j + 1}}}$" for j in range(pod_modes)]

    return CaseData(
        name="reacDiff",
        train_data=train_data,
        eval_data=eval_data,
        T=T_tau,
        ylim=pod_ylim,
        plot_kind="stacked_coordinates",
        metadata={
            "case": "reacdiff",
            "filename": str(filename),
            "D": float(D),
            "pod_modes": int(pod_modes),
            "tau": int(tau),
            "observable": "POD coefficients",
            "ylim": pod_ylim,
            "coord_labels": coord_labels,
            "stack_layout": "wide",
        },
    )