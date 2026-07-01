from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiments.arrays import check_case_data
from src.experiments.engine import CaseData


def load_case(
    *,
    filename: str | Path = "sinus/data/sinusX.npy",
    Tf: float = 4 * 2 * np.pi,
    N_t: int = 200,
) -> CaseData:
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Sinus data file not found: {filename}")

    X = np.load(filename)

    # Current sinus file is one scalar trajectory. Canonicalize to [1, 1, K].
    X = np.asarray(X).reshape(1, 1, -1)

    T = np.linspace(0.0, Tf, X.shape[-1])

    # If N_t is kept for backwards compatibility, check rather than force.
    if N_t != X.shape[-1]:
        T = np.linspace(0.0, Tf, X.shape[-1])

    train_data, eval_data, T = check_case_data(X, X.copy(), T)

    return CaseData(
        name="sinus",
        train_data=train_data,
        eval_data=eval_data,
        T=T,
        ylim=(-1.25, 1.35),
        plot_kind="scalar_single",
        metadata={
            "case": "sinus",
            "filename": str(filename),
            "Tf": float(Tf),
            "N_t_requested": int(N_t),
            "N_t_used": int(X.shape[-1]),
            "observable": "x",
        },
    )