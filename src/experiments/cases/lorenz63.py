from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiments.arrays import check_case_data
from src.experiments.engine import CaseData


def load_case(
    *,
    filename: str | Path = "lorenz63/data/lorenz63X.npy",
    t_span: tuple[float, float] = (0.0, 100.0),
    dt: float = 0.01,
    tau: int = 16,
    train_frac: float = 0.9,
    observable_index: int = 0,
    observable_name: str = "x",
) -> CaseData:
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Lorenz63 data file not found: {filename}")

    if tau < 1:
        raise ValueError(f"tau must be >= 1, got tau={tau}")

    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be between 0 and 1, got {train_frac}")

    raw = np.load(filename)

    if raw.ndim != 3:
        raise ValueError(
            f"Expected Lorenz data with shape [N, K, d] or [N, d, K], got {raw.shape}."
        )

    # Current repository format is [N_all, K, d].
    # Robustly convert to [N_all, d, K].
    if raw.shape[-1] <= 10:
        data = np.transpose(raw, (0, 2, 1))
    else:
        data = raw

    if observable_index < 0 or observable_index >= data.shape[1]:
        raise IndexError(
            f"observable_index={observable_index} is invalid for d={data.shape[1]}."
        )

    # Keep one scalar observable but preserve [N, 1, K].
    data = data[:, observable_index:observable_index + 1, :]

    T_full = np.arange(t_span[0], t_span[1], dt)

    # Existing script discards the first half of T and then uses the loaded data.
    T_full = T_full[len(T_full) // 2 :]

    data_tau = data[:, :, ::tau]
    T_tau = T_full[::tau]

    K = min(data_tau.shape[-1], T_tau.shape[0])
    data_tau = data_tau[:, :, :K]
    T_tau = T_tau[:K]

    N_all = data_tau.shape[0]
    train_split = int(train_frac * N_all)

    if train_split < 1 or train_split >= N_all:
        raise ValueError(
            f"Invalid train split: N_all={N_all}, train_frac={train_frac}, "
            f"train_split={train_split}."
        )

    train_data = data_tau[:train_split]
    eval_data = data_tau[train_split:]

    train_data, eval_data, T_tau = check_case_data(train_data, eval_data, T_tau)

    return CaseData(
        name="lorenz63",
        train_data=train_data,
        eval_data=eval_data,
        T=T_tau,
        ylim=(-20, 20),
        plot_kind="scalar_many",
        metadata={
            "case": "lorenz63",
            "filename": str(filename),
            "t_span": tuple(float(x) for x in t_span),
            "dt": float(dt),
            "tau": int(tau),
            "train_frac": float(train_frac),
            "observable_index": int(observable_index),
            "observable": observable_name,
            "N_train": int(train_data.shape[0]),
            "N_eval": int(eval_data.shape[0]),
        },
    )