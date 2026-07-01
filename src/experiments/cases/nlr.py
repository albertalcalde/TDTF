from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.experiments.arrays import check_case_data
from src.experiments.engine import CaseData


QOI_TO_CHANNEL = {
    "drag": 0,
    "lift": 1,
}

DEFAULT_TEST_TRAJ_INDICES = [4, 8]


def load_case(
    qoi: str,
    *,
    data_file: str | Path = "nlr/data/nlrX.npy",
    time_file: str | Path = "nlr/data/nlrTime.npy",
    tau: int = 30,
    test_traj_ind: Sequence[int] = DEFAULT_TEST_TRAJ_INDICES,
    variant: str = "interpolation",
) -> CaseData:
    qoi = qoi.lower().strip()

    if qoi not in QOI_TO_CHANNEL:
        raise ValueError(f"QoI must be 'drag' or 'lift', got {qoi!r}")

    if tau < 1:
        raise ValueError(f"tau must be >= 1, got tau={tau}")

    data_file = Path(data_file)
    time_file = Path(time_file)

    if not data_file.exists():
        raise FileNotFoundError(f"NLR data file not found: {data_file}")

    if not time_file.exists():
        raise FileNotFoundError(f"NLR time file not found: {time_file}")

    raw = np.load(data_file)
    T = np.load(time_file).reshape(-1)

    if raw.ndim != 3:
        raise ValueError(
            f"Expected NLR data with shape [N, 2, K] or [N, K, 2], got {raw.shape}."
        )

    # Current format is [N_all, 2, K].
    # Robust fallback handles [N_all, K, 2].
    if raw.shape[1] == 2:
        data = raw
    elif raw.shape[2] == 2:
        data = np.transpose(raw, (0, 2, 1))
    else:
        raise ValueError(
            f"Could not infer NLR channel axis from shape {raw.shape}."
        )

    N_all = data.shape[0]
    test_traj_ind = [int(i) for i in test_traj_ind]

    # Allow Python-style negative trajectory indices, e.g. [-2, -1],
    test_traj_ind = [i if i >= 0 else N_all + i for i in test_traj_ind]


    if len(test_traj_ind) == 0:
        raise ValueError("test_traj_ind must contain at least one trajectory.")

    bad = [i for i in test_traj_ind if i < 0 or i >= N_all]
    if bad:
        raise IndexError(
            f"Invalid test trajectory indices {bad}; available range is [0, {N_all - 1}]."
        )

    if len(set(test_traj_ind)) != len(test_traj_ind):
        raise ValueError(f"Duplicate test trajectory indices: {test_traj_ind}")

    train_traj_ind = [i for i in range(N_all) if i not in set(test_traj_ind)]

    if len(train_traj_ind) == 0:
        raise ValueError("No training trajectories remain after test split.")

    channel = QOI_TO_CHANNEL[qoi]

    # Keep one scalar observable, preserving [N, 1, K].
    data = data[:, channel:channel + 1, :]

    data_tau = data[:, :, ::tau]
    T_tau = T[::tau]

    K = min(data_tau.shape[-1], T_tau.shape[0])
    data_tau = data_tau[:, :, :K]
    T_tau = T_tau[:K]

    train_data = data_tau[train_traj_ind]
    eval_data = data_tau[test_traj_ind]

    train_data, eval_data, T_tau = check_case_data(train_data, eval_data, T_tau)

    ylim = (-0.06, 0.02) if qoi == "drag" else (-0.25, 1.25)

    return CaseData(
        # name=f"nlr_{qoi}",
        name=f"nlr_{qoi}" if variant == "interpolation" else f"nlr_{qoi}_{variant}",
        train_data=train_data,
        eval_data=eval_data,
        T=T_tau,
        ylim=ylim,
        plot_kind="scalar_many",
        metadata={
            "case": "nlr",
            "qoi": qoi,
            "channel": int(channel),
            "data_file": str(data_file),
            "time_file": str(time_file),
            "tau": int(tau),
            "variant": variant,
            "test_traj_ind": list(test_traj_ind),
            "train_traj_ind": list(train_traj_ind),
            "N_train": int(train_data.shape[0]),
            "N_eval": int(eval_data.shape[0]),
            "legacy_reproduction": {
                "replace": True,
                "m": 1,
                "J": 5000,
                "dataset_seed": 0,
                "n_values": list(range(5, 16)),
                "h_values": [10, 50, 100, 150],
                "seeds": [0],
                "batch_size": 100,
                "lr": 1e-2,
                "weight_decay": 1e-2,
                "epochs": 500,
            },
        },
    )