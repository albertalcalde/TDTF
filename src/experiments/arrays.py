from __future__ import annotations

import numpy as np


def require_NdK(X, *, name: str) -> np.ndarray:
    """
    Require canonical trajectory data format [N, d, K].
    This is intentionally strict for saved case data.
    """
    X = np.asarray(X)

    if X.ndim != 3:
        raise ValueError(
            f"{name} must have shape [N, d, K], got {X.shape}."
        )

    N, d, K = X.shape

    if N < 1:
        raise ValueError(f"{name} must contain at least one trajectory, got N={N}.")

    if d < 1:
        raise ValueError(f"{name} must contain at least one observable, got d={d}.")

    if K < 1:
        raise ValueError(f"{name} must contain at least one time step, got K={K}.")

    return X


def require_T(T, *, K: int | None = None, name: str = "T") -> np.ndarray:
    """
    Require canonical time format [K].
    """
    T = np.asarray(T).reshape(-1)

    if T.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {T.shape}.")

    if K is not None and len(T) != K:
        raise ValueError(
            f"{name} length mismatch: expected K={K}, got len({name})={len(T)}."
        )

    return T


# def trim_to_common_time(X: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Trim [N, d, Kx] and [Kt] to a shared K.
#     """
#     X = require_NdK(X, name="X")
#     T = require_T(T)

#     K = min(X.shape[-1], len(T))
#     return X[:, :, :K], T[:K]


def prediction_to_dK(pred, *, d: int, K: int, name: str) -> np.ndarray:
    """
    Convert a single predicted trajectory to canonical [d, K].

    Handles common outputs from reconstruct_TDDMD and AR_sliding:
      [d, K]
      [1, d, K]
      [d, 1, K]
      [K] when d == 1
      [K, d]
    """
    arr = np.asarray(pred)

    # Remove only singleton axes introduced by broadcasting, not real axes.
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        if d == 1 and arr.shape[0] == K:
            return arr[None, :]
        raise ValueError(
            f"{name} has shape {arr.shape}; expected [d, K]=[{d}, {K}]."
        )

    if arr.ndim == 2:
        if arr.shape == (d, K):
            return arr

        if arr.shape == (K, d):
            return arr.T

        if d == 1 and arr.shape == (1, K):
            return arr

        if d == 1 and arr.shape == (K, 1):
            return arr.T

        raise ValueError(
            f"{name} has shape {arr.shape}; expected [d, K]=[{d}, {K}]."
        )

    if arr.ndim == 3:
        # Examples: [1, d, K], [d, 1, K]
        squeezed = np.squeeze(arr)
        return prediction_to_dK(squeezed, d=d, K=K, name=name)

    raise ValueError(
        f"{name} has unsupported shape {arr.shape}; expected [d, K]=[{d}, {K}]."
    )


def check_case_data(train_data, eval_data, T):
    """
    Validate the full case-loader contract.
    """
    train_data = require_NdK(train_data, name="train_data")
    eval_data = require_NdK(eval_data, name="eval_data")

    if train_data.shape[1] != eval_data.shape[1]:
        raise ValueError(
            "train_data and eval_data must have the same observable dimension d: "
            f"{train_data.shape[1]} != {eval_data.shape[1]}."
        )

    if train_data.shape[2] != eval_data.shape[2]:
        raise ValueError(
            "train_data and eval_data must have the same time length K: "
            f"{train_data.shape[2]} != {eval_data.shape[2]}."
        )

    T = require_T(T, K=train_data.shape[2])

    return train_data, eval_data, T