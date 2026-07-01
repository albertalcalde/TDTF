import torch
import numpy as np
from torch.utils.data import TensorDataset
from src.modelClass import ModelClass

def burstsMultistepDataset(X, n, m, J, replace=False, seed=None):
    """
    Samples bursts of n delays plus m multistep predictions from a dataset of trajectories.

    Parameters:
    - X: np.ndarray of shape (N, d, K)
    - n: int, number of delays (input steps - 1)
    - m: int, number of multisteps predicted (output steps)
    - J: int, total number of bursts to sample
    - seed: int or None, random seed

    Returns:
    - bursts: np.ndarray of shape (J, d, n+m)
    """
    if seed is not None:
        np.random.seed(seed)
    N, d, K = X.shape
    bursts_per_traj = J // N
    remainder = J % N
    bursts = []
    burst_len = n + m  # total length of each burst
    for i in range(N):
        num_bursts = bursts_per_traj + (1 if i < remainder else 0)
        max_start = K - burst_len
        if max_start < 0:
            raise ValueError(f"Trajectory {i} is too short for burst length {burst_len}")
        starts = np.random.choice(max_start + 1, size=num_bursts, replace=replace)
        for s in starts:
            burst = X[i, :, s:s+burst_len]
            bursts.append(burst)
    bursts = np.stack(bursts, axis=0)
    return bursts

# def burstsMultistepDataset(X, n, m, J, replace=False, seed=None):
#     rng = np.random.default_rng(seed)

#     N, d, K = X.shape
#     bursts_per_traj = J // N
#     remainder = J % N
#     bursts = []
#     burst_len = n + m

#     for i in range(N):
#         num_bursts = bursts_per_traj + (1 if i < remainder else 0)
#         max_start = K - burst_len

#         if max_start < 0:
#             raise ValueError(f"Trajectory {i} is too short for burst length {burst_len}")

#         starts = rng.choice(max_start + 1, size=num_bursts, replace=replace)

#         for s in starts:
#             bursts.append(X[i, :, s:s + burst_len])

#     return np.stack(bursts, axis=0)

def prepareDataset(XtrainNorm, n):
    """
        Prepare the Tensor dataset
        XtrainNorm: Normalized bursts of shape [J, d, n+m] 
    """
    Xtilde = torch.tensor(XtrainNorm[:, :, :n], dtype=torch.float32).permute(0, 2, 1)  # [J, n, d]
    Ytilde = torch.tensor(XtrainNorm[:, :, n:], dtype=torch.float32).permute(0, 2, 1)   # [J, m, d]

    return TensorDataset(Xtilde, Ytilde)

def train_model(train_dataset, model_class, model_kwargs=None, multistep = True, lr=1e-1, weight_decay=0.01, batch_size=32, epochs=200, model_name='model'):
    """
    Trains a model using the provided model class and training dataset.

    Parameters:
    - train_dataset: TensorDataset
    - model_class: class (e.g., TDTF)
    - model_kwargs: dict of arguments to pass to the model_class
    - multistep: bool, whether to use multistep training
    - lr: learning rate
    - epochs: number of training epochs
    - dtype: data type for model parameters
    - model_name: name for saving results

    Returns:
    - trained ModelClass instance
    """
    if model_kwargs is None:
        model_kwargs = {}

    model = ModelClass(model_name)
    model_instance = model_class(**model_kwargs)
    model.set_dynamics_dataset(train_dataset)
    model.set_model(model_instance, multistep=multistep, weight_decay=weight_decay, opt_kwargs=dict(lr=lr))
    model.train(epochs=epochs, batch_size=batch_size)
    return model
