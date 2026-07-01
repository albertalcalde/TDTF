import numpy as np
from numpy.linalg import svd
import torch
import torch.nn as nn


def TDDMD(Xall): # the option of <r=None> option is disabled for now
    """
    Time Delayed Dynamic Mode Decomposition (TD-DMD)
    Xall: np.ndarray of shape (J, d, n + 1)
    A_TD: np.ndarray of shape (d*n , d*n)
    """

    X = Xall[:, :, :-1]
    Y = Xall[:, :, 1:]

    J, d, n = X.shape
    
    # Flatten with column-major so blocks are [delay0(d entries), delay1(d entries), ...]
    Xdelay = np.reshape(X, (J, d * n), order='F').T  # (d*n, J)
    Ydelay = np.reshape(Y, (J, d * n), order='F').T  # (d*n, J)

    U, S, Vh = svd(Xdelay, full_matrices=False)
    A_TD = Ydelay @ Vh.T @ np.diag(1/S) @ U.T
    return A_TD


class TDTF(nn.Module):
    """
    Implementation of the time-delayed transformer, where we optionally append a normalized time feature t/n
    to each token (so the input to the FF is d+1). 
    Output remains size d.
    """
    def __init__(self, d, h, pos_encoding = False):
        super().__init__()
        self.d = d
        self.h = h
        self.pos_encoding = pos_encoding

        if pos_encoding:
            din = d + 1                                         # + 1 for time feature
        else:
            din = d

        self.to_B = nn.Linear(din, din, bias=False)             # query-key matrix
        self.to_V = nn.Linear(din, d, bias=False)               # value matrix

        self.ff = nn.Sequential(
            nn.Linear(din, h),
            nn.Tanh(),                                          # hyperbolic tangent activation
            nn.Linear(h, din, bias=False)
        )

    def forward(self, x):
        batch, n, d = x.shape
        assert d == self.d, f"Expected last dim {self.d}, got {d}"

        if self.pos_encoding:
            # build normalized time feature t/n and append as an extra dimension
            t = torch.arange(n, device=x.device, dtype=x.dtype) / n
            t = t.view(1, n, 1).expand(batch, n, 1)             # (B, n, 1)
            x_aug = torch.cat([x, t], dim=-1)                   # (B, n, din)
            xn_aug = x_aug[:, -1:, :]                           # (B, 1, din)
        else:
            x_aug = x
            xn_aug = x[:, -1:, :]
        
        # feedforward
        ffx = self.ff(x_aug)                                    # (B, n, din)            
        ffxn = self.ff(xn_aug)                                  # (B, 1, din)
        Bffx = self.to_B(ffx)                                   # (B, n, din)
        Vffx = self.to_V(ffx)                                   # (B, n, d)
        
        # attention
        scores = ffxn @ Bffx.transpose(-2, -1)                  # (B, 1, n)
        attn = torch.softmax(scores, dim=-1)                    # (B, 1, n)

        out = x[:, -1:, :] + attn @ Vffx                        # (B, 1, d)
        return out


class DelayEmbeddedMLP(nn.Module):
    """
    Shared delay-embedded MLP baseline.

    Computes:
        y = sum_{i=1}^n MLP(x_i)

    Input:  x   [B, n, d]
    Output: y   [B, 1, d]
    """
    def __init__(self, d, n=None, h=100, pos_encoding=False):
        super().__init__()
        self.d = d
        self.n = n
        self.h = h
        self.pos_encoding = pos_encoding

        din = d + 1 if pos_encoding else d

        self.shared_mlp = nn.Sequential(
            nn.Linear(din, h),
            nn.Tanh(),
            nn.Linear(h, d, bias=False),
        )

    def forward(self, x):
        batch, n, d = x.shape

        if d != self.d:
            raise ValueError(f"Expected last dim {self.d}, got {d}")

        if self.n is not None and n != self.n:
            raise ValueError(f"Expected delay length n={self.n}, got n={n}")

        if self.pos_encoding:
            t = torch.arange(n, device=x.device, dtype=x.dtype) / n
            t = t.view(1, n, 1).expand(batch, n, 1)
            x = torch.cat([x, t], dim=-1)   # [B, n, d+1]

        z = self.shared_mlp(x)              # [B, n, d]
        out = z.sum(dim=1, keepdim=True)    # [B, 1, d]

        return out