# Copyright 2025. Licensed under the MIT License.
"""Feature construction: standardization, env-subspace removal (h_res), env
coeffs (e), and alignment-aware frame flattening.

ALIGNMENT (resolved in Phase 0): ``actions[:, t]`` is paired with ``h[:, t+shift]``.
The frozen GR00T head's convention is shift=0 (h_t -> a_t); Phase 0 confirms this
empirically and all later phases read ``alignment.shift`` from alignment.json.
"""
from __future__ import annotations

import numpy as np
import torch


def standardize(x, mean, std):
    return (x - mean) / std


def remove_subspace(x, W, mean, std):
    """Remove span(W) from x in standardized space, return STANDARDIZED residual.

    x:[...,D], W:[D,K]. Unlike the prior experiment's operator (which mapped back
    to raw space), here we keep the standardized residual h_res = h_hat - P_env h_hat
    because all downstream linear probes consume standardized features.
    """
    xn = (x - mean) / std
    proj = (xn @ W) @ W.T
    return xn - proj


def env_coeffs(x, W, mean, std):
    """e = W^T standardized(x).  x:[...,D], W:[D,K] -> [...,K]."""
    return ((x - mean) / std) @ W


def flatten_frames(feat: torch.Tensor, idx, shift: int = 0, action: torch.Tensor | None = None):
    """Flatten per-clip features/targets to per-frame rows under an alignment shift.

    feat:[N,T,D] selected at clip indices ``idx``; pair frame t's feature with
    action[:, t] using ``h[:, t+shift]``. Frames without a valid partner (at the
    sequence ends, depending on shift) are dropped per clip.

    Returns (X[M,D], clip_ids[M]) and, if ``action`` given, (X, Y[M,7], clip_ids).
    """
    N, T, D = feat.shape
    t0 = max(0, -shift)
    t1 = min(T, T - shift)            # valid action-frame range [t0, t1)
    xs, ys, cids = [], [], []
    for i in idx:
        ts = np.arange(t0, t1)
        xs.append(feat[i, ts + shift])           # h_{t+shift}
        cids.append(np.full(len(ts), int(i)))
        if action is not None:
            ys.append(action[i, ts])             # a_t
    X = torch.cat(xs, 0)
    clip_ids = np.concatenate(cids)
    if action is not None:
        Y = torch.cat(ys, 0)
        return X, Y, clip_ids
    return X, clip_ids


def delta(feat: torch.Tensor):
    """Δfeat_t = feat_{t+1} - feat_t  -> [N,T-1,D] (used in Phase 2)."""
    return feat[:, 1:] - feat[:, :-1]
