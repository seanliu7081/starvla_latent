# Copyright 2025. Licensed under the MIT License.
"""
Prediction-drop validation (Phase 10).

Two complementary tests:

  1. learned latent dynamics — fit a ridge map  z_t (variant) -> Δz_t (true)  on
     train clips and measure val MSE per ablation condition.  Removing the action
     subspace should hurt transition prediction more than random removal.

  2. action-head readout drop (model-based, run in 05) — feed ablated DiT token
     hidden into the FROZEN GR00T action head and measure action-prediction MSE.
"""
from __future__ import annotations

import torch


def _augment(X):
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1)


def fit_ridge(X, Y, ridge=1.0):
    """Closed-form ridge regression with bias.  X:[M,D], Y:[M,K] -> W:[D+1,K]."""
    Xa = _augment(X.float())
    D = Xa.shape[1]
    A = Xa.T @ Xa + ridge * torch.eye(D, dtype=Xa.dtype)
    return torch.linalg.solve(A, Xa.T @ Y.float())


def eval_ridge(W, X, Y):
    pred = _augment(X.float()) @ W
    mse = float(((pred - Y.float()) ** 2).mean().item())
    ss_res = ((Y.float() - pred) ** 2).sum()
    ss_tot = ((Y.float() - Y.float().mean(0, keepdim=True)) ** 2).sum() + 1e-8
    return mse, float((1 - ss_res / ss_tot).item())


def dynamics_drop(z_variant_train, z_true_train, z_variant_val, z_true_val, ridge=1.0):
    """Fit z_t(variant)->Δz_t(true) on train, eval on val.

    Each arg is [N,T,D].  Δz is computed from the TRUE (unablated) sequence so all
    conditions predict the same target; only the input (current latent) varies.
    Returns dict(mse, r2).
    """
    def pairs(zv, zt):
        X = zv[:, :-1].reshape(-1, zv.shape[-1])
        Y = (zt[:, 1:] - zt[:, :-1]).reshape(-1, zt.shape[-1])
        return X, Y
    Xtr, Ytr = pairs(z_variant_train, z_true_train)
    Xva, Yva = pairs(z_variant_val, z_true_val)
    W = fit_ridge(Xtr, Ytr, ridge=ridge)
    mse, r2 = eval_ridge(W, Xva, Yva)
    return {"val_mse": mse, "val_r2": r2}
