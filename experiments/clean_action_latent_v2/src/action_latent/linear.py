# Copyright 2025. Licensed under the MIT License.
"""Linear regression utilities (closed-form ridge with val-selected lambda).

Phase 2's PLS / reduced-rank regression / CCA live here too (added in M2); for
M1 we only need ridge + one-hot helpers for the confound gate.
"""
from __future__ import annotations

import numpy as np
import torch


def _augment(X):
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1)


def fit_ridge(X, Y, ridge=1.0):
    """Closed-form ridge with bias. X:[M,D], Y:[M,K] -> W:[D+1,K] (bias not penalized)."""
    Xa = _augment(X.double())
    D = Xa.shape[1]
    reg = ridge * torch.eye(D, dtype=Xa.dtype)
    reg[-1, -1] = 0.0  # do not penalize bias
    A = Xa.T @ Xa + reg
    return torch.linalg.solve(A, Xa.T @ Y.double())


def predict_ridge(W, X):
    return (_augment(X.double()) @ W).float()


def fit_ridge_cv(Xtr, Ytr, Xva, Yva, lambdas):
    """Pick ridge lambda by validation aggregate-R^2; refit on train. Returns (W, lam, val_r2)."""
    best = (None, None, -1e18)
    for lam in lambdas:
        W = fit_ridge(Xtr, Ytr, ridge=float(lam))
        pv = predict_ridge(W, Xva)
        r2 = _agg_r2(Yva, pv)
        if r2 > best[2]:
            best = (W, float(lam), r2)
    return best


def _agg_r2(Y, P):
    Y = Y.double()
    P = P.double()
    ss_res = ((Y - P) ** 2).sum()
    ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum() + 1e-12
    return float((1 - ss_res / ss_tot).item())


# --------------------------------------------------------------------------- #
# Supervised linear action subspaces (projectable: return ORTHONORMAL W[D,d])
# and unsupervised baselines. Pure torch (no sklearn).
#
# NOTE: the action target is 7-D, so any *supervised* linear subspace of h that
# predicts action has intrinsic rank <= 7. RRR/CCA are therefore capped at 7;
# PLS can extract d>7 X-covariance directions but action R^2 saturates by ~7.
# This is a finding, not a bug — reported in the d-sweep.
# --------------------------------------------------------------------------- #
def _ridge_coef(Xs, Y, ridge):
    """OLS-with-ridge coefficient B[D,K] mapping standardized X -> centered Y."""
    Yc = Y.double() - Y.double().mean(0, keepdim=True)
    D = Xs.shape[1]
    A = Xs.double().T @ Xs.double() + ridge * torch.eye(D, dtype=torch.float64)
    return torch.linalg.solve(A, Xs.double().T @ Yc)          # [D,K]


def rrr_subspace(Xs, Y, d, ridge=10.0):
    """Reduced-rank regression: top-d action-relevance-ordered orthonormal dirs."""
    B = _ridge_coef(Xs, Y, ridge)                              # [D,K]
    U, S, _ = torch.linalg.svd(B, full_matrices=False)        # U:[D,K] orthonormal
    k = min(d, U.shape[1])
    return U[:, :k].float()


def pls_subspace(Xs, Y, d):
    """NIPALS PLS; return an orthonormal basis [D,d] of the weight span."""
    X = Xs.double().clone()
    Yc = (Y.double() - Y.double().mean(0, keepdim=True)).clone()
    Ws = []
    for _ in range(d):
        M = X.T @ Yc                                           # [D,K]
        u, s, vt = torch.linalg.svd(M, full_matrices=False)
        w = u[:, 0]
        t = X @ w
        tt = (t @ t).clamp_min(1e-12)
        p = (X.T @ t) / tt
        q = (Yc.T @ t) / tt
        X = X - torch.outer(t, p)
        Yc = Yc - torch.outer(t, q)
        Ws.append(w)
    W = torch.stack(Ws, 1)                                     # [D,d]
    Q, _ = torch.linalg.qr(W)
    return Q[:, :d].float()


def cca_subspace(Xs, Y, d, ridge=1.0):
    """Canonical X-directions (capped at K=dim(Y)); orthonormalized."""
    Xs = Xs.double(); Yc = Y.double() - Y.double().mean(0, keepdim=True)
    M = Xs.shape[0]
    Dx, K = Xs.shape[1], Yc.shape[1]
    Sxx = Xs.T @ Xs / M + ridge * torch.eye(Dx, dtype=torch.float64)
    Syy = Yc.T @ Yc / M + ridge * torch.eye(K, dtype=torch.float64)
    Sxy = Xs.T @ Yc / M
    A = torch.linalg.solve(Sxx, Sxy) @ torch.linalg.solve(Syy, Sxy.T)   # [Dx,Dx]
    evals, evecs = torch.linalg.eigh((A + A.T) / 2)
    order = torch.argsort(evals, descending=True)
    k = min(d, K)
    W = evecs[:, order[:k]]
    Q, _ = torch.linalg.qr(W)
    return Q[:, :k].float()


def pca_subspace(Xs, d):
    """Top-d principal directions of standardized X (unsupervised baseline)."""
    U, S, Vt = torch.linalg.svd(Xs.double() - Xs.double().mean(0, keepdim=True),
                                full_matrices=False)
    return Vt[:d].T.float()                                    # [D,d]


def random_subspace(D, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, d, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :d].float()


def supervised_subspace(Xs, Y, d, method, ridge=10.0):
    if method == "pls":
        return pls_subspace(Xs, Y, d)
    if method == "rrr":
        return rrr_subspace(Xs, Y, d, ridge)
    if method == "cca":
        return cca_subspace(Xs, Y, d)
    raise ValueError(method)


def one_hot(labels: np.ndarray, n_classes: int | None = None) -> torch.Tensor:
    labels = np.asarray(labels).astype(int)
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    M = np.zeros((len(labels), n_classes), np.float32)
    M[np.arange(len(labels)), labels] = 1.0
    return torch.from_numpy(M)
