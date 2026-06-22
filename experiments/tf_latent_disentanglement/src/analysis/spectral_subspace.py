# Copyright 2025. Licensed under the MIT License.
"""Spectral subspace decomposition (Phase 5): find low/high-frequency directions."""
from __future__ import annotations

import torch


def fft_low_high_pass(z: torch.Tensor, fps: float = 20.0, cutoff_hz: float = 2.0):
    """z: [N,T,D] -> (z_low, z_high) via ideal FFT band split along time."""
    assert z.ndim == 3, z.shape
    N, T, D = z.shape
    Zf = torch.fft.rfft(z, dim=1, norm="ortho")
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(z.device)
    low_mask = (freqs <= cutoff_hz).float()[None, :, None]
    high_mask = 1.0 - low_mask
    z_low = torch.fft.irfft(Zf * low_mask, n=T, dim=1, norm="ortho")
    z_high = torch.fft.irfft(Zf * high_mask, n=T, dim=1, norm="ortho")
    return z_low, z_high


def covariance_from_sequence(x: torch.Tensor) -> torch.Tensor:
    """x: [N,T,D] -> [D,D] covariance over all (clip,time) samples."""
    N, T, D = x.shape
    x = x.reshape(N * T, D)
    x = x - x.mean(dim=0, keepdim=True)
    return (x.T @ x) / max(x.shape[0] - 1, 1)


def generalized_top_directions(A: torch.Tensor, B: torch.Tensor, k: int, eps: float = 1e-4):
    """Top-k subspace maximising w^T A w / w^T B w.

    Returns (W [D,k], scores [k]) where:
      * ``scores`` are the TRUE top-k generalized eigenvalues of  A w = lambda B w
        (sorted descending) — the spectrum of the generalized problem on the
        selected subspace.
      * ``W`` is a EUCLIDEAN-orthonormal basis spanning that subspace (so the
        orthogonal projector is simply ``W @ W.T``).  Note: because the
        generalized eigenvectors are B-orthogonal (not Euclidean-orthonormal),
        the QR step mixes them — individual W columns are therefore NOT
        per-column eigenvectors, but span() and the eigenvalue *set* are exact.
    """
    D = A.shape[0]
    k = min(k, D)
    I = torch.eye(D, device=A.device, dtype=A.dtype)
    B_reg = B + eps * I
    evals_B, U_B = torch.linalg.eigh(B_reg)
    evals_B = evals_B.clamp_min(eps)
    B_inv_sqrt = U_B @ torch.diag(evals_B.rsqrt()) @ U_B.T
    M = B_inv_sqrt @ A @ B_inv_sqrt
    M = 0.5 * (M + M.T)
    evals, V = torch.linalg.eigh(M)            # ascending
    V_top = V[:, -k:].flip(1)                  # top-k, descending
    scores = evals[-k:].flip(0)                # TRUE generalized eigenvalues, descending
    W = B_inv_sqrt @ V_top
    W, _ = torch.linalg.qr(W, mode="reduced")  # orthonormal basis for the same subspace
    W = W[:, :k]
    return W, scores


@torch.no_grad()
def spectral_subspace_decomposition(z: torch.Tensor, fps: float = 20.0, cutoff_hz: float = 2.0,
                                    env_dim: int = 128, act_dim: int = 16, eps: float = 1e-4,
                                    orthogonalize_action_against_env: bool = True) -> dict:
    """z: [N,T,D].  Returns env/action bases (W_env,W_act) and coefficients (e,m,u)."""
    assert z.ndim == 3, z.shape
    z = z.float()
    mean = z.mean(dim=(0, 1), keepdim=True)
    std = z.std(dim=(0, 1), keepdim=True) + 1e-6
    z_norm = (z - mean) / std

    z_low, z_high = fft_low_high_pass(z_norm, fps=fps, cutoff_hz=cutoff_hz)
    C_low = covariance_from_sequence(z_low)
    C_high = covariance_from_sequence(z_high)

    W_env, env_scores = generalized_top_directions(C_low, C_high, env_dim, eps=eps)
    W_act_raw, act_scores = generalized_top_directions(C_high, C_low, act_dim, eps=eps)

    if orthogonalize_action_against_env:
        W_act = W_act_raw - W_env @ (W_env.T @ W_act_raw)
        W_act, _ = torch.linalg.qr(W_act, mode="reduced")
        W_act = W_act[:, :W_act_raw.shape[1]]
    else:
        W_act = W_act_raw

    e = z_norm @ W_env
    m = z_norm @ W_act
    u = (z_norm[:, 1:] - z_norm[:, :-1]) @ W_act
    return {
        "W_env": W_env.cpu(), "W_act": W_act.cpu(),
        "e": e.cpu(), "m": m.cpu(), "u": u.cpu(),
        "mean": mean.cpu(), "std": std.cpu(),
        "env_scores": env_scores.cpu(), "act_scores": act_scores.cpu(),
        "C_low": C_low.cpu(), "C_high": C_high.cpu(),
        "cutoff_hz": float(cutoff_hz), "fps": float(fps),
    }


def random_orthonormal_basis(D: int, k: int, seed: int = 0, device="cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(D, min(k, D), generator=g)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q[:, :k].to(device)


def pca_basis(z: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k PCA directions of standardized [N,T,D] (baseline)."""
    z = z.float()
    mean = z.mean(dim=(0, 1), keepdim=True)
    std = z.std(dim=(0, 1), keepdim=True) + 1e-6
    zn = ((z - mean) / std).reshape(-1, z.shape[-1])
    C = covariance_from_sequence(zn[None])
    evals, V = torch.linalg.eigh(C)
    return V[:, -k:].flip(1)
