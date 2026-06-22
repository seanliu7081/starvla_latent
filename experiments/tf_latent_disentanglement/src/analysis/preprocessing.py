# Copyright 2025. Licensed under the MIT License.
"""Latent preprocessing for temporal/frequency analysis."""
from __future__ import annotations

import torch


def prepare_latent_for_temporal_analysis(z: torch.Tensor) -> torch.Tensor:
    """Convert common latent shapes into [N, T, D_flat].

    Supported: [N,T,D] | [N,T,S,D] (token mean-pool) | [N,T,C,H,W] (spatial mean-pool).
    """
    if z.ndim == 3:
        return z
    if z.ndim == 4:
        return z.mean(dim=2)            # [N,T,S,D] -> [N,T,D]
    if z.ndim == 5:
        return z.mean(dim=(-1, -2))     # [N,T,C,H,W] -> [N,T,C]
    raise ValueError(f"Unsupported latent shape: {tuple(z.shape)}")


def standardize_latents(z: torch.Tensor, eps: float = 1e-6):
    """Per-dimension z-score over (clips, time).  Returns (z_norm, mean, std)."""
    z = z.float()
    mean = z.mean(dim=(0, 1), keepdim=True)
    std = z.std(dim=(0, 1), keepdim=True) + eps
    return (z - mean) / std, mean, std


def remove_temporal_mean(z: torch.Tensor) -> torch.Tensor:
    """Remove per-clip temporal mean (kills the DC component before FFT)."""
    return z - z.mean(dim=1, keepdim=True)
