# Copyright 2025. Licensed under the MIT License.
"""Token-wise frequency analysis (Phase 6): which spatial tokens carry motion."""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_token_frequency_scores(token_z: torch.Tensor, fps: float = 20.0, cutoff_hz: float = 2.0):
    """token_z: [N,T,S,D] -> (token_low_ratio [S], token_high_ratio [S])."""
    assert token_z.ndim == 4, token_z.shape
    N, T, S, D = token_z.shape
    z = token_z.float()
    z = z - z.mean(dim=1, keepdim=True)
    Zf = torch.fft.rfft(z, dim=1, norm="ortho")     # [N,F,S,D]
    power = Zf.abs().pow(2)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(z.device)
    low_mask = freqs <= cutoff_hz
    high_mask = freqs > cutoff_hz
    low_energy = power[:, low_mask, :, :].sum(dim=(0, 1, 3))     # [S]
    high_energy = power[:, high_mask, :, :].sum(dim=(0, 1, 3))   # [S]
    total = low_energy + high_energy + 1e-8
    return (low_energy / total).cpu(), (high_energy / total).cpu()
