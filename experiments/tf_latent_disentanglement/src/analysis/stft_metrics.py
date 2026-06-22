# Copyright 2025. Licensed under the MIT License.
"""STFT local-action analysis (Phase 4)."""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_stft_power(z: torch.Tensor, fps: float = 20.0, n_fft: int = 16, hop_length: int = 4):
    """z: [N,T,D] -> power [N,D,F,W], freqs [F]."""
    assert z.ndim == 3, z.shape
    N, T, D = z.shape
    device = z.device
    z = z.float()
    z = z - z.mean(dim=1, keepdim=True)
    z = z / (z.std(dim=1, keepdim=True) + 1e-6)

    x = z.permute(0, 2, 1).reshape(N * D, T)
    window = torch.hann_window(n_fft, device=device)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                   window=window, return_complex=True, center=True)
    power = X.abs().pow(2)
    F, W = power.shape[-2], power.shape[-1]
    power = power.reshape(N, D, F, W)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fps).to(device)
    return power.cpu(), freqs.cpu()


@torch.no_grad()
def compute_local_high_frequency_score(z: torch.Tensor, fps: float = 20.0, n_fft: int = 16,
                                       hop_length: int = 4, high_cutoff_hz: float = 4.0):
    """z: [N,T,D] -> score [N,W] : mean high-band STFT power per window."""
    power, freqs = compute_stft_power(z, fps=fps, n_fft=n_fft, hop_length=hop_length)
    high_mask = freqs >= high_cutoff_hz
    if high_mask.sum() == 0:                 # fall back to top half of the band
        high_mask = freqs >= freqs.median()
    score = power[:, :, high_mask, :].mean(dim=(1, 2))   # [N,W]
    return score
