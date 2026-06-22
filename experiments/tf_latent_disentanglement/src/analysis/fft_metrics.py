# Copyright 2025. Licensed under the MIT License.
"""Dimension-wise FFT metrics (Phase 3)."""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_fft_metrics(z: torch.Tensor, fps: float = 20.0, cutoff_hz: float = 2.0,
                        remove_dc: bool = True) -> dict:
    """z: [N, T, D].  Returns power/freqs and per-dim low/high ratio + centroid."""
    assert z.ndim == 3, z.shape
    z = z.float()
    if remove_dc:
        z = z - z.mean(dim=1, keepdim=True)

    Zf = torch.fft.rfft(z, dim=1, norm="ortho")
    power = Zf.abs().pow(2)                          # [N,F,D]
    T = z.shape[1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(z.device)

    low_mask = freqs <= cutoff_hz
    high_mask = freqs > cutoff_hz
    total = power.sum(dim=(0, 1)) + 1e-8             # [D]
    low_energy = power[:, low_mask, :].sum(dim=(0, 1))
    high_energy = power[:, high_mask, :].sum(dim=(0, 1))
    low_ratio = low_energy / total
    high_ratio = high_energy / total
    centroid = (power * freqs[None, :, None]).sum(dim=(0, 1)) / total

    return {
        "power": power.cpu(), "freqs": freqs.cpu(),
        "low_ratio": low_ratio.cpu(), "high_ratio": high_ratio.cpu(),
        "centroid_hz": centroid.cpu(), "total_energy": total.cpu(),
        "cutoff_hz": float(cutoff_hz), "fps": float(fps),
    }


@torch.no_grad()
def low_ratio_of_signal(z: torch.Tensor, fps: float, cutoff_hz: float, remove_dc: bool = True) -> float:
    """Aggregate low-frequency energy ratio of a whole [N,T,D] signal (scalar)."""
    z = z.float()
    if remove_dc:
        z = z - z.mean(dim=1, keepdim=True)
    Zf = torch.fft.rfft(z, dim=1, norm="ortho")
    power = Zf.abs().pow(2)
    T = z.shape[1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(z.device)
    low = power[:, freqs <= cutoff_hz, :].sum()
    tot = power.sum() + 1e-8
    return float((low / tot).item())
