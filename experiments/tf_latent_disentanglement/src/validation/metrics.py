# Copyright 2025. Licensed under the MIT License.
"""Shared metrics: R^2, between/within-video variance ratio, etc."""
from __future__ import annotations

import torch


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred, target = pred.float(), target.float()
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum() + 1e-8
    return float((1 - ss_res / ss_tot).item())


def between_within_ratio(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """y: [N,T,D].  R_d = Var_n(clip-mean_d) / E_{n,t}[(y - clip-mean)^2].

    High R_d => dimension is stable within a clip but varies across clips
    (environment-like).  Returns [D].
    """
    y = y.float()
    clip_mean = y.mean(dim=1)                       # [N,D]
    between = clip_mean.var(dim=0, unbiased=False)  # [D]
    within = ((y - clip_mean[:, None, :]) ** 2).mean(dim=(0, 1))  # [D]
    return between / (within + eps)


def mean_low_ratio(low_ratio: torch.Tensor) -> float:
    return float(low_ratio.float().mean().item())
