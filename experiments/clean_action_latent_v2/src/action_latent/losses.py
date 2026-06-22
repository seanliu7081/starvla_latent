# Copyright 2025. Licensed under the MIT License.
"""Loss functions for the nonlinear bottleneck (Phase 3).

action + transition + decorrelation(vs e) + smoothness + L2, plus a gradient-
reversal env adversary (Stage B). The discrete adversary is necessary-not-
sufficient (plan §3.3): the real cleanliness check is the continuous u->e probe
in Phase 4, which we recompute on the trained u.
"""
from __future__ import annotations

import torch
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grad_reverse(x, lamb=1.0):
    return _GradReverse.apply(x, lamb)


def decorrelation_loss(u, e):
    """Penalize cross-correlation between u dims and env coeffs e (centered)."""
    u = u - u.mean(0, keepdim=True)
    e = e - e.mean(0, keepdim=True)
    us = u / (u.std(0, keepdim=True) + 1e-6)
    es = e / (e.std(0, keepdim=True) + 1e-6)
    C = (us.T @ es) / u.shape[0]
    return (C ** 2).mean()


def smoothness_loss(u_seq):
    """u_seq:[B,T,d] -> mean squared temporal difference (kept tiny; plan warns)."""
    return ((u_seq[:, 1:] - u_seq[:, :-1]) ** 2).mean()
