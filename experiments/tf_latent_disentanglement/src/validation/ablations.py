# Copyright 2025. Licensed under the MIT License.
"""Subspace projection / ablation operators (Phase 9)."""
from __future__ import annotations

import torch


@torch.no_grad()
def project_to_subspace(z, W, mean=None, std=None):
    """Keep only the component of z inside span(W).  z:[...,D], W:[D,K].

    Returns (z_projected [...,D], coeff [...,K]) in the ORIGINAL latent space.
    """
    z0 = z.float()
    if mean is not None and std is not None:
        z0 = (z0 - mean) / std
    coeff = z0 @ W
    z_proj = coeff @ W.T
    if mean is not None and std is not None:
        z_proj = z_proj * std + mean
    return z_proj, coeff


@torch.no_grad()
def remove_subspace(z, W, mean=None, std=None):
    """Remove the component of z inside span(W).  Returns z_removed [...,D]."""
    z0 = z.float()
    if mean is not None and std is not None:
        zn = (z0 - mean) / std
        proj = (zn @ W) @ W.T
        return (zn - proj) * std + mean
    proj = (z0 @ W) @ W.T
    return z0 - proj
