# Copyright 2025. Licensed under the MIT License.
"""Environment/action subspace swapping (Phase 11)."""
from __future__ import annotations

import torch


@torch.no_grad()
def swap_env_action_subspaces(z_A, z_B, W_env, W_act, mean, std, alpha: float = 1.0):
    """Environment (+ residual) from A, action/motion subspace from B.

    z_A, z_B : [...,D] latents (same shape).  Returns z_swap [...,D].

    We keep ALL of A (environment subspace + the residual orthogonal to both
    subspaces) and replace only A's *action-subspace* component with B's:

        z_swap = z_A + alpha * (z_B - z_A) @ P_act
               = P_env z_A + (residual_A) + [alpha P_act z_B + (1-alpha) P_act z_A]

    This is equivalent to ``P_env z_A + P_act z_B + residual_A`` at alpha=1 and,
    crucially, preserves the (often large) residual so the edited latent stays
    on-manifold — important for high-D layers where env_dim+act_dim << D.
    alpha=1 => full B action, alpha=0 => unchanged A.
    """
    y_A = (z_A.float() - mean) / std
    y_B = (z_B.float() - mean) / std
    P_act = W_act @ W_act.T
    y_swap = y_A + alpha * ((y_B - y_A) @ P_act)
    return y_swap * std + mean
