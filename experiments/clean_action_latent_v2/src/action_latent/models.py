# Copyright 2025. Licensed under the MIT License.
"""Post-hoc trainable models.

M1 needs only ``MLPRegressor`` + ``train_regressor`` (the nonlinear action
*ceiling* for the confound gate). ``ActionBottleneck`` and adversaries (Phase 3)
are stubbed below and implemented in M4.

NOTE: these are POST-HOC heads only. They never touch frozen-model weights
(Prime Directive 1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _agg_r2(Y, P):
    ss_res = ((Y - P) ** 2).sum()
    ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum() + 1e-12
    return float((1 - ss_res / ss_tot).item())


def train_regressor(Xtr, Ytr, Xva, Yva, *, hidden=512, dropout=0.1, lr=3e-4,
                    weight_decay=1e-4, epochs=200, batch_size=1024, patience=15,
                    device="cuda", seed=42, verbose=False):
    """Train an MLP X->Y with early stopping on val aggregate-R^2.

    Returns (best_model_on_cpu_eval, info). Inputs are CPU tensors; standardize
    BEFORE calling (train-only stats).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device if torch.cuda.is_available() else "cpu"
    model = MLPRegressor(Xtr.shape[1], Ytr.shape[1], hidden, dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr_d, Ytr_d = Xtr.to(dev), Ytr.to(dev)
    Xva_d, Yva_d = Xva.to(dev), Yva.to(dev)
    n = Xtr.shape[0]
    best_r2, best_state, bad = -1e18, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            loss = ((model(Xtr_d[b]) - Ytr_d[b]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            r2 = _agg_r2(Yva_d, model(Xva_d))
        if r2 > best_r2 + 1e-5:
            best_r2, bad = r2, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"  ep{ep:3d} val_r2={r2:.4f} best={best_r2:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model.to("cpu"), {"best_val_r2": best_r2, "epochs_ran": ep + 1}


@torch.no_grad()
def regressor_predict(model, X, device="cpu", batch_size=4096):
    model.eval()
    outs = []
    for s in range(0, X.shape[0], batch_size):
        outs.append(model(X[s:s + batch_size]))
    return torch.cat(outs, 0)


# --------------------------------------------------------------------------- #
# Phase 3 — nonlinear action bottleneck (post-hoc; never touches frozen weights)
# --------------------------------------------------------------------------- #
class ActionBottleneck(nn.Module):
    """2-layer GELU+LayerNorm encoder -> u; action + transition decoders.

    A NONLINEAR u cannot be projected out of h, so it can only reach Outcome-B
    (T2) evidence — used here for the nonlinear ceiling + adversarial tradeoff.
    """

    def __init__(self, input_dim, u_dim, hidden=512, action_dim=7, trans_dim=128, dropout=0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, u_dim))
        self.action_dec = nn.Linear(u_dim, action_dim)
        self.trans_dec = nn.Sequential(nn.Linear(u_dim, hidden), nn.GELU(), nn.Linear(hidden, trans_dim))

    def forward(self, x):
        u = self.enc(x)
        return u, self.action_dec(u), self.trans_dec(u)


class EnvAdversary(nn.Module):
    """Classifier on u (via gradient reversal) that tries to predict env labels."""

    def __init__(self, u_dim, n_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(u_dim, hidden), nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, u):
        return self.net(u)
