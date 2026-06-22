# Copyright 2025. Licensed under the MIT License.
"""Linear probes (Phase 8) — pure torch, no sklearn."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import r2_score


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def _split(n, train_ratio, seed, groups=None):
    """Return (train_idx, val_idx).  If ``groups`` is given (one id per sample),
    the split is done over unique groups (clips) so correlated frames never leak
    across the train/val boundary."""
    g = torch.Generator().manual_seed(seed)
    if groups is None:
        perm = torch.randperm(n, generator=g)
        n_tr = int(round(n * train_ratio))
        return perm[:n_tr], perm[n_tr:]
    groups = torch.as_tensor(groups)
    uniq = torch.unique(groups)
    gperm = uniq[torch.randperm(len(uniq), generator=g)]
    n_tr_g = max(1, int(round(len(uniq) * train_ratio)))
    train_groups = set(gperm[:n_tr_g].tolist())
    is_train = torch.tensor([int(int(x) in train_groups) for x in groups])
    all_idx = torch.arange(n)
    return all_idx[is_train.bool()], all_idx[~is_train.bool()]


def _standardize(tr, va, mode="per_dim"):
    """Normalize features using TRAIN statistics.

    mode='per_dim': z-score each feature (well-conditioned, but NOT invariant to
        an orthonormal rotation of the feature space — use only for natural-basis
        features like the full latent).
    mode='global': subtract per-dim mean, divide by a single global scalar (mean
        per-dim std).  This IS rotation-invariant, so probes on subspace
        coefficients e/u give the same answer regardless of the (arbitrary)
        orthonormal basis chosen for the subspace.
    """
    mu = tr.mean(0, keepdim=True)
    if mode == "per_dim":
        sd = tr.std(0, keepdim=True) + 1e-6
    elif mode == "global":
        sd = tr.std(0).mean() + 1e-6        # scalar
    else:
        sd = 1.0
    return (tr - mu) / sd, (va - mu) / sd


@torch.no_grad()
def _acc(logits, y):
    return float((logits.argmax(-1) == y).float().mean().item())


def train_classification_probe(features, labels, num_classes=None, train_ratio=0.8,
                               epochs=200, lr=1e-3, weight_decay=1e-4, seed=42, device="cpu",
                               groups=None, whiten="per_dim"):
    """features [M,D], labels [M] (int).  Returns metrics dict.

    ``groups`` (one id per sample) enables clip-grouped train/val splitting.
    ``whiten`` selects feature normalization ('per_dim' | 'global' | 'none').
    """
    features = features.float()
    labels = labels.long()
    # remap labels to contiguous [0..C)
    uniq = torch.unique(labels)
    remap = {int(u): i for i, u in enumerate(uniq.tolist())}
    labels = torch.tensor([remap[int(l)] for l in labels])
    C = num_classes or len(uniq)
    M, D = features.shape
    tr, va = _split(M, train_ratio, seed, groups=groups)
    Xtr, Xva = _standardize(features[tr], features[va], mode=whiten)
    ytr, yva = labels[tr], labels[va]
    Xtr, Xva, ytr, yva = [t.to(device) for t in (Xtr, Xva, ytr, yva)]

    torch.manual_seed(seed)              # deterministic probe init (run-to-run stable)
    probe = LinearProbe(D, C).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        probe.train(); opt.zero_grad()
        loss = F.cross_entropy(probe(Xtr), ytr)
        loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        tr_acc = _acc(probe(Xtr), ytr); va_acc = _acc(probe(Xva), yva)
    # majority-class baseline on val
    maj = float(torch.bincount(yva, minlength=C).max().item() / max(len(yva), 1))
    return {"task": "classification", "n_classes": int(C), "n": int(M),
            "train_acc": tr_acc, "val_acc": va_acc, "majority_baseline": maj,
            "val_acc_above_chance": va_acc - maj}


def train_regression_probe(features, targets, train_ratio=0.8, epochs=200, lr=1e-3,
                           weight_decay=1e-4, seed=42, device="cpu", groups=None, whiten="per_dim"):
    """features [M,D], targets [M,K] (float).  Returns metrics dict (MSE, R^2)."""
    features = features.float(); targets = targets.float()
    if targets.ndim == 1:
        targets = targets[:, None]
    M, D = features.shape; K = targets.shape[1]
    tr, va = _split(M, train_ratio, seed, groups=groups)
    Xtr, Xva = _standardize(features[tr], features[va], mode=whiten)
    ytr_raw, yva_raw = targets[tr], targets[va]
    ymu = ytr_raw.mean(0, keepdim=True); ysd = ytr_raw.std(0, keepdim=True) + 1e-6
    ytr = (ytr_raw - ymu) / ysd; yva = (yva_raw - ymu) / ysd
    Xtr, Xva, ytr, yva = [t.to(device) for t in (Xtr, Xva, ytr, yva)]

    torch.manual_seed(seed)              # deterministic probe init
    probe = LinearProbe(D, K).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        probe.train(); opt.zero_grad()
        loss = F.mse_loss(probe(Xtr), ytr)
        loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        pred_va = probe(Xva)
        mse = float(F.mse_loss(pred_va, yva).item())
        r2 = r2_score(pred_va.cpu(), yva.cpu())
    return {"task": "regression", "out_dim": int(K), "n": int(M),
            "val_mse_standardized": mse, "val_r2": r2}
