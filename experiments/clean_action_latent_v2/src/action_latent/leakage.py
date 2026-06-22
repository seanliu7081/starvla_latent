# Copyright 2025. Licensed under the MIT License.
"""Leakage probes (Phase 4): discrete (suite/task/object) + continuous (u->e R^2)
+ conditional (env <- [action_bin, u]).

Review-critical points baked in (plan §3.3 / §4):
- Discrete-at-chance is necessary, NOT sufficient — always report the CONTINUOUS
  probe (u->e R^2), which catches entanglement the label probes miss.
- The defensible target is LOW CONDITIONAL leakage, not zero raw leakage (in
  robotics the correct action genuinely depends on task/object).
- Action binning separates the near-binary gripper; positions/rotations are
  k-means binned (rotations are small deltas, ~angular≈euclidean here).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import features as F
from . import linear as L
from . import metrics as M


def _flatten(feat, idx, shift, labels=None):
    """feat:[N,T,D] at clip idx -> X[M,D], clip_ids[M]; labels:[N] -> y[M] per-frame."""
    N, T, D = feat.shape
    t0 = max(0, -shift); t1 = min(T, T - shift)
    xs, ys, cids = [], [], []
    for i in idx:
        ts = np.arange(t0, t1)
        xs.append(feat[i, ts + shift])
        cids.append(np.full(len(ts), int(i)))
        if labels is not None:
            ys.append(np.full(len(ts), int(labels[i])))
    X = torch.cat(xs, 0)
    cid = np.concatenate(cids)
    if labels is not None:
        return X, np.concatenate(ys), cid
    return X, cid


class _Softmax(nn.Module):
    def __init__(self, d, c, hidden=0):
        super().__init__()
        self.net = nn.Linear(d, c) if hidden == 0 else nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, c))

    def forward(self, x):
        return self.net(x)


def _train_classifier(Xtr, ytr, Xva, yva, c, *, hidden=0, epochs=300, lr=1e-2,
                      wd=1e-3, device="cuda", seed=0):
    torch.manual_seed(seed)
    dev = device if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xva = ((Xtr - mu) / sd).to(dev), ((Xva - mu) / sd).to(dev)
    ytr = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    yva = torch.as_tensor(yva, dtype=torch.long, device=dev)
    model = _Softmax(Xtr.shape[1], c, hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = nn.functional.cross_entropy(model(Xtr), ytr)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1) == yva).float().mean().item()
        if acc > best + 1e-4:
            best, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 40:
                break
    model.load_state_dict(best_state)
    return model.to(dev), (mu.to(dev), sd.to(dev))


def _balanced_acc(yt, yp, c):
    accs = []
    for k in range(c):
        m = yt == k
        if m.sum() > 0:
            accs.append((yp[m] == k).mean())
    return float(np.mean(accs)) if accs else 0.0


def discrete_probe(u_feat, labels, split, shift, n_classes, *, hidden=0, device="cuda",
                   seed=0, n_boot=1000):
    """Probe u -> discrete label, held-out clips. acc, majority, balanced acc, clip-bootstrap CI."""
    tr, va, te = split["train"], split["val"], split["test"]
    Xtr, ytr, _ = _flatten(u_feat, tr, shift, labels)
    Xva, yva, _ = _flatten(u_feat, va, shift, labels)
    Xte, yte, cte = _flatten(u_feat, te, shift, labels)
    model, (mu, sd) = _train_classifier(Xtr, ytr, Xva, yva, n_classes, hidden=hidden,
                                        device=device, seed=seed)
    dev = mu.device
    with torch.no_grad():
        yp = model((Xte.to(dev) - mu) / sd).argmax(1).cpu().numpy()
    acc = float((yp == yte).mean())
    vals, cnts = np.unique(ytr, return_counts=True)
    maj = vals[cnts.argmax()]
    majority = float((yte == maj).mean())
    bal = _balanced_acc(yte, yp, n_classes)
    uniq = np.unique(cte); idx_by = {ccc: np.where(cte == ccc)[0] for ccc in uniq}
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        ch = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[ccc] for ccc in ch])
        boots.append(float((yp[rows] == yte[rows]).mean()))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"acc": acc, "majority_baseline": majority, "balanced_acc": bal,
            "acc_above_chance": acc - majority, "acc_ci": [float(lo), float(hi)],
            "n_classes": int(n_classes)}


def continuous_probe(u_feat, e, split, shift, cfg, seed=0):
    """REVIEW-CRITICAL continuous leakage: ridge u -> e (env coeffs), report R^2."""
    tr, va, te = split["train"], split["val"], split["test"]
    flat = u_feat[tr].reshape(-1, u_feat.shape[-1])
    mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-6)
    uf = (u_feat - mu) / sd
    Xtr, Ytr, _ = F.flatten_frames(uf, tr, shift=shift, action=e)
    Xva, Yva, _ = F.flatten_frames(uf, va, shift=shift, action=e)
    Xte, Yte, cte = F.flatten_frames(uf, te, shift=shift, action=e)
    W, lam, _ = L.fit_ridge_cv(Xtr, Ytr, Xva, Yva, list(cfg.linear.ridge_lambdas))
    pte = L.predict_ridge(W, Xte)
    rep = M.r2_report(Yte.numpy(), pte.numpy())
    ci = M.bootstrap_r2_ci(Yte.numpy(), pte.numpy(), cte, n_boot=int(cfg.eval.bootstrap_n), seed=seed)
    return {"u_to_e_r2": rep["r2"], "u_to_e_r2_equal": rep["r2_equal"], "u_to_e_r2_ci": [ci["lo"], ci["hi"]]}


def _kmeans(X, k, iters=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:k]
    C = X[idx].clone()
    a = torch.zeros(X.shape[0], dtype=torch.long)
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    return a, C


def action_bins(actions, split, cfg, seed=0):
    """Composite action bin per frame: (pos kmeans) x (rot kmeans) x (gripper 0/1).
    Centroids fit on TRAIN frames only. Returns labels[N,T] (int) and n_bins."""
    ab = cfg.action_binning
    tr = split["train"]
    A = actions.reshape(-1, 7)
    Atr = actions[tr].reshape(-1, 7)
    k = int(ab.k)
    k_pos = max(2, k // 4); k_rot = 2
    _, Cp = _kmeans(Atr[:, 0:3], k_pos, seed=seed)
    _, Cr = _kmeans(Atr[:, 3:6], k_rot, seed=seed + 1)
    pos = torch.cdist(A[:, 0:3], Cp).argmin(1)
    rot = torch.cdist(A[:, 3:6], Cr).argmin(1)
    grip = (A[:, 6] > 0.5).long()
    comp = (pos * (k_rot * 2) + rot * 2 + grip).reshape(actions.shape[0], actions.shape[1])
    nb = k_pos * k_rot * 2
    return comp, nb


def conditional_leakage(u_feat, actions, env_labels, split, shift, cfg, n_env, seed=0):
    """leak_cond = acc(env <- [action_bin_onehot, u]) - acc(env <- action_bin_onehot).
    Small => u adds little env info beyond what the action itself implies."""
    abins, nb = action_bins(actions, split, cfg, seed=seed)
    oh = L.one_hot(abins.reshape(-1).long().numpy(), nb).reshape(actions.shape[0], actions.shape[1], nb)
    p_base = discrete_probe(oh, env_labels, split, shift, n_env, device=cfg.device, seed=seed)
    p_aug = discrete_probe(torch.cat([oh, u_feat], dim=-1), env_labels, split, shift, n_env,
                           device=cfg.device, seed=seed)
    return {"env_from_action_bin_acc": p_base["acc"], "env_from_action_bin_plus_u_acc": p_aug["acc"],
            "leak_cond": p_aug["acc"] - p_base["acc"], "n_action_bins": int(nb), "n_env": int(n_env)}
