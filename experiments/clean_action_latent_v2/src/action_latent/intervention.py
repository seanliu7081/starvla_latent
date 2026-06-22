# Copyright 2025. Licensed under the MIT License.
"""Phase 6 — frozen-head edited-h interface + projective edits (tier-1 causal).

Edits are applied to TOKEN-level h in the SUBSPACE-normalized standardized space
(the space W_env / W_act_sup live in), then mapped back to raw h for the frozen
GR00T head (which consumes token-level h, per the Phase-0 interface probe).

PAIRED-NOISE design (critical): the flow-matching head is stochastic (Phase 0
found cross-seed |Δ|≈1). For each (clip, noise-seed) we set the SAME torch seed
before every condition's head call, so the noise realization is shared and the
per-condition action-MSE DELTA vs `original` largely cancels the noise — making a
~5e-4 causal effect detectable. Deltas are averaged over seeds and bootstrapped
over the (few) token clips.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def edit_remove(h_tokens, W, mean, std):
    """Remove span(W) from token-h; return RAW-space edited h. h:[T,Ntok,D], W:[D,k]."""
    xn = (h_tokens.float() - mean) / std
    proj = (xn @ W) @ W.T
    return (xn - proj) * std + mean


@torch.no_grad()
def edit_keep(h_tokens, W, mean, std):
    """Keep ONLY span(W) (zero the complement) in standardized space; raw-space out."""
    xn = (h_tokens.float() - mean) / std
    proj = (xn @ W) @ W.T
    return proj * std + mean


def build_conditions(W_env, W_sup_list, W_prev, mean, std, D, seed):
    """dict[name] -> (op, W). W_sup_list: list of (name, W) supervised candidates."""
    conds = {"original": ("none", None), "no_env": ("remove", W_env),
             "no_prev_highfreq": ("remove", W_prev)}
    g = torch.Generator().manual_seed(seed)
    for nm, W in W_sup_list:
        d = W.shape[1]
        conds[f"no_sup_{nm}"] = ("remove", W)
        A = torch.randn(D, d, generator=g, dtype=torch.float64)
        Q, _ = torch.linalg.qr(A)
        conds[f"no_random_{nm}_d{d}"] = ("remove", Q[:, :d].float())
    if W_sup_list:
        conds[f"sup_{W_sup_list[0][0]}_only"] = ("keep", W_sup_list[0][1])
    return conds


@torch.no_grad()
def run_intervention(model, action_from_hidden, h_tokens_all, gt_actions, conds, mean, std,
                     *, n_seeds=10, base_seed=0):
    """Paired-noise readout-drop. Returns per-condition action MSE (mean over clips&seeds),
    delta vs original, and per-clip delta lists for bootstrapping."""
    names = list(conds.keys())
    per_clip_mse = {n: [] for n in names}
    per_clip_delta = {n: [] for n in names}
    for j in range(h_tokens_all.shape[0]):
        h = h_tokens_all[j].float()
        gt = gt_actions[j]
        edited = {}
        for n, (op, W) in conds.items():
            if op == "none":
                edited[n] = h
            elif op == "remove":
                edited[n] = edit_remove(h, W, mean, std)
            else:
                edited[n] = edit_keep(h, W, mean, std)
        seed_mse = {n: [] for n in names}
        for s in range(n_seeds):
            sd = base_seed + s
            for n in names:
                torch.manual_seed(sd)                # SHARED noise across conditions
                pred = action_from_hidden(model, edited[n])      # [T,horizon,7]
                p0 = pred[:, 0, :]
                m = min(p0.shape[0], gt.shape[0])
                seed_mse[n].append(float(np.mean((p0[:m] - gt[:m]) ** 2)))
        orig_m = float(np.mean(seed_mse["original"]))
        for n in names:
            cm = float(np.mean(seed_mse[n]))
            per_clip_mse[n].append(cm)
            per_clip_delta[n].append(cm - orig_m)
    res = {}
    for n in names:
        arr = np.array(per_clip_mse[n]); darr = np.array(per_clip_delta[n])
        res[n] = {"action_mse_mean": float(arr.mean()),
                  "delta_vs_original_mean": float(darr.mean()),
                  "delta_per_clip": darr.tolist()}
    return res


def decoder_recombination(u, e, action, split, shift, seed=0, device="cpu"):
    """Phase 6B (tier-2): train a-hat = g(e, u); swap u across frame pairs and test
    whether the predicted action follows u (the candidate action code) or e (env).

    Tests the TRAINED decoder, NOT the frozen model -> cannot establish Outcome A.
    Returns the fraction of swaps where pred([e_A,u_B]) is closer to a_B than a_A,
    and the action-MSE when reading [e_A,u_B] against a_B.
    """
    import torch.nn as nn
    from . import features as F

    tr, te = split["train"], split["test"]
    def flat(feat):
        X, Y, c = F.flatten_frames(feat, te, shift=shift, action=action)
        return X.numpy(), Y.numpy(), c
    Xu_tr, A_tr, _ = F.flatten_frames(u, tr, shift=shift, action=action)
    Xe_tr, _, _ = F.flatten_frames(e, tr, shift=shift, action=action)
    Ztr = torch.cat([Xe_tr, Xu_tr], 1)
    dev = device if torch.cuda.is_available() else "cpu"
    mu, sd = Ztr.mean(0, keepdim=True), Ztr.std(0, keepdim=True).clamp_min(1e-6)
    g = nn.Sequential(nn.Linear(Ztr.shape[1], 256), nn.GELU(), nn.Linear(256, 7)).to(dev)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3, weight_decay=1e-4)
    Zt = ((Ztr - mu) / sd).to(dev); At = A_tr.to(dev)
    torch.manual_seed(seed)
    for _ in range(300):
        opt.zero_grad(); loss = ((g(Zt) - At) ** 2).mean(); loss.backward(); opt.step()
    # test recombination
    Xu_te, A_te, c_te = flat(u); Xe_te, _, _ = flat(e)
    rng = np.random.default_rng(seed)
    n = len(A_te); perm = rng.permutation(n)
    eA, uA, aA = Xe_te, Xu_te, A_te
    eB, uB, aB = Xe_te[perm], Xu_te[perm], A_te[perm]
    def pred(ef, uf):
        Z = (torch.tensor(np.concatenate([ef, uf], 1), dtype=torch.float32).to(dev) - mu.to(dev)) / sd.to(dev)
        with torch.no_grad():
            return g(Z).cpu().numpy()
    p_AB = pred(eA, uB)          # env A, action-code B -> should follow B
    closer_to_B = (np.linalg.norm(p_AB - aB, axis=1) < np.linalg.norm(p_AB - aA, axis=1)).mean()
    mse_to_B = float(((p_AB - aB) ** 2).mean()); mse_to_A = float(((p_AB - aA) ** 2).mean())
    return {"swap_eA_uB_closer_to_B_frac": float(closer_to_B),
            "swap_eA_uB_mse_to_aB": mse_to_B, "swap_eA_uB_mse_to_aA": mse_to_A,
            "reading": ("if u is the action code, [e_A,u_B] follows action B: closer_to_B>>0.5 and "
                        "mse_to_aB << mse_to_aA. Tests the trained decoder only (T2).")}


def bootstrap_delta_diff(delta_a, delta_b, seed=0, n_boot=2000):
    """Clip-bootstrap CI for mean(delta_a) - mean(delta_b) (paired over clips)."""
    a = np.asarray(delta_a); b = np.asarray(delta_b); n = len(a)
    rng = np.random.default_rng(seed)
    point = float(a.mean() - b.mean())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}
