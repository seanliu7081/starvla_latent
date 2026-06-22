# Copyright 2025. Licensed under the MIT License.
"""Phase 5 — pair / retrieval deconfounding.

Because task/object/action are correlated in LIBERO, these isolate action from
environment:
  - partial correlation corr(d_u, d_action | same_env)  <- key deconfounded stat
  - retrieval: NN in u-space; action-bin recall@k, env diversity among neighbors
  - pair tests via the raw correlations (d_u vs same_env, d_u vs d_action)
All with clip-aware subsampling and bootstrap CIs (small N). [T4-retr]
"""
from __future__ import annotations

import numpy as np


def _pdist(X):
    G = X @ X.T
    sq = np.diag(G)
    d2 = sq[:, None] + sq[None, :] - 2 * G
    return np.sqrt(np.clip(d2, 0, None))


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.sqrt((a @ a) * (b @ b)) + 1e-12))


def partial_corr(d_u, d_a, same_env):
    """partial corr(d_u, d_a | same_env)."""
    r_ua = _corr(d_u, d_a); r_ue = _corr(d_u, same_env); r_ae = _corr(d_a, same_env)
    denom = np.sqrt((1 - r_ue ** 2) * (1 - r_ae ** 2)) + 1e-12
    return (r_ua - r_ue * r_ae) / denom, {"r_du_da": r_ua, "r_du_env": r_ue, "r_da_env": r_ae}


def pair_stats(u, a, env, clip_ids, *, seed=0, n_sub=600, n_boot=500):
    """Subsample frames, compute cross-clip all-pairs d_u/d_a/same_env, partial corr
    + raw corrs, with a frame-bootstrap CI on the partial correlation."""
    rng = np.random.default_rng(seed)
    n = u.shape[0]
    sub = rng.choice(n, size=min(n_sub, n), replace=False)
    U, A, E, C = u[sub], a[sub], env[sub], clip_ids[sub]
    iu = np.triu_indices(len(sub), k=1)
    cross = C[iu[0]] != C[iu[1]]
    du = _pdist(U)[iu][cross]; da = _pdist(A)[iu][cross]
    se = (E[iu[0]] == E[iu[1]]).astype(float)[cross]
    pc, raw = partial_corr(du, da, se)
    boots = []
    for _ in range(n_boot):
        bs = rng.choice(len(sub), size=len(sub), replace=True)
        Ub, Ab, Eb, Cb = U[bs], A[bs], E[bs], C[bs]
        jj = np.triu_indices(len(bs), k=1)
        cr = Cb[jj[0]] != Cb[jj[1]]
        if cr.sum() < 50:
            continue
        dub = _pdist(Ub)[jj][cr]; dab = _pdist(Ab)[jj][cr]
        seb = (Eb[jj[0]] == Eb[jj[1]]).astype(float)[cr]
        boots.append(partial_corr(dub, dab, seb)[0])
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan))
    return {"partial_corr_du_da_given_env": pc, "raw_corrs": raw,
            "partial_corr_ci": [float(lo), float(hi)], "n_pairs": int(cross.sum())}


def retrieval(u, action_bin, env, clip_ids, *, k=5, seed=0):
    """For each query frame, k nearest cross-clip neighbors in u-space. Report
    action-bin recall@k and neighbor env-diversity vs a random-neighbor baseline."""
    n = u.shape[0]
    D = _pdist(u)
    same_clip = clip_ids[:, None] == clip_ids[None, :]
    D[same_clip] = np.inf
    nn = np.argsort(D, axis=1)[:, :k]
    abin_match = (action_bin[nn] == action_bin[:, None]).mean()
    env_diff = (env[nn] != env[:, None]).mean()
    rng = np.random.default_rng(seed)
    rnd = rng.integers(0, n, size=(n, k))
    abin_rand = (action_bin[rnd] == action_bin[:, None]).mean()
    env_diff_rand = (env[rnd] != env[:, None]).mean()
    return {"action_bin_recall_at_k": float(abin_match), "action_bin_recall_random": float(abin_rand),
            "neighbor_env_diversity": float(env_diff), "neighbor_env_diversity_random": float(env_diff_rand),
            "k": int(k)}
