# Copyright 2025. Licensed under the MIT License.
"""Metrics: R^2 (aggregate + per-dim), MSE, and CLIP-level bootstrap CIs.

Small-N discipline (Prime Directive 4 / §7): every test-set metric here can be
bootstrapped by RESAMPLING TEST CLIPS (never individual frames), because frames
within a clip are highly dependent and frame-level bootstrap would understate
the CI.
"""
from __future__ import annotations

import numpy as np


def _agg_r2(y, pred):
    """Variance-weighted (aggregate) R^2 over all dims, 1 - SS_res/SS_tot."""
    y = np.asarray(y, np.float64)
    pred = np.asarray(pred, np.float64)
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean(0, keepdims=True)) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)


def _perdim_r2(y, pred):
    y = np.asarray(y, np.float64)
    pred = np.asarray(pred, np.float64)
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0, keepdims=True)) ** 2).sum(0) + 1e-12
    return (1.0 - ss_res / ss_tot).tolist()


def r2_report(y, pred):
    pd = _perdim_r2(y, pred)
    return {"r2": _agg_r2(y, pred),            # variance-weighted (raw-scale pooling)
            "r2_equal": float(np.mean(pd)),     # equal-weight (mean of per-dim) — per M1 verification caveat
            "r2_per_dim": pd,
            "mse": float(((np.asarray(y) - np.asarray(pred)) ** 2).mean())}


def bootstrap_r2_ci(y, pred, clip_ids, n_boot=1000, seed=0, alpha=0.05):
    """Clip-level bootstrap CI for aggregate R^2.

    y,pred : [M, K] arrays of per-frame targets/predictions on the test set.
    clip_ids : [M] integer clip id per frame (frames of one clip share an id).
    Resamples the set of UNIQUE clips with replacement, recomputes R^2.
    Returns dict(point, lo, hi, std, n_clips).
    """
    y = np.asarray(y, np.float64)
    pred = np.asarray(pred, np.float64)
    clip_ids = np.asarray(clip_ids)
    uniq = np.unique(clip_ids)
    # precompute per-clip frame index lists
    idx_by_clip = {c: np.where(clip_ids == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    point = _agg_r2(y, pred)
    boots = np.empty(n_boot, np.float64)
    nU = len(uniq)
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=nU, replace=True)
        rows = np.concatenate([idx_by_clip[c] for c in chosen])
        boots[b] = _agg_r2(y[rows], pred[rows])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "std": float(boots.std()), "n_clips": int(nU)}


def bootstrap_gain_ci(y, pred_u, pred_e, clip_ids, n_boot=1000, seed=0, alpha=0.05):
    """Clip-level bootstrap CI for the GAIN = R2(pred_u) - R2(pred_e).

    The headline action metric is always gain over the env-only shortcut
    (Prime Directive 3); its CI must be computed on the SAME resampled clips so
    the two R^2's are paired.
    """
    y = np.asarray(y, np.float64)
    pu = np.asarray(pred_u, np.float64)
    pe = np.asarray(pred_e, np.float64)
    clip_ids = np.asarray(clip_ids)
    uniq = np.unique(clip_ids)
    idx_by_clip = {c: np.where(clip_ids == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    point = _agg_r2(y, pu) - _agg_r2(y, pe)
    boots = np.empty(n_boot, np.float64)
    nU = len(uniq)
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=nU, replace=True)
        rows = np.concatenate([idx_by_clip[c] for c in chosen])
        boots[b] = _agg_r2(y[rows], pu[rows]) - _agg_r2(y[rows], pe[rows])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "std": float(boots.std()), "n_clips": int(nU),
            "excludes_zero": bool(lo > 0 or hi < 0)}
