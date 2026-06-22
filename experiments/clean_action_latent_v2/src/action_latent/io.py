# Copyright 2025. Licensed under the MIT License.
"""I/O, splits, normalization for the clean-action-latent v2 experiment.

The prior experiment cached latents as ``latent_cache/{train,val}.pt`` (120 + 40
clips). The canonical 160-clip ordering used by the frozen env subspace
(``subspaces/h_subspace_decomposition.pt``) is the concatenation ``[train; val]``
— this is *verified* at load time by reconstructing the cached env coeffs ``e``
from ``W_env``/``mean``/``std`` (Prime Directive 8: fail loud on any mismatch).

All paths in the config that are not absolute are resolved relative to the
starVLA repo root inferred from this file's location.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

# .../clean_action_latent_v2/src/action_latent/io.py
#   parents[0]=action_latent [1]=src [2]=clean_action_latent_v2 [3]=experiments [4]=<repo root>
_HERE = Path(__file__).resolve()
EXP_ROOT = _HERE.parents[2]
REPO_ROOT = _HERE.parents[4]


# --------------------------------------------------------------------------- #
# config / path helpers
# --------------------------------------------------------------------------- #
class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v


def load_config(path: str | Path) -> Cfg:
    with open(path) as f:
        return Cfg(yaml.safe_load(f))


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(_to_native(obj), f, indent=2)
    return path


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def _to_native(o):
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return o


# --------------------------------------------------------------------------- #
# cache loading  (concat [train; val] -> canonical 160-clip order)
# --------------------------------------------------------------------------- #
_LABEL_KEYS = ["suite_id", "task_global", "task_local", "episode_uid", "clip_idx"]


def load_global_cache(cfg) -> dict:
    """Return the full 160-clip cache in canonical [train; val] order.

    keys: h_pooled[N,T,2048], z_pooled[N,T,16], actions[N,T,7], states[N,T,8],
          suite_id/task_global/task_local/episode_uid/clip_idx [N], task_str(list),
          n_train, n_val, T
    """
    cdir = resolve_path(cfg.inputs.latent_cache_dir)
    tr = torch.load(cdir / "train.pt", map_location="cpu", weights_only=False)
    va = torch.load(cdir / "val.pt", map_location="cpu", weights_only=False)
    out = {}
    for k in ["z_pooled", "h_pooled", "actions", "states"]:
        out[k] = torch.cat([tr[k], va[k]], 0).float()
    for k in _LABEL_KEYS:
        out[k] = torch.cat([tr[k], va[k]], 0).long()
    out["task_str"] = list(tr["task_str"]) + list(va["task_str"])
    out["n_train"] = int(tr["h_pooled"].shape[0])
    out["n_val"] = int(va["h_pooled"].shape[0])
    out["T"] = int(tr["h_pooled"].shape[1])
    return out


def load_env_subspace(cfg, verify_against_h: torch.Tensor | None = None) -> dict:
    """Load the inherited frozen env subspace + negative-control high-freq subspace.

    If ``verify_against_h`` (the canonical-order h_pooled) is given, RECONSTRUCT
    the cached env coeffs from W_env/mean/std and assert an exact match — this
    pins the row alignment between the cache and the subspace artifacts.
    """
    es = cfg.env_subspace
    sdir = resolve_path(cfg.inputs.subspace_dir)
    dec = torch.load(sdir / es.file, map_location="cpu", weights_only=False)
    sub = {
        "W_env": dec[es.W_env_key].float(),          # [2048, 128]
        "mean": dec[es.mean_key].float(),            # [1,1,2048]
        "std": dec[es.std_key].float(),
        "e": dec[es.e_key].float(),                  # [160,48,128]
        "W_act_hf": dec[es.highfreq_W_act_key].float(),  # [2048,16] negative control
        "u_hf": dec[es.highfreq_u_key].float(),      # [160,47,16]
        "env_dim": int(dec[es.W_env_key].shape[1]),
    }
    if verify_against_h is not None:
        hn = (verify_against_h - sub["mean"]) / sub["std"]
        e_recon = hn @ sub["W_env"]
        err = (e_recon - sub["e"]).abs().max().item()
        if err > 1e-3:
            raise RuntimeError(
                f"env-coeff reconstruction mismatch (max abs err={err:.4g}); "
                f"the cache [train;val] order does NOT match subspace 'e'. HALT.")
        sub["e_reconstruct_max_abs_err"] = err
    return sub


def load_token_cache(cfg) -> dict:
    return torch.load(resolve_path(cfg.inputs.token_cache),
                      map_location="cpu", weights_only=False, mmap=True)


# --------------------------------------------------------------------------- #
# clip-grouped split  (Prime Directive 6)
# --------------------------------------------------------------------------- #
def make_clip_split(cache, cfg) -> dict:
    """Stratified-by-suite, clip-grouped 70/15/15 split over the 160 clips.

    Returns dict with int index arrays (into canonical order):
      {train, val, test} plus 'group_key', 'stratify_key', and an assertion record.
    Each clip is its own group (episode_uid unique), so a row-level split cannot
    let a clip's frames cross splits; we still assert it explicitly.
    """
    n = cache["h_pooled"].shape[0]
    rng = np.random.default_rng(int(cfg.seed))
    group = cache[cfg.splits.group_key].numpy()
    strat = cache[cfg.splits.stratify_key].numpy()

    r_tr, r_va = float(cfg.splits.train), float(cfg.splits.val)
    train, val, test = [], [], []
    for s in np.unique(strat):
        idx = np.where(strat == s)[0]
        rng.shuffle(idx)
        n_s = len(idx)
        n_tr = int(round(r_tr * n_s))
        n_va = int(round(r_va * n_s))
        train += idx[:n_tr].tolist()
        val += idx[n_tr:n_tr + n_va].tolist()
        test += idx[n_tr + n_va:].tolist()
    train, val, test = sorted(train), sorted(val), sorted(test)

    # ---- assert no group crosses splits (Prime Directive 6) ----
    g_tr, g_va, g_te = set(group[train]), set(group[val]), set(group[test])
    assert not (g_tr & g_va) and not (g_tr & g_te) and not (g_va & g_te), \
        "GROUP LEAK: a clip/episode appears in more than one split"
    assert len(train) + len(val) + len(test) == n
    assert len(set(train) | set(val) | set(test)) == n

    return {
        "train": np.array(train, dtype=int),
        "val": np.array(val, dtype=int),
        "test": np.array(test, dtype=int),
        "group_key": cfg.splits.group_key,
        "stratify_key": cfg.splits.stratify_key,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "no_group_leak_verified": True,
    }


def train_norm_stats(feat: torch.Tensor, train_idx) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dim mean/std over TRAIN frames only. feat:[N,T,D] -> ([1,1,D],[1,1,D])."""
    x = feat[train_idx].reshape(-1, feat.shape[-1])
    mu = x.mean(0, keepdim=True)
    sd = x.std(0, keepdim=True).clamp_min(1e-6)
    return mu.view(1, 1, -1), sd.view(1, 1, -1)
