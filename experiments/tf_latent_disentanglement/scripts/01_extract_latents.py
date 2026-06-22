#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""
Phase 1 — extract latent trajectories from the frozen WM4A checkpoint.

Samples contiguous frame clips across the four LIBERO suites, runs the frozen
Cosmos-Predict2 backbone per frame, and caches:

  - pooled trajectories  z_pooled [N,T,16], h_pooled [N,T,2048]
  - spatial-token tensors for a subset of clips (z_tokens, h_tokens)
  - per-frame actions [N,T,7] / states [N,T,8]
  - per-clip scene labels (suite_id, task_global, task_local, episode_uid)

Usage:
    python scripts/01_extract_latents.py --config configs/experiment.yaml
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP.parents[2]))  # starVLA repo

from src.data.libero_lerobot import LiberoLeRobotReader  # noqa: E402
from src.models.extract import extract_clip_latents, pool_h, pool_z, tokens_z  # noqa: E402
from src.models.load_model import load_frozen_wm4a  # noqa: E402
from src.utils.common import (EXP_ROOT, ensure_dir, load_config, resolve_path,  # noqa: E402
                              save_json, save_pt, set_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP / "configs/experiment.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    out_dir = ensure_dir(resolve_path(cfg.output_dir) / "latent_cache")
    dcfg, ecfg = cfg.data, cfg.extraction

    # ---- data index ----
    reader = LiberoLeRobotReader(resolve_path(dcfg.data_root), list(dcfg.suites))
    print("[data]", reader.summary())
    fps = reader.fps
    specs = reader.build_clip_index(
        seq_len=dcfg.seq_len, stride=dcfg.stride, n_clips=dcfg.n_clips,
        clips_per_episode=dcfg.clips_per_episode, seed=cfg.seed)
    print(f"[data] sampled {len(specs)} clips, seq_len={dcfg.seq_len}, fps={fps}")

    # ---- model ----
    print("[model] loading frozen WM4A checkpoint ...")
    model, mcfg, norm_stats = load_frozen_wm4a(
        str(resolve_path(cfg.checkpoint.path)), device=cfg.device,
        sigma_data=float(cfg.checkpoint.sigma_data))

    # token-clip subset (first N after shuffle -> already random)
    n_tok = int(ecfg.n_token_clips) if ecfg.store_tokens else 0
    tok_clip_set = set(range(min(n_tok, len(specs))))

    # ---- extract ----
    z_pooled, h_pooled = [], []
    actions, states = [], []
    suite_id, task_global, task_local, episode_uid, task_str = [], [], [], [], []
    z_tokens, h_tokens, tok_idx = [], [], []
    z_grid = h_grid = None

    t0 = time.time()
    for ci, spec in enumerate(tqdm(specs, desc="clips")):
        clip = reader.load_clip(spec, load_wrist=False)
        feat = extract_clip_latents(model, clip.images, spec.task_str,
                                    frame_batch_size=int(ecfg.frame_batch_size))
        z_grid, h_grid = feat["z_grid"], feat["h_grid"]
        z_pooled.append(pool_z(feat["z_latent"]))      # [T,16]
        h_pooled.append(pool_h(feat["h_hidden"]))      # [T,2048]
        actions.append(torch.from_numpy(clip.actions)) # [T,7]
        states.append(torch.from_numpy(clip.states))   # [T,8]
        suite_id.append(spec.suite_id); task_global.append(spec.task_global)
        task_local.append(spec.task_index); episode_uid.append(spec.episode_uid)
        task_str.append(spec.task_str)
        if ci in tok_clip_set:
            z_tokens.append(tokens_z(feat["z_latent"]).half())   # [T,Hl*Wl,16]
            h_tokens.append(feat["h_hidden"].half())             # [T,N_tok,2048]
            tok_idx.append(ci)

    z_pooled = torch.stack(z_pooled)   # [N,T,16]
    h_pooled = torch.stack(h_pooled)   # [N,T,2048]
    actions = torch.stack(actions).float()
    states = torch.stack(states).float()
    meta_arr = lambda x: torch.tensor(x)
    N = z_pooled.shape[0]
    print(f"[extract] done {N} clips in {time.time()-t0:.1f}s | "
          f"z {tuple(z_pooled.shape)} h {tuple(h_pooled.shape)} | z_grid {z_grid} h_grid {h_grid}")
    assert torch.isfinite(z_pooled).all() and torch.isfinite(h_pooled).all()

    # ---- train/val split (by clip) ----
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(N)
    n_val = max(1, int(round(N * dcfg.val_ratio)))
    val_idx = np.sort(perm[:n_val]); train_idx = np.sort(perm[n_val:])

    def pack(idx):
        idx_t = torch.from_numpy(idx)
        return {
            "z_pooled": z_pooled[idx_t], "h_pooled": h_pooled[idx_t],
            "actions": actions[idx_t], "states": states[idx_t],
            "suite_id": meta_arr([suite_id[i] for i in idx]),
            "task_global": meta_arr([task_global[i] for i in idx]),
            "task_local": meta_arr([task_local[i] for i in idx]),
            "episode_uid": meta_arr([episode_uid[i] for i in idx]),
            "task_str": [task_str[i] for i in idx],
            "clip_idx": idx_t,
        }

    save_pt(pack(train_idx), out_dir / "train.pt")
    save_pt(pack(val_idx), out_dir / "val.pt")

    # ---- token subset (kept whole; tagged with split membership) ----
    if z_tokens:
        tok_idx_t = torch.tensor(tok_idx)
        train_set = set(train_idx.tolist())
        save_pt({
            "z_tokens": torch.stack(z_tokens), "h_tokens": torch.stack(h_tokens),
            "clip_idx": tok_idx_t,
            "is_train": torch.tensor([int(i in train_set) for i in tok_idx]),
            "suite_id": meta_arr([suite_id[i] for i in tok_idx]),
            "task_global": meta_arr([task_global[i] for i in tok_idx]),
            "task_str": [task_str[i] for i in tok_idx],
            "z_grid": list(z_grid), "h_grid": list(h_grid),
        }, out_dir / "tokens.pt")

    save_json({
        "fps": fps, "seq_len": dcfg.seq_len, "stride": dcfg.stride,
        "n_clips": N, "n_train": len(train_idx), "n_val": len(val_idx),
        "layers": {"z": {"D": z_pooled.shape[-1], "grid": list(z_grid),
                         "desc": "VAE latent (spatial mean-pool); decodable"},
                   "h": {"D": h_pooled.shape[-1], "grid": list(h_grid),
                         "desc": "DiT last hidden (token mean-pool); predictive"}},
        "shapes": {"z_pooled": list(z_pooled.shape), "h_pooled": list(h_pooled.shape),
                   "actions": list(actions.shape), "states": list(states.shape)},
        "n_token_clips": len(tok_idx), "suites": list(dcfg.suites),
        "sigma_data": float(cfg.checkpoint.sigma_data),
        "data_summary": reader.summary(),
    }, out_dir / "metadata.json")
    print(f"[save] -> {out_dir}")


if __name__ == "__main__":
    main()
