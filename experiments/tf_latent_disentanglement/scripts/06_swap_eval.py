#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""
Phase 11 — environment / action subspace swapping.

(z) VAE-decode swap: env from A + action from B -> decode to pixels.  We expect
    the swap to look like A's scene (env subspace carries appearance) and measure
    appearance transfer via pixel-MSE to decode(A) vs decode(B).

(h) action-head readout swap: env from A + action from B in DiT hidden -> predict
    action; report whether the predicted action tracks A or B.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP.parents[2]))

from src.utils import visualization as V  # noqa
from src.utils.common import (ensure_dir, load_config, load_pt, resolve_path,  # noqa
                              save_json)
from src.validation.swaps import swap_env_action_subspaces  # noqa


def select_pairs(task_global, n_pairs):
    pairs = []
    idx = list(range(len(task_global)))
    for a, b in itertools.combinations(idx, 2):
        if task_global[a] != task_global[b]:           # different scene/task
            pairs.append((a, b))
        if len(pairs) >= n_pairs:
            break
    if not pairs and len(idx) >= 2:                    # fallback: any pairs
        pairs = list(itertools.combinations(idx, 2))[:n_pairs]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP / "configs/experiment.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_path(cfg.output_dir)
    cache = out / "latent_cache"; sdir = out / "subspaces"
    vdir = ensure_dir(out / "validation"); figdir = ensure_dir(vdir / "figures")

    tok_path = cache / "tokens.pt"
    if not tok_path.exists():
        save_json({"status": "no token cache"}, vdir / "swap_results.json"); return
    tok = load_pt(tok_path)
    task_global = tok["task_global"].tolist()
    n_pairs = int(cfg.validation.n_swap_examples)
    pairs = select_pairs(task_global, n_pairs)

    from src.models.decode import action_from_hidden, decode_vae_latent
    from src.models.load_model import load_frozen_wm4a
    print("[model] loading for swaps ...")
    model, _, _ = load_frozen_wm4a(str(resolve_path(cfg.checkpoint.path)),
                                   device=cfg.device, sigma_data=float(cfg.checkpoint.sigma_data))

    dz = load_pt(sdir / "z_subspace_decomposition.pt")
    dh = load_pt(sdir / "h_subspace_decomposition.pt")
    Hl, Wl = tok["z_grid"]
    frame_t = 0

    # ---- (z) decode swaps ----
    rows, row_labels = [], []
    z_app = {"mse_swap_to_A": [], "mse_swap_to_B": []}
    for (a, b) in pairs:
        za = tok["z_tokens"][a].float()[frame_t:frame_t + 1]   # [1,S,16] (keep frame dim)
        zb = tok["z_tokens"][b].float()[frame_t:frame_t + 1]
        to_grid = lambda z: z.permute(0, 2, 1).reshape(1, 16, Hl, Wl)
        zl_a = to_grid(za); zl_b = to_grid(zb)
        zsw = swap_env_action_subspaces(za, zb, dz["W_env"], dz["W_act"], dz["mean"], dz["std"])
        zl_sw = to_grid(zsw)
        dA = decode_vae_latent(model.backbone, zl_a)[0]
        dB = decode_vae_latent(model.backbone, zl_b)[0]
        dS = decode_vae_latent(model.backbone, zl_sw)[0]
        rows.append([dA, dB, dS]); row_labels.append(f"A:{tok['task_str'][a][:12]} / B:{tok['task_str'][b][:12]}")
        z_app["mse_swap_to_A"].append(float(np.mean((dS.astype(np.float32) - dA) ** 2)))
        z_app["mse_swap_to_B"].append(float(np.mean((dS.astype(np.float32) - dB) ** 2)))
    V.image_grid(rows, figdir / "swap_examples_grid.png", row_labels=row_labels,
                 col_labels=["A (env src)", "B (action src)", "swap envA+actB"],
                 title="env(A) + action(B) swap, VAE-decoded (z)")

    # ---- (h) action-head readout swaps ----
    h_swap = {"dist_swap_to_A": [], "dist_swap_to_B": []}
    for (a, b) in pairs:
        ha = tok["h_tokens"][a].float()                   # [T,Ntok,2048]
        hb = tok["h_tokens"][b].float()
        T = min(ha.shape[0], hb.shape[0])
        hsw = swap_env_action_subspaces(ha[:T], hb[:T], dh["W_env"], dh["W_act"], dh["mean"], dh["std"])
        aA = action_from_hidden(model, ha[:T])[:, 0, :]
        aB = action_from_hidden(model, hb[:T])[:, 0, :]
        aS = action_from_hidden(model, hsw)[:, 0, :]
        h_swap["dist_swap_to_A"].append(float(np.mean((aS - aA) ** 2)))
        h_swap["dist_swap_to_B"].append(float(np.mean((aS - aB) ** 2)))

    res = {
        "n_pairs": len(pairs),
        "z_appearance_transfer": {
            "mse_swap_to_A_mean": float(np.mean(z_app["mse_swap_to_A"])),
            "mse_swap_to_B_mean": float(np.mean(z_app["mse_swap_to_B"])),
            "note": "lower-to-A => swap keeps A's appearance (env subspace carries scene)",
        },
        "h_action_readout_swap": {
            "dist_swap_to_A_mean": float(np.mean(h_swap["dist_swap_to_A"])),
            "dist_swap_to_B_mean": float(np.mean(h_swap["dist_swap_to_B"])),
            "note": "lower-to-B => predicted action follows B's action subspace",
        },
    }
    save_json(res, vdir / "swap_results.json")
    print(f"[z] swap->A mse={res['z_appearance_transfer']['mse_swap_to_A_mean']:.1f} "
          f"swap->B mse={res['z_appearance_transfer']['mse_swap_to_B_mean']:.1f}")
    print(f"[save] -> {vdir / 'swap_results.json'}")


if __name__ == "__main__":
    main()
