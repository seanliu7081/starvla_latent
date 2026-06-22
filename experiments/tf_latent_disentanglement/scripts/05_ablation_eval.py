#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""
Phase 9/10 — ablation + prediction-drop validation.

(A) Learned-dynamics prediction-drop (both layers, data-only): fit a ridge map
    current-latent -> Δlatent under each ablation condition; removing the action
    subspace should hurt transition prediction more than random/PCA removal.

(B) Action-head readout drop (h, model-based): feed env/act/random-ablated DiT
    token-hidden into the FROZEN GR00T head; removing the action subspace should
    raise action-prediction error more than removing env/random.

(C) VAE-decode ablation grids (z, model-based): decode env-only / act-only /
    no-env / no-act edited VAE latents to pixels for qualitative inspection.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP.parents[2]))

from src.analysis.spectral_subspace import random_orthonormal_basis  # noqa
from src.utils import visualization as V  # noqa
from src.utils.common import (ensure_dir, load_config, load_json, load_pt,  # noqa
                              resolve_path, save_json)
from src.validation.ablations import project_to_subspace, remove_subspace  # noqa
from src.validation.prediction_drop import dynamics_drop  # noqa


def make_conditions(Z, dec, seed):
    """Return dict[name] -> ablated Z (same shape [.,.,D]).  Z is the *normalized*-
    space-agnostic raw pooled latent; ablation uses dec mean/std internally."""
    mean, std = dec["mean"], dec["std"]
    We, Wa = dec["W_env"], dec["W_act"]
    D = Z.shape[-1]
    Wre = random_orthonormal_basis(D, We.shape[1], seed=seed)
    Wra = random_orthonormal_basis(D, Wa.shape[1], seed=seed + 1)
    Wpe, Wpa = dec["W_pca_env"], dec["W_pca_act"]
    return {
        "original": Z,
        "env_only": project_to_subspace(Z, We, mean, std)[0],
        "act_only": project_to_subspace(Z, Wa, mean, std)[0],
        "no_env": remove_subspace(Z, We, mean, std),
        "no_act": remove_subspace(Z, Wa, mean, std),
        "no_rand_env": remove_subspace(Z, Wre, mean, std),
        "no_rand_act": remove_subspace(Z, Wra, mean, std),
        "no_pca_act": remove_subspace(Z, Wpa, mean, std),
    }


def part_A_dynamics(cfg, cache, sdir, out):
    tr = load_pt(cache / "train.pt"); va = load_pt(cache / "val.pt")
    res = {}
    for layer in cfg.extraction.layers:
        dec = load_pt(sdir / f"{layer}_subspace_decomposition.pt")
        Ztr = tr[f"{layer}_pooled"].float(); Zva = va[f"{layer}_pooled"].float()
        cond_tr = make_conditions(Ztr, dec, cfg.seed)
        cond_va = make_conditions(Zva, dec, cfg.seed)
        layer_res = {}
        for name in cond_tr:
            layer_res[name] = dynamics_drop(cond_tr[name], Ztr, cond_va[name], Zva, ridge=1.0)
        base = layer_res["original"]["val_mse"]
        for name in layer_res:
            layer_res[name]["mse_increase_vs_original"] = layer_res[name]["val_mse"] - base
        res[layer] = layer_res
        print(f"[A:{layer}] dynamics val_mse  orig={base:.4f}  "
              f"no_act={layer_res['no_act']['val_mse']:.4f}  "
              f"no_env={layer_res['no_env']['val_mse']:.4f}  "
              f"no_rand_act={layer_res['no_rand_act']['val_mse']:.4f}")
    return res


def part_B_readout(cfg, cache, sdir, model):
    tok_path = cache / "tokens.pt"
    if not tok_path.exists():
        return {}
    from src.models.decode import action_from_hidden
    tok = load_pt(tok_path)
    dec = load_pt(sdir / "h_subspace_decomposition.pt")
    mean, std = dec["mean"], dec["std"]; We, Wa = dec["W_env"], dec["W_act"]
    D = We.shape[0]
    Wre = random_orthonormal_basis(D, We.shape[1], seed=cfg.seed)
    Wra = random_orthonormal_basis(D, Wa.shape[1], seed=cfg.seed + 1)

    # global GT actions by original clip index
    tr = load_pt(cache / "train.pt"); va = load_pt(cache / "val.pt")
    N = tr["actions"].shape[0] + va["actions"].shape[0]
    T = tr["actions"].shape[1]
    g_actions = torch.zeros(N, T, tr["actions"].shape[-1])
    g_actions[tr["clip_idx"]] = tr["actions"]; g_actions[va["clip_idx"]] = va["actions"]
    horizon = int(model.action_horizon)

    conds = {"original": None, "no_env": We, "no_act": Wa, "no_rand_env": Wre, "no_rand_act": Wra}
    per_clip = {k: [] for k in conds}
    clip_ids = tok["clip_idx"].tolist()
    for j, ci in enumerate(clip_ids):
        h = tok["h_tokens"][j].float()           # [T,Ntok,2048]
        gt = g_actions[ci]                        # [T,7]  (action at frame t)
        for name, W in conds.items():
            h_in = h if W is None else remove_subspace(h, W, mean, std)
            pred = action_from_hidden(model, h_in)            # [T,horizon,7]
            pred_t0 = pred[:, 0, :]                            # first step of each chunk
            # compare to GT action at the same frame (chunk start)
            m = min(pred_t0.shape[0], gt.shape[0])
            mse = float(np.mean((pred_t0[:m] - gt[:m].numpy()) ** 2))
            per_clip[name].append(mse)
    res = {name: {"action_mse_vs_gt_mean": float(np.mean(v))} for name, v in per_clip.items()}
    base = res["original"]["action_mse_vs_gt_mean"]
    for name in res:
        res[name]["mse_increase_vs_original"] = res[name]["action_mse_vs_gt_mean"] - base
    print(f"[B:h] action-head readout MSE  orig={base:.4f}  no_act={res['no_act']['action_mse_vs_gt_mean']:.4f}  "
          f"no_env={res['no_env']['action_mse_vs_gt_mean']:.4f}  no_rand_act={res['no_rand_act']['action_mse_vs_gt_mean']:.4f}")
    return res


def part_C_decode_grids(cfg, cache, sdir, model, figdir):
    tok_path = cache / "tokens.pt"
    if not tok_path.exists():
        return {}
    from src.models.decode import decode_vae_latent
    tok = load_pt(tok_path)
    dec = load_pt(sdir / "z_subspace_decomposition.pt")
    mean, std = dec["mean"], dec["std"]; We, Wa = dec["W_env"], dec["W_act"]
    Hl, Wl = tok["z_grid"]
    n_show = min(3, tok["z_tokens"].shape[0])
    pixel_mse = {k: [] for k in ["env_only", "act_only", "no_env", "no_act"]}
    rows, row_labels = [], []
    frame_t = 0
    for j in range(n_show):
        zt = tok["z_tokens"][j].float()                   # [T,S,16]
        T = zt.shape[0]
        zl = zt.permute(0, 2, 1).reshape(T, 16, Hl, Wl)   # [T,16,Hl,Wl]
        variants = {
            "original": zl,
            "env_only": project_to_subspace(zt, We, mean, std)[0].permute(0, 2, 1).reshape(T, 16, Hl, Wl),
            "act_only": project_to_subspace(zt, Wa, mean, std)[0].permute(0, 2, 1).reshape(T, 16, Hl, Wl),
            "no_env": remove_subspace(zt, We, mean, std).permute(0, 2, 1).reshape(T, 16, Hl, Wl),
            "no_act": remove_subspace(zt, Wa, mean, std).permute(0, 2, 1).reshape(T, 16, Hl, Wl),
        }
        decoded = {k: decode_vae_latent(model.backbone, v[frame_t:frame_t + 1])[0] for k, v in variants.items()}
        rows.append([decoded[k] for k in ["original", "env_only", "act_only", "no_env", "no_act"]])
        row_labels.append(tok["task_str"][j][:18])
        for k in pixel_mse:
            pixel_mse[k].append(float(np.mean((decoded[k].astype(np.float32) - decoded["original"].astype(np.float32)) ** 2)))
    V.image_grid(rows, figdir / "ablation_decode_grid.png", row_labels=row_labels,
                 col_labels=["original", "env_only", "act_only", "no_env", "no_act"],
                 title="VAE-decoded subspace ablations (z)")
    return {k: float(np.mean(v)) for k, v in pixel_mse.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP / "configs/experiment.yaml"))
    ap.add_argument("--no_model", action="store_true", help="run only data-only part A")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_path(cfg.output_dir)
    cache = out / "latent_cache"; sdir = out / "subspaces"
    vdir = ensure_dir(out / "validation"); figdir = ensure_dir(vdir / "figures")

    results = {"part_A_learned_dynamics_drop": part_A_dynamics(cfg, cache, sdir, out)}

    if not args.no_model:
        from src.models.load_model import load_frozen_wm4a
        print("[model] loading for readout-drop + decode grids ...")
        model, _, _ = load_frozen_wm4a(str(resolve_path(cfg.checkpoint.path)),
                                       device=cfg.device, sigma_data=float(cfg.checkpoint.sigma_data))
        results["part_B_action_head_readout_drop"] = part_B_readout(cfg, cache, sdir, model)
        results["part_C_decode_pixel_mse"] = part_C_decode_grids(cfg, cache, sdir, model, figdir)

    save_json(results, vdir / "ablation_results.json")
    print(f"[save] -> {vdir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
