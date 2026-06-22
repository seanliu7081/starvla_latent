#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""Phase 5/7 — spectral subspace decomposition; build e/m/u; PCA & random baselines."""
import argparse
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from src.analysis.fft_metrics import low_ratio_of_signal  # noqa
from src.analysis.spectral_subspace import (  # noqa
    pca_basis, random_orthonormal_basis, spectral_subspace_decomposition)
from src.utils.common import ensure_dir, load_config, load_json, load_pt, resolve_path, save_json, save_pt  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP / "configs/experiment.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_path(cfg.output_dir)
    cache = out / "latent_cache"
    sdir = ensure_dir(out / "subspaces")
    meta = load_json(cache / "metadata.json")
    fps = float(meta["fps"]); cutoff = float(cfg.frequency.cutoff_hz)

    tr = load_pt(cache / "train.pt"); va = load_pt(cache / "val.pt")
    dims = {"z": (int(cfg.subspace.env_dim_z), int(cfg.subspace.act_dim_z)),
            "h": (int(cfg.subspace.env_dim_h), int(cfg.subspace.act_dim_h))}

    summary = {"fps": fps, "cutoff_hz": cutoff, "layers": {}}
    for layer in cfg.extraction.layers:
        key = f"{layer}_pooled"
        Z = torch.cat([tr[key], va[key]], dim=0).float()    # [N,T,D]
        env_dim, act_dim = dims[layer]

        dec = spectral_subspace_decomposition(
            Z, fps=fps, cutoff_hz=cutoff, env_dim=env_dim, act_dim=act_dim,
            eps=float(cfg.subspace.eps),
            orthogonalize_action_against_env=bool(cfg.subspace.orthogonalize_action_against_env))

        # orthonormality checks
        We, Wa = dec["W_env"], dec["W_act"]
        assert torch.allclose(We.T @ We, torch.eye(We.shape[1]), atol=1e-3)
        assert torch.allclose(Wa.T @ Wa, torch.eye(Wa.shape[1]), atol=1e-3)
        assert torch.isfinite(We).all() and torch.isfinite(Wa).all()

        # spectral character of env vs action coefficients (success criterion #1)
        e_low = low_ratio_of_signal(dec["e"], fps, cutoff)            # env coeffs
        m_low = low_ratio_of_signal(dec["m"], fps, cutoff)            # action-dir state coeffs
        u_low = low_ratio_of_signal(dec["u"], fps, cutoff)            # action transition coeffs

        # baselines: PCA + random subspaces (same dims)
        W_pca_env = pca_basis(Z, env_dim)
        W_pca_act = pca_basis(Z, act_dim)
        W_rand_env = random_orthonormal_basis(Z.shape[-1], env_dim, seed=cfg.seed)
        W_rand_act = random_orthonormal_basis(Z.shape[-1], act_dim, seed=cfg.seed + 1)

        save_pt({
            "W_env": We, "W_act": Wa, "mean": dec["mean"], "std": dec["std"],
            "e": dec["e"], "m": dec["m"], "u": dec["u"],
            "env_scores": dec["env_scores"], "act_scores": dec["act_scores"],
            "W_pca_env": W_pca_env, "W_pca_act": W_pca_act,
            "W_rand_env": W_rand_env, "W_rand_act": W_rand_act,
            "env_dim": env_dim, "act_dim": act_dim, "D": Z.shape[-1],
            "n_train": tr[key].shape[0], "n_val": va[key].shape[0],
        }, sdir / f"{layer}_subspace_decomposition.pt")

        summary["layers"][layer] = {
            "D": int(Z.shape[-1]), "env_dim": env_dim, "act_dim": act_dim,
            "env_low_ratio": e_low, "action_state_low_ratio": m_low, "action_delta_low_ratio": u_low,
            "env_minus_action_low_ratio": e_low - u_low,   # >0 => env slower than action (expected)
            "env_scores_top5": dec["env_scores"][:5].tolist(),
            "act_scores_top5": dec["act_scores"][:5].tolist(),
        }
        print(f"[{layer}] env_low={e_low:.3f} action(u)_low={u_low:.3f} "
              f"(env should be > action): {'OK' if e_low > u_low else 'NO'}")

    save_json(summary, sdir / "subspace_summary.json")
    print(f"[save] -> {sdir}")


if __name__ == "__main__":
    main()
