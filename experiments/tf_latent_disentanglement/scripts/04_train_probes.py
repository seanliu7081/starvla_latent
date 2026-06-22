#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""
Phase 8 — probe validation.

Tests whether the environment coefficients ``e`` and action/transition
coefficients ``u`` encode the EXPECTED information:

  scene/task/object  -> e should beat u   (environment is scene-predictive)
  robot action       -> u should beat e   (action is transition-predictive)

Methodology notes:
  * All probes use FRAME-LEVEL features with CLIP-GROUPED train/val splits so the
    (few-hundred-clip) sample size becomes thousands of frames while held-out
    clips never leak.  Scene/task labels are clip-constant.
  * The action probe is reported both at full dim and CAPACITY-MATCHED (e
    restricted to act_dim) because e naturally has more dimensions than u.
"""
import argparse
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from src.utils.common import (ensure_dir, load_config, load_pt, resolve_path,  # noqa
                              save_json)
from src.validation.metrics import between_within_ratio  # noqa
from src.validation.probes import (train_classification_probe,  # noqa
                                    train_regression_probe)


def frame_flat(x):
    """[N,T,K] -> ([N*T,K], group_ids[N*T])."""
    N, T, K = x.shape
    g = torch.arange(N).repeat_interleave(T)
    return x.reshape(N * T, K), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP / "configs/experiment.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_path(cfg.output_dir)
    cache = out / "latent_cache"; sdir = out / "subspaces"
    vdir = ensure_dir(out / "validation")
    vcfg = cfg.validation
    pk = dict(train_ratio=float(vcfg.probe_train_ratio), epochs=int(vcfg.probe_epochs),
              lr=float(vcfg.probe_lr), weight_decay=float(vcfg.probe_weight_decay), seed=cfg.seed)
    # subspace coefficients (e,u) live in an arbitrary orthonormal basis -> use a
    # rotation-invariant ('global') normalization so probe accuracy does not depend
    # on the basis.  Full-latent features (natural basis) use per-dim z-scoring.
    pk_sub = {**pk, "whiten": "global"}
    pk_full = {**pk, "whiten": "per_dim"}

    tr = load_pt(cache / "train.pt"); va = load_pt(cache / "val.pt")
    suite = torch.cat([tr["suite_id"], va["suite_id"]]).long()       # [N]
    task = torch.cat([tr["task_global"], va["task_global"]]).long()
    task_local = torch.cat([tr["task_local"], va["task_local"]]).long()
    actions = torch.cat([tr["actions"], va["actions"]]).float()      # [N,T,7]

    results = {"layers": {}, "notes": "frame-level features, clip-grouped splits"}
    for layer in cfg.extraction.layers:
        dec = load_pt(sdir / f"{layer}_subspace_decomposition.pt")
        e = dec["e"].float()           # [N,T,env_dim]
        u = dec["u"].float()           # [N,T-1,act_dim]
        N, T, env_dim = e.shape
        act_dim = u.shape[-1]

        # frame-level features (+ clip groups) ; for scene/task replicate clip label per frame
        e_f, e_g = frame_flat(e)                       # [N*T, env_dim]
        u_f, u_g = frame_flat(u)                       # [N*(T-1), act_dim]
        # action target aligned with transitions
        act_f = actions[:, :-1].reshape(-1, actions.shape[-1])        # [N*(T-1),7]
        e_align = e[:, :-1].reshape(-1, env_dim)                      # align e to transitions
        e_align_match = e_align[:, :act_dim]                          # capacity-matched
        suite_f = suite.repeat_interleave(T)
        task_f = task.repeat_interleave(T)

        lr_ = {}
        # scene (4-way) / task (40-way): expect e >> u
        lr_["scene_suite_from_e"] = train_classification_probe(e_f, suite_f, groups=e_g, **pk_sub)
        lr_["scene_suite_from_u"] = train_classification_probe(u_f, suite.repeat_interleave(T - 1), groups=u_g, **pk_sub)
        lr_["task_from_e"] = train_classification_probe(e_f, task_f, groups=e_g, **pk_sub)
        lr_["task_from_u"] = train_classification_probe(u_f, task.repeat_interleave(T - 1), groups=u_g, **pk_sub)

        # object identity (libero_object suite)
        obj = suite == 0
        if int(obj.sum()) >= 8 and len(torch.unique(task_local[obj])) > 1:
            ef_o, eg_o = frame_flat(e[obj]); uf_o, ug_o = frame_flat(u[obj])
            tl_e = task_local[obj].repeat_interleave(T); tl_u = task_local[obj].repeat_interleave(T - 1)
            lr_["object_from_e"] = train_classification_probe(ef_o, tl_e, groups=eg_o, **pk_sub)
            lr_["object_from_u"] = train_classification_probe(uf_o, tl_u, groups=ug_o, **pk_sub)

        # action regression: expect u >> e ; report capacity-matched e too
        lr_["action_from_u"] = train_regression_probe(u_f, act_f, groups=u_g, **pk_sub)
        lr_["action_from_e_full"] = train_regression_probe(e_align, act_f, groups=u_g, **pk_sub)
        lr_["action_from_e_matched"] = train_regression_probe(e_align_match, act_f, groups=u_g, **pk_sub)
        # diagnostics: is the action linearly present in the FULL latent at all?
        Z = torch.cat([tr[f"{layer}_pooled"], va[f"{layer}_pooled"]]).float()   # [N,T,D]
        dZ_f = (Z[:, 1:] - Z[:, :-1]).reshape(-1, Z.shape[-1])                   # full transition
        Z_f = Z[:, :-1].reshape(-1, Z.shape[-1])                                # full state
        lr_["action_from_full_delta"] = train_regression_probe(dZ_f, act_f, groups=u_g, **pk_full)
        lr_["action_from_full_state"] = train_regression_probe(Z_f, act_f, groups=u_g, **pk_full)

        # between/within variance ratio (env should be much higher)
        bw_e = between_within_ratio(e); bw_u = between_within_ratio(u)
        lr_["between_within_ratio"] = {"env_mean": float(bw_e.mean()), "action_mean": float(bw_u.mean())}

        lr_["_disentanglement"] = {
            "scene_e_minus_u_acc": lr_["scene_suite_from_e"]["val_acc"] - lr_["scene_suite_from_u"]["val_acc"],
            "task_e_minus_u_acc": lr_["task_from_e"]["val_acc"] - lr_["task_from_u"]["val_acc"],
            "action_u_minus_e_matched_r2": lr_["action_from_u"]["val_r2"] - lr_["action_from_e_matched"]["val_r2"],
        }
        results["layers"][layer] = lr_
        d = lr_["_disentanglement"]
        print(f"[{layer}] env_dim={env_dim} act_dim={act_dim} | "
              f"scene e/u={lr_['scene_suite_from_e']['val_acc']:.2f}/{lr_['scene_suite_from_u']['val_acc']:.2f} | "
              f"task e/u={lr_['task_from_e']['val_acc']:.2f}/{lr_['task_from_u']['val_acc']:.2f} | "
              f"action R² u/e_match/e_full="
              f"{lr_['action_from_u']['val_r2']:.2f}/{lr_['action_from_e_matched']['val_r2']:.2f}/{lr_['action_from_e_full']['val_r2']:.2f}")

    save_json(results, vdir / "probe_results.json")
    print(f"[save] -> {vdir / 'probe_results.json'}")


if __name__ == "__main__":
    main()
