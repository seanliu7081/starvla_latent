#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""Phase 4 — Leakage (discrete + continuous + conditional) on every candidate u.

For each feature (linear u's + references) report:
  discrete  : u -> suite / task / object (held-out clips), acc vs majority, balanced acc, CI
  continuous: u -> e  (R^2)   <-- the review-critical entanglement check
  conditional: leak_cond = acc(suite <- [action_bin, u]) - acc(suite <- action_bin)

Features: cca_d4 (chosen), pls_d16, full_h (ref), e (positive control), random_d16,
prev_highfreq_u (negative control). All [T3-corr].
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT / "src"))

from action_latent import features as F          # noqa: E402
from action_latent import io                     # noqa: E402
from action_latent import leakage as LK          # noqa: E402
from action_latent import linear as L            # noqa: E402
from action_latent.reporting import write_status  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_ROOT / "configs/clean_action_latent.yaml"))
    args = ap.parse_args()
    cfg = io.load_config(args.config)
    out = io.resolve_path(cfg.output_dir)
    out0, out4 = out / "phase0", io.ensure_dir(out / "phase4")
    seed = int(cfg.seed)

    align = io.load_json(out0 / "alignment.json")
    shift = int(align["final_decision"]["final_shift"])
    sm = io.load_json(out0 / "split_metadata.json")
    split = {k: np.array(sm[f"{k}_idx"]) for k in ["train", "val", "test"]}

    cache = io.load_global_cache(cfg)
    sub = io.load_env_subspace(cfg, verify_against_h=cache["h_pooled"])
    h, action, e = cache["h_pooled"], cache["actions"], sub["e"]
    Wenv, smean, sstd = sub["W_env"], sub["mean"], sub["std"]
    suite = cache["suite_id"].numpy(); task = cache["task_global"].numpy()

    sp = torch.load(out / "phase2" / "linear_subspaces.pt", weights_only=False)
    h_res = F.remove_subspace(h, Wenv, smean, sstd)
    h_std = F.standardize(h, smean, sstd)

    feats = {
        "cca_d4_chosen": h_res @ sp["W_act_sup_chosen"],
        "pls_d16": h_res @ sp["pls_d16"],
        "random_d16": h_res @ L.random_subspace(h_res.shape[-1], 16, seed=seed),
        "full_h": h_std,
        "e_env_POSCTRL": e,
        "prev_highfreq_u_NEGCTRL": torch.cat([sub["u_hf"], sub["u_hf"][:, -1:]], 1),  # pad to T
    }

    # object: libero_object subset (suite 0), task_local as object id
    obj_mask = (suite == 0)
    obj_split = {k: np.array([i for i in split[k] if obj_mask[i]]) for k in split}
    obj_lab = cache["task_local"].numpy()

    results = {"evidence_tier": "T3-corr", "alignment_shift": shift, "features": {}}
    for name, u in feats.items():
        rec = {}
        rec["discrete_suite"] = LK.discrete_probe(u, suite, split, shift, 4, device=cfg.device, seed=seed,
                                                  n_boot=int(cfg.eval.bootstrap_n))
        rec["discrete_task"] = LK.discrete_probe(u, task, split, shift, 40, device=cfg.device, seed=seed,
                                                 n_boot=int(cfg.eval.bootstrap_n))
        rec["continuous_u_to_e"] = LK.continuous_probe(u, e, split, shift, cfg, seed=seed)
        rec["conditional_suite"] = LK.conditional_leakage(u, action, suite, split, shift, cfg, 4, seed=seed)
        if all(len(obj_split[k]) > 0 for k in obj_split):
            rec["discrete_object_subset"] = LK.discrete_probe(u, obj_lab, obj_split, shift, 10,
                                                              device=cfg.device, seed=seed,
                                                              n_boot=int(cfg.eval.bootstrap_n))
        results["features"][name] = rec
        print(f"[{name:24s}] suite_acc={rec['discrete_suite']['acc']:.3f}"
              f"(maj {rec['discrete_suite']['majority_baseline']:.2f}) "
              f"task_acc={rec['discrete_task']['acc']:.3f}(maj {rec['discrete_task']['majority_baseline']:.2f}) "
              f"u->e_R2={rec['continuous_u_to_e']['u_to_e_r2']:+.3f} "
              f"leak_cond={rec['conditional_suite']['leak_cond']:+.3f}")

    # interpretation summary
    chosen = results["features"]["cca_d4_chosen"]
    results["summary"] = {
        "chosen_u_continuous_leak_r2": chosen["continuous_u_to_e"]["u_to_e_r2"],
        "full_h_continuous_leak_r2": results["features"]["full_h"]["continuous_u_to_e"]["u_to_e_r2"],
        "chosen_u_suite_above_chance": chosen["discrete_suite"]["acc_above_chance"],
        "chosen_u_task_above_chance": chosen["discrete_task"]["acc_above_chance"],
        "chosen_u_leak_cond": chosen["conditional_suite"]["leak_cond"],
        "reading": ("clean if: u->e R^2 materially below full_h, discrete suite/task near chance, "
                    "and leak_cond small. Defensible target is LOW CONDITIONAL leakage, not zero raw."),
    }
    io.save_json(results, out4 / "leakage.json")
    print(f"\n[summary] chosen u: u->e R2={results['summary']['chosen_u_continuous_leak_r2']:.3f} "
          f"(full_h {results['summary']['full_h_continuous_leak_r2']:.3f}), "
          f"suite+{results['summary']['chosen_u_suite_above_chance']:+.3f}, "
          f"leak_cond={results['summary']['chosen_u_leak_cond']:+.3f}")
    write_status(out, phase_reached="phase4",
                 gates_passed=io.load_json(out / "STATUS.json")["gates_passed"],
                 interface=io.load_json(out0 / "interface_probe.json")["interface"],
                 verdict=None, extra={"leakage_summary": results["summary"]})
    print(f"[save] -> {out4/'leakage.json'}")


if __name__ == "__main__":
    main()
