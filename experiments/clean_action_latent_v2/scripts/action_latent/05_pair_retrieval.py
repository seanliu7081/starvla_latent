#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""Phase 5 — Pair / retrieval deconfounding.

Isolates action from environment for the chosen u (cca_d4), vs e (env) and full_h:
  - partial corr(d_u, d_action | same_env)  (key deconfounded stat)
  - retrieval recall@k for action bins + neighbor env-diversity
  - leave-task-out: refit cca_d4 excluding held-out tasks, eval on those tasks.
[T4-retr], bootstrap CIs.
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
from action_latent import retrieval as R          # noqa: E402
from action_latent.reporting import write_status  # noqa: E402


def _np(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def frames_of(feat, idx, shift, *others):
    """Flatten clip idx -> per-frame rows. feat uses h_{t+shift}; per-frame `others`
    ([N,T,...]) align to a_t (frame t); per-clip `others` ([N]) broadcast over t."""
    N, T = feat.shape[0], feat.shape[1]
    t0 = max(0, -shift); t1 = min(T, T - shift)
    feat = _np(feat); others = [_np(o) for o in others]
    xs, extra, cids = [], [[] for _ in others], []
    for i in idx:
        ts = np.arange(t0, t1)
        xs.append(feat[i, ts + shift])
        cids.append(np.full(len(ts), int(i)))
        for j, o in enumerate(others):
            extra[j].append(o[i, ts] if o.ndim >= 2 else np.full(len(ts), o[i]))
    return np.concatenate(xs, 0), np.concatenate(cids), [np.concatenate(e) for e in extra]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_ROOT / "configs/clean_action_latent.yaml"))
    args = ap.parse_args()
    cfg = io.load_config(args.config)
    out = io.resolve_path(cfg.output_dir)
    out0, out5 = out / "phase0", io.ensure_dir(out / "phase5")
    seed = int(cfg.seed)

    shift = int(io.load_json(out0 / "alignment.json")["final_decision"]["final_shift"])
    sm = io.load_json(out0 / "split_metadata.json")
    split = {k: np.array(sm[f"{k}_idx"]) for k in ["train", "val", "test"]}
    cache = io.load_global_cache(cfg)
    sub = io.load_env_subspace(cfg, verify_against_h=cache["h_pooled"])
    h, action = cache["h_pooled"], cache["actions"]
    suite = cache["suite_id"].numpy(); task = cache["task_global"].numpy()
    sp = torch.load(out / "phase2" / "linear_subspaces.pt", weights_only=False)
    h_res = F.remove_subspace(h, sub["W_env"], sub["mean"], sub["std"])
    h_std = F.standardize(h, sub["mean"], sub["std"])

    # action bins (fit on train)
    abins, nb = LK.action_bins(action, split, cfg, seed=seed)
    abins = abins.numpy()

    feats = {"cca_d4_chosen": (h_res @ sp["W_act_sup_chosen"]),
             "e_env": sub["e"], "full_h": h_std}

    results = {"evidence_tier": "T4-retr", "n_action_bins": int(nb), "features": {}}
    te = split["test"]
    for name, U in feats.items():
        Uf, cids, (Af, Eb, Tb, abf2) = frames_of(U, te, shift, action, suite, task, abins)
        ps = R.pair_stats(Uf, Af, Eb.astype(int), cids, seed=seed, n_sub=600, n_boot=int(cfg.eval.bootstrap_n)//2)
        rt = R.retrieval(Uf, abf2.astype(int), Eb.astype(int), cids, k=5, seed=seed)
        rt_task = R.retrieval(Uf, abf2.astype(int), Tb.astype(int), cids, k=5, seed=seed)
        results["features"][name] = {"pair": ps, "retrieval_suite": rt,
                                     "neighbor_task_diversity": rt_task["neighbor_env_diversity"]}
        print(f"[{name:16s}] partial_corr(du,da|env)={ps['partial_corr_du_da_given_env']:+.3f} "
              f"CI={[round(x,3) for x in ps['partial_corr_ci']]} | raw du-env={ps['raw_corrs']['r_du_env']:+.3f} "
              f"du-da={ps['raw_corrs']['r_du_da']:+.3f} | recall@5={rt['action_bin_recall_at_k']:.3f}"
              f"(rand {rt['action_bin_recall_random']:.3f}) env_div={rt['neighbor_env_diversity']:.3f}")

    # ---- leave-task-out: refit cca_d4 excluding held-out tasks, eval on them ----
    rng = np.random.default_rng(seed)
    held_tasks = []
    for s in range(4):
        ts = np.unique(task[suite == s])
        held_tasks += rng.choice(ts, size=2, replace=False).tolist()
    held = set(held_tasks)
    train_lto = np.array([i for i in split["train"] if task[i] not in held])
    eval_lto = np.array([i for i in range(160) if task[i] in held])
    Xtr, Ytr, _ = F.flatten_frames(h_res, train_lto, shift=shift, action=action)
    Wlto = L.supervised_subspace(Xtr, Ytr, 4, "cca")
    Ulto = h_res @ Wlto
    Uf, cids, (Af, Eb, abf3) = frames_of(Ulto, eval_lto, shift, action, task, abins)
    ps_lto = R.pair_stats(Uf, Af, Eb.astype(int), cids, seed=seed, n_sub=600, n_boot=int(cfg.eval.bootstrap_n)//2)
    rt_lto = R.retrieval(Uf, abf3.astype(int), Eb.astype(int), cids, k=5, seed=seed)
    results["leave_task_out"] = {"held_out_tasks": [int(t) for t in held_tasks],
                                 "n_train_clips": int(len(train_lto)), "n_eval_clips": int(len(eval_lto)),
                                 "pair": ps_lto, "retrieval_task": rt_lto}
    print(f"[LTO cca_d4]    partial_corr(du,da|env)={ps_lto['partial_corr_du_da_given_env']:+.3f} "
          f"CI={[round(x,3) for x in ps_lto['partial_corr_ci']]} recall@5={rt_lto['action_bin_recall_at_k']:.3f}"
          f"(rand {rt_lto['action_bin_recall_random']:.3f})")

    c = results["features"]["cca_d4_chosen"]["pair"]
    results["summary"] = {
        "chosen_partial_corr_du_da_given_env": c["partial_corr_du_da_given_env"],
        "chosen_partial_corr_ci": c["partial_corr_ci"],
        "chosen_du_env_corr": c["raw_corrs"]["r_du_env"],
        "lto_partial_corr": ps_lto["partial_corr_du_da_given_env"],
        "reading": ("clean action u: partial corr(d_u,d_action|env) strongly positive (u-distance "
                    "tracks action even within an environment), low d_u-env corr, action-bin "
                    "recall@5 >> random, high neighbor env-diversity; holds under leave-task-out."),
    }
    io.save_json(results, out5 / "pair_retrieval.json")
    write_status(out, phase_reached="phase5", gates_passed=io.load_json(out/"STATUS.json")["gates_passed"],
                 interface=io.load_json(out0/"interface_probe.json")["interface"], verdict=None,
                 extra={"phase5_summary": results["summary"]})
    print(f"[save] -> {out5/'pair_retrieval.json'}")


if __name__ == "__main__":
    main()
