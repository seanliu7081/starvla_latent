#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""Phase 2 — Linear-first action subspace (the projectable candidate).

Fits supervised linear subspaces W_act_sup[2048,d] on the GATE-1 primary feature
(h_res = env-removed standardized hidden) via PLS / RRR / CCA, plus matched-d
baselines (random, PCA(h_res), PCA(Δh_res), previous high-freq u, e-only). Each
yields u = h_res @ W and a ridge decoder u->a. Reports action R^2 (variance- AND
equal-weighted; per M1 verification caveat), gain over e (paired clip bootstrap),
and transition R^2 (u -> PCA128(Δh_res)). Evaluates GATE-2.

The subspace is fit in the SUBSPACE-normalized standardized-h space (same space as
W_env) so W_act_sup applies per-token in Phase 6A (projectable -> Outcome A path).
All numbers are [T3-corr].
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
from action_latent import linear as L            # noqa: E402
from action_latent import metrics as M           # noqa: E402
from action_latent.reporting import write_status  # noqa: E402


def fit_decoder_eval(u_feat, action, split, shift, cfg, seed):
    """Fit ridge decoder u->target on train (val-selected lambda), eval on test.
    u_feat:[N,T,d], action:[N,T,K]. Returns (metrics, (y,pred,cids))."""
    tr, va, te = split["train"], split["val"], split["test"]
    mu, sd = io.train_norm_stats(u_feat, tr)
    uf = (u_feat - mu) / sd
    Xtr, Ytr, _ = F.flatten_frames(uf, tr, shift=shift, action=action)
    Xva, Yva, _ = F.flatten_frames(uf, va, shift=shift, action=action)
    Xte, Yte, cte = F.flatten_frames(uf, te, shift=shift, action=action)
    W, lam, val_r2 = L.fit_ridge_cv(Xtr, Ytr, Xva, Yva, list(cfg.linear.ridge_lambdas))
    pte = L.predict_ridge(W, Xte)
    rep = M.r2_report(Yte.numpy(), pte.numpy())
    out = {"test_r2": rep["r2"], "test_r2_equal": rep["r2_equal"],
           "test_r2_per_dim": rep["r2_per_dim"],
           "test_mae": float(np.abs(Yte.numpy() - pte.numpy()).mean()),
           "val_r2": val_r2, "ridge_lambda": lam, "u_dim": int(u_feat.shape[-1])}
    return out, (Yte.numpy(), pte.numpy(), cte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_ROOT / "configs/clean_action_latent.yaml"))
    args = ap.parse_args()
    cfg = io.load_config(args.config)
    out = io.resolve_path(cfg.output_dir)
    out0, out2 = out / "phase0", io.ensure_dir(out / "phase2")
    seed = int(cfg.seed)

    align = io.load_json(out0 / "alignment.json")
    shift = int(align["final_decision"]["final_shift"])
    sm = io.load_json(out0 / "split_metadata.json")
    split = {k: np.array(sm[f"{k}_idx"]) for k in ["train", "val", "test"]}
    p1 = io.load_json(out / "phase1" / "confound_gate.json")
    ceiling = p1["headline"]["ceiling_R2"]
    shortcut = p1["headline"]["shortcut_R2"]

    cache = io.load_global_cache(cfg)
    sub = io.load_env_subspace(cfg, verify_against_h=cache["h_pooled"])
    h, action = cache["h_pooled"], cache["actions"]
    Wenv, smean, sstd = sub["W_env"], sub["mean"], sub["std"]

    # GATE-1 primary feature: h_res (env-removed, subspace-normalized residual ⟂ W_env)
    h_res = F.remove_subspace(h, Wenv, smean, sstd)
    dh = h_res[:, 1:] - h_res[:, :-1]                          # [N,T-1,2048]
    Vt = L.pca_subspace(dh[split["train"]].reshape(-1, dh.shape[-1]), 128)   # [2048,128]
    dh_target = dh @ Vt                                        # [N,T-1,128] transition target

    tr = split["train"]
    Xtr_full, Ytr_full, _ = F.flatten_frames(h_res, tr, shift=shift, action=action)
    print(f"[phase2] fit pool: train frames={Xtr_full.shape[0]}  ceiling(var)={ceiling:.3f} shortcut={shortcut:.3f}")

    dims = list(cfg.latent.dims)
    methods = list(cfg.linear.methods)

    e = sub["e"]
    e_metrics, e_preds = fit_decoder_eval(e, action, split, shift, cfg, seed)
    print(f"[phase2] e-only decoder r2={e_metrics['test_r2']:.4f}")

    results = {"ceiling_R2_var": ceiling, "shortcut_R2": shortcut,
               "gate2_threshold": float(cfg.gates.gate2_action_fraction_of_ceiling) * ceiling,
               "supervised": {}, "baselines": {}}
    subspaces_to_save = {}

    def gain_ci(preds):
        y, pu, c = preds
        return M.bootstrap_gain_ci(y, pu, e_preds[1], c, n_boot=int(cfg.eval.bootstrap_n), seed=seed)

    # ---- supervised subspaces ----
    for method in methods:
        results["supervised"][method] = {}
        for d in dims:
            W = L.supervised_subspace(Xtr_full, Ytr_full, d, method)   # [2048,k<=d]
            k = int(W.shape[1])
            u = h_res @ W
            m, preds = fit_decoder_eval(u, action, split, shift, cfg, seed)
            g = gain_ci(preds)
            tm, _ = fit_decoder_eval(u[:, :dh_target.shape[1]], dh_target, split, shift, cfg, seed)
            m.update({"effective_dim": k, "gain_over_e": g["point"], "gain_ci": [g["lo"], g["hi"]],
                      "gain_excludes_zero": g["excludes_zero"], "frac_of_ceiling": m["test_r2"]/ceiling,
                      "transition_r2": tm["test_r2"]})
            results["supervised"][method][str(d)] = m
            if method == "pls":
                subspaces_to_save[f"pls_d{d}"] = W
            print(f"  {method} d={d:>2}(k={k}) r2={m['test_r2']:.4f} equal={m['test_r2_equal']:.3f} "
                  f"gain={g['point']:+.3f} excl0={g['excludes_zero']} trans={tm['test_r2']:+.3f}")

    # ---- matched-d baselines ----
    Dh = h_res.shape[-1]
    pca_hres = L.pca_subspace(Xtr_full, max(dims))
    for d in dims:
        results["baselines"][str(d)] = {}
        bset = {"random": L.random_subspace(Dh, d, seed=seed),
                "pca_h_res": pca_hres[:, :d],
                "pca_delta_h_res": Vt[:, :d]}
        for name, W in bset.items():
            u = h_res @ W
            m, preds = fit_decoder_eval(u, action, split, shift, cfg, seed)
            g = gain_ci(preds)
            results["baselines"][str(d)][name] = {"test_r2": m["test_r2"], "test_r2_equal": m["test_r2_equal"],
                                                  "gain_over_e": g["point"], "frac_of_ceiling": m["test_r2"]/ceiling}
        print(f"  [base d={d}] rand={results['baselines'][str(d)]['random']['test_r2']:.3f} "
              f"pca_h={results['baselines'][str(d)]['pca_h_res']['test_r2']:.3f} "
              f"pca_dh={results['baselines'][str(d)]['pca_delta_h_res']['test_r2']:.3f}")

    # fixed-dim references
    u_hf = sub["u_hf"]
    m_hf, _ = fit_decoder_eval(u_hf, action[:, :u_hf.shape[1]], split, 0, cfg, seed)
    results["reference_prev_highfreq_u"] = {"test_r2": m_hf["test_r2"], "u_dim": 16}
    results["reference_e_only"] = {"test_r2": e_metrics["test_r2"], "test_r2_equal": e_metrics["test_r2_equal"], "u_dim": int(e.shape[-1])}
    print(f"  [ref] prev_highfreq_u r2={m_hf['test_r2']:+.3f}  e_only r2={e_metrics['test_r2']:.3f}")

    # ---- GATE-2 ----
    thr = results["gate2_threshold"]
    maxd = int(cfg.gates.gate2_max_dim)
    passers = []
    for method in methods:
        for d in dims:
            if d > maxd:
                continue
            mm = results["supervised"][method][str(d)]
            base_best = max(results["baselines"][str(d)][b]["test_r2"]
                            for b in ["random", "pca_h_res", "pca_delta_h_res"])
            if (mm["test_r2"] >= thr and mm["gain_excludes_zero"]
                    and mm["test_r2"] > base_best + 0.02 and mm["test_r2"] > m_hf["test_r2"] + 0.05):
                passers.append({"method": method, "d": d, "r2": mm["test_r2"],
                                "gain": mm["gain_over_e"], "beats_baseline_by": mm["test_r2"]-base_best})
    gate2_pass = len(passers) > 0
    chosen = None
    if gate2_pass:
        chosen = sorted(passers, key=lambda p: (p["d"], -p["r2"]))[0]
        Wc = L.supervised_subspace(Xtr_full, Ytr_full, chosen["d"], chosen["method"])
        subspaces_to_save["W_act_sup_chosen"] = Wc
        subspaces_to_save["chosen_meta"] = {"method": chosen["method"], "d": int(Wc.shape[1])}
    results["gate2"] = {"passed": gate2_pass, "threshold": thr, "passers": passers,
                        "chosen_projectable": chosen,
                        "note": ("supervised linear action subspace is intrinsically <=7-D (action is 7-D); "
                                 "R^2 saturates by d~7, so d>=8 reaches the h_res ceiling.")}
    subspaces_to_save.update({"W_env": Wenv, "W_act_highfreq": sub["W_act_hf"],
                              "sub_mean": smean, "sub_std": sstd})
    torch.save(subspaces_to_save, out2 / "linear_subspaces.pt")
    io.save_json({"evidence_tier": "T3-corr", **results}, out2 / "linear_metrics.json")

    print(f"\n[GATE-2] passed={gate2_pass}  chosen={chosen}")
    write_status(out, phase_reached="phase2",
                 gates_passed=["GATE-0", "GATE-1"] + (["GATE-2"] if gate2_pass else []),
                 interface=io.load_json(out0 / "interface_probe.json")["interface"],
                 verdict=None, extra={"gate2_passed": gate2_pass, "chosen_projectable": chosen})
    print(f"[save] -> {out2/'linear_metrics.json'} , {out2/'linear_subspaces.pt'}")


if __name__ == "__main__":
    main()
