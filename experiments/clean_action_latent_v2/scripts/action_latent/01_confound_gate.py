#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""Phase 1 — Confound gate.

How much of "predicting action" is just "predicting the environment"? Held-out
(test-clip) action R^2 for each predictor, with clip-level bootstrap CIs and the
headline GAIN-over-e (Prime Directive 3). Every number here is correlational
[T3-corr]: a probe reads action off a feature.

Predictors (plan Phase-1 table):
  mean | e->a (shortcut) | task_onehot->a | object_onehot->a (libero_object subset)
  | h_full linear | h_full MLP (ceiling) | h_res linear | + previous high-freq u
  (negative control) | proprio state (reference)

Saves outputs/.../phase1/confound_gate.json with shortcut_R2, ceiling_R2,
headroom, hres_retains, and the two GATE-1 decisions consumed by Phases 2-3.
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
from action_latent import models as Mod          # noqa: E402
from action_latent.reporting import write_status  # noqa: E402


def expand_label(label_vec, T, n_classes):
    """[N] per-clip labels -> [N,T,n_classes] per-frame one-hot (constant over t)."""
    oh = L.one_hot(label_vec.numpy(), n_classes)        # [N,C]
    return oh.unsqueeze(1).expand(-1, T, -1).contiguous()


def ridge_predictor(feat, action, split, shift, cfg, seed, standardize=None):
    """Flatten (shift-aligned), fit ridge (val-selected lambda), eval on test.
    Returns dict with test r2 + per-dim + CI, and the test (Y,pred,clip_ids)."""
    tr, va, te = split["train"], split["val"], split["test"]
    if standardize is not None:
        mu, sd = standardize
        feat = (feat - mu) / sd
    Xtr, Ytr, _ = F.flatten_frames(feat, tr, shift=shift, action=action)
    Xva, Yva, _ = F.flatten_frames(feat, va, shift=shift, action=action)
    Xte, Yte, cte = F.flatten_frames(feat, te, shift=shift, action=action)
    W, lam, val_r2 = L.fit_ridge_cv(Xtr, Ytr, Xva, Yva, list(cfg.linear.ridge_lambdas))
    pte = L.predict_ridge(W, Xte)
    rep = M.r2_report(Yte.numpy(), pte.numpy())
    ci = M.bootstrap_r2_ci(Yte.numpy(), pte.numpy(), cte,
                           n_boot=int(cfg.eval.bootstrap_n), seed=seed)
    return {"test_r2": rep["r2"], "test_r2_per_dim": rep["r2_per_dim"],
            "test_mse": rep["mse"], "test_r2_ci": [ci["lo"], ci["hi"]],
            "val_r2": val_r2, "ridge_lambda": lam}, (Yte.numpy(), pte.numpy(), cte)


def mlp_predictor(feat, action, split, shift, cfg, seed, standardize):
    tr, va, te = split["train"], split["val"], split["test"]
    mu, sd = standardize
    feat = (feat - mu) / sd
    Xtr, Ytr, _ = F.flatten_frames(feat, tr, shift=shift, action=action)
    Xva, Yva, _ = F.flatten_frames(feat, va, shift=shift, action=action)
    Xte, Yte, cte = F.flatten_frames(feat, te, shift=shift, action=action)
    mc = cfg.mlp_ceiling
    model, info = Mod.train_regressor(
        Xtr, Ytr, Xva, Yva, hidden=int(mc.hidden_dim), dropout=float(mc.dropout),
        lr=float(mc.lr), weight_decay=float(mc.weight_decay), epochs=int(mc.epochs),
        batch_size=int(mc.batch_size), patience=int(mc.early_stopping_patience),
        device=cfg.device, seed=seed)
    pte = Mod.regressor_predict(model, Xte).numpy()
    rep = M.r2_report(Yte.numpy(), pte)
    ci = M.bootstrap_r2_ci(Yte.numpy(), pte, cte, n_boot=int(cfg.eval.bootstrap_n), seed=seed)
    return {"test_r2": rep["r2"], "test_r2_per_dim": rep["r2_per_dim"],
            "test_mse": rep["mse"], "test_r2_ci": [ci["lo"], ci["hi"]],
            **info}, (Yte.numpy(), pte, cte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_ROOT / "configs/clean_action_latent.yaml"))
    args = ap.parse_args()
    cfg = io.load_config(args.config)
    out = io.resolve_path(cfg.output_dir)
    out0, out1 = out / "phase0", io.ensure_dir(out / "phase1")
    seed = int(cfg.seed)

    # inputs from Phase 0
    align = io.load_json(out0 / "alignment.json")
    shift = int(align["final_decision"]["final_shift"])
    split = {k: np.array(io.load_json(out0 / "split_metadata.json")[f"{k}_idx"])
             for k in ["train", "val", "test"]}
    ns = torch.load(out0 / "norm_stats.pt", weights_only=False)
    mu_h, std_h = ns["mu_h"], ns["std_h"]

    cache = io.load_global_cache(cfg)
    sub = io.load_env_subspace(cfg, verify_against_h=cache["h_pooled"])
    h, action = cache["h_pooled"], cache["actions"]
    T = cache["T"]
    print(f"[phase1] alignment shift={shift}  split={ {k: len(v) for k,v in split.items()} }")

    results, preds = {}, {}

    # --- mean baseline (predict train-mean action) ---
    tr_act = action[split["train"]].reshape(-1, 7).mean(0)
    Yte = []
    cte = []
    for i in split["test"]:
        Yte.append(action[i]); cte.append(np.full(T, int(i)))
    Yte = torch.cat(Yte, 0).numpy(); cte = np.concatenate(cte)
    pmean = np.tile(tr_act.numpy(), (Yte.shape[0], 1))
    rep = M.r2_report(Yte, pmean)
    ci = M.bootstrap_r2_ci(Yte, pmean, cte, n_boot=int(cfg.eval.bootstrap_n), seed=seed)
    results["mean"] = {"test_r2": rep["r2"], "test_r2_ci": [ci["lo"], ci["hi"]]}
    print(f"  mean              r2={rep['r2']:+.4f}")

    # --- e -> a  (env shortcut) ---
    e = sub["e"]                                            # [160,48,128]
    mu_e, sd_e = io.train_norm_stats(e, split["train"])
    results["e_to_a"], preds["e"] = ridge_predictor(e, action, split, shift, cfg, seed, (mu_e, sd_e))
    print(f"  e->a (shortcut)   r2={results['e_to_a']['test_r2']:+.4f}  CI={results['e_to_a']['test_r2_ci']}")

    # --- task_onehot -> a ---
    task_feat = expand_label(cache["task_global"], T, 40)
    results["task_to_a"], _ = ridge_predictor(task_feat, action, split, shift, cfg, seed)
    print(f"  task->a           r2={results['task_to_a']['test_r2']:+.4f}")

    # --- object_onehot -> a  (libero_object subset, suite_id==0) ---
    suite = cache["suite_id"].numpy()
    obj_mask = (suite == 0)
    obj_clips = np.where(obj_mask)[0]
    obj_split = {k: np.array([i for i in split[k] if obj_mask[i]]) for k in ["train", "val", "test"]}
    obj_feat = expand_label(cache["task_local"], T, 10)     # within libero_object, task_local==object id
    if all(len(obj_split[k]) > 0 for k in obj_split):
        r_obj, _ = ridge_predictor(obj_feat, action, obj_split, shift, cfg, seed)
        # within-subset full-h ceiling for a meaningful fraction
        r_obj_h, _ = ridge_predictor(h, action, obj_split, shift, cfg, seed, (mu_h, std_h))
        results["object_to_a_libero_object_subset"] = {
            **r_obj, "subset_full_h_linear_r2": r_obj_h["test_r2"],
            "n_clips": {k: int(len(obj_split[k])) for k in obj_split}}
        print(f"  object->a (sub)   r2={r_obj['test_r2']:+.4f}  (subset h_full={r_obj_h['test_r2']:+.4f})")

    # --- h_full linear -> a  (linear ceiling) ---
    results["h_full_linear"], preds["h_lin"] = ridge_predictor(h, action, split, shift, cfg, seed, (mu_h, std_h))
    print(f"  h_full linear     r2={results['h_full_linear']['test_r2']:+.4f}  CI={results['h_full_linear']['test_r2_ci']}")

    # --- h_res linear -> a  (env subspace removed) ---
    h_res = F.remove_subspace(h, sub["W_env"], sub["mean"], sub["std"])   # standardized residual
    # standardize residual with its own train stats for ridge conditioning
    mu_r, sd_r = io.train_norm_stats(h_res, split["train"])
    results["h_res_linear"], preds["h_res"] = ridge_predictor(h_res, action, split, shift, cfg, seed, (mu_r, sd_r))
    print(f"  h_res linear      r2={results['h_res_linear']['test_r2']:+.4f}  CI={results['h_res_linear']['test_r2_ci']}")

    # --- previous high-freq u -> a  (NEGATIVE CONTROL) ---
    u_hf = sub["u_hf"]                                      # [160,47,16] transitions t->t+1
    # u indexed by transition t (frame t..t+1); pair u[:,t] with a_t (shift handled inside)
    mu_u, sd_u = io.train_norm_stats(u_hf, split["train"])
    # u has T-1 frames; flatten with shift 0 maps u[:,t]->a_t for t in [0, T-1)
    results["previous_highfreq_u_NEGCTRL"], _ = ridge_predictor(u_hf, action[:, :u_hf.shape[1]], split, 0, cfg, seed, (mu_u, sd_u))
    print(f"  prev high-freq u  r2={results['previous_highfreq_u_NEGCTRL']['test_r2']:+.4f}  (negative control)")

    # --- proprio state -> a  (reference) ---
    st = cache["states"]
    mu_s, sd_s = io.train_norm_stats(st, split["train"])
    results["proprio_state_reference"], _ = ridge_predictor(st, action, split, shift, cfg, seed, (mu_s, sd_s))
    print(f"  proprio state     r2={results['proprio_state_reference']['test_r2']:+.4f}  (reference)")

    # --- h_full MLP -> a  (nonlinear CEILING) ---
    print("  [mlp] training nonlinear ceiling ...")
    results["h_full_mlp"], preds["h_mlp"] = mlp_predictor(h, action, split, shift, cfg, seed, (mu_h, std_h))
    print(f"  h_full MLP        r2={results['h_full_mlp']['test_r2']:+.4f}  CI={results['h_full_mlp']['test_r2_ci']}  "
          f"(val={results['h_full_mlp']['best_val_r2']:.3f}, epochs={results['h_full_mlp']['epochs_ran']})")

    # --- headline quantities ---
    shortcut_R2 = results["e_to_a"]["test_r2"]
    ceiling_lin = results["h_full_linear"]["test_r2"]
    ceiling_R2 = max(results["h_full_mlp"]["test_r2"], ceiling_lin)   # ceiling = best of MLP / linear
    ceiling_src = "mlp" if results["h_full_mlp"]["test_r2"] >= ceiling_lin else "linear"
    headroom = ceiling_R2 - shortcut_R2
    hres_retains = results["h_res_linear"]["test_r2"] / ceiling_R2 if ceiling_R2 > 0 else float("nan")

    # GAINS over e (paired clip-level bootstrap CIs; e and h_* share test order/ids)
    gains = {}
    for name, key in [("h_full_linear", "h_lin"), ("h_full_mlp", "h_mlp"), ("h_res_linear", "h_res")]:
        y, pu, c = preds[key]
        _, pe, _ = preds["e"]
        # align rows: e and h_* share the same flatten (shift,test) -> same order/ids
        g = M.bootstrap_gain_ci(y, pu, pe, c, n_boot=int(cfg.eval.bootstrap_n), seed=seed)
        gains[name] = g
        print(f"  GAIN[{name} - e] = {g['point']:+.4f}  CI=[{g['lo']:+.4f},{g['hi']:+.4f}]  excl0={g['excludes_zero']}")

    headline = {
        "alignment_shift": shift,
        "shortcut_R2": shortcut_R2,
        "ceiling_R2": ceiling_R2,
        "ceiling_source": ceiling_src,
        "ceiling_R2_linear": ceiling_lin,
        "ceiling_R2_mlp": results["h_full_mlp"]["test_r2"],
        "headroom": headroom,
        "hres_retains": hres_retains,
        "shortcut_fraction_of_ceiling": shortcut_R2 / ceiling_R2 if ceiling_R2 > 0 else float("nan"),
        "gain_over_e": {k: gains[k]["point"] for k in gains},
        "gain_over_e_ci": {k: [gains[k]["lo"], gains[k]["hi"]] for k in gains},
        "gain_over_e_excludes_zero": {k: gains[k]["excludes_zero"] for k in gains},
    }

    # ---- GATE-1 decisions (consumed by Phases 2-3) ----
    g1 = cfg.gates
    drop_to_CD = headline["shortcut_fraction_of_ceiling"] > float(g1.gate1_shortcut_fraction_for_CD)
    switch_to_hfull = hres_retains < float(g1.gate1_hres_retains_min)
    decisions = {
        "shortcut_largely_explains_action": bool(drop_to_CD),
        "expected_verdict_drops_to_C_or_D": bool(drop_to_CD),
        "primary_feature_for_phase2_3": ("h_full" if switch_to_hfull else "h_res"),
        "env_removal_destroys_action_info": bool(switch_to_hfull),
        "thresholds": {"shortcut_fraction_for_CD": float(g1.gate1_shortcut_fraction_for_CD),
                       "hres_retains_min": float(g1.gate1_hres_retains_min)},
        "rationale": {
            "headroom": ("small -> action is largely an env shortcut; conditional-leakage + "
                         "pair tests become primary" if drop_to_CD else
                         "substantial -> a clean action latent has room to exist beyond e"),
            "hres": ("env removal destroys shared action variance -> use full h with "
                     "adversary/decorrelation instead of pre-projection" if switch_to_hfull else
                     "h_res still retains most action info -> safe to project out W_env"),
        },
    }

    payload = {"evidence_tier": "T3-corr",
               "predictors": results, "headline": headline, "gate1_decisions": decisions}
    io.save_json(payload, out1 / "confound_gate.json")
    print("\n[GATE-1] shortcut/ceiling = %.3f  headroom = %.3f  hres_retains = %.3f" %
          (headline["shortcut_fraction_of_ceiling"], headroom, hres_retains))
    print("[GATE-1] expected_verdict_drops_to_C/D =", drop_to_CD,
          "| primary_feature =", decisions["primary_feature_for_phase2_3"])

    iface = io.load_json(out0 / "interface_probe.json")["interface"] if (out0 / "interface_probe.json").exists() else None
    write_status(out, phase_reached="phase1", gates_passed=["GATE-0", "GATE-1"],
                 interface=iface, verdict=None,
                 extra={"alignment_shift": shift, "gate1_decisions": decisions,
                        "headline": headline})
    print(f"[save] -> {out1 / 'confound_gate.json'}")


if __name__ == "__main__":
    main()
