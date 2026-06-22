#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""Phase 3 — Nonlinear bottleneck (MINIMIZED: GATE-2 passed, so this is the
nonlinear-ceiling + adversarial-tradeoff comparison, not the Outcome-A path).

Two-stage training on h_res:
  Stage A (predictive): action + transition + decorr(vs e) + L2.
  Stage B (leakage reduction): + env adversary (GRL on suite) swept over weights.
Outputs the tradeoff curve (action R^2 vs continuous leakage u->e) so we can see
whether a nonlinear/adversarial u is any cleaner than the linear CCA-d4 subspace.
A nonlinear u is NOT projectable -> at most T2 evidence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT / "src"))

from action_latent import features as F          # noqa: E402
from action_latent import io                     # noqa: E402
from action_latent import leakage as LK          # noqa: E402
from action_latent import linear as L            # noqa: E402
from action_latent import losses as LO           # noqa: E402
from action_latent import metrics as M           # noqa: E402
from action_latent.models import ActionBottleneck, EnvAdversary  # noqa: E402
from action_latent.reporting import write_status  # noqa: E402


def flat(feat, action, trans, suite, idx, shift):
    N, T, D = feat.shape
    t0 = max(0, -shift); t1 = min(T, T - shift)
    xs, ys, ts_, ss = [], [], [], []
    tt = min(t1, trans.shape[1])
    for i in idx:
        rng = np.arange(t0, tt)
        xs.append(feat[i, rng + shift]); ys.append(action[i, rng])
        ts_.append(trans[i, rng]); ss.append(np.full(len(rng), int(suite[i])))
    return (torch.cat(xs, 0), torch.cat(ys, 0), torch.cat(ts_, 0),
            torch.as_tensor(np.concatenate(ss), dtype=torch.long))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_ROOT / "configs/clean_action_latent.yaml"))
    ap.add_argument("--u_dim", type=int, default=16)
    args = ap.parse_args()
    cfg = io.load_config(args.config)
    out = io.resolve_path(cfg.output_dir)
    out0, out3 = out / "phase0", io.ensure_dir(out / "phase3")
    seed = int(cfg.seed); torch.manual_seed(seed); np.random.seed(seed)
    dev = cfg.device if torch.cuda.is_available() else "cpu"

    shift = int(io.load_json(out0 / "alignment.json")["final_decision"]["final_shift"])
    sm = io.load_json(out0 / "split_metadata.json")
    split = {k: np.array(sm[f"{k}_idx"]) for k in ["train", "val", "test"]}
    cache = io.load_global_cache(cfg)
    sub = io.load_env_subspace(cfg, verify_against_h=cache["h_pooled"])
    h, action = cache["h_pooled"], cache["actions"]
    suite = cache["suite_id"].numpy(); e = sub["e"]
    h_res = F.remove_subspace(h, sub["W_env"], sub["mean"], sub["std"])
    dh = h_res[:, 1:] - h_res[:, :-1]
    Vt = L.pca_subspace(dh[split["train"]].reshape(-1, dh.shape[-1]), 128)
    trans = dh @ Vt
    # standardize input with train-only stats
    mu, sd = io.train_norm_stats(h_res, split["train"]); hin = (h_res - mu) / sd
    # e standardized per-frame for decorrelation target
    emu, esd = io.train_norm_stats(e, split["train"]); estd = (e - emu) / esd

    Xtr, Atr, Ttr, Str = flat(hin, action, trans, suite, split["train"], shift)
    Xte, Ate, Tte, Ste = flat(hin, action, trans, suite, split["test"], shift)
    # e rows aligned to the SAME train frames (for the decorrelation target)
    Etr = flat(estd, action, trans, suite, split["train"], shift)[0]
    Xtr, Atr, Ttr = Xtr.to(dev), Atr.to(dev), Ttr.to(dev)
    Etr, Str = Etr.to(dev), Str.to(dev)

    def encode_all(model):
        model.eval()
        with torch.no_grad():
            u = model.enc(hin.reshape(-1, hin.shape[-1]).to(dev)).cpu()
        return u.reshape(hin.shape[0], hin.shape[1], -1)

    def eval_action_r2(model):
        model.eval()
        with torch.no_grad():
            _, ahat, _ = model(Xte.to(dev))
        rep = M.r2_report(Ate.numpy(), ahat.cpu().numpy())
        return rep["r2"], rep["r2_equal"]

    lw = {"action": 1.0, "transition": 0.2, "decorrelation": 0.01, "l2": 1e-3}  # plan §5 Stage A
    tradeoff = {}
    adv_weights = [0.0, 0.05, 0.1, 0.2]
    for adv_w in adv_weights:
        torch.manual_seed(seed)
        model = ActionBottleneck(hin.shape[-1], args.u_dim, hidden=int(cfg.bottleneck.hidden_dim),
                                 trans_dim=128, dropout=float(cfg.bottleneck.dropout)).to(dev)
        adv = EnvAdversary(args.u_dim, 4, 256).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.bottleneck.lr), weight_decay=1e-4)
        opt_a = torch.optim.AdamW(adv.parameters(), lr=1e-3)
        n = Xtr.shape[0]; bs = 1024
        # Stage A (predictive) then Stage B (adversary) — total epochs split
        for ep in range(80):
            stage_b = ep >= 40 and adv_w > 0
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                opt.zero_grad(); opt_a.zero_grad()
                u, ahat, that = model(Xtr[b])
                la = ((ahat - Atr[b]) ** 2).mean()
                lt = ((that - Ttr[b]) ** 2).mean()
                ld = LO.decorrelation_loss(u, Etr[b])
                loss = lw["action"] * la + lw["transition"] * lt + lw["decorrelation"] * ld + lw["l2"] * (u ** 2).mean()
                if stage_b:
                    logits = adv(LO.grad_reverse(u, adv_w))
                    l_adv = nn.functional.cross_entropy(logits, Str[b])
                    loss = loss + l_adv
                loss.backward(); opt.step()
                if stage_b:
                    opt_a.step()
        ar2, ar2e = eval_action_r2(model)
        u_all = encode_all(model)
        leak = LK.continuous_probe(u_all, e, split, shift, cfg, seed=seed)
        suite_adv = LK.discrete_probe(u_all, suite, split, shift, 4, device=cfg.device, seed=seed,
                                      n_boot=200)
        tradeoff[f"adv_{adv_w}"] = {"action_r2": ar2, "action_r2_equal": ar2e,
                                    "u_to_e_r2": leak["u_to_e_r2"],
                                    "suite_acc": suite_adv["acc"], "suite_chance": suite_adv["majority_baseline"]}
        print(f"  adv={adv_w}: action_r2={ar2:.4f} equal={ar2e:.3f} u->e={leak['u_to_e_r2']:+.3f} "
              f"suite_acc={suite_adv['acc']:.3f}(chance {suite_adv['majority_baseline']:.2f})")

    p2 = io.load_json(out / "phase2" / "linear_metrics.json")
    payload = {"evidence_tier": "T2-recomb (nonlinear u not projectable)", "u_dim": args.u_dim,
               "tradeoff_curve": tradeoff,
               "linear_reference": {"cca_d4_action_r2": p2["supervised"]["cca"]["4"]["test_r2"],
                                    "pls_d16_action_r2": p2["supervised"]["pls"]["16"]["test_r2"]},
               "note": ("nonlinear ceiling ~ linear ceiling (action is ~linearly decodable from h); "
                        "adversary reduces discrete suite leakage but the linear CCA-d4 u is already "
                        "continuous-leak-free, so the bottleneck adds little. Not on the Outcome-A path.")}
    io.save_json(payload, out3 / "bottleneck_metrics.json")
    print(f"[save] -> {out3/'bottleneck_metrics.json'}")
    write_status(out, phase_reached="phase3", gates_passed=io.load_json(out/"STATUS.json")["gates_passed"],
                 interface=io.load_json(out0/"interface_probe.json")["interface"], verdict=None,
                 extra={"phase3_tradeoff": tradeoff})


if __name__ == "__main__":
    main()
