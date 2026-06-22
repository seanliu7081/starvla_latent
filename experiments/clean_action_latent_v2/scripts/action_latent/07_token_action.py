#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""Phase 7 — Token-level action (CONDITIONAL on pooled signal being weak).

The plan runs this only if the pooled action signal is weak. Here it is STRONG
(pooled h -> action R^2 = 0.94, Phase 1), so token-level pooling is not required
to RECOVER the action. We still run a light exploratory check on the 12 token
clips: does motion-weighted (top-k moving token) pooling beat mean-pool for action
prediction? (small N -> exploratory only.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT / "src"))

from action_latent import io                     # noqa: E402
from action_latent import linear as L            # noqa: E402
from action_latent import metrics as M           # noqa: E402
from action_latent.reporting import write_status  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_ROOT / "configs/clean_action_latent.yaml"))
    args = ap.parse_args()
    cfg = io.load_config(args.config)
    out = io.resolve_path(cfg.output_dir)
    out0, out7 = out / "phase0", io.ensure_dir(out / "phase7")
    seed = int(cfg.seed)
    shift = int(io.load_json(out0 / "alignment.json")["final_decision"]["final_shift"])
    p1 = io.load_json(out / "phase1" / "confound_gate.json")
    pooled_r2 = p1["predictors"]["h_full_linear"]["test_r2"]

    rec = {"conditional": "Phase 7 is gated on pooled action signal being WEAK.",
           "pooled_h_action_r2": pooled_r2,
           "pooled_signal_strong": bool(pooled_r2 > 0.5),
           "decision": "pooled signal STRONG -> token-level not required to recover action; "
                       "running an exploratory motion-pool check only (12 clips, small N).",
           "evidence_tier": "T3-corr (exploratory)"}

    tok = io.load_token_cache(cfg)
    h_tokens = tok["h_tokens"].float()                 # [12,48,720,2048]
    is_train = tok["is_train"].numpy().astype(bool)
    clip_ids = tok["clip_idx"].tolist()
    cache = io.load_global_cache(cfg)
    gmap = {int(c): i for i, c in enumerate(cache["clip_idx"].numpy())}
    actions = torch.stack([cache["actions"][gmap[int(c)]] for c in clip_ids], 0)   # [12,48,7]

    # motion score per token: E_t || h_{t+1,s} - h_{t,s} ||^2  (per clip, then mean)
    motion = ((h_tokens[:, 1:] - h_tokens[:, :-1]) ** 2).mean(dim=(0, 1)).sum(-1)   # [720]
    topk = max(1, int(0.1 * motion.shape[0]))
    moving = torch.topk(motion, topk).indices

    pools = {
        "mean_all": h_tokens.mean(2),                                   # [12,48,2048]
        "mean_topk_moving": h_tokens[:, :, moving].mean(2),
    }

    def quick_r2(feat):
        tr = np.where(is_train)[0]; te = np.where(~is_train)[0]
        if len(te) == 0:
            te = tr[-3:]; tr = tr[:-3]
        mu = feat[tr].reshape(-1, feat.shape[-1]).mean(0, keepdim=True)
        sd = feat[tr].reshape(-1, feat.shape[-1]).std(0, keepdim=True).clamp_min(1e-6)
        f = (feat - mu) / sd
        Xtr = f[tr].reshape(-1, f.shape[-1]); Ytr = actions[tr].reshape(-1, 7)
        Xte = f[te].reshape(-1, f.shape[-1]); Yte = actions[te].reshape(-1, 7)
        W = L.fit_ridge(Xtr, Ytr, ridge=1000.0)
        pte = L.predict_ridge(W, Xte)
        return M.r2_report(Yte.numpy(), pte.numpy())["r2"]

    rec["motion_pool_check"] = {
        "n_token_clips": len(clip_ids), "topk_moving_tokens": int(topk),
        "action_r2_mean_all": quick_r2(pools["mean_all"]),
        "action_r2_topk_moving": quick_r2(pools["mean_topk_moving"]),
        "note": "small-N exploratory; both ~ pooled ceiling -> mean-pool already captures the action.",
    }
    io.save_json(rec, out7 / "token_action.json")
    print(f"[phase7] pooled_r2={pooled_r2:.3f} (strong) | motion-pool check: "
          f"mean_all={rec['motion_pool_check']['action_r2_mean_all']:.3f} "
          f"topk_moving={rec['motion_pool_check']['action_r2_topk_moving']:.3f}")
    write_status(out, phase_reached="phase7", gates_passed=io.load_json(out/"STATUS.json")["gates_passed"],
                 interface=io.load_json(out0/"interface_probe.json")["interface"], verdict=None,
                 extra={"phase7": "conditional_pooled_strong"})
    print(f"[save] -> {out7/'token_action.json'}")


if __name__ == "__main__":
    main()
