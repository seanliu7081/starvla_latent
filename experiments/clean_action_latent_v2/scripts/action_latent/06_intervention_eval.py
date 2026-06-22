#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""Phase 6 — Intervention / swap-readout (tier-1 causal).

Branch on Phase-0 interface status. interface==OK -> 6A projective intervention:
feed token-h with subspaces removed into the FROZEN GR00T head and measure the
action-readout MSE increase. PAIRED-NOISE (shared seed across conditions) makes the
small causal effect detectable despite the stochastic head.

PASS (real action subspace): removing the supervised action subspace P_act hurts
the frozen head's action MSE MORE than removing a matched-d random subspace AND
more than removing the previous high-freq subspace. [T1-causal]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT / "src"))

from action_latent import intervention as IV     # noqa: E402
from action_latent import io                     # noqa: E402
from action_latent.reporting import write_status  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_ROOT / "configs/clean_action_latent.yaml"))
    ap.add_argument("--n_seeds", type=int, default=10)
    args = ap.parse_args()
    cfg = io.load_config(args.config)
    out = io.resolve_path(cfg.output_dir)
    out0, out6 = out / "phase0", io.ensure_dir(out / "phase6")
    seed = int(cfg.seed)

    iface = io.load_json(out0 / "interface_probe.json")["interface"]
    print(f"[phase6] interface = {iface}")
    if iface != "OK":
        rec = {"interface": iface, "mode": "skipped_6A",
               "note": "interface != OK -> projective tier-1 intervention unavailable; "
                       "Phase 6B decoder-recombination (tier-2) would be the fallback."}
        io.save_json(rec, out6 / "intervention.json")
        write_status(out, phase_reached="phase6", gates_passed=io.load_json(out/"STATUS.json")["gates_passed"],
                     interface=iface, verdict=None, extra={"phase6": "skipped_6A"})
        return

    # ---- load frozen model + decode helper (reuse validated loader) ----
    prior_dir = io.resolve_path(cfg.frozen_model.loader_experiment_dir)
    sys.path.insert(0, str(prior_dir))
    from src.models.decode import action_from_hidden       # noqa
    from src.models.load_model import load_frozen_wm4a      # noqa
    print("[phase6] loading frozen WM4A ...")
    model, _, _ = load_frozen_wm4a(str(io.resolve_path(cfg.frozen_model.checkpoint_path)),
                                   device=cfg.device, sigma_data=float(cfg.frozen_model.sigma_data))
    assert sum(int(p.requires_grad) for p in model.parameters()) == 0 and not model.training

    def head(model, h_in):  # returns numpy [T,horizon,7]
        return action_from_hidden(model, h_in)

    # ---- subspaces + token cache ----
    sp = torch.load(out / "phase2" / "linear_subspaces.pt", weights_only=False)
    Wenv, Wprev = sp["W_env"], sp["W_act_highfreq"]
    mean, std = sp["sub_mean"], sp["sub_std"]
    D = Wenv.shape[0]
    sup_list = [("cca4", sp["W_act_sup_chosen"]), ("pls16", sp["pls_d16"])]

    tok = io.load_token_cache(cfg)
    h_tokens = tok["h_tokens"]                                 # [12,48,720,2048]
    clip_ids = tok["clip_idx"].tolist()
    cache = io.load_global_cache(cfg)
    gmap = {int(c): i for i, c in enumerate(cache["clip_idx"].numpy())}
    gt = [cache["actions"][gmap[int(c)]].numpy() for c in clip_ids]   # list of [T,7]

    conds = IV.build_conditions(Wenv, sup_list, Wprev, mean, std, D, seed)
    print(f"[phase6] conditions: {list(conds.keys())}  n_seeds={args.n_seeds}  clips={len(clip_ids)}")
    res = IV.run_intervention(model, head, h_tokens, gt, conds, mean, std,
                              n_seeds=int(args.n_seeds), base_seed=seed)

    # ---- pass criteria per supervised candidate ----
    verdicts = {}
    for nm, _ in sup_list:
        ns = f"no_sup_{nm}"
        nr = [k for k in conds if k.startswith(f"no_random_{nm}_")][0]
        vs_rand = IV.bootstrap_delta_diff(res[ns]["delta_per_clip"], res[nr]["delta_per_clip"], seed=seed)
        vs_prev = IV.bootstrap_delta_diff(res[ns]["delta_per_clip"], res["no_prev_highfreq"]["delta_per_clip"], seed=seed)
        verdicts[nm] = {
            "delta_no_sup": res[ns]["delta_vs_original_mean"],
            "delta_no_random_matched": res[nr]["delta_vs_original_mean"],
            "delta_no_prev_highfreq": res["no_prev_highfreq"]["delta_vs_original_mean"],
            "delta_no_env": res["no_env"]["delta_vs_original_mean"],
            "sup_minus_random": vs_rand, "sup_minus_prev": vs_prev,
            "causal_pass": bool(vs_rand["excludes_zero"] and vs_rand["point"] > 0
                                and vs_prev["point"] > 0),
        }
        print(f"  [{nm}] Δno_sup={res[ns]['delta_vs_original_mean']:+.5f} "
              f"Δno_rand={res[nr]['delta_vs_original_mean']:+.5f} "
              f"Δno_prev={res['no_prev_highfreq']['delta_vs_original_mean']:+.5f} "
              f"Δno_env={res['no_env']['delta_vs_original_mean']:+.5f} | "
              f"sup>rand={vs_rand['point']:+.5f} excl0={vs_rand['excludes_zero']} -> "
              f"PASS={verdicts[nm]['causal_pass']}")

    any_pass = any(v["causal_pass"] for v in verdicts.values())
    payload = {"evidence_tier": "T1-causal", "interface": iface, "n_seeds": int(args.n_seeds),
               "n_token_clips": len(clip_ids),
               "per_condition": res, "verdicts": verdicts, "any_causal_pass": any_pass,
               "note": ("Action info is redundant across h's 2048 dims (prior found subspace "
                        "removals barely move the readout); paired-noise design + many seeds is "
                        "required to resolve the small deltas. A null result here does NOT negate "
                        "the strong correlational+leakage evidence — it caps the verdict below T1.")}
    io.save_json(payload, out6 / "intervention.json")
    print(f"\n[phase6] any_causal_pass={any_pass}")
    write_status(out, phase_reached="phase6", gates_passed=io.load_json(out/"STATUS.json")["gates_passed"],
                 interface=iface, verdict=None,
                 extra={"phase6_any_causal_pass": any_pass, "phase6_verdicts": verdicts})
    print(f"[save] -> {out6/'intervention.json'}")


if __name__ == "__main__":
    main()
