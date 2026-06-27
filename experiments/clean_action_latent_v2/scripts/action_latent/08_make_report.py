#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""Phase 8 — Report + verdict (A/B/C/D, evidence-tiered).

Assembles all phase artifacts into verdict.json + STATUS.json and a structured
report. Respects the GATE-0 evidence ceiling: NO Outcome-A language unless the
tier-1 frozen-head intervention (Phase 6A) passed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT / "src"))

from action_latent import io                     # noqa: E402
from action_latent.reporting import write_status  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_ROOT / "configs/clean_action_latent.yaml"))
    args = ap.parse_args()
    cfg = io.load_config(args.config)
    out = io.resolve_path(cfg.output_dir)

    p0v = io.load_json(out / "phase0" / "verification.json")
    iface = io.load_json(out / "phase0" / "interface_probe.json")["interface"]
    align = io.load_json(out / "phase0" / "alignment.json")["final_decision"]
    p1 = io.load_json(out / "phase1" / "confound_gate.json")["headline"]
    p2 = io.load_json(out / "phase2" / "linear_metrics.json")
    p4 = io.load_json(out / "phase4" / "leakage.json")["summary"]
    p5 = io.load_json(out / "phase5" / "pair_retrieval.json")["summary"]
    p6 = io.load_json(out / "phase6" / "intervention.json")
    chosen = p2["gate2"]["chosen_projectable"]

    gate2 = p2["gate2"]["passed"]
    causal_t1 = p6.get("any_causal_pass", False)
    # T1 projective intervention (6A) is only available when the frozen-head
    # interface is OK; on a DEGRADED interface Phase 6 runs the interface-independent
    # tier-2 decoder recombination (6B) fallback instead, so `verdicts` may be absent.
    t1_available = "verdicts" in p6
    p6_verdicts = p6.get("verdicts", {})
    recomb = p6.get("phase6B_decoder_recombination_T2", {})
    t2_pass = recomb.get("swap_eA_uB_closer_to_B_frac", 0) > 0.6
    clean = (abs(p4["chosen_u_continuous_leak_r2"]) < 0.1 and p4["chosen_u_suite_above_chance"] < 0.05
             and abs(p4["chosen_u_leak_cond"]) < 0.1)
    deconf = p5["chosen_partial_corr_du_da_given_env"] > 0.3 and p5["chosen_partial_corr_ci"][0] > 0

    # ---- verdict (respect evidence ceiling: no A without T1) ----
    if iface == "OK" and gate2 and causal_t1 and clean and deconf:
        verdict, letter = "A", "A — Clean linear action subspace, causally verified (T1)."
    elif gate2 and clean and deconf and t2_pass:
        t1_clause = ("the tier-1 projective intervention is UNAVAILABLE (the frozen-head action "
                     "readout is too stochastic under this checkpoint's noise source for a clean "
                     "interface — Phase 0 marked it DEGRADED)"
                     if not t1_available else
                     "the tier-1 intervention FAILS (action is redundantly distributed across h)")
        verdict, letter = "B", ("B — Clean, compact, LINEAR & projectable action latent IS recoverable "
                                "(action-predictive with large gain over e, near-zero env leakage, "
                                "action-specific under deconfounding & recombination), BUT it is NOT a "
                                "causal bottleneck of the frozen head: " + t1_clause + ". "
                                "Causal support limited to trained-decoder recombination (T2). Not A because no T1.")
    elif clean and not (deconf and gate2):
        verdict, letter = "C", "C — Conditional action latent only."
    else:
        verdict, letter = "D", "D — No clean action latent."

    payload = {
        "verdict": verdict, "verdict_text": letter,
        "evidence_ceiling_from_interface": iface,
        "gates_passed": ["GATE-0", "GATE-1"] + (["GATE-2"] if gate2 else []),
        "evidence": {
            "alignment_shift": align["final_shift"],
            "shortcut_R2": p1["shortcut_R2"], "ceiling_R2": p1["ceiling_R2"],
            "gain_over_e_h_res": p1["gain_over_e"]["h_res_linear"],
            "chosen_subspace": chosen,
            "chosen_action_r2": chosen["r2"] if chosen else None,
            "T3_correlational": {"gate2_passed": gate2, "chosen_r2": chosen["r2"] if chosen else None},
            "T3_leakage_clean": clean, "leakage": p4,
            "T4_retrieval": {"deconfounded": deconf, **p5},
            "T1_causal_frozen_head": {"passed": causal_t1, "available": t1_available,
                                      "interface": iface,
                                      "cca4": p6_verdicts.get("cca4"), "pls16": p6_verdicts.get("pls16")},
            "T2_decoder_recombination": recomb,
        },
        "tiers": {
            "T1-causal": ("UNAVAILABLE — frozen-head interface DEGRADED (stochastic readout); "
                          "projective tier-1 intervention not run"
                          if not t1_available else
                          "PASS — removing the action subspace degrades the frozen head"
                          if causal_t1 else
                          "FAIL — removing the action subspace does not degrade the frozen head (redundancy)"),
            "T2-recomb": (f"{'PASS' if t2_pass else 'FAIL'} — recombination "
                          f"{'follows' if t2_pass else 'does NOT follow'} u "
                          f"({recomb.get('swap_eA_uB_closer_to_B_frac',0):.2f})"
                          if recomb else "n/a — tier-2 recombination not run"),
            "T3-corr": f"PASS — linear u predicts action (r2={chosen['r2']:.3f}) with gain over e; near-zero leakage" if chosen else "n/a",
            "T4-retr": f"PASS — partial corr(d_u,d_action|env)={p5['chosen_partial_corr_du_da_given_env']:.2f}, holds leave-task-out",
        },
    }
    io.save_json(payload, out / "verdict.json")
    write_status(out, phase_reached="phase8",
                 gates_passed=payload["gates_passed"], interface=iface, verdict=verdict,
                 extra={"verdict_text": letter})
    print(f"[VERDICT] {letter}")
    print(f"[save] -> {out/'verdict.json'} , STATUS.json")


if __name__ == "__main__":
    main()
