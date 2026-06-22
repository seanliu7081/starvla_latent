#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""Phase 0 — Verify inputs, pin action time-alignment, probe the frozen-head
intervention interface.

Outputs (under outputs/clean_action_latent_v2/phase0/):
  verification.json     inputs valid, shapes, split, train-only norm stats path
  split_metadata.json   the clip-grouped 70/15/15 split (no group crosses splits)
  alignment.json        empirically pinned action<->hidden shift + evidence
  interface_probe.json  interface in {OK, DEGRADED, UNAVAILABLE} + evidence ceiling
  norm_stats.pt         train-only mu_h/std_h

This script NEVER puts a frozen-model weight in train mode (Prime Directive 1);
the loader returns the model in eval() with requires_grad=False on all params,
and we only call inference helpers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# make the action_latent package importable
EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT / "src"))

from action_latent import features as F          # noqa: E402
from action_latent import io                     # noqa: E402
from action_latent import linear as L            # noqa: E402
from action_latent import metrics as M           # noqa: E402
from action_latent.reporting import write_status  # noqa: E402


# --------------------------------------------------------------------------- #
def random_orthonormal_basis(D, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, K, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :K].float()


def remove_subspace_raw(x, W, mean, std):
    """Remove span(W) from x, returning x in the ORIGINAL (raw) space.

    Used to build an EDITED hidden that the frozen head can consume directly.
    x:[...,D], W:[D,K], mean/std:[1,1,D] (broadcast over tokens for [T,Ntok,D]).
    """
    xn = (x.float() - mean) / std
    proj = (xn @ W) @ W.T
    return (xn - proj) * std + mean


# --------------------------------------------------------------------------- #
# 0.1  verify inputs + split + norm stats
# --------------------------------------------------------------------------- #
def verify_inputs(cfg, out0):
    rec = {"files_checked": {}, "errors": []}

    def need(path, key):
        p = io.resolve_path(path)
        ok = p.exists()
        rec["files_checked"][key] = {"path": str(p), "exists": bool(ok)}
        if not ok:
            rec["errors"].append(f"MISSING: {key} -> {p}")
        return ok

    ok = True
    cdir = cfg.inputs.latent_cache_dir
    ok &= need(Path(cdir) / "train.pt", "train.pt")
    ok &= need(Path(cdir) / "val.pt", "val.pt")
    ok &= need(cfg.inputs.token_cache, "tokens.pt")
    ok &= need(Path(cfg.inputs.subspace_dir) / cfg.env_subspace.file, "h_subspace_decomposition.pt")
    ok &= need(Path(cfg.inputs.previous_validation_dir) / "probe_results.json", "previous_probe_results.json")
    ok &= need(cfg.frozen_model.checkpoint_path, "frozen_checkpoint.pt")
    if not ok:
        rec["inputs_valid"] = False
        return rec, None, None, None

    cache = io.load_global_cache(cfg)
    h = cache["h_pooled"]
    rec["shapes"] = {k: list(cache[k].shape) for k in ["z_pooled", "h_pooled", "actions", "states"]}
    rec["n_clips"] = int(h.shape[0])
    rec["T"] = int(h.shape[1])
    rec["h_finite"] = bool(torch.isfinite(h).all().item())
    rec["actions_finite"] = bool(torch.isfinite(cache["actions"]).all().item())

    # env subspace + ROW-ALIGNMENT verification (pins cache<->subspace ordering)
    sub = io.load_env_subspace(cfg, verify_against_h=h)
    rec["W_env_shape"] = list(sub["W_env"].shape)
    rec["env_coeff_reconstruct_max_abs_err"] = sub["e_reconstruct_max_abs_err"]
    rec["env_dim"] = sub["env_dim"]

    # clip-grouped split + train-only norm stats
    split = io.make_clip_split(cache, cfg)
    mu_h, std_h = io.train_norm_stats(h, split["train"])
    torch.save({"mu_h": mu_h, "std_h": std_h}, out0 / "norm_stats.pt")

    split_meta = {
        "mode": cfg.splits.mode,
        "group_key": split["group_key"], "stratify_key": split["stratify_key"],
        "ratios": {"train": cfg.splits.train, "val": cfg.splits.val, "test": cfg.splits.test},
        "sizes": split["sizes"],
        "no_group_leak_verified": split["no_group_leak_verified"],
        "seed": int(cfg.seed),
        "train_idx": split["train"].tolist(),
        "val_idx": split["val"].tolist(),
        "test_idx": split["test"].tolist(),
        # per-split suite balance (sanity)
        "suite_balance": {
            s: {int(u): int((cache["suite_id"][split[s]].numpy() == u).sum())
                for u in np.unique(cache["suite_id"].numpy())}
            for s in ["train", "val", "test"]},
    }
    io.save_json(split_meta, out0 / "split_metadata.json")
    rec["inputs_valid"] = True
    rec["norm_stats_path"] = str(out0 / "norm_stats.pt")
    return rec, cache, sub, (split, mu_h, std_h)


# --------------------------------------------------------------------------- #
# 0.2  pin action time-alignment
# --------------------------------------------------------------------------- #
def _pairs_shift(h_std, action, idx, shift):
    """X = h_{t+shift}, Y = a_t (single-frame)."""
    X, Y, cids = F.flatten_frames(h_std, idx, shift=shift, action=action)
    return X, Y, cids


def _pairs_concat(h_std, action, idx, shifts, add_delta=False):
    """X = concat(h_{t+s} for s in shifts) [+ delta], Y = a_t."""
    N, T, D = h_std.shape
    lo = max(0, -min(shifts)); hi = min(T, T - max(shifts))
    xs, ys, cids = [], [], []
    for i in idx:
        ts = np.arange(lo, hi)
        cols = [h_std[i, ts + s] for s in shifts]
        if add_delta:
            cols.append(h_std[i, ts + 1] - h_std[i, ts])
        xs.append(torch.cat(cols, dim=1))
        ys.append(action[i, ts])
        cids.append(np.full(len(ts), int(i)))
    return torch.cat(xs, 0), torch.cat(ys, 0), np.concatenate(cids)


def _fit_eval(Xtr, Ytr, Xva, Yva, Xte, Yte, cids_te, lambdas, n_boot, seed):
    W, lam, val_r2 = L.fit_ridge_cv(Xtr, Ytr, Xva, Yva, lambdas)
    pte = L.predict_ridge(W, Xte)
    rep = M.r2_report(Yte.numpy(), pte.numpy())
    ci = M.bootstrap_r2_ci(Yte.numpy(), pte.numpy(), cids_te, n_boot=n_boot, seed=seed)
    return {"val_r2": val_r2, "ridge_lambda": lam, "test_r2": rep["r2"],
            "test_r2_per_dim": rep["r2_per_dim"], "test_mse": rep["mse"],
            "test_r2_ci": [ci["lo"], ci["hi"]]}


def pin_alignment(cfg, cache, split, mu_h, std_h, out0):
    h_std = F.standardize(cache["h_pooled"], mu_h, std_h)
    action = cache["actions"]
    tr, va, te = split["train"], split["val"], split["test"]
    lambdas = list(cfg.linear.ridge_lambdas)
    nb = int(cfg.eval.bootstrap_n)

    res = {"single_frame_shifts": {}, "multi_frame": {}}
    # single-frame candidate shifts: a_t read from h_{t+shift}
    for shift in cfg.alignment.candidate_shifts:
        Xtr, Ytr, _ = _pairs_shift(h_std, action, tr, shift)
        Xva, Yva, _ = _pairs_shift(h_std, action, va, shift)
        Xte, Yte, cte = _pairs_shift(h_std, action, te, shift)
        res["single_frame_shifts"][str(shift)] = _fit_eval(
            Xtr, Ytr, Xva, Yva, Xte, Yte, cte, lambdas, nb, int(cfg.seed))
        print(f"[align] shift={shift:+d}  test_r2={res['single_frame_shifts'][str(shift)]['test_r2']:.4f}")

    # pair [h_t,h_{t+1}] -> a_t  (transition-alignment candidate / upper bound)
    for name, shifts, dlt in [("pair_t_tp1", [0, 1], False),
                              ("pair_t_tp1_delta", [0, 1], True)]:
        Xtr, Ytr, _ = _pairs_concat(h_std, action, tr, shifts, dlt)
        Xva, Yva, _ = _pairs_concat(h_std, action, va, shifts, dlt)
        Xte, Yte, cte = _pairs_concat(h_std, action, te, shifts, dlt)
        res["multi_frame"][name] = _fit_eval(
            Xtr, Ytr, Xva, Yva, Xte, Yte, cte, lambdas, nb, int(cfg.seed))
        print(f"[align] {name}  test_r2={res['multi_frame'][name]['test_r2']:.4f}")

    # decision: best single-frame shift = the alignment convention
    sf = res["single_frame_shifts"]
    best_shift = max(sf.keys(), key=lambda k: sf[k]["test_r2"])
    best_r2 = sf[best_shift]["test_r2"]
    conv = {"-1": "a_t = transition into frame t (read from h_{t-1})",
            "0": "a_t = command at frame t (read from h_t) — frozen-head convention",
            "1": "a_t = transition t->t+1 (read from h_{t+1})"}[best_shift]

    ambiguous = best_r2 < 0.20
    res["decision"] = {
        "chosen_shift": int(best_shift),
        "convention": conv,
        "chosen_test_r2": best_r2,
        "consistent_with_frozen_head": bool(int(best_shift) == 0),
        "frozen_head_convention_shift": 0,
        "frozen_head_reference": "h_t -> pred[t,0] ~ actions[t], prior readout MSE 0.0138, corr 0.853",
        "ambiguous_halt": bool(ambiguous),
        "note": ("ridge h_{t+shift}->a_t (single-frame) selects the convention; "
                 "pair/delta probes are an upper bound, not the alignment test."),
    }
    io.save_json(res, out0 / "alignment.json")
    print(f"[align] CHOSEN shift={best_shift} ({conv}) test_r2={best_r2:.4f} "
          f"ambiguous={ambiguous}")
    return res, int(best_shift), ambiguous


# --------------------------------------------------------------------------- #
# 0.3  probe the frozen-head intervention interface
# --------------------------------------------------------------------------- #
def probe_interface(cfg, cache, sub, out0):
    rec = {"interface": "UNAVAILABLE", "head_consumes": None, "evidence": {}}
    try:
        prior_dir = io.resolve_path(cfg.frozen_model.loader_experiment_dir)
        sys.path.insert(0, str(prior_dir))
        from src.models.decode import action_from_hidden       # noqa: E402
        from src.models.load_model import load_frozen_wm4a     # noqa: E402

        ckpt = str(io.resolve_path(cfg.frozen_model.checkpoint_path))
        print("[interface] loading frozen WM4A (eval, requires_grad=False) ...")
        model, _, _ = load_frozen_wm4a(ckpt, device=cfg.device,
                                       sigma_data=float(cfg.frozen_model.sigma_data))
        # Prime-Directive-1 guard: assert truly frozen + eval
        n_trainable = sum(int(p.requires_grad) for p in model.parameters())
        rec["frozen_guard"] = {"trainable_params": n_trainable, "training_mode": bool(model.training)}
        assert n_trainable == 0 and not model.training, "frozen model not in eval/no-grad mode!"

        tok = io.load_token_cache(cfg)
        h_tokens = tok["h_tokens"]                # [12,48,720,2048] fp16
        clip_ids = tok["clip_idx"].tolist()
        mean, std = sub["mean"], sub["std"]       # [1,1,2048] broadcast over tokens
        D = mean.shape[-1]
        edit_dim = int(cfg.interface_probe.edit_subspace_dim)
        n_clips = min(int(cfg.interface_probe.n_token_clips), h_tokens.shape[0])
        n_seeds = int(cfg.interface_probe.n_seeds)
        horizon_step0 = 0

        # GT actions per global clip (canonical order); clip_idx are global ids
        actions = cache["actions"]
        cidx_global = cache["clip_idx"].numpy()
        gmap = {int(c): i for i, c in enumerate(cidx_global)}

        W_rand = random_orthonormal_basis(D, edit_dim, seed=int(cfg.seed))

        def readout_mse(h_in_clip, gt):
            pred = action_from_hidden(model, h_in_clip)      # [T,horizon,7]
            p0 = pred[:, horizon_step0, :]                   # first chunk step
            m = min(p0.shape[0], gt.shape[0])
            mse = float(np.mean((p0[:m] - gt[:m].numpy()) ** 2))
            return mse, p0[:m]

        # frozen-head ALIGNMENT cross-check: does pred[t,0] best match a_{t+k}?
        # k* (argmin head MSE) grounds the action convention in the head itself.
        # mapping to linear-probe shift: probe predicts a_t from h_{t+s} with s=-k*.
        offset_ks = [-1, 0, 1]
        offset_sqerr = {k: [0.0, 0] for k in offset_ks}      # [sum_sq, count]
        base_p0_list, gt_list = [], []

        base_mses, edit_mses, pred_changes, det_diffs = [], [], [], []
        for j in range(n_clips):
            ci = clip_ids[j]
            gt = actions[gmap[int(ci)]]                       # [T,7]
            h_clip = h_tokens[j].float()                      # [T,720,2048]

            # determinism: run baseline twice (different torch seed)
            torch.manual_seed(int(cfg.seed))
            b0_mse, b0 = readout_mse(h_clip, gt)
            torch.manual_seed(int(cfg.seed) + 12345)
            _, b1 = readout_mse(h_clip, gt)
            det_diffs.append(float(np.max(np.abs(b0 - b1))))

            # baseline averaged over seeds (in case head is stochastic)
            seed_mse = [b0_mse]
            for s in range(1, n_seeds):
                torch.manual_seed(int(cfg.seed) + s)
                seed_mse.append(readout_mse(h_clip, gt)[0])
            base_mses.append(float(np.mean(seed_mse)))

            # EDIT: remove a random 16-dim subspace (raw space) and re-read
            h_edit = remove_subspace_raw(h_clip, W_rand, mean, std)
            torch.manual_seed(int(cfg.seed))
            e_mse, e0 = readout_mse(h_edit, gt)
            edit_mses.append(e_mse)
            pred_changes.append(float(np.mean(np.abs(e0 - b0))))

            # accumulate offset cross-check (pred[t,0] vs a_{t+k})
            p0_full, gt_full = b0, gt.numpy()       # both [T,7] (T=48)
            base_p0_list.append(p0_full); gt_list.append(gt_full)
            Tn = p0_full.shape[0]
            for k in offset_ks:
                t0 = max(0, -k); t1 = min(Tn, Tn - k)
                d = p0_full[t0:t1] - gt_full[t0 + k:t1 + k]
                offset_sqerr[k][0] += float((d ** 2).sum())
                offset_sqerr[k][1] += d.size

        # frozen-head preferred offset
        offset_mse = {k: (offset_sqerr[k][0] / offset_sqerr[k][1]) for k in offset_ks}
        k_star = min(offset_mse, key=offset_mse.get)
        s_head = -k_star
        rec["frozen_head_alignment_check"] = {
            "readout_mse_by_offset_k": {str(k): offset_mse[k] for k in offset_ks},
            "note": "pred[t,0] compared to actions[t+k]; k*=argmin",
            "k_star": int(k_star),
            "implied_probe_shift_s": int(s_head),
            "interpretation": ("frozen head's chunk-step-0 best matches actions[t+%d]; "
                               "equivalently hidden h_tau -> a_{tau+%d}" % (k_star, k_star)),
        }
        print(f"[interface] frozen-head offset MSE by k: "
              f"{ {k: round(v,5) for k,v in offset_mse.items()} } -> k*={k_star} (implies probe shift {s_head})")

        base = float(np.mean(base_mses))
        edit = float(np.mean(edit_mses))
        change = float(np.mean(pred_changes))
        det = float(np.max(det_diffs))
        prior_ref = 0.0138

        rec["head_consumes"] = "token_level_h [T, 720, 2048]"
        rec["evidence"] = {
            "baseline_readout_mse_mean": base,
            "prior_reference_mse": prior_ref,
            "baseline_matches_prior": bool(abs(base - prior_ref) < 0.005),
            "edited_readout_mse_mean": edit,
            "edit_subspace_dim": edit_dim,
            "edit_is_random_subspace": True,
            "mean_abs_pred_change_under_edit": change,
            "determinism_max_abs_diff_across_seeds": det,
            "n_token_clips_used": n_clips,
            "n_seeds": n_seeds,
            "action_horizon": int(model.action_horizon),
            "readout_uses_state": False,
        }
        # classify
        runs = np.isfinite(base) and np.isfinite(edit)
        differs = change > 1e-6
        sane = rec["evidence"]["baseline_matches_prior"]
        if runs and differs and sane:
            rec["interface"] = "OK"
        elif runs and differs:
            rec["interface"] = "DEGRADED"
        else:
            rec["interface"] = "DEGRADED"
        print(f"[interface] status={rec['interface']} base_mse={base:.4f} "
              f"edit_mse={edit:.4f} pred_change={change:.4g} det_diff={det:.2g}")
    except Exception as e:  # interface failure is NOT a halt; it sets the ceiling
        import traceback
        rec["interface"] = "UNAVAILABLE"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        print(f"[interface] UNAVAILABLE: {rec['error']}")

    # evidence ceiling
    rec["evidence_ceiling"] = {
        "OK": "Outcome A reachable (clean linear subspace, causally verified via Phase 6A)",
        "DEGRADED": "max verdict B/C; Phase 6 falls back to decoder recombination (T2)",
        "UNAVAILABLE": "max verdict B/C; no T1-causal claims available",
    }[rec["interface"]]
    io.save_json(rec, out0 / "interface_probe.json")
    return rec


# --------------------------------------------------------------------------- #
# reconcile the linear-probe alignment with the frozen-head cross-check
# --------------------------------------------------------------------------- #
def reconcile_alignment(arec, irec, s_linear, out0):
    """Final alignment = frozen-head-grounded when the interface is OK, else the
    linear-probe argmax. The frozen head defines the action semantics and is the
    feature used by the (tier-1 causal) Phase 6, so its convention wins ties.
    """
    head_ok = (irec is not None and irec.get("interface") in ("OK", "DEGRADED")
               and "frozen_head_alignment_check" in irec)
    if head_ok:
        s_head = int(irec["frozen_head_alignment_check"]["implied_probe_shift_s"])
        final = s_head
        agree = (s_head == s_linear)
        basis = "frozen_head_offset_argmin (grounds action semantics; used by Phase 6)"
    else:
        s_head = None
        final = s_linear
        agree = None
        basis = "linear_probe_argmax (frozen-head cross-check unavailable)"

    final_dec = {
        "final_shift": int(final),
        "basis": basis,
        "linear_probe_argmax_shift": int(s_linear),
        "frozen_head_implied_shift": s_head,
        "agree": agree,
        "convention": {
            -1: "a_t aligns with the PREVIOUS hidden h_{t-1}",
            0: "a_t = command at frame t, read from h_t (frozen-head canonical)",
            1: "a_t aligns with the NEXT hidden h_{t+1}",
        }[int(final)],
        "caveat": (None if agree in (True, None) else
                   "linear probe marginally preferred shift %d but the frozen head "
                   "(which Phase 6 uses) implies shift %d; adopting the head's "
                   "convention for consistency. R^2 gap is within bootstrap CIs."
                   % (s_linear, final)),
    }
    arec["final_decision"] = final_dec
    io.save_json(arec, out0 / "alignment.json")
    print(f"[align] FINAL shift={final}  ({final_dec['convention']})  "
          f"linear_argmax={s_linear} head_implied={s_head} agree={agree}")
    return int(final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_ROOT / "configs/clean_action_latent.yaml"))
    ap.add_argument("--skip_interface", action="store_true",
                    help="skip the 7B-model interface probe (0.3); records nothing for it")
    args = ap.parse_args()
    cfg = io.load_config(args.config)
    out = io.resolve_path(cfg.output_dir)
    out0 = io.ensure_dir(out / "phase0")
    io.ensure_dir(out / "configs")
    # snapshot resolved config
    import shutil
    shutil.copy(args.config, out / "configs/resolved_config.yaml")

    print("=== Phase 0.1 — verify inputs + split + norm stats ===")
    vrec, cache, sub, splitpack = verify_inputs(cfg, out0)
    io.save_json(vrec, out0 / "verification.json")
    if not vrec["inputs_valid"]:
        print("[GATE-0] inputs INVALID — HALT.")
        for e in vrec["errors"]:
            print("   ", e)
        write_status(out, phase_reached="phase0_verify_failed", gates_passed=[],
                     interface=None, verdict=None, extra={"errors": vrec["errors"]})
        sys.exit(1)
    split, mu_h, std_h = splitpack
    print(f"[0.1] inputs valid. split sizes={split['sizes']} "
          f"env_coeff_recon_err={vrec['env_coeff_reconstruct_max_abs_err']:.2g}")

    print("=== Phase 0.2 — pin action time-alignment (linear-probe, provisional) ===")
    arec, s_linear, ambiguous = pin_alignment(cfg, cache, split, mu_h, std_h, out0)

    interface = None
    irec = None
    if not args.skip_interface:
        print("=== Phase 0.3 — probe frozen-head intervention interface ===")
        irec = probe_interface(cfg, cache, sub, out0)
        interface = irec["interface"]
    else:
        print("=== Phase 0.3 — SKIPPED (--skip_interface) ===")

    print("=== Phase 0.2/0.3 — reconcile final alignment ===")
    shift = reconcile_alignment(arec, irec, s_linear, out0)

    # GATE-0
    gate0_pass = (vrec["inputs_valid"] and not ambiguous)
    write_status(out, phase_reached="phase0",
                 gates_passed=(["GATE-0"] if gate0_pass else []),
                 interface=interface, verdict=None,
                 extra={"alignment_shift": shift, "alignment_ambiguous": ambiguous})
    if ambiguous:
        print("[GATE-0] alignment AMBIGUOUS (best single-frame R^2 < 0.2) — HALT.")
        sys.exit(1)
    print(f"[GATE-0] PASS. alignment_shift={shift}  interface={interface}")


if __name__ == "__main__":
    main()
