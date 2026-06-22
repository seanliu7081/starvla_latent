#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""Phase 2/3/4/6 — frequency analysis (FFT, STFT, token-frequency) per layer."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from src.analysis.fft_metrics import compute_fft_metrics, low_ratio_of_signal  # noqa
from src.analysis.preprocessing import remove_temporal_mean, standardize_latents  # noqa
from src.analysis.stft_metrics import compute_local_high_frequency_score  # noqa
from src.analysis.token_frequency import compute_token_frequency_scores  # noqa
from src.utils import visualization as V  # noqa
from src.utils.common import ensure_dir, load_config, load_json, load_pt, resolve_path, save_json, save_pt  # noqa


def load_layer(cache_dir, layer):
    tr = load_pt(cache_dir / "train.pt"); va = load_pt(cache_dir / "val.pt")
    key = f"{layer}_pooled"
    Z = torch.cat([tr[key], va[key]], dim=0).float()      # [N,T,D]
    actions = torch.cat([tr["actions"], va["actions"]], dim=0).float()
    return Z, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP / "configs/experiment.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_path(cfg.output_dir)
    cache = out / "latent_cache"
    fdir = ensure_dir(out / "frequency_analysis"); pdir = ensure_dir(fdir / "plots")
    meta = load_json(cache / "metadata.json")
    fps = float(meta["fps"]); cutoff = float(cfg.frequency.cutoff_hz)

    summary = {"fps": fps, "cutoff_hz": cutoff, "layers": {}}

    # --- frequency content of the ACTION signal itself (key interpretive control) ---
    tr = load_pt(cache / "train.pt"); va = load_pt(cache / "val.pt")
    act_all = torch.cat([tr["actions"], va["actions"]], dim=0).float()   # [N,T,7]
    act_fft = compute_fft_metrics(act_all, fps=fps, cutoff_hz=cutoff)
    summary["action_signal"] = {
        "mean_centroid_hz": float(act_fft["centroid_hz"].mean()),
        "mean_low_ratio": float(act_fft["low_ratio"].mean()),
        "per_dim_centroid_hz": [round(x, 3) for x in act_fft["centroid_hz"].tolist()],
        "note": "If action energy is mostly low-frequency, the 'action=high-freq' premise fails.",
    }
    print(f"[action] mean centroid={act_fft['centroid_hz'].mean():.2f}Hz "
          f"mean low_ratio(<{cutoff}Hz)={act_fft['low_ratio'].mean():.2f}")

    for layer in cfg.extraction.layers:
        Z, actions = load_layer(cache, layer)
        N, T, D = Z.shape
        Zn, _, _ = standardize_latents(Z)
        Zn = remove_temporal_mean(Zn)

        fft = compute_fft_metrics(Zn, fps=fps, cutoff_hz=cutoff, remove_dc=False)
        save_pt(fft, fdir / f"fft_metrics_{layer}.pt")

        # sanity: ratios in [0,1], sum~1
        lr, hr = fft["low_ratio"], fft["high_ratio"]
        assert (lr >= -1e-4).all() and (lr <= 1 + 1e-4).all()
        assert torch.allclose(lr + hr, torch.ones_like(lr), atol=1e-3)

        # plots: histograms
        V.hist(lr.numpy(), pdir / f"low_ratio_hist_{layer}.png",
               f"[{layer}] low-freq energy ratio (cutoff {cutoff} Hz)", "low_ratio")
        V.hist(fft["centroid_hz"].numpy(), pdir / f"spectral_centroid_hist_{layer}.png",
               f"[{layer}] spectral centroid", "Hz")

        # example dim curves: top-3 slow & top-3 fast dims, first clip
        slow = torch.argsort(lr, descending=True)[:3].tolist()
        fast = torch.argsort(hr, descending=True)[:3].tolist()
        curves = [Zn[0, :, d].numpy() for d in slow] + [Zn[0, :, d].numpy() for d in fast]
        labels = [f"slow d{d}" for d in slow] + [f"fast d{d}" for d in fast]
        V.lines(curves, pdir / f"example_dim_curves_{layer}.png",
                f"[{layer}] example slow/fast dim trajectories (clip 0)", labels=labels)

        # STFT local action score for a few clips + action-speed overlay
        stft = compute_local_high_frequency_score(
            Zn, fps=fps, n_fft=int(cfg.frequency.stft_n_fft),
            hop_length=int(cfg.frequency.stft_hop_length),
            high_cutoff_hz=float(cfg.frequency.high_cutoff_hz))   # [N,W]
        save_pt({"stft_high_score": stft}, fdir / f"stft_scores_{layer}.pt")
        # action speed per frame = L2 of action delta (proxy for motion intensity)
        act_speed = (actions[:, 1:] - actions[:, :-1]).norm(dim=-1)  # [N,T-1]
        ex = list(range(min(3, N)))
        hop = int(cfg.frequency.stft_hop_length)
        for i in ex:
            w = stft[i].numpy()
            w_frames = np.arange(len(w)) * hop
            sc = (w - w.min()) / (w.max() - w.min() + 1e-8)
            a = act_speed[i].numpy(); a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            V.lines([np.interp(np.arange(T), w_frames, sc), np.concatenate([[a[0]], a])],
                    pdir / f"stft_vs_action_{layer}_clip{i}.png",
                    f"[{layer}] STFT high-freq score vs action speed (clip {i})",
                    labels=["STFT high-freq (norm)", "action speed (norm)"])

        # cutoff sweep (sensitivity): mean low-ratio of the whole signal vs cutoff
        sweep = {float(c): low_ratio_of_signal(Zn, fps, float(c), remove_dc=False)
                 for c in cfg.frequency.cutoff_sweep}

        summary["layers"][layer] = {
            "D": int(D), "mean_low_ratio": float(lr.mean()),
            "mean_high_ratio": float(hr.mean()), "mean_centroid_hz": float(fft["centroid_hz"].mean()),
            "frac_dims_low_dominant": float((lr > 0.5).float().mean()),
            "top_slow_dims": slow, "top_fast_dims": fast,
            "cutoff_sweep_low_ratio": sweep,
        }
        print(f"[{layer}] D={D} mean_low_ratio={lr.mean():.3f} centroid={fft['centroid_hz'].mean():.3f}Hz "
              f"frac_low_dominant={(lr>0.5).float().mean():.2f}")

    # token-frequency heatmaps
    tok_path = cache / "tokens.pt"
    if tok_path.exists():
        tok = load_pt(tok_path)
        tfreq = {}
        for layer, key, grid in [("z", "z_tokens", tuple(tok["z_grid"])),
                                 ("h", "h_tokens", tuple(tok["h_grid"]))]:
            if key not in tok:
                continue
            tz = tok[key].float()          # [Nt,T,S,D]
            low, high = compute_token_frequency_scores(tz, fps=fps, cutoff_hz=cutoff)
            Hp, Wp = grid
            V.heatmap(high.reshape(Hp, Wp).numpy(), pdir / f"token_high_freq_heatmap_{layer}.png",
                      f"[{layer}] token high-freq ratio ({Hp}x{Wp})", "high_ratio")
            V.heatmap(low.reshape(Hp, Wp).numpy(), pdir / f"token_low_freq_heatmap_{layer}.png",
                      f"[{layer}] token low-freq ratio ({Hp}x{Wp})", "low_ratio")
            tfreq[layer] = {"low": low, "high": high, "grid": list(grid)}
        save_pt(tfreq, fdir / "token_frequency_scores.pt")
        summary["token_frequency"] = {
            l: {"mean_high_ratio": float(d["high"].mean()),
                "high_ratio_std_over_tokens": float(d["high"].std())}
            for l, d in tfreq.items()}

    save_json(summary, fdir / "frequency_summary.json")
    print(f"[save] -> {fdir}")


if __name__ == "__main__":
    main()
