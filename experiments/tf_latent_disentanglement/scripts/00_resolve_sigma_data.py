#!/usr/bin/env python
# Copyright 2025. Licensed under the MIT License.
"""
Acceptance test + sigma_data resolver.

Loads the frozen WM4A checkpoint via our gated-free loader and validates the
ENTIRE loading path end-to-end by checking that the model predicts actions that
correlate with the ground-truth actions recorded in the LIBERO dataset.

The only architecture unknown is the scheduler's ``sigma_data`` (a scalar that
rescales VAE latents feeding the DiT).  We sweep candidate values and pick the
one that maximises action-prediction correlation.

Usage:
    python scripts/00_resolve_sigma_data.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]          # /home/haotian/code/starVLA
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiment root

from src.models.load_model import load_frozen_wm4a  # noqa: E402

CKPT = REPO / "pretrained/WM4A-CosmoPredict-GR00T-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
DATA = REPO / "playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot"


def read_frames_pyav(video_path, indices):
    """Decode specific frame indices from an av1 mp4 via pyav."""
    import av
    from PIL import Image

    indices = sorted(set(int(i) for i in indices))
    want = set(indices)
    out = {}
    container = av.open(str(video_path))
    for i, frame in enumerate(container.decode(video=0)):
        if i in want:
            out[i] = Image.fromarray(frame.to_ndarray(format="rgb24"))
            if len(out) == len(want):
                break
    container.close()
    return [out[i] for i in indices]


def q99_normalize(x, stats):
    q01 = np.asarray(stats["q01"]); q99 = np.asarray(stats["q99"])
    mask = q01 != q99
    out = x.astype(np.float64).copy()
    out[..., mask] = 2 * (x[..., mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1
    return np.clip(out, -2.2, 2.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 0.5, 2.0])
    ap.add_argument("--n_frames", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # ---- ground-truth: one episode, a handful of frames ----
    tasks = {}
    with open(DATA / "meta/tasks.jsonl") as f:
        for line in f:
            d = json.loads(line); tasks[d["task_index"]] = d["task"]
    ep_idx = 0
    df = pd.read_parquet(DATA / f"data/chunk-000/episode_{ep_idx:06d}.parquet")
    horizon = 8
    n = len(df)
    # frame anchors spaced across the episode (leave room for an 8-step chunk)
    anchors = np.linspace(5, n - horizon - 1, args.n_frames).astype(int)
    video = DATA / f"videos/chunk-000/observation.images.image/episode_{ep_idx:06d}.mp4"
    imgs = read_frames_pyav(video, anchors)
    gt_actions = np.stack([df["action"].iloc[t:t + horizon].to_numpy() for t in anchors])  # [F,8,7]
    gt_actions = np.stack([np.stack(a) for a in gt_actions])
    langs = [tasks[int(df["task_index"].iloc[t])] for t in anchors]

    stats = json.load(open(REPO / "pretrained/WM4A-CosmoPredict-GR00T-LIBERO-4in1/dataset_statistics.json"))
    action_stats = stats["franka"]["action"]
    gt_norm = q99_normalize(gt_actions, action_stats)  # [F,8,7]

    print(f"episode {ep_idx} len={n}, anchors={list(anchors)}")
    print(f"task examples: {langs[0]!r}")

    results = {}
    for sigma in args.sigmas:
        print(f"\n===== sigma_data = {sigma} =====")
        model, cfg, _ = load_frozen_wm4a(str(CKPT), device=args.device, sigma_data=sigma)
        preds = []
        for img, lang in zip(imgs, langs):
            out = model.predict_action([{"image": img, "lang": lang}])
            preds.append(np.asarray(out["normalized_actions"])[0])  # [8,7]
        del model
        torch.cuda.empty_cache()
        preds = np.stack(preds)  # [F,8,7]

        # overall correlation (flattened) + per-dim correlation
        pf, gf = preds.reshape(-1), gt_norm.reshape(-1)
        overall = np.corrcoef(pf, gf)[0, 1]
        per_dim = [float(np.corrcoef(preds[..., d].reshape(-1), gt_norm[..., d].reshape(-1))[0, 1])
                   for d in range(7)]
        mae = float(np.mean(np.abs(preds - gt_norm)))
        results[sigma] = {"overall_corr": float(overall), "per_dim_corr": per_dim,
                          "mae_norm": mae, "pred_range": [float(preds.min()), float(preds.max())]}
        print(f"  overall corr = {overall:.3f} | MAE(norm) = {mae:.3f} | pred range = "
              f"[{preds.min():.2f}, {preds.max():.2f}]")
        print(f"  per-dim corr = {[round(c,2) for c in per_dim]}")

    best = max(results, key=lambda s: results[s]["overall_corr"])
    print(f"\n>>> BEST sigma_data = {best}  (corr={results[best]['overall_corr']:.3f})")
    out_path = Path(__file__).resolve().parents[1] / "outputs" / "sigma_data_acceptance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "best_sigma_data": best}, open(out_path, "w"), indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
