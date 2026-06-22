# Copyright 2025. Licensed under the MIT License.
"""
Decode helpers for ablation / swap visualisation.

  - decode_vae_latent: invert the interface's latent normalisation and run the
    Wan VAE decoder to turn a (possibly edited) normalised VAE latent back into
    pixels.
  - action_from_hidden: run the FROZEN GR00T action head on a (possibly edited)
    DiT token-hidden tensor to get a predicted action chunk (readout drop test).
"""
from __future__ import annotations

import numpy as np
import torch


def _latent_stats(backbone, device, dtype):
    vae = backbone.vae
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    sigma = backbone.scheduler.config.sigma_data
    return mean, std, sigma


@torch.inference_mode()
def decode_vae_latent(backbone, norm_latent: torch.Tensor) -> np.ndarray:
    """norm_latent: [B,16,Hl,Wl] normalised VAE latent -> [B,H,W,3] uint8 RGB frames.

    Inverts ``latents = (raw - mean)/std * sigma_data`` then runs vae.decode.
    """
    vae = backbone.vae
    device = next(vae.parameters()).device
    dtype = vae.dtype
    z = norm_latent.to(device=device, dtype=dtype)
    if z.ndim == 4:
        z = z.unsqueeze(2)                       # [B,16,1,Hl,Wl]
    mean, std, sigma = _latent_stats(backbone, device, dtype)
    raw = z / sigma * std + mean
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = vae.decode(raw).sample             # [B,3,T,H,W] in ~[-1,1]
    out = out.float().clamp(-1, 1)
    out = (out + 1) / 2                          # [0,1]
    frames = (out[:, :, 0].permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    return frames                                # [B,H,W,3]


@torch.inference_mode()
def action_from_hidden(model, h_tokens: torch.Tensor, state: torch.Tensor | None = None) -> np.ndarray:
    """h_tokens: [T, N_tok, 2048] DiT hidden -> predicted normalized actions [T, horizon, 7]."""
    device = next(model.action_model.parameters()).device
    dtype = next(model.action_model.parameters()).dtype
    h = h_tokens.to(device=device, dtype=dtype)
    with torch.autocast("cuda", dtype=torch.float32):
        pred = model.action_model.predict_action(h, state)   # [T, horizon, 7]
    return pred.float().cpu().numpy()
