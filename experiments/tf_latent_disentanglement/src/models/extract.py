# Copyright 2025. Licensed under the MIT License.
"""
Latent extraction from the frozen Cosmos-Predict2 world model.

For each video frame we run the backbone once and capture two latent layers:

  z : VAE latent      -> [16, Hl, Wl]   (Hl=40, Wl=72 at the model's 320x576)
      The Wan VAE latent; decodable back to pixels via ``vae.decode``.  Visual /
      environment-leaning.  This is the DiT *input*.

  h : DiT last hidden -> [N_tok, 2048]  (N_tok = (Hl/2)*(Wl/2) = 720)
      Last transformer-block activations — the spatiotemporal representation the
      action head consumes.  Predictive / dynamics-leaning.

We process frames in batches (each frame = a 1-frame "video", T_latent=1).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


@torch.inference_mode()
def extract_clip_latents(model, images_np: np.ndarray, lang: str, frame_batch_size: int = 16):
    """Run the frozen backbone over a clip's frames.

    Args:
        images_np: [T, H, W, 3] uint8 RGB frames.
        lang: instruction string for the clip.
    Returns dict with:
        z_latent : [T, 16, Hl, Wl]  float32 (cpu)
        h_hidden : [T, N_tok, 2048] float32 (cpu)
        z_grid   : (Hl, Wl)
        h_grid   : (Hp, Wp)
    """
    backbone = model.backbone
    device = next(backbone.transformer.parameters()).device
    T = images_np.shape[0]
    pil = [Image.fromarray(images_np[t]) for t in range(T)]

    z_chunks, h_chunks = [], []
    for s in range(0, T, frame_batch_size):
        batch_imgs = pil[s:s + frame_batch_size]
        b = len(batch_imgs)
        inp = backbone.build_inputs(images=batch_imgs, instructions=[lang] * b)
        # VAE latent (DiT input): [b, 16, 1, Hl, Wl]
        z = inp["hidden_states"]
        z = z.squeeze(2)  # T_latent == 1 -> [b, 16, Hl, Wl]
        out = backbone(**inp, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]  # [b, N_tok, 2048]
        z_chunks.append(z.float().cpu())
        h_chunks.append(h.float().cpu())

    z_latent = torch.cat(z_chunks, dim=0)   # [T,16,Hl,Wl]
    h_hidden = torch.cat(h_chunks, dim=0)   # [T,N_tok,2048]

    Hl, Wl = z_latent.shape[-2], z_latent.shape[-1]
    patch = list(backbone.transformer.config.patch_size)  # [1,2,2]
    Hp, Wp = Hl // patch[1], Wl // patch[2]
    assert Hp * Wp == h_hidden.shape[1], (
        f"token grid mismatch: {Hp}x{Wp}={Hp*Wp} vs N_tok={h_hidden.shape[1]}")
    return {
        "z_latent": z_latent,
        "h_hidden": h_hidden,
        "z_grid": (Hl, Wl),
        "h_grid": (Hp, Wp),
    }


def pool_z(z_latent: torch.Tensor) -> torch.Tensor:
    """[T,16,Hl,Wl] -> [T,16] spatial mean-pool."""
    return z_latent.mean(dim=(-1, -2))


def pool_h(h_hidden: torch.Tensor) -> torch.Tensor:
    """[T,N_tok,2048] -> [T,2048] token mean-pool."""
    return h_hidden.mean(dim=1)


def tokens_z(z_latent: torch.Tensor) -> torch.Tensor:
    """[T,16,Hl,Wl] -> [T, Hl*Wl, 16] token layout (S, D)."""
    T, C, Hl, Wl = z_latent.shape
    return z_latent.permute(0, 2, 3, 1).reshape(T, Hl * Wl, C)
