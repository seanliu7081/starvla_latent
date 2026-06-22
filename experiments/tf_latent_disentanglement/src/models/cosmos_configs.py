# Copyright 2025. Licensed under the MIT License.
"""
Reconstructed architecture configs for ``nvidia/Cosmos-Predict2-2B-Video2World``.

The base model repo is HF-gated (our account lacks access), but the released
``WM4A-CosmoPredict-GR00T-LIBERO-4in1`` checkpoint already contains *all* the
backbone weights (``backbone.transformer.*``, ``backbone.vae.*``,
``backbone.text_encoder.*``).  We therefore only need the *architecture* configs
to construct empty modules, then load the checkpoint weights into them.

Every config below was validated to produce an EXACT state-dict key + shape
match against the checkpoint (transformer 567/567, vae 194/194, text_encoder
196/196).  See ``load_model.py`` for the assembly.

Sources:
  - transformer config: HF diffusers ``scripts/convert_cosmos_to_diffusers.py``
    (``TRANSFORMER_CONFIGS["Cosmos-2.0-Diffusion-2B-Video2World"]``).
  - vae config (Wan2.1): identical across Cosmos-Predict2 variants; pulled from
    an ungated diffusers mirror and cached locally.
  - text_encoder: ``google-t5/t5-11b`` encoder (d_model=1024, vocab=32128,
    num_heads=128, d_kv=128 -> inner 16384) — matches checkpoint exactly.
"""

# Exact official transformer config for Cosmos-2.0-Diffusion-2B-Video2World.
# hidden_size = num_attention_heads * attention_head_dim = 16 * 128 = 2048.
# in_channels = 17 = 16 VAE latent channels + 1 condition_mask channel
#   (concat_padding_mask=True adds a further +1 inside the model -> patch_embed
#    sees 18 channels -> proj weight (2048, 18*1*2*2=72), matching checkpoint).
TRANSFORMER_CONFIG = {
    "_class_name": "CosmosTransformer3DModel",
    "in_channels": 17,
    "out_channels": 16,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 28,
    "mlp_ratio": 4.0,
    "text_embed_dim": 1024,
    "adaln_lora_dim": 256,
    "max_size": [128, 240, 240],
    "patch_size": [1, 2, 2],
    "rope_scale": [1.0, 3.0, 3.0],
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
}

# Wan2.1 VAE (AutoencoderKLWan).  z_dim = 16, spatial compression 8x,
# temporal compression 4x.  latents_mean / latents_std are baked into the config
# (per-channel), used by the interface's latent normalisation.  Loaded from the
# bundled, checkpoint-validated JSON asset (single source of truth).
import json as _json
from pathlib import Path as _Path

_VAE_CONFIG_JSON = _Path(__file__).parent / "assets" / "wan_vae_config.json"


def load_vae_config() -> dict:
    with open(_VAE_CONFIG_JSON) as f:
        return _json.load(f)

# Text encoder = encoder of google-t5/t5-11b (loaded by id, ungated).
T5_MODEL_ID = "google-t5/t5-11b"

# FlowMatchEulerDiscreteScheduler.  The only config value the interface reads is
# ``sigma_data`` (used to rescale VAE latents).  diffusers' scheduler does not
# register ``sigma_data`` by default; the correct value is resolved empirically
# by ``scripts/00_resolve_sigma_data.py`` (action-prediction acceptance test) and
# defaults to 1.0 (Cosmos-Predict2 convention).
SCHEDULER_KWARGS = {"use_karras_sigmas": True}
DEFAULT_SIGMA_DATA = 1.0
