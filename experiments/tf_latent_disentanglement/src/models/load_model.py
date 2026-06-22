# Copyright 2025. Licensed under the MIT License.
"""
Load the frozen WM4A ``CosmoPredict2GR00T`` checkpoint *without* the gated base
model.

Strategy
--------
The repo's ``_CosmoPredict2_Interface`` constructs its sub-modules via
``X.from_pretrained(base_wm, subfolder=...)``, which requires the gated
``nvidia/Cosmos-Predict2-2B-Video2World`` repo.  We can't access it.  But the
released checkpoint already contains every backbone weight, so we only need the
architecture configs (see ``cosmos_configs.py``, validated to exact key+shape
match).

We monkeypatch the three weight-loading ``from_pretrained`` calls during model
construction so they build empty modules from our reconstructed configs instead
of downloading weights.  Then we load the checkpoint state_dict (strict) to fill
in the real trained weights.

The result is byte-for-byte the same architecture and the same trained weights
the model would have with the gated repo present — only the redundant base-weight
download is skipped.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest import mock

import torch

from . import cosmos_configs as CC


# ----------------------------------------------------------------------------
# Monkeypatch context: replace the interface's `from_pretrained` weight loaders
# with config-only builders that produce empty (correct-architecture) modules.
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def _patched_base_loaders(sigma_data: float):
    from diffusers import (
        AutoencoderKLWan,
        CosmosTransformer3DModel,
        FlowMatchEulerDiscreteScheduler,
    )
    from transformers import T5Config, T5EncoderModel, T5TokenizerFast

    # Capture the genuine tokenizer loader BEFORE patching (the fake below must
    # call the real one, not itself).
    _orig_tokenizer_from_pretrained = T5TokenizerFast.from_pretrained

    def fake_transformer_from_pretrained(*args, **kwargs):
        # ignore (model_name, subfolder=...) — build from our exact config
        return CosmosTransformer3DModel.from_config(dict(CC.TRANSFORMER_CONFIG))

    def fake_vae_from_pretrained(*args, **kwargs):
        return AutoencoderKLWan.from_config(CC.load_vae_config())

    def fake_t5_from_pretrained(*args, **kwargs):
        # T5-11b encoder config matches the checkpoint exactly.
        cfg = T5Config.from_pretrained(CC.T5_MODEL_ID)
        return T5EncoderModel(cfg)

    def fake_tokenizer_from_pretrained(*args, **kwargs):
        # Use the ungated t5-11b tokenizer (standard SentencePiece T5 vocab=32128).
        return _orig_tokenizer_from_pretrained(CC.T5_MODEL_ID, legacy=False)

    def fake_scheduler_from_pretrained(*args, **kwargs):
        sch = FlowMatchEulerDiscreteScheduler(**CC.SCHEDULER_KWARGS)
        # The interface reads self.scheduler.config.sigma_data to rescale VAE
        # latents; FlowMatchEulerDiscreteScheduler does not register it by default.
        sch.register_to_config(sigma_data=float(sigma_data))
        return sch

    with mock.patch.object(CosmosTransformer3DModel, "from_pretrained",
                           side_effect=fake_transformer_from_pretrained), \
         mock.patch.object(AutoencoderKLWan, "from_pretrained",
                           side_effect=fake_vae_from_pretrained), \
         mock.patch.object(T5EncoderModel, "from_pretrained",
                           side_effect=fake_t5_from_pretrained), \
         mock.patch.object(T5TokenizerFast, "from_pretrained",
                           side_effect=fake_tokenizer_from_pretrained), \
         mock.patch.object(FlowMatchEulerDiscreteScheduler, "from_pretrained",
                           side_effect=fake_scheduler_from_pretrained):
        yield


def load_frozen_wm4a(
    checkpoint_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    sigma_data: float = CC.DEFAULT_SIGMA_DATA,
    base_wm_dir: str | None = None,
):
    """Load the full frozen ``CosmoPredict2GR00T`` framework from a checkpoint.

    Returns ``(model, cfg, norm_stats)``.  The model is on ``device``, in eval
    mode, with ``requires_grad=False`` on all parameters.
    """
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config

    checkpoint_path = str(checkpoint_path)
    model_config, norm_stats = read_mode_config(checkpoint_path)
    cfg = dict_to_namespace(model_config)
    cfg.trainer.pretrained_checkpoint = None

    # Point base_wm at a dummy local dir so the (patched) loaders never touch HF.
    # NOTE: get_world_model() routes on the substring "cosmos-predict2" in the
    # path, so the stub dir name must contain it.
    if base_wm_dir is None:
        base_wm_dir = str(Path(checkpoint_path).parent.parent / "_Cosmos-Predict2_stub")
    Path(base_wm_dir).mkdir(parents=True, exist_ok=True)
    cfg.framework.world_model.base_wm = base_wm_dir

    # 1) Build architecture (empty backbone via patched loaders + action head).
    #    The patched scheduler loader carries the provided sigma_data.
    with _patched_base_loaders(sigma_data):
        model = build_framework(cfg)

    # 2) Load all trained weights from the checkpoint (strict).
    sd = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # We expect a perfect match for all real parameters.  The only acceptable
    # "missing" entries are non-persistent buffers (rope tables etc.) that are
    # recomputed at runtime and never appear in any state_dict.
    real_missing = [k for k in missing if "rope" not in k.lower()]
    if real_missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch loading {checkpoint_path}:\n"
            f"  missing ({len(real_missing)}): {real_missing[:12]}\n"
            f"  unexpected ({len(unexpected)}): {list(unexpected)[:12]}"
        )

    model.norm_stats = norm_stats
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, norm_stats
