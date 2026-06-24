# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
EquiCosmoPredict2-GR00T — CosmoPredict2GR00T with a structured ("equi-noise") source.

This is a SOURCE-ONLY probe: it is byte-for-byte identical to ``CosmoPredict2GR00T``
except that the flow-matching Gaussian source ``z0`` is drawn from a block-isotropic,
normalization-aware distribution (:class:`EquiNoise`) instead of plain
``torch.randn``. Everything else — the frozen Cosmos-Predict2 backbone, the
cross-attention DiT velocity field, ``sample_time`` / Beta schedule,
``repeated_diffusion_steps``, the loss, the data normalization — is unchanged.

Opt-in / default-off contract
-----------------------------
When ``framework.action_model.equi_noise`` is absent or ``enable: false`` the head
falls back to ``torch.randn`` and is bit-for-bit equivalent to ``CosmoPredict2GR00T``
(same train + eval behavior). The same structured source is used at BOTH training
time (``forward``) and inference time (``predict_action``) — they MUST match.

Auto-discovery: lives in ``WM4A/`` and does not start with ``_``, so
``base_framework._auto_import_framework_modules()`` imports it via ``pkgutil`` and
the ``@FRAMEWORK_REGISTRY.register("EquiCosmoPredict2GR00T")`` decorator runs with
no edits to any ``__init__.py``.

----------------------------------------------------------------------------------
PROVENANCE (vendored methods)
  ``EquiFlowmatchingActionHead.forward`` and ``.predict_action`` are vendored
  VERBATIM from:
      starVLA/model/modules/action_model/GR00T_ActionHeader.py
      :: FlowmatchingActionHead.forward / .predict_action
  at git commit 0adcd74 (short SHA). Vendored verbatim; ONLY the noise-source
  line changed (``torch.randn(...)`` -> ``self._source(...)``). If the upstream
  bodies change, this copy will drift — re-vendor and re-diff the two methods.
----------------------------------------------------------------------------------
"""

import torch

from starVLA.model.framework.WM4A.CosmoPredict2GR00T import CosmoPredict2_GR00T
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.action_model.equi_noise import EquiNoise, libero_7dof_blocks
from starVLA.model.tools import FRAMEWORK_REGISTRY


class EquiFlowmatchingActionHead(FlowmatchingActionHead):
    """``FlowmatchingActionHead`` whose flow-matching source is a structured
    (block-isotropic, normalization-aware) :class:`EquiNoise` module.

    Default-off: if ``config.equi_noise`` is missing or ``enable: false`` the
    source is plain ``torch.randn`` and behavior is identical to the parent head.
    """

    def __init__(self, full_config):
        # Builds DiT, encoders, beta-dist, and sets self.config = full_config.framework.action_model
        super().__init__(full_config)

        ec = self.config.get("equi_noise", None)
        if ec is not None and ec.get("enable", False):
            # Convert OmegaConf containers to plain Python before handing to the module.
            ranges = list(ec.get("ranges", None) or []) or None
            scales = dict(ec.get("scales", None)) if ec.get("scales", None) is not None else None
            mode = ec.get("mode", "normalized")
            world_frame_rotation = bool(ec.get("world_frame_rotation", False))

            blocks = libero_7dof_blocks(
                mode=mode,
                ranges=ranges,
                scales=scales,
                world_frame_rotation=world_frame_rotation,
            )
            self.equi_noise = EquiNoise(action_dim=self.action_dim, blocks=blocks, mode=mode)
        else:
            self.equi_noise = None

    def _source(self, shape, device, dtype):
        if self.equi_noise is not None:
            return self.equi_noise.sample(shape, device=device, dtype=dtype)
        return torch.randn(shape, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Vendored from FlowmatchingActionHead.forward (commit 0adcd74).
    # ONLY change: the `noise = torch.randn(...)` line -> `self._source(...)`.
    # ------------------------------------------------------------------
    def forward(
        self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None
    ):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, action_dim)
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = self._source(actions.shape, actions.device, actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    # ------------------------------------------------------------------
    # Vendored from FlowmatchingActionHead.predict_action (commit 0adcd74).
    # ONLY change: the `actions = torch.randn(...)` seed line -> `self._source(...)`.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = self._source(
            (batch_size, self.action_horizon, self.action_dim),
            device, vl_embs.dtype,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions


@FRAMEWORK_REGISTRY.register("EquiCosmoPredict2GR00T")
class EquiCosmoPredict2_GR00T(CosmoPredict2_GR00T):
    """``CosmoPredict2GR00T`` with the equi-noise flow-matching source.

    Builds the standard backbone + head via the parent, then swaps in the
    structured-source head. ``cross_attention_dim`` was already aligned to the
    world-model hidden size by ``super().__init__``; weights are loaded AFTER
    ``__init__`` (see ``baseframework.from_pretrained``), so re-initialising the
    head here loses nothing. The discarded standard head is tiny vs the backbone
    and is GC'd.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)  # builds backbone + standard head
        self.action_model = EquiFlowmatchingActionHead(full_config=self.config)


if __name__ == "__main__":
    import argparse
    import os

    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/bar/equinoise_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.name = "EquiCosmoPredict2GR00T"
    cfg.framework.world_model = {
        "base_wm": "./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World",
        "extract_layers": [-1],
    }

    model: EquiCosmoPredict2_GR00T = EquiCosmoPredict2_GR00T(cfg)
    print(model)
    print("equi_noise module:", model.action_model.equi_noise)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
