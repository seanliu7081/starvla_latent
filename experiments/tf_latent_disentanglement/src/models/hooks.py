# Copyright 2025. Licensed under the MIT License.
"""Forward-hook utility for capturing intermediate features (per the plan)."""
from __future__ import annotations


class FeatureCatcher:
    def __init__(self):
        self.features = {}
        self.handles = []

    def add_hook(self, module, name):
        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, dict):
                self.features[name] = {
                    k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in output.items()
                }
            else:
                self.features[name] = output.detach().cpu()
        self.handles.append(module.register_forward_hook(hook_fn))

    def clear(self):
        self.features = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
