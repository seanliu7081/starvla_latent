# Copyright 2025. Licensed under the MIT License.
"""Small shared utilities: config loading, seeding, IO, repo paths."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Repo root = .../starVLA  (this file is experiments/tf_latent_disentanglement/src/utils/common.py)
REPO_ROOT = Path(__file__).resolve().parents[4]
EXP_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    cfg = OmegaConf.load(path)
    # resolve ${oc.env:...} / interpolations
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path: absolute as-is, else relative to the starVLA repo root."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pt(obj, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(obj, str(path))


def load_pt(path: str | Path, map_location="cpu"):
    return torch.load(str(path), map_location=map_location, weights_only=False)


def save_json(obj, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    return obj
