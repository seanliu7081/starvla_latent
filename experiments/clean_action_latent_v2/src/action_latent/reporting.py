# Copyright 2025. Licensed under the MIT License.
"""Reporting helpers: STATUS.json writer + evidence-tier tagging (§7).

Evidence tiers (every empirical claim must be tagged):
  T1-causal : frozen-head projective intervention (edit h -> frozen head output)
  T2-recomb : trained-decoder recombination
  T3-corr   : a probe reads action/labels off a feature
  T4-retr   : retrieval / partial-correlation
"""
from __future__ import annotations

from pathlib import Path

from .io import save_json

TIERS = {
    "T1-causal": "frozen-head projective intervention",
    "T2-recomb": "trained-decoder recombination",
    "T3-corr": "probe reads target off a feature",
    "T4-retr": "retrieval / partial correlation",
}


def write_status(output_dir, *, phase_reached, gates_passed, interface=None,
                 verdict=None, extra=None):
    status = {
        "phase_reached": phase_reached,
        "gates_passed": gates_passed,
        "interface": interface,
        "verdict": verdict,
    }
    if extra:
        status.update(extra)
    return save_json(status, Path(output_dir) / "STATUS.json")
