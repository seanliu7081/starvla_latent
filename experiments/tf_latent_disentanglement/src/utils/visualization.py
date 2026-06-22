# Copyright 2025. Licensed under the MIT License.
"""Plotting helpers (headless matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def hist(values, path, title, xlabel, bins=40):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.asarray(values), bins=bins, color="#4C72B0", alpha=0.85)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("count")
    _save(fig, path)


def hist_two(a, b, path, title, xlabel, labels=("a", "b")):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.asarray(a), bins=40, alpha=0.6, label=labels[0], color="#4C72B0")
    ax.hist(np.asarray(b), bins=40, alpha=0.6, label=labels[1], color="#C44E52")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("count"); ax.legend()
    _save(fig, path)


def lines(curves, path, title, xlabel="frame", ylabel="value", labels=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, c in enumerate(curves):
        ax.plot(np.asarray(c), label=(labels[i] if labels else None), lw=1.3)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if labels:
        ax.legend(fontsize=8, ncol=2)
    _save(fig, path)


def heatmap(mat, path, title, cbar_label=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(np.asarray(mat), aspect="auto", cmap="viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    _save(fig, path)


def bar_groups(categories, series, path, title, ylabel):
    """series: dict[label] -> list aligned with categories."""
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(categories)), 4))
    n = len(series); w = 0.8 / max(n, 1)
    x = np.arange(len(categories))
    for i, (lab, vals) in enumerate(series.items()):
        ax.bar(x + i * w, vals, width=w, label=lab)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(fontsize=8)
    _save(fig, path)


def image_grid(rows, path, row_labels=None, col_labels=None, title=None):
    """rows: list of list of HxWx3 uint8 images."""
    nr = len(rows); nc = max(len(r) for r in rows)
    fig, axes = plt.subplots(nr, nc, figsize=(2.0 * nc, 2.0 * nr), squeeze=False)
    for i in range(nr):
        for j in range(nc):
            ax = axes[i][j]; ax.axis("off")
            if j < len(rows[i]) and rows[i][j] is not None:
                ax.imshow(np.asarray(rows[i][j]))
            if i == 0 and col_labels and j < len(col_labels):
                ax.set_title(col_labels[j], fontsize=9)
            if j == 0 and row_labels and i < len(row_labels):
                ax.text(-0.1, 0.5, row_labels[i], rotation=90, va="center",
                        ha="right", transform=ax.transAxes, fontsize=9)
    if title:
        fig.suptitle(title)
    _save(fig, path)
