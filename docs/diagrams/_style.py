"""Shared helpers for the diagram scripts (matplotlib only; red = fraud, blue = normal)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

RED, BLUE, GREY, LIGHT = "#d62728", "#1f77b4", "#555555", "#f2f2f2"
OUT = Path(__file__).resolve().parent


_ASPECT = [1.0]
_HAS_NODES = [False]


def canvas(w: float = 10, h: float = 5):
    """Blank axes in a 0-100 x 0-100 coordinate system without frame."""
    fig, ax = plt.subplots(figsize=(w, h))
    _ASPECT[0] = w / h
    _HAS_NODES[0] = False
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc=LIGHT, ec=GREY, fs=10, tc="black", bold=False):
    """Rounded box with centred text; (x, y) is the lower-left corner."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q, color=GREY, lw=1.6, style="-|>", rad=0.0, ls="-"):
    """Arrow from p to q (data coordinates)."""
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=22 if _HAS_NODES[0] else 3, shrinkB=22 if _HAS_NODES[0] else 3))


def node(ax, x, y, label, color=BLUE, r=3.2):
    """Circular account node."""
    _HAS_NODES[0] = True
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * _ASPECT[0], fc=color, ec="white", lw=1.5, zorder=3))
    ax.text(x, y, label, color="white", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)


def save(fig, name: str) -> None:
    """Save PNG and SVG next to the scripts."""
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=170 if ext == "png" else None)
    plt.close(fig)
    print(f"saved {name}.png/.svg")
