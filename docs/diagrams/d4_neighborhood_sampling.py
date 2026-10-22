"""Diagram 4: 2-hop time-aware neighbourhood sampling around one target transaction."""
import numpy as np

from _style import BLUE, GREY, RED, arrow, canvas, node, save


def main() -> None:
    fig, ax = canvas(8.5, 6)
    u, v = (38, 50), (62, 50)
    node(ax, *u, "u", RED)
    node(ax, *v, "v", RED)
    arrow(ax, u, v, RED, 3)
    ax.text(50, 54.5, "target tx (step t)", ha="center", color=RED, fontsize=9)
    rng = np.random.default_rng(1)
    for centre, side in ((u, -1), (v, 1)):
        for k, ang in enumerate(np.linspace(-1.0, 1.0, 3)):
            a1 = (centre[0] + side * 15 * np.cos(ang), centre[1] + 15 * np.sin(ang) * 1.3)
            node(ax, *a1, "", BLUE, 2.3)
            arrow(ax, a1, centre, GREY, 1.4)
            for j in (-0.5, 0.5):
                a2 = (a1[0] + side * 11 * np.cos(ang + j), a1[1] + 11 * np.sin(ang + j) * 1.3)
                node(ax, *a2, "", BLUE, 1.6)
                arrow(ax, a2, a1, GREY, 1.0)
    ax.text(8, 94, "hop 1: up to 10 past edges per account", fontsize=9)
    ax.text(8, 89, "hop 2: up to 5 past edges per neighbour", fontsize=9)
    ax.text(50, 5, "only edges from steps < t are sampled (most recent first); later edges are never used",
            ha="center", fontsize=9, color=BLUE)
    save(fig, "d4_neighborhood_sampling")


if __name__ == "__main__":
    main()
