"""Diagram 5: GATv2 edge-classifier architecture."""
from _style import BLUE, RED, arrow, box, canvas, save


def main() -> None:
    fig, ax = canvas(12, 5)
    box(ax, 2, 60, 16, 22, "Sampled past\nsubgraph\n(edge features\n+ age + direction)", fc="#e8f1fa", ec=BLUE, fs=9)
    box(ax, 24, 62, 15, 18, "Constant node\ninput -> Linear", fs=9)
    box(ax, 45, 62, 15, 18, "GATv2Conv\nlayer 1 (+skip,\nLayerNorm)", fc="#e8f1fa", ec=BLUE, fs=9)
    box(ax, 66, 62, 15, 18, "GATv2Conv\nlayer 2 (+skip,\nLayerNorm)", fc="#e8f1fa", ec=BLUE, fs=9)
    for a, b in ((18, 24), (39, 45), (60, 66)):
        arrow(ax, (a + 0.5, 71), (b - 0.5, 71))
    box(ax, 60, 36, 14, 14, "h_sender", fs=9)
    box(ax, 78, 36, 14, 14, "h_receiver", fs=9)
    arrow(ax, (70, 62), (67, 50.5))
    arrow(ax, (77, 62), (85, 50.5))
    box(ax, 60, 16, 32, 12, "concat  ->  MLP", fc="#fdecea", ec=RED, fs=10, bold=True)
    arrow(ax, (67, 35.5), (70, 28.5))
    arrow(ax, (85, 35.5), (82, 28.5))
    box(ax, 28, 16, 24, 12, "features of the\ntarget transaction", fc="#fdecea", ec=RED, fs=9)
    arrow(ax, (52.5, 22), (59.5, 22))
    arrow(ax, (76, 15.5), (76, 8), RED)
    ax.text(76, 3, "fraud logit", ha="center", fontsize=10, color=RED, fontweight="bold")
    save(fig, "d5_gat_architecture")


if __name__ == "__main__":
    main()
