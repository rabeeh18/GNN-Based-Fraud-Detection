"""Diagram 3: temporal split timeline."""
from _style import BLUE, GREY, RED, arrow, box, canvas, save


def main() -> None:
    fig, ax = canvas(11, 3.8)
    total = 743.0
    cuts = [(1, 500, "Train", "#c6dbef"), (501, 620, "Validation", "#fdd0a2"), (621, 743, "Test", "#d9d9d9")]
    for a, b, name, c in cuts:
        x0, x1 = 4 + 92 * a / total, 4 + 92 * b / total
        box(ax, x0, 40, x1 - x0, 18, name, fc=c, ec=GREY, bold=True)
        ax.text((x0 + x1) / 2, 34, f"steps {a}-{b}", ha="center", fontsize=9)
    ax.text(4, 66, "time (step = 1 hour)  ->", fontsize=9)
    tx = 4 + 92 * 640 / total
    ax.plot([tx, tx], [38, 62], color=RED, lw=2)
    ax.text(tx, 66, "scored transaction at t", color=RED, ha="center", fontsize=9)
    arrow(ax, (tx - 1, 20), (6, 20), BLUE, 2.4)
    ax.text(50, 11, "only edges from steps < t are visible (past-only features and GNN neighbourhood)", ha="center",
            fontsize=10, color=BLUE)
    ax.text(tx + 1, 20, "", fontsize=9)
    save(fig, "d3_temporal_split")


if __name__ == "__main__":
    main()
