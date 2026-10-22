"""Diagram 2: project pipeline."""
from _style import BLUE, arrow, box, canvas, save


def main() -> None:
    fig, ax = canvas(12, 4.4)
    steps = [("Load + filter\nPaySim", "TRANSFER / CASH_OUT"), ("Temporal split\n+ past-only\nfeatures", "train < val < test"),
             ("EDA", "graph sparsity,\nleakage checks"), ("Baselines\nB0, B0b, B1", "XGBoost / RF"),
             ("GATv2\nedge classifier", "time-aware sampling"), ("Evaluation\n+ plots", "PR-AUC, recall@P")]
    w, gap = 13.5, 3.2
    for i, (t, sub) in enumerate(steps):
        x = 2 + i * (w + gap)
        box(ax, x, 45, w, 24, t, fc="#e8f1fa", ec=BLUE, fs=10, bold=True)
        ax.text(x + w / 2, 37, sub, ha="center", va="top", fontsize=8, color="#444444")
        if i:
            arrow(ax, (x - gap, 57), (x, 57), lw=2)
    ax.text(50, 12, "every model is run twice: with and without balance features, over several seeds",
            ha="center", fontsize=10, style="italic")
    save(fig, "d2_pipeline")


if __name__ == "__main__":
    main()
