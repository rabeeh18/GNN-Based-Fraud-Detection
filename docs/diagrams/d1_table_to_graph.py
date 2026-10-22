"""Diagram 1: transaction table -> account graph."""
from _style import BLUE, RED, arrow, box, canvas, node, save


def main() -> None:
    fig, ax = canvas(11, 4.6)
    ax.text(22, 96, "Transactions (table)", ha="center", fontsize=12, fontweight="bold")
    ax.text(77, 96, "Account graph", ha="center", fontsize=12, fontweight="bold")
    rows = [("1", "TRANSFER", "A", "B", "9,000", True), ("1", "CASH_OUT", "B", "M", "9,000", True),
            ("2", "PAYMENT", "C", "S", "40", False), ("3", "CASH_OUT", "D", "M", "120", False)]
    hdr = ["step", "type", "from", "to", "amount"]
    xs = [3, 11, 27, 35, 43]
    for x, h in zip(xs, hdr):
        ax.text(x, 86, h, fontsize=9, fontweight="bold", color="#333333")
    for i, r in enumerate(rows):
        y = 74 - i * 15
        box(ax, 1, y - 5, 54, 11, "", fc="#fdecea" if r[5] else "#e8f1fa", ec=RED if r[5] else BLUE)
        for x, v in zip(xs, r[:5]):
            ax.text(x, y, v, fontsize=9, va="center")
    arrow(ax, (57, 50), (60, 50), lw=2.5)
    pos = {"A": (68, 70), "B": (80, 52), "M": (93, 70), "C": (68, 28), "S": (80, 14), "D": (93, 32)}
    for k, (x, y) in pos.items():
        fraud = k in ("A", "B")
        node(ax, x, y, k, RED if fraud else BLUE)
    arrow(ax, pos["A"], pos["B"], RED, 2.2)
    arrow(ax, pos["B"], pos["M"], RED, 2.2)
    arrow(ax, pos["C"], pos["S"], BLUE)
    arrow(ax, pos["D"], pos["M"], BLUE, rad=-0.2)
    ax.text(72, 63, "tx 1", color=RED, fontsize=8)
    ax.text(90, 58, "tx 2", color=RED, fontsize=8)
    ax.text(3, 4, "node = account    arrow = one transaction (sender -> receiver)    red = fraud, blue = normal",
            fontsize=9, ha="left")
    save(fig, "d1_table_to_graph")


if __name__ == "__main__":
    main()
