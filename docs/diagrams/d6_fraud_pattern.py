"""Diagram 6: typical PaySim fraud pattern, victim -> mule -> cash-out."""
from _style import BLUE, RED, arrow, canvas, node, save


def main() -> None:
    fig, ax = canvas(10, 3.8)
    node(ax, 14, 55, "victim", BLUE, 6)
    node(ax, 50, 55, "mule", RED, 6)
    node(ax, 86, 55, "cash-out\npoint", BLUE, 6)
    arrow(ax, (20.5, 55), (43.5, 55), RED, 3)
    arrow(ax, (56.5, 55), (79.5, 55), RED, 3)
    ax.text(32, 63, "1. TRANSFER\n(drains the account)", ha="center", fontsize=9, color=RED)
    ax.text(68, 63, "2. CASH_OUT\n(same amount, later step)", ha="center", fontsize=9, color=RED)
    ax.text(50, 22, "the mule is usually a fresh account: little history, balance goes to 0 right after",
            ha="center", fontsize=10)
    ax.text(50, 12, "red = fraud transactions / involved accounts, blue = normal", ha="center", fontsize=9, color="#555555")
    save(fig, "d6_fraud_pattern")


if __name__ == "__main__":
    main()
