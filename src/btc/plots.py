"""Figures for the Bitcoin part: python -m src.btc.plots  ->  figures/btc_*.png"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..utils import build_parser, ensure_dir, get_logger, resolve_config  # noqa: E402

log = get_logger()
COLORS = {"XGB": "#7f7f7f", "XGB+graph": "#2ca02c", "GAT": "#9467bd", "Hybrid": "#d62728"}
TASKS = [("transactions", "Criminal transactions"), ("wallets", "Criminal wallets"),
         ("wallets_early", "Early warning: wallet at first appearance")]


def plot_tasks(res: Path, out: Path) -> Path:
    """Test PR-AUC and recall at 90% precision per model for the three tasks (mean ± std over seeds)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey="row")
    for j, (task, title) in enumerate(TASKS):
        p = res / task / "metrics_per_run.csv"
        if not p.exists():
            continue
        r = pd.read_csv(p)
        models = [m for m in COLORS if m in set(r["model"])]
        for i, (metric, label) in enumerate((("pr_auc", "test PR-AUC"), ("recall_at_p90", "recall at 90% precision"))):
            g = r.groupby("model")[metric].agg(["mean", "std"]).reindex(models)
            ax = axes[i, j]
            ax.bar(models, g["mean"], yerr=g["std"].fillna(0), capsize=3, color=[COLORS[m] for m in models])
            for k, (v, e) in enumerate(zip(g["mean"], g["std"].fillna(0))):
                ax.text(k, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=8)
            ax.set_ylim(0, 1)
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(title, fontsize=10)
            ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Elliptic++ (real Bitcoin data), test period steps 35-49, 5 seeds", fontsize=11)
    fig.tight_layout()
    path = out / "btc_results.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_leak(res: Path, out: Path) -> Path | None:
    """Wallet PR-AUC: provided lifetime features (paper protocol / one split per wallet) vs past-only features."""
    p1, p2 = res / "wallet_leak" / "metrics_per_run.csv", res / "wallets" / "metrics_per_run.csv"
    if not (p1.exists() and p2.exists()):
        return None
    a = pd.read_csv(p1)
    b = pd.read_csv(p2)
    paper = a[(a["mode"] == "paper") & (a["model"] == "XGB")]
    rows = [("Provided, paper\nprotocol: wallets\nalso in training", paper["pr_auc_seen_wallets"]),
            ("Provided, paper\nprotocol: new\nwallets", paper["pr_auc_new_wallets"]),
            ("Provided,\none split\nper wallet", a[(a["mode"] == "by_wallet") & (a["model"] == "XGB")]["pr_auc"]),
            ("Past-only\n(own activity)", b[b["model"] == "XGB"]["pr_auc"]),
            ("Past-only\n+ its transactions", b[b["model"] == "XGB+graph"]["pr_auc"])]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([r[0] for r in rows], [r[1].mean() for r in rows], yerr=[r[1].std() for r in rows], capsize=3,
           color=["#dddddd", "#bbbbbb", "#999999", "#7f7f7f", "#2ca02c"])
    for k, (_, v) in enumerate(rows):
        ax.text(k, v.mean() + v.std() + 0.02, f"{v.mean():.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test PR-AUC (XGBoost)")
    ax.set_title("Wallets: the provided lifetime features vs features rebuilt from the past only", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    path = out / "btc_wallet_features.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    """Write every figure whose inputs exist."""
    cfg = resolve_config(build_parser("Bitcoin figures").parse_args())
    res = Path(cfg["paths"]["results_dir"]) / "btc"
    out = ensure_dir(cfg["paths"]["figures_dir"])
    for f in (plot_tasks(res, out), plot_leak(res, out)):
        if f:
            log.info("saved %s", f)


if __name__ == "__main__":
    main()
