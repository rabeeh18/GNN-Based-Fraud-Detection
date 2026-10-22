"""Figures generated from files in results/ (EDA tables, metrics, predictions, attention examples)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import build_parser, ensure_dir, get_logger, mode_name, resolve_config

log = get_logger()
RED, BLUE, GREY = "#d62728", "#1f77b4", "#7f7f7f"
MODEL_COLORS = {"B0": "#7f7f7f", "B0b": "#bcbd22", "B1": "#2ca02c", "GAT": "#9467bd", "FlaggedRule": "#8c564b"}
MODEL_ORDER = ["FlaggedRule", "B0", "B0b", "B1", "GAT"]
MODE_TITLE = {"balance": "with balance features", "nobalance": "without balance features"}


def _save(fig: plt.Figure, cfg: dict, name: str) -> Path:
    out = ensure_dir(cfg["paths"]["figures_dir"]) / name
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("saved %s", out)
    return out


def _res(cfg: dict) -> Path:
    return Path(cfg["paths"]["results_dir"])


# ---------------------------------------------------------------- EDA figures
def plot_fraud_by_type(cfg: dict) -> Path:
    """Bar chart of fraud rate and fraud count per transaction type."""
    t = pd.read_csv(_res(cfg) / "eda_fraud_by_type.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].bar(t["type"], t["rows"], color=BLUE)
    ax[0].set_yscale("log")
    ax[0].set_title("Transactions per type")
    ax[1].bar(t["type"], t["frauds"], color=RED)
    ax[1].set_title("Frauds per type")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    return _save(fig, cfg, "eda_fraud_by_type.png")


def plot_fraud_over_time(cfg: dict) -> Path:
    """Fraud count and fraud rate over time steps."""
    t = pd.read_csv(_res(cfg) / "eda_fraud_over_time.csv")
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax[0].bar(t["step"], t["rows"], color=BLUE, width=1.0)
    ax[0].set_ylabel("transactions")
    ax[1].bar(t["step"], t["frauds"], color=RED, width=1.0)
    ax[1].set_ylabel("frauds")
    ax[1].set_xlabel("step (hour)")
    ax[0].set_title("Volume and fraud over time")
    return _save(fig, cfg, "eda_fraud_over_time.png")


def plot_degree_distributions(cfg: dict) -> Path:
    """Log-log degree histograms (in / out) for fraud-involved vs other accounts."""
    d = pd.read_csv(_res(cfg) / "eda_degree.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for a, kind in zip(ax, ("in", "out")):
        for group, color in (("other", BLUE), ("fraud_involved", RED)):
            s = d[(d["group"] == group) & (d["kind"] == kind)]
            if s.empty:
                continue
            a.loglog(s["degree"].clip(lower=1), s["count"] / s["count"].sum(), "o", ms=4, color=color,
                     alpha=0.8, label=group.replace("_", " "))
        a.set_title(f"{kind}-degree distribution")
        a.set_xlabel("degree (log)")
    ax[0].set_ylabel("fraction of accounts (log)")
    ax[0].legend()
    return _save(fig, cfg, "eda_degree_distributions.png")


# ---------------------------------------------------------------- result figures
def _load_runs(cfg: dict) -> pd.DataFrame:
    p = _res(cfg) / "metrics_per_run.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run baselines / train_gat first")
    return pd.read_csv(p)


def plot_pr_curves(cfg: dict) -> list[Path]:
    """PR curves on the test split for all models, one figure per feature mode (lowest seed available)."""
    from sklearn.metrics import precision_recall_curve

    runs = _load_runs(cfg)
    outs = []
    for mode in runs["mode"].unique():
        fig, ax = plt.subplots(figsize=(6, 4.6))
        for model in MODEL_ORDER:
            sub = runs[(runs["mode"] == mode) & (runs["model"] == model)]
            if sub.empty:
                continue
            seed = int(sub["seed"].min())
            p = _res(cfg) / f"predictions_{model}_{mode}_seed{seed}.parquet"
            if not p.exists():
                continue
            pr = pd.read_parquet(p)
            prec, rec, _ = precision_recall_curve(pr["y_true"], pr["score"])
            ap = float(sub.loc[sub["seed"] == seed, "pr_auc"].iloc[0])
            ax.plot(rec, prec, color=MODEL_COLORS.get(model), label=f"{model} (AP {ap:.3f})")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Test PR curves, {MODE_TITLE.get(mode, mode)} (seed-min)")
        ax.legend(loc="lower left")
        outs.append(_save(fig, cfg, f"pr_curves_{mode}.png"))
    return outs


def plot_summary_bars(cfg: dict) -> Path:
    """Mean +- std over seeds of PR-AUC and recall@90% precision per model and mode."""
    runs = _load_runs(cfg)
    metrics = [("pr_auc", "PR-AUC (test)"), ("recall_at_p90", "Recall @ 90% precision"),
               ("f1", "F1 @ val-tuned threshold")]
    modes = list(runs["mode"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4))
    for ax, (m, title) in zip(np.atleast_1d(axes), metrics):
        width = 0.8 / max(len(modes), 1)
        models = [x for x in MODEL_ORDER if x in set(runs["model"])]
        for j, mode in enumerate(modes):
            g = runs[runs["mode"] == mode].groupby("model")[m].agg(["mean", "std"]).reindex(models)
            x = np.arange(len(models)) + j * width
            ax.bar(x, g["mean"], width, yerr=g["std"].fillna(0), capsize=3,
                   color=[MODEL_COLORS[k] for k in models], alpha=1.0 if j == 0 else 0.55,
                   hatch="" if j == 0 else "//", edgecolor="white", label=MODE_TITLE.get(mode, mode))
        ax.set_xticks(np.arange(len(models)) + width * (len(modes) - 1) / 2)
        ax.set_xticklabels(models, rotation=20)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    return _save(fig, cfg, "summary_bars.png")


def plot_confusion_matrices(cfg: dict) -> list[Path]:
    """Row-normalised confusion matrices (mean over seeds) at the validation-tuned threshold."""
    runs = _load_runs(cfg)
    outs = []
    for mode in runs["mode"].unique():
        models = [m for m in MODEL_ORDER if ((runs["mode"] == mode) & (runs["model"] == m)).any()]
        fig, axes = plt.subplots(1, len(models), figsize=(3.1 * len(models), 3.2))
        for ax, model in zip(np.atleast_1d(axes), models):
            r = runs[(runs["mode"] == mode) & (runs["model"] == model)][["tn", "fp", "fn", "tp"]].mean()
            cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
            norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:.0f}\n({norm[i, j]:.1%})", ha="center", va="center",
                            color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
            ax.set_xticks([0, 1], ["pred normal", "pred fraud"], fontsize=7)
            ax.set_yticks([0, 1], ["normal", "fraud"], fontsize=7)
            ax.set_title(model)
        outs.append(_save(fig, cfg, f"confusion_{mode}.png"))
    return outs


def plot_neighborhood_graphs(cfg: dict) -> list[Path]:
    """networkx drawing of the sampled neighbourhood of flagged frauds, edge width = GAT attention."""
    import networkx as nx

    outs = []
    for p in sorted(_res(cfg).glob("attention_examples_*.json")):
        ex = json.loads(p.read_text())
        items = ex["examples"][:6]
        if not items:
            continue
        cols = min(3, len(items))
        rows = int(np.ceil(len(items) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows))
        for ax, it in zip(np.atleast_1d(axes).ravel(), items):
            g = nx.MultiDiGraph()
            for e in it["edges"]:
                g.add_edge(e["src"], e["dst"], att=e["attention"], fraud=e["is_fraud"])
            g.add_edge(it["target"]["src"], it["target"]["dst"], att=1.0, fraud=1, target=True)
            pos = nx.spring_layout(nx.Graph(g), seed=0)
            nx.draw_networkx_nodes(g, pos, ax=ax, node_size=90, node_color=BLUE, alpha=0.9)
            seeds = [it["target"]["src"], it["target"]["dst"]]
            nx.draw_networkx_nodes(g, pos, nodelist=seeds, ax=ax, node_size=170, node_color=RED)
            for u, v, k, d in g.edges(keys=True, data=True):
                is_t = d.get("target", False)
                ax.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(
                    arrowstyle="-|>", color=RED if (d["fraud"] or is_t) else GREY,
                    lw=0.6 + 5 * float(d["att"]), alpha=0.9 if is_t else 0.7,
                    linestyle="--" if is_t else "-", shrinkA=5, shrinkB=5,
                    connectionstyle=f"arc3,rad={0.12 * (k + 1)}"))
            ax.set_title(f"tx {it['target']['tx_id']}  score {it['score']:.2f}", fontsize=9)
            ax.axis("off")
        for ax in np.atleast_1d(axes).ravel()[len(items):]:
            ax.axis("off")
        fig.suptitle("Past-only neighbourhood of flagged frauds (red nodes = target accounts; edge width = attention; red edge = past fraud)",
                     fontsize=10)
        outs.append(_save(fig, cfg, p.stem.replace("attention_examples", "neighborhood") + ".png"))
    return outs


def main() -> None:
    """CLI entry point: regenerate every figure whose inputs exist."""
    args = build_parser("Generate result figures from results/").parse_args()
    cfg = resolve_config(args)
    jobs = [plot_fraud_by_type, plot_fraud_over_time, plot_degree_distributions, plot_pr_curves,
            plot_summary_bars, plot_confusion_matrices, plot_neighborhood_graphs]
    for job in jobs:
        try:
            job(cfg)
        except FileNotFoundError as e:
            log.warning("skipping %s: %s", job.__name__, e)


if __name__ == "__main__":
    main()
