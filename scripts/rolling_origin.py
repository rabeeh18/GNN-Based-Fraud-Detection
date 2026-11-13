"""Rolling-origin check of the PaySim result: is the GAT's gain over B1 stable across time?

The main split (train <= 500, val 501-620, test 621-743) gives one test window, and there the validation and
test splits disagree about GAT vs B1. This script re-trains B0, B1, MLP and GAT from scratch on earlier
windows (train up to T, validation the next 80 steps, test the 80 steps after that), without balance
features, and reports the paired GAT-vs-B1 difference per window on validation and test.

python -m scripts.rolling_origin --data_path data/PS_20174392719_1491204439457_log.csv
  -> <results_dir>/rolling/o<T>/metrics_per_run.csv for each window, and <results_dir>/rolling/summary.csv
Finished runs are skipped on a rerun, like every other entry point.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import run_baselines
from src.data import prepare_dataset
from src.train_gat import run_gat
from src.utils import build_parser, ensure_dir, get_logger, resolve_config

log = get_logger()

WINDOW = 80                                 # length of the validation and of the test window, in steps
TRAIN_ENDS = [300, 380, 460]                # three windows ending before the main split's test period
MODELS = ["B0", "B1", "MLP", "GAT"]


def window_cfg(cfg: dict, train_end: int) -> dict:
    """Config for one window, with its own results folder."""
    c = copy.deepcopy(cfg)
    c["split"].update(train_end_step=train_end, val_end_step=train_end + WINDOW,
                      test_end_step=train_end + 2 * WINDOW)
    c["use_balance"] = False
    c["baselines"]["models"] = ["B0", "B1"]
    c["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]) / "rolling" / f"o{train_end}")
    return c


def summarize_windows(cfg: dict, train_ends: list[int]) -> pd.DataFrame:
    """One row per window and model (mean/std over seeds), plus the paired GAT - B1 difference."""
    rows = []
    for t in train_ends:
        runs = pd.read_csv(Path(cfg["paths"]["results_dir"]) / "rolling" / f"o{t}" / "metrics_per_run.csv")
        runs = runs[(runs["mode"] == "nobalance") & runs["model"].isin(MODELS)]
        piv = {k: runs.pivot(index="seed", columns="model", values=k) for k in ("pr_auc", "val_pr_auc")}
        for m in MODELS:
            if m not in piv["pr_auc"]:
                continue
            rows.append({"train_end": t, "test_steps": f"{t + WINDOW + 1}-{t + 2 * WINDOW}", "model": m,
                         "test_pr_auc": piv["pr_auc"][m].mean(), "test_std": piv["pr_auc"][m].std(),
                         "val_pr_auc": piv["val_pr_auc"][m].mean(), "n_seeds": int(piv["pr_auc"][m].notna().sum()),
                         "n_test_fraud": int(runs.loc[runs["model"] == m, "n_test_fraud"].iloc[0])})
        if {"GAT", "B1"} <= set(piv["pr_auc"]):
            for k, name in (("pr_auc", "test"), ("val_pr_auc", "val")):
                d = (piv[k]["GAT"] - piv[k]["B1"]).dropna()
                rows.append({"train_end": t, "test_steps": f"{t + WINDOW + 1}-{t + 2 * WINDOW}",
                             "model": f"GAT-B1 ({name})", "test_pr_auc": d.mean(), "test_std": d.std(),
                             "n_seeds": len(d), "gat_wins": int((d > 0).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(ensure_dir(Path(cfg["paths"]["results_dir"]) / "rolling") / "summary.csv", index=False)
    return out


def main() -> None:
    """Run every window (B0/B1 baselines, MLP control, GAT) and print the summary."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = build_parser("Rolling-origin evaluation of GAT vs B1 on PaySim (no balance features)")
    p.add_argument("--train_ends", type=int, nargs="+", default=TRAIN_ENDS)
    args = p.parse_args()
    cfg = resolve_config(args)
    if args.seeds is None:
        cfg["seeds"] = [0, 1, 2]
    for t in args.train_ends:
        c = window_cfg(cfg, t)
        log.info("=== window: train <= %d, val %d-%d, test %d-%d ===", t, t + 1, t + WINDOW, t + WINDOW + 1,
                 t + 2 * WINDOW)
        ds = prepare_dataset(c)
        run_baselines(c, ds)
        run_gat(c, ds, no_graph=True)
        run_gat(c, ds)
    s = summarize_windows(cfg, args.train_ends)
    log.info("saved %s", plot_windows(cfg, args.train_ends))
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()


def plot_windows(cfg: dict, train_ends: list[int]) -> Path:
    """Test PR-AUC per window and model (seeds 0-2), including the main split as the last window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plots import MODEL_COLORS

    rd = Path(cfg["paths"]["results_dir"])
    frames = []
    for t in train_ends:
        r = pd.read_csv(rd / "rolling" / f"o{t}" / "metrics_per_run.csv")
        frames.append(r.assign(window=f"test {t + WINDOW + 1}-{t + 2 * WINDOW}"))
    main = pd.read_csv(rd / "metrics_per_run.csv")
    frames.append(main[main["seed"].isin([0, 1, 2])].assign(window="test 621-743\n(main split)"))
    runs = pd.concat(frames)
    runs = runs[(runs["mode"] == "nobalance") & runs["model"].isin(MODELS)]
    g = runs.groupby(["window", "model"], sort=False)["pr_auc"].agg(["mean", "std"]).reset_index()
    wins = list(dict.fromkeys(g["window"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(MODELS)
    for i, m in enumerate(MODELS):
        s = g[g["model"] == m].set_index("window").reindex(wins)
        ax.bar(np.arange(len(wins)) + i * w, s["mean"], w, yerr=s["std"], capsize=2, color=MODEL_COLORS[m], label=m)
    ax.set_xticks(np.arange(len(wins)) + w * (len(MODELS) - 1) / 2)
    ax.set_xticklabels(wins, fontsize=8)
    ax.set_ylabel("test PR-AUC (no balance features)")
    ax.set_ylim(0, 0.8)
    ax.set_title("Rolling-origin check: B1 wins 3 of 4 time windows; GAT only in the main split")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    out = ensure_dir(cfg["paths"]["figures_dir"]) / "rolling_origin.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
