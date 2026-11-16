"""Part 1: criminal transactions on Elliptic++ with four models.

* XGB        XGBoost on the transaction's own features (93 local + 15 augmented)
* XGB+graph  XGBoost on own + pre-computed neighbourhood features (72 aggregated + in/out degree)
* GAT        GATv2 on the transaction graph, own features only (learns the neighbourhood itself)
* Hybrid     XGB+graph plus the GAT's score as one extra feature (stacked: training rows get out-of-fold
             GAT scores from models trained on other time-step blocks; validation/test rows get the score of
             the GAT trained on the whole training split)

Split by time: train steps 1-29, validation 30-34, test 35-49. Every edge stays inside one time step.
python -m src.btc.transactions  ->  results/btc/transactions/
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines import fit_xgb
from ..evaluate import completed_runs, evaluate_run, markdown_table, save_run, summarize
from ..utils import build_parser, ensure_dir, get_logger, resolve_config
from .data import TEST, TRAIN, VAL, load_transactions
from .gnn import out_of_fold_scores, standardize, train_node_gat

log = get_logger()
MODELS = ["XGB", "XGB+graph", "GAT", "Hybrid"]
MODE = "transactions"


def time_blocks(step: np.ndarray, n: int) -> np.ndarray:
    """Fold id per row: ``n`` contiguous blocks of time steps."""
    edges = np.quantile(np.unique(step), np.linspace(0, 1, n + 1)[1:-1])
    return np.searchsorted(edges, step, side="right")


def fit_predict(m: str, t, cfg: dict, seed: int, cache: dict) -> tuple[np.ndarray, dict]:
    """Fit model ``m`` for ``seed`` and return scores for every transaction (``cache`` shares the GAT run)."""
    bc = cfg["btc"]
    tr, va = np.flatnonzero(t.split == TRAIN), np.flatnonzero(t.split == VAL)
    y = t.y.astype(np.int64)
    X_own, X_all = t.own, np.hstack([t.own, t.graph])
    if m in ("GAT", "Hybrid") and "gat" not in cache:
        cache["X_gnn"] = standardize(t.own, tr)
        cache["gat"] = train_node_gat(cache["X_gnn"], t.edge_index, y, tr, va, bc["gat"], seed)
    if m == "XGB":
        mdl, extra = fit_xgb(X_own[tr], y[tr], X_own[va], y[va], cfg, seed)
        return mdl.predict_proba(X_own)[:, 1], extra
    if m == "XGB+graph":
        mdl, extra = fit_xgb(X_all[tr], y[tr], X_all[va], y[va], cfg, seed)
        return mdl.predict_proba(X_all)[:, 1], extra
    gat_s, n_ep = cache["gat"]
    if m == "GAT":
        return gat_s, {"best_iteration": n_ep}
    folds = time_blocks(t.step[tr], bc["hybrid_folds"])
    g = gat_s.copy()
    g[tr] = out_of_fold_scores(cache["X_gnn"], t.edge_index, y, tr, va, folds, bc["gat"], seed)
    Xh = np.hstack([X_all, g[:, None]])
    mdl, extra = fit_xgb(Xh[tr], y[tr], Xh[va], y[va], cfg, seed)
    return mdl.predict_proba(Xh)[:, 1], extra


def run(cfg: dict) -> pd.DataFrame:
    """Train and evaluate the four models for every seed; finished (model, seed) runs are skipped."""
    t = load_transactions(cfg["btc"]["data_dir"])
    rdir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "btc" / MODE)
    done = set() if cfg.get("rerun") else completed_runs(rdir)
    va, te = np.flatnonzero(t.split == VAL), np.flatnonzero(t.split == TEST)
    y = t.y.astype(np.int64)
    targets = cfg["metrics"]["precision_targets"]
    for seed in cfg["seeds"]:
        cache: dict = {}
        for m in [m for m in MODELS if (m, MODE, seed) not in done]:
            t0 = time.time()
            s, extra = fit_predict(m, t, cfg, seed, cache)
            row = evaluate_run(m, MODE, seed, y[va], s[va], y[te], s[te], targets,
                               train_seconds=round(time.time() - t0, 1), **extra)
            save_run(rdir, row, t.tx[te], y[te], s[te])
            log.info("[transactions] %s seed %d: PR-AUC %.4f R@P90 %.4f F1 %.4f", m, seed, row["pr_auc"],
                     row["recall_at_p90"], row["f1"])
    out = summarize(rdir)
    print(markdown_table(out))
    return out


def main() -> None:
    """CLI entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run(resolve_config(build_parser("Elliptic++ part 1: criminal transactions").parse_args()))


if __name__ == "__main__":
    main()
