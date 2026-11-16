"""Parts 2 and 3: criminal wallets on Elliptic++, with honest (past-only) features, and early warning.

Why not use ``wallets_features.csv`` directly: its 56 features are lifetime totals. A wallet's row at time step
25 already holds its total transaction count, last active block, lifetime and so on up to the end of the data,
so a model trained on them sees the future. Instead, every feature here is rebuilt from the address-transaction
links (``AddrTx_edgelist.csv``, ``TxAddr_edgelist.csv``) using only transactions up to the time step in question.

Unit of prediction: a *wallet occurrence* = (wallet, time step in which it was active). Its features use only
time steps <= that step. The graph joins each occurrence to the transactions it took part in at that step, and
transactions to each other (Elliptic money flow, always inside one step), so the graph never reaches forward.

Splits by the wallet's FIRST appearance: train 1-29, validation 30-34, test 35-49, so a wallet is never in two
splits. A wallet's window runs from its first appearance to the end of its split. Its label is recomputed from
the transaction labels inside that window (the Elliptic++ rule "illicit if it touched an illicit transaction"):
illicit if any of its window transactions is illicit, licit if it has a licit one and no illicit one, else unknown.
Training labels therefore never use transaction labels from the validation or test periods.

Two tasks, same four models (XGB, XGB+graph, GAT, Hybrid), each trained for its own task:
* ``wallets``        the wallet at its last activity inside its window (everything known by then)
* ``wallets_early``  the wallet at its FIRST appearance (early warning: is this new wallet criminal?)

python -m src.btc.wallets  ->  results/btc/wallets/ and results/btc/wallets_early/
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines import fit_xgb
from ..evaluate import completed_runs, evaluate_run, markdown_table, save_run, summarize
from ..utils import build_parser, ensure_dir, get_logger, resolve_config, resolve_path
from .data import TEST, TRAIN, TRAIN_END, VAL, VAL_END, Transactions, load_transactions, step_split
from .gnn import out_of_fold_scores, standardize, train_node_gat
from .transactions import time_blocks

log = get_logger()
MODELS = ["XGB", "XGB+graph", "GAT", "Hybrid"]
WINDOW_END = {TRAIN: TRAIN_END, VAL: VAL_END, TEST: 49}

OWN_COLS = ["n_sent_step", "n_recv_step", "n_tx_step", "cum_sent", "cum_recv", "cum_tx", "steps_active",
            "age_steps", "gap_steps", "is_first", "counterparties_step", "cum_counterparties"]


@dataclass
class WalletGraph:
    """Everything the wallet experiments and the demo need."""

    address: np.ndarray        # address string per wallet id
    occ_wallet: np.ndarray     # wallet id per occurrence
    occ_step: np.ndarray       # time step per occurrence
    own: np.ndarray            # [n_occ, len(OWN_COLS)] past-only wallet features
    graph: np.ndarray          # [n_occ, k] aggregates of the occurrence's transactions at that step
    graph_cols: list[str]
    inc_occ: np.ndarray        # incidence list: occurrence id ...
    inc_tx: np.ndarray         # ... transaction row ...
    inc_role: np.ndarray       # ... 0 = sender (input), 1 = receiver (output)
    wallet_split: np.ndarray   # split of each wallet by first appearance
    wallet_label: np.ndarray   # 1 / 0 / -1 inside the wallet's window
    first_occ: np.ndarray      # occurrence id of each wallet's first appearance
    last_occ: np.ndarray       # occurrence id of each wallet's last appearance inside its window


def build_wallet_graph(t: Transactions, root: str | Path) -> WalletGraph:
    """Join wallets to transactions, build occurrences, past-only features and window labels."""
    root = resolve_path(root)
    pos = t.index_of()
    a_in = pd.read_csv(root / "AddrTx_edgelist.csv")
    a_out = pd.read_csv(root / "TxAddr_edgelist.csv")
    inc = pd.concat([pd.DataFrame({"address": a_in["input_address"], "tx": pos.reindex(a_in["txId"]).to_numpy(), "role": 0}),
                     pd.DataFrame({"address": a_out["output_address"], "tx": pos.reindex(a_out["txId"]).to_numpy(), "role": 1})],
                    ignore_index=True)
    if inc["tx"].isna().any():
        raise ValueError("address edge list refers to unknown txId")
    inc["tx"] = inc["tx"].astype(np.int64)
    inc["step"] = t.step[inc["tx"].to_numpy()]
    wid, address = pd.factorize(inc["address"])
    inc["w"] = wid
    inc = inc.drop(columns="address")

    # occurrences = (wallet, step)
    occ = inc[["w", "step"]].drop_duplicates().sort_values(["w", "step"]).reset_index(drop=True)
    occ["occ"] = np.arange(len(occ))
    inc = inc.merge(occ, on=["w", "step"])
    first = occ.groupby("w")["step"].transform("min").to_numpy()
    wallet_first = occ.groupby("w")["step"].min().to_numpy()
    wsplit = step_split(wallet_first)

    # own features (counts per occurrence, then cumulative over the wallet's occurrences in time order)
    per = inc.assign(n_sent_step=(inc["role"] == 0).astype(int), n_recv_step=(inc["role"] == 1).astype(int))         .groupby("occ")[["n_sent_step", "n_recv_step"]].sum()
    occ = occ.join(per, on="occ")
    occ["n_tx_step"] = inc.groupby("occ")["tx"].nunique().reindex(occ["occ"]).to_numpy()
    g = occ.groupby("w")
    occ["cum_sent"] = g["n_sent_step"].cumsum()
    occ["cum_recv"] = g["n_recv_step"].cumsum()
    occ["cum_tx"] = g["n_tx_step"].cumsum()
    occ["steps_active"] = g.cumcount() + 1
    occ["age_steps"] = occ["step"] - first
    occ["gap_steps"] = (occ["step"] - g["step"].shift(1)).fillna(0)
    occ["is_first"] = (occ["steps_active"] == 1).astype(int)
    tx_size = inc.groupby("tx")["w"].nunique()
    inc["others"] = tx_size.reindex(inc["tx"]).to_numpy() - 1
    occ["counterparties_step"] = inc.groupby("occ")["others"].sum().reindex(occ["occ"]).to_numpy()
    occ["cum_counterparties"] = occ.groupby("w")["counterparties_step"].cumsum()
    own = occ[OWN_COLS].to_numpy(np.float32)

    # graph features: mean and max of the occurrence's transactions' own features + mean of their graph features
    tx_rows = inc["tx"].to_numpy()
    occ_ids = inc["occ"].to_numpy()
    n_occ = len(occ)
    own_df = pd.DataFrame(t.own[tx_rows])
    gb = own_df.groupby(occ_ids)
    mean_own = gb.mean().reindex(np.arange(n_occ)).to_numpy(np.float32)
    max_own = gb.max().reindex(np.arange(n_occ)).to_numpy(np.float32)
    mean_graph = pd.DataFrame(t.graph[tx_rows]).groupby(occ_ids).mean().reindex(np.arange(n_occ)).to_numpy(np.float32)
    del own_df, gb
    graph = np.hstack([mean_own, max_own, mean_graph])
    graph_cols = ([f"tx_mean_{c}" for c in t.own_cols] + [f"tx_max_{c}" for c in t.own_cols] +
                  [f"tx_mean_{c}" for c in t.graph_cols])

    # labels inside each wallet's window [first appearance, end of its split]
    wend = np.vectorize(WINDOW_END.get)(wsplit)
    in_win = inc["step"].to_numpy() <= wend[inc["w"].to_numpy()]
    ty = t.y[tx_rows]
    w_arr = inc["w"].to_numpy()
    n_w = len(address)
    has_ill = np.zeros(n_w, bool)
    has_lic = np.zeros(n_w, bool)
    np.logical_or.at(has_ill, w_arr[in_win & (ty == 1)], True)
    np.logical_or.at(has_lic, w_arr[in_win & (ty == 0)], True)
    wlabel = np.where(has_ill, 1, np.where(has_lic, 0, -1)).astype(np.int8)

    occ_in_win = occ["step"].to_numpy() <= wend[occ["w"].to_numpy()]
    first_occ = occ.groupby("w")["occ"].min().to_numpy()
    last_occ = occ[occ_in_win].groupby("w")["occ"].max().reindex(np.arange(n_w)).to_numpy()
    log.info("wallets: %d, occurrences: %d, address-transaction links: %d; window labels: %d illicit, %d licit, %d unknown",
             n_w, n_occ, len(inc), int((wlabel == 1).sum()), int((wlabel == 0).sum()), int((wlabel == -1).sum()))
    return WalletGraph(np.asarray(address), occ["w"].to_numpy(), occ["step"].to_numpy(), own, graph, graph_cols,
                       occ_ids, tx_rows, inc["role"].to_numpy(), wsplit, wlabel, first_occ, last_occ.astype(np.int64))


def gnn_inputs(t: Transactions, wg: WalletGraph, fit_rows_occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One graph with transaction nodes [0, n_tx) and occurrence nodes [n_tx, n_tx + n_occ).

    Node features are block-diagonal (transaction features | wallet features | node-type flags), so the GAT's
    first linear layer acts as a separate input projection per node type. Standardisation uses training rows only.
    """
    n_tx, n_occ = len(t.tx), len(wg.occ_wallet)
    tx_fit = np.flatnonzero(t.step <= TRAIN_END)
    Xt = standardize(t.own, tx_fit)
    Xo = standardize(wg.own, fit_rows_occ)
    X = np.zeros((n_tx + n_occ, Xt.shape[1] + Xo.shape[1] + 2), np.float32)
    X[:n_tx, :Xt.shape[1]] = Xt
    X[n_tx:, Xt.shape[1]:Xt.shape[1] + Xo.shape[1]] = Xo
    X[:n_tx, -2] = 1.0
    X[n_tx:, -1] = 1.0
    ei = np.hstack([t.edge_index, np.stack([wg.inc_occ + n_tx, wg.inc_tx])])
    return X, ei.astype(np.int64)


def run_task(cfg: dict, t: Transactions, wg: WalletGraph, task: str) -> pd.DataFrame:
    """Train/evaluate the four models for ``wallets`` (last activity) or ``wallets_early`` (first appearance)."""
    bc = cfg["btc"]
    gc = {**bc["gat"], **bc.get("wallet_gat", {})}
    rdir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "btc" / task)
    done = set() if cfg.get("rerun") else completed_runs(rdir)
    target_occ = wg.last_occ if task == "wallets" else wg.first_occ
    lab = wg.wallet_label
    idx = {s: np.flatnonzero((wg.wallet_split == s) & (lab >= 0)) for s in (TRAIN, VAL, TEST)}
    tr_w, va_w, te_w = idx[TRAIN], idx[VAL], idx[TEST]
    y = lab.astype(np.int64)
    targets = cfg["metrics"]["precision_targets"]
    Xo = target_occ                                   # occurrence id per wallet
    X_own = wg.own[Xo]
    X_all = np.hstack([X_own, wg.graph[Xo]])
    n_tx = len(t.tx)
    occ_fit = np.flatnonzero(wg.occ_step <= TRAIN_END)
    X_gnn, ei = gnn_inputs(t, wg, occ_fit)
    y_nodes = np.full(len(X_gnn), -1, np.int64)
    y_nodes[n_tx + Xo] = y                            # each wallet's label on its target occurrence node
    node = n_tx + Xo                                  # graph node per wallet
    folds = time_blocks(wg.occ_step[Xo[tr_w]], bc.get("wallet_hybrid_folds", bc["hybrid_folds"]))
    log.info("[%s] train/val/test wallets with labels: %d/%d/%d (illicit %d/%d/%d)", task, len(tr_w), len(va_w),
             len(te_w), int(y[tr_w].sum()), int(y[va_w].sum()), int(y[te_w].sum()))
    for seed in cfg["seeds"]:
        todo = [m for m in MODELS if (m, task, seed) not in done]
        if not todo:
            continue
        gat_w = None
        for m in todo:
            t0 = time.time()
            extra: dict = {}
            if m in ("GAT", "Hybrid") and gat_w is None:
                s_nodes, n_ep = train_node_gat(X_gnn, ei, y_nodes, node[tr_w], node[va_w], gc, seed,
                                               tag=f"GAT[{task}]")
                gat_w = s_nodes[node]
            if m == "XGB":
                mdl, extra = fit_xgb(X_own[tr_w], y[tr_w], X_own[va_w], y[va_w], cfg, seed)
                s = mdl.predict_proba(X_own)[:, 1]
            elif m == "XGB+graph":
                mdl, extra = fit_xgb(X_all[tr_w], y[tr_w], X_all[va_w], y[va_w], cfg, seed)
                s = mdl.predict_proba(X_all)[:, 1]
            elif m == "GAT":
                s, extra = gat_w, {"best_iteration": n_ep}
            else:
                oof = out_of_fold_scores(X_gnn, ei, y_nodes, node[tr_w], node[va_w], folds, gc, seed)
                g = gat_w.copy()
                g[tr_w] = oof
                Xh = np.hstack([X_all, g[:, None]])
                mdl, extra = fit_xgb(Xh[tr_w], y[tr_w], Xh[va_w], y[va_w], cfg, seed)
                s = mdl.predict_proba(Xh)[:, 1]
            row = evaluate_run(m, task, seed, y[va_w], s[va_w], y[te_w], s[te_w], targets,
                               train_seconds=round(time.time() - t0, 1), **extra)
            save_run(rdir, row, wg.address[te_w], y[te_w], s[te_w])
            log.info("[%s] %s seed %d: PR-AUC %.4f R@P90 %.4f F1 %.4f", task, m, seed, row["pr_auc"],
                     row["recall_at_p90"], row["f1"])
    out = summarize(rdir)
    print(markdown_table(out))
    return out


def main() -> None:
    """CLI entry point: both wallet tasks."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = build_parser("Elliptic++ parts 2-3: criminal wallets and early warning")
    p.add_argument("--tasks", nargs="+", default=["wallets", "wallets_early"])
    args = p.parse_args()
    cfg = resolve_config(args)
    t = load_transactions(cfg["btc"]["data_dir"])
    wg = build_wallet_graph(t, cfg["btc"]["data_dir"])
    for task in args.tasks:
        run_task(cfg, t, wg, task)


if __name__ == "__main__":
    main()
