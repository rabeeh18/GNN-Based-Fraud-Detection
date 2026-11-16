"""Export the data behind the investigator app (app/investigator.py).

* The wallet model is the XGBoost-based model (XGB+graph or Hybrid) with the higher mean VALIDATION PR-AUC in
  results/btc/wallets; it is refitted with seed 0. XGBoost is used for the demo because its per-feature
  contributions give an explanation for every score.
* Transaction risk in the money trail comes from the transaction model with the higher mean validation PR-AUC in
  results/btc/transactions, refitted with seed 0 so that every transaction (labelled or not) gets a score.

python -m src.btc.export_demo  ->  results/btc/demo/{wallets,links,txs}.parquet, meta.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from ..baselines import fit_xgb
from ..evaluate import best_f1_threshold
from ..utils import build_parser, ensure_dir, get_logger, resolve_config
from .data import TEST, TRAIN, VAL, load_transactions
from .gnn import out_of_fold_scores, train_node_gat
from .transactions import fit_predict, time_blocks
from .wallets import OWN_COLS, build_wallet_graph, gnn_inputs

log = get_logger()
SPLIT_NAME = {TRAIN: "train", VAL: "val", TEST: "test"}


def pick(results: Path, candidates: list[str]) -> str:
    """Model with the highest mean validation PR-AUC among ``candidates``."""
    r = pd.read_csv(results / "metrics_per_run.csv")
    v = r[r["model"].isin(candidates)].groupby("model")["val_pr_auc"].mean()
    log.info("validation PR-AUC used for the choice:\n%s", v.round(4).to_string())
    return str(v.idxmax())


def main() -> None:
    """Refit the chosen wallet model, score every wallet, and write the demo tables."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = resolve_config(build_parser("Export data for the investigator app").parse_args())
    bc = cfg["btc"]
    res = Path(cfg["paths"]["results_dir"]) / "btc"
    wallet_model = pick(res / "wallets", ["XGB+graph", "Hybrid"])
    tx_model = pick(res / "transactions", ["XGB", "XGB+graph", "GAT", "Hybrid"])
    log.info("demo wallet model: %s, transaction model: %s", wallet_model, tx_model)

    t = load_transactions(bc["data_dir"])
    wg = build_wallet_graph(t, bc["data_dir"])
    occ = wg.last_occ
    lab = wg.wallet_label.astype(np.int64)
    tr = np.flatnonzero((wg.wallet_split == TRAIN) & (lab >= 0))
    va = np.flatnonzero((wg.wallet_split == VAL) & (lab >= 0))
    X = np.hstack([wg.own[occ], wg.graph[occ]])
    groups = ["own"] * len(OWN_COLS) + ["tx"] * len(wg.graph_cols)
    if wallet_model == "Hybrid":
        gc = {**bc["gat"], **bc.get("wallet_gat", {})}
        Xg, ei = gnn_inputs(t, wg, np.flatnonzero(wg.occ_step <= 29))
        n_tx = len(t.tx)
        y_nodes = np.full(len(Xg), -1, np.int64)
        y_nodes[n_tx + occ] = lab
        node = n_tx + occ
        s_nodes, _ = train_node_gat(Xg, ei, y_nodes, node[tr], node[va], gc, 0, tag="GAT[demo]")
        g = s_nodes[node]
        g[tr] = out_of_fold_scores(Xg, ei, y_nodes, node[tr], node[va],
                                   time_blocks(wg.occ_step[occ[tr]], bc.get("wallet_hybrid_folds", bc["hybrid_folds"])), gc, 0)
        X = np.hstack([X, g[:, None]])
        groups.append("gnn")
    model, _ = fit_xgb(X[tr], lab[tr], X[va], lab[va], cfg, 0)
    score = model.predict_proba(X)[:, 1]
    thr = best_f1_threshold(lab[va], score[va])
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]   # last column = bias
    gsum = pd.DataFrame(contrib).T.groupby(np.array(groups)).sum().T

    test_w = wg.wallet_split == TEST
    # links: every address-transaction link of test-period transactions (steps 35-49)
    keep = t.step[wg.inc_tx] >= 35
    links = pd.DataFrame({"address": wg.address[wg.occ_wallet[wg.inc_occ[keep]]], "txId": t.tx[wg.inc_tx[keep]],
                          "role": wg.inc_role[keep], "step": t.step[wg.inc_tx[keep]]})
    shown = np.zeros(len(wg.address), bool)
    shown[test_w] = True
    shown[pd.Index(wg.address).get_indexer(links["address"].unique())] = True
    wallets = pd.DataFrame({"address": wg.address, "score": score.astype(np.float32), "label": lab,
                            "split": pd.Series(wg.wallet_split).map(SPLIT_NAME).to_numpy(),
                            "first_step": wg.occ_step[wg.first_occ], "n_tx": wg.own[occ, OWN_COLS.index("cum_tx")],
                            "contrib_own": gsum["own"].to_numpy(np.float32), "contrib_tx": gsum["tx"].to_numpy(np.float32)})
    if "gnn" in gsum:
        wallets["contrib_gnn"] = gsum["gnn"].to_numpy(np.float32)
    wallets = wallets[shown]

    tx_score, _ = fit_predict(tx_model, t, cfg, 0, {})
    tx_runs = pd.read_csv(res / "transactions" / "metrics_per_run.csv")
    tx_thr = float(tx_runs[(tx_runs["model"] == tx_model) & (tx_runs["seed"] == 0)]["threshold"].iloc[0])
    test_tx = pd.DataFrame({"txId": t.tx, "step": t.step, "score": tx_score.astype(np.float32)})
    test_tx = test_tx[test_tx["step"] >= 35]
    out = ensure_dir(res / "demo")
    wallets.to_parquet(out / "wallets.parquet", index=False)
    links.to_parquet(out / "links.parquet", index=False)
    test_tx.to_parquet(out / "txs.parquet", index=False)
    (out / "meta.json").write_text(json.dumps({"model": wallet_model, "tx_model": tx_model, "threshold": float(thr),
                                               "tx_threshold": tx_thr}, indent=2))
    log.info("demo data: %d wallets (%d test), %d links, %d transactions -> %s", len(wallets),
             int((wallets["split"] == "test").sum()), len(links), len(test_tx), out)


if __name__ == "__main__":
    main()
