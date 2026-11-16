"""Bitcoin (Elliptic++) part on a tiny fake copy of the raw files: loading, past-only wallet features, window
labels, and the transaction / wallet pipelines end to end."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils import ROOT


def _fake(root, n_tx=900, n_addr=500, seed=0, drop_after=None):
    """Write fake Elliptic++ files. ``drop_after`` removes every transaction after that step."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    tx = np.arange(1000, 1000 + n_tx)
    step = np.sort(rng.integers(1, 50, n_tx))
    bad = rng.random(n_tx) < 0.12
    cls = np.where(rng.random(n_tx) < 0.35, 3, np.where(bad, 1, 2))
    f = pd.DataFrame({"txId": tx, "Time step": step})
    for i in range(1, 6):
        f[f"Local_feature_{i}"] = rng.normal(size=n_tx) + bad * 1.5
    for i in range(1, 3):
        f[f"Aggregate_feature_{i}"] = rng.normal(size=n_tx)
    f["in_txs_degree"], f["out_txs_degree"] = rng.integers(0, 3, n_tx), rng.integers(0, 3, n_tx)
    f["total_BTC"] = rng.exponential(size=n_tx)
    e = [(tx[i], tx[j]) for s in np.unique(step) for idx in [np.flatnonzero(step == s)]
         for i, j in zip(idx[:-1], idx[1:])]
    addr = np.array([f"addr{i:04d}" for i in range(n_addr)])
    a_in = pd.DataFrame({"input_address": addr[rng.integers(0, n_addr, n_tx)], "txId": tx})
    a_out = pd.DataFrame({"txId": np.repeat(tx, 2), "output_address": addr[rng.integers(0, n_addr, 2 * n_tx)]})
    if drop_after is not None:
        keep = set(tx[step <= drop_after])
        f, cls = f[f["txId"].isin(keep)], cls[step <= drop_after]
        e = [p for p in e if p[0] in keep]
        a_in, a_out = a_in[a_in["txId"].isin(keep)], a_out[a_out["txId"].isin(keep)]
    f.to_csv(root / "txs_features.csv", index=False)
    pd.DataFrame({"txId": f["txId"], "class": cls}).to_csv(root / "txs_classes.csv", index=False)
    pd.DataFrame(e, columns=["txId1", "txId2"]).to_csv(root / "txs_edgelist.csv", index=False)
    a_in.to_csv(root / "AddrTx_edgelist.csv", index=False)
    a_out.to_csv(root / "TxAddr_edgelist.csv", index=False)
    return root


@pytest.fixture()
def cfg(tmp_path):
    c = copy.deepcopy(yaml.safe_load(open(ROOT / "configs" / "default.yaml")))
    c["paths"]["results_dir"] = str(tmp_path / "res")
    c["seeds"] = [0]
    c["btc"]["data_dir"] = str(_fake(tmp_path / "epp"))
    c["btc"]["gat"].update(epochs=4, patience=4, hidden=16)
    c["btc"]["wallet_gat"] = {"hidden": 8}
    c["btc"]["hybrid_folds"] = 2
    c["baselines"]["xgb"].update(n_estimators=10, device="cpu")
    return c


def test_transactions_load(cfg):
    from src.btc.data import TX_GRAPH_EXTRA, load_transactions

    t = load_transactions(cfg["btc"]["data_dir"])
    assert not set(TX_GRAPH_EXTRA) & set(t.own_cols) and set(TX_GRAPH_EXTRA) <= set(t.graph_cols)
    assert (t.step[t.edge_index[0]] == t.step[t.edge_index[1]]).all()
    assert (t.split[t.y < 0] == -1).all()


def test_wallet_features_are_past_only(cfg, tmp_path):
    """Removing every transaction after step 30 must not change any occurrence at step <= 30."""
    from src.btc.data import load_transactions
    from src.btc.wallets import build_wallet_graph

    full_root = cfg["btc"]["data_dir"]
    cut_root = _fake(tmp_path / "cut", drop_after=30)
    a, b = load_transactions(full_root), load_transactions(cut_root)
    wa, wb = build_wallet_graph(a, full_root), build_wallet_graph(b, cut_root)

    def table(w):
        df = pd.DataFrame(np.hstack([w.own, w.graph]))
        df["address"], df["step"] = w.address[w.occ_wallet], w.occ_step
        return df[df["step"] <= 30].set_index(["address", "step"]).sort_index()

    pd.testing.assert_frame_equal(table(wa), table(wb), check_dtype=False)


def test_window_labels_ignore_later_transactions(cfg):
    """A training wallet's label only uses transaction labels up to the end of the training period (step 29)."""
    from src.btc.data import TRAIN, TRAIN_END, load_transactions
    from src.btc.wallets import build_wallet_graph

    t = load_transactions(cfg["btc"]["data_dir"])
    wg = build_wallet_graph(t, cfg["btc"]["data_dir"])
    w_of_inc = wg.occ_wallet[wg.inc_occ]
    early_ill = np.zeros(len(wg.address), bool)
    m = (t.step[wg.inc_tx] <= TRAIN_END) & (t.y[wg.inc_tx] == 1)
    early_ill[w_of_inc[m]] = True
    tr = wg.wallet_split == TRAIN
    assert ((wg.wallet_label[tr] == 1) == early_ill[tr]).all()


def test_pipelines_run_and_resume(cfg, monkeypatch):
    import src.btc.transactions as bt
    import src.btc.wallets as bw
    from src.btc.data import load_transactions

    out = bt.run(cfg)
    assert set(out["model"]) == {"XGB", "XGB+graph", "GAT", "Hybrid"}
    t = load_transactions(cfg["btc"]["data_dir"])
    wg = bw.build_wallet_graph(t, cfg["btc"]["data_dir"])
    for task in ("wallets", "wallets_early"):
        assert set(bw.run_task(cfg, t, wg, task)["model"]) == {"XGB", "XGB+graph", "GAT", "Hybrid"}
    monkeypatch.setattr(bw, "train_node_gat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrained")))
    bw.run_task(cfg, t, wg, "wallets")
